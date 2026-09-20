#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agents/synthesizer.py —— 角色① 程序合成。**唯一产出控制流的角色。**

它做两件事，都不需要执行程序：
    1. 把问题、场景清单、工具文档、上一次的失败原因拼成提示词，调一次 LLM
    2. 对拿回来的源码做 **AST 四项静态检查**（§13.3(2)）

为什么静态检查值钱
================
四项检查全部零成本、不需要 GPU、不需要场景：**能否 `ast.parse` / 函数名是否在工具白名单 /
参数名是否存在于 schema / 是否调用 `submit`**。

这让「模型守不守协议」变成一个**在花钱之前**就能回答的问题：

    · 对实验：PRTS（协议遵守率）可以在不执行程序、不建场景图的情况下测出来
    · 对 Phase 8 的数据合成：它是**自动验收闸门** —— 一条合成样本在入库前先过这四项
    · 对调试：把「模型不会用工具」与「几何算错了」彻底分开，这是两类完全不同的问题

第 3 项（参数名）在本项目里被**加强**了一条：**工具一律关键字传参**。
理由不是洁癖 —— 工具的第一个参数是 `scene_id`，写 `list_objects("chair")`
会把 `"chair"` 静默绑到 `scene_id` 上，得到一个语法正确、语义错位、
**运行起来不报错**的程序。这类错误靠运行期捕获是抓不到的，
只能在 AST 这层拦（或在运行期拿到一个莫名其妙的 `NOT_FOUND`）。
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from agents.executor import ALLOWED_MODULES, QA_TOOLSET
from agents.prompts import system as prompt_system

__all__ = [
    "StaticCheckReport",
    "extract_code",
    "static_check",
    "SynthesisResult",
    "build_messages",
    "synthesize",
]

#: 代码块提取。优先 fenced（模型照提示词做的时候）；没有 fence 也不报错 ——
#: 大量真实模型会直接吐裸代码，为这个丢掉一次调用不划算，所以兜底成"整段都是代码"。
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


# ============================================================================
# 1. 静态检查
# ============================================================================


@dataclass(frozen=True)
class StaticCheckReport:
    """四项检查的结果。`errors` 里的每一条都必须是**模型读完就能改**的句子。"""

    parse_ok: bool
    has_submit: bool
    tool_calls: tuple[str, ...] = ()
    unknown_calls: tuple[str, ...] = ()
    bad_kwargs: tuple[tuple[str, str], ...] = ()
    positional_tool_calls: tuple[str, ...] = ()
    bad_imports: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    syntax_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.parse_ok and not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "parse_ok": self.parse_ok,
            "has_submit": self.has_submit,
            "tool_calls": list(self.tool_calls),
            "n_tool_calls": len(self.tool_calls),
            "unknown_calls": list(self.unknown_calls),
            "bad_kwargs": [list(p) for p in self.bad_kwargs],
            "positional_tool_calls": list(self.positional_tool_calls),
            "bad_imports": list(self.bad_imports),
            "errors": list(self.errors),
            "syntax_error": self.syntax_error,
        }

    def feedback(self) -> str:
        """渲染成给模型看的一段话（进重试提示词）。"""
        if self.syntax_error:
            return "语法错误，程序无法编译：%s" % self.syntax_error
        return "\n".join("- %s" % e for e in self.errors)


def _defined_names(tree: ast.AST) -> set[str]:
    """程序自己定义的名字 —— 它们不是"未知函数"，调用它们完全合法。

    这一趟必须走得比较全，否则「自己写了个 helper 函数」会被误判成幻觉工具名，
    而那会让静态检查变成一个**假阳性机器**（模型每次都被拒，然后开始瞎改）。
    """
    names: set[str] = set()

    def bind_target(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                bind_target(elt)
        elif isinstance(node, ast.Starred):
            bind_target(node.value)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            args = getattr(node, "args", None)
            if args is not None:
                for a in list(getattr(args, "posonlyargs", [])) + list(args.args) \
                        + list(args.kwonlyargs):
                    names.add(a.arg)
                if args.vararg:
                    names.add(args.vararg.arg)
                if args.kwarg:
                    names.add(args.kwarg.arg)
        elif isinstance(node, ast.Lambda):
            for a in list(node.args.args) + list(node.args.kwonlyargs):
                names.add(a.arg)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for tgt in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                bind_target(tgt)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            bind_target(node.target)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            bind_target(node.optional_vars)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def static_check(
    source: str,
    *,
    tools: Sequence[str] | None = None,
    params: Mapping[str, Sequence[str]] | None = None,
    modules: Sequence[str] = ALLOWED_MODULES,
) -> StaticCheckReport:
    """AST 四项静态检查 + 两条本项目加强项。**不执行任何代码。**

    `tools` / `params` 留成参数（默认取 `QA_TOOLSET` + 注册表）是为了让测试能构造
    一个"世界上只有两个工具"的假环境 —— 检查逻辑本身与工具集解耦。
    ⚠ 默认值必须是 `QA_TOOLSET` 而不是"全部已注册工具"：否则 L5 元工具会被放行，
    而运行期命名空间里根本没有它们（见 `executor.build_namespace`）。
    """
    if tools is None or params is None:
        from llm.schema import params_of

        tools = list(QA_TOOLSET) if tools is None else list(tools)
        params = {n: params_of(n) for n in tools} if params is None else params
    tool_set = set(tools)

    try:
        tree = ast.parse(source or "")
    except SyntaxError as exc:
        return StaticCheckReport(
            parse_ok=False,
            has_submit=False,
            syntax_error="第 %s 行：%s" % (exc.lineno, exc.msg),
            errors=("程序有语法错误，无法编译（%s）。" % exc.msg,),
        )

    local_names = _defined_names(tree)
    tool_calls: list[str] = []
    unknown_calls: list[str] = []
    bad_kwargs: list[tuple[str, str]] = []
    positional: list[str] = []
    bad_imports: list[str] = []
    errors: list[str] = []
    has_submit = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in modules:
                    bad_imports.append(alias.name)
                    errors.append(
                        "第 %d 行：`import %s` 不在白名单内。可用：%s。"
                        "（空间量请用工具算，不要自己导库实现。）"
                        % (node.lineno, alias.name, ", ".join(modules))
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level and node.level > 0:
                bad_imports.append("." * node.level + (node.module or ""))
                errors.append("第 %d 行：不允许相对导入。" % node.lineno)
            elif root not in modules:
                bad_imports.append(node.module or "")
                errors.append(
                    "第 %d 行：`from %s import ...` 不在白名单内（可用：%s）。"
                    % (node.lineno, node.module, ", ".join(modules))
                )
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "submit":
                has_submit = True
                continue
            if name in tool_set:
                tool_calls.append(name)
                if node.args:
                    positional.append(name)
                    errors.append(
                        "第 %d 行：`%s()` 用了位置参数。工具**一律关键字传参** —— "
                        "第一个参数是 scene_id，位置传参会把值静默绑错参数。"
                        "正确写法示例：%s(%s)。"
                        % (node.lineno, name, name,
                           ", ".join("%s=..." % p for p in (params.get(name) or ("scene_id",))[:2]))
                    )
                allowed = set(params.get(name) or ())
                for kw in node.keywords:
                    if kw.arg is None:
                        errors.append("第 %d 行：`%s()` 用了 `**kwargs` 展开 —— 参数名无法静态核对。"
                                      % (node.lineno, name))
                    elif kw.arg not in allowed:
                        bad_kwargs.append((name, kw.arg))
                        errors.append(
                            "第 %d 行：`%s()` 没有参数 `%s`。可用参数：%s。"
                            % (node.lineno, name, kw.arg,
                               ", ".join(allowed) if allowed else "（无）")
                        )
            elif name in local_names or name in _BUILTIN_OK:
                continue
            else:
                unknown_calls.append(name)
                errors.append(
                    "第 %d 行：调用了未知函数 `%s()`。只能用上面的工具、"
                    "白名单内置函数，或你自己在本程序里定义的函数。" % (node.lineno, name)
                )

    if not has_submit:
        errors.append(
            "程序里没有调用 `submit()`。答案必须由 "
            "`submit(answer, target_ids=[...], evidence=[\"...\"])` 交出 —— "
            "没有它就没有答案（VADAR 用『命名空间里有没有 final_result』取答案，"
            "缺失时静默算成错答案，本项目明确判失败）。"
        )

    return StaticCheckReport(
        parse_ok=True,
        has_submit=has_submit,
        tool_calls=tuple(tool_calls),
        unknown_calls=tuple(unknown_calls),
        bad_kwargs=tuple(bad_kwargs),
        positional_tool_calls=tuple(positional),
        bad_imports=tuple(bad_imports),
        errors=tuple(errors),
    )


#: 静态检查放行的内置名。**故意比 executor 的 `SAFE_BUILTIN_NAMES` 宽一点**：
#: 这里多放行几个（`open` 之类仍不在内），是为了让"能写但运行会失败"的情况
#: 尽量少 —— 静态检查的价值在于准确，不在这里争严格。
#: 真正的边界在 `executor.build_namespace()`，那是唯一有安全含义的地方。
_BUILTIN_OK: frozenset[str] = frozenset(
    ("abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float",
     "format", "frozenset", "int", "isinstance", "issubclass", "len", "list", "map",
     "max", "min", "pow", "print", "range", "repr", "reversed", "round", "set",
     "slice", "sorted", "str", "sum", "tuple", "zip",
     "Exception", "ValueError", "KeyError", "IndexError", "TypeError",
     "ZeroDivisionError", "ArithmeticError", "AssertionError", "RuntimeError",
     "getattr", "setattr", "type", "sorted")
)


# ============================================================================
# 2. 提取代码
# ============================================================================


def extract_code(reply_text: str) -> tuple[str, bool]:
    """从模型回复里取出程序源码，返回 `(source, fenced)`。

    `fenced=False` 说明模型没按提示词给代码块。**不当成失败**，
    但要记录下来 —— 它是「提示词遵守率」的一个观测点，
    而且这类回复里经常混着解释性文字，会表现为语法错误，需要能区分。
    """
    text = reply_text or ""
    blocks = _FENCE_RE.findall(text)
    if blocks:
        return max(blocks, key=len).strip(), True
    return text.strip(), False


# ============================================================================
# 3. 合成
# ============================================================================


@dataclass(frozen=True)
class SynthesisResult:
    source: str
    fenced: bool
    check: StaticCheckReport
    reply: Any | None = None
    messages: tuple[Mapping[str, Any], ...] = ()

    @property
    def ok(self) -> bool:
        return self.check.ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "fenced": self.fenced,
            "chars": len(self.source),
            "check": self.check.to_dict(),
            "reply": self.reply.to_dict() if self.reply is not None else None,
        }


def build_messages(
    question: str,
    scene_hint: Mapping[str, Any],
    *,
    tool_docs_text: str | None = None,
    answer_type: str | None = None,
    feedback: str | None = None,
    previous_source: str | None = None,
    tools: Sequence[str] | None = None,
    plan_text: str | None = None,
) -> list[dict[str, Any]]:
    """拼消息。**工具文档放 system**（逐字节稳定 → 能吃服务端前缀缓存）。

    `tools` 只影响工具文档那一段（静态检查与运行期各自取同一份名单）。
    `plan_text` 是臂 G 的前置计划，插在 **user** 里 —— 它是每题都变的内容，
    放进 system 会把前缀缓存打碎（那会让「开计划」顺带改变成本结构，消融就不纯了）。
    """
    from llm.schema import docs_text

    docs = tool_docs_text if tool_docs_text is not None else docs_text(
        tools=tools if tools is not None else QA_TOOLSET,
        heading="（返回类型统一是 ToolResult：先判 res.ok 再读 res.value）",
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt_system.build_system_prompt(docs, ALLOWED_MODULES)},
        {"role": "user", "content": prompt_system.build_user_prompt(
            question, scene_hint, answer_type=answer_type, plan_text=plan_text)},
    ]
    if feedback:
        messages.append({"role": "assistant", "content": previous_source or ""})
        messages.append({"role": "user", "content": prompt_system.build_retry_prompt(
            feedback, previous_source or "")})
    return messages


def synthesize(
    client: Any,
    *,
    question: str,
    scene_hint: Mapping[str, Any],
    tool_docs_text: str | None = None,
    answer_type: str | None = None,
    feedback: str | None = None,
    previous_source: str | None = None,
    purpose: str = "synthesize",
    tools: Sequence[str] | None = None,
    plan_text: str | None = None,
) -> SynthesisResult:
    """调一次 LLM 生成程序，然后静态检查。**不做重试** —— 重试策略在 `loop.py`。

    为什么把重试留在循环层：重试需要知道「执行结果」这种循环层才有的信息，
    塞进这里会让 synthesizer 变成半个循环，而它本该只负责"生成 + 检查"。
    """
    messages = build_messages(
        question, scene_hint,
        tool_docs_text=tool_docs_text,
        answer_type=answer_type,
        feedback=feedback,
        previous_source=previous_source,
        tools=tools,
        plan_text=plan_text,
    )
    reply = client.chat(messages, purpose=purpose)
    source, fenced = extract_code(reply.text)
    check = static_check(source, tools=tools)
    return SynthesisResult(
        source=source,
        fenced=fenced,
        check=check,
        reply=reply,
        messages=tuple(messages),
    )
