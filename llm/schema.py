#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""llm/schema.py —— 把 `tools/` 注册表渲染成喂给 LLM 的工具文档（只读、可版本化）。

为什么值得单独一个文件
====================
工具文档的长度**直接决定 prompt 长度**，而 prompt 长度决定成本与「模型记不记得住」。
VADAR 的 program prompt 实测已达 6965 字符（§13.3(2)），主要就是被工具说明吃掉的。

所以这里的取舍是明确的：**每个工具只给一行摘要 + 参数表**。
工具 docstring 里那些「为什么这么设计」的长段落是写给**人**看的，
它们留在源码里、留在方案文档里，但**不进 prompt** ——
`_first_paragraph()` 就是这条纪律的执行点（它只取第一个空行之前的内容）。

两个消费方：
    `agents/synthesizer.py`  渲染成文本进 system prompt（稳定前缀 → 可吃服务端缓存）
    `agents/synthesizer.py`  同时取 `tool_names()` / `params_of()` 做 AST 静态检查

⚠ 工具集变了必须升 `tools/version.py::TOOLS_VERSION` —— 否则「两次实验用的是不是
同一套工具」这个问题就答不了（§11 设计原则 5）。
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterator

__all__ = [
    "ToolDoc",
    "SEPARATOR",
    "first_paragraph",
    "iter_docs",
    "tool_docs",
    "params_of",
    "tool_names",
    "docs_text",
    "signature_line",
]

#: 摘要与参数表之间的缩进标记。选一个**中文里不会出现**的字符，免得被后处理吃掉。
SEPARATOR = " · "


def _ensure_loaded() -> None:
    """确保工具已注册。

    `tools.load_tools()` 只导入零 GPU 依赖的模块（geometry/spatial/scene_report），
    L1 感知工具由各自的实验臂按需导入 —— 所以这里不会把 torch 拉进进程。
    """
    import tools
    tools.load_tools()


def first_paragraph(doc: str | None) -> str:
    """取 docstring 的第一个段落，压成一行。

    这是「prompt 里不放设计理由」这条纪律的实现点。
    参数表里已经写清了默认值，所以摘要里**再重复一遍参数说明**是纯浪费。
    """
    if not doc:
        return ""
    for chunk in doc.strip().split("\n\n"):
        text = " ".join(line.strip() for line in chunk.splitlines()).strip()
        if text:
            return text
    return ""


def _fmt_default(value: Any) -> str:
    if value is inspect.Parameter.empty:
        return ""
    if isinstance(value, str):
        return "=%r" % value
    return "=%r" % (value,)


@dataclass(frozen=True)
class ToolDoc:
    """一个工具对 LLM 的可见面。"""

    name: str
    #: `((参数名, "参数名: 类型=默认值"), ...)` —— 第二个字段是渲染好的片段，
    #: 让「怎么显示」只有一处实现。
    params: tuple[tuple[str, str], ...]
    summary: str
    signature: str

    def param_names(self) -> tuple[str, ...]:
        return tuple(p[0] for p in self.params)

    def line(self) -> str:
        return "%s\n    %s" % (self.signature, self.summary or "（无摘要）")


def _render_param(p: inspect.Parameter) -> str:
    ann = ""
    if p.annotation is not inspect.Parameter.empty:
        ann = ": %s" % (p.annotation if isinstance(p.annotation, str) else _ann_str(p.annotation))
    return "%s%s%s" % (p.name, ann, _fmt_default(p.default))


def _ann_str(ann: Any) -> str:
    """把非字符串注解（`str | None` 在运行时是 `types.UnionType`）还原成可读文本。

    有了 `from __future__ import annotations`，注释其实都是字符串；
    但 `tools/` 里凡是**没写** future import 的模块就会给出真正的类型对象，
    两条路都要能渲染 —— 否则工具文档会随机地变成 `<class 'str'>`。
    """
    if isinstance(ann, type):
        return ann.__name__
    text = str(ann)
    return re.sub(r"<class '([\w.]+)'>", lambda m: m.group(1).split(".")[-1], text)


def iter_docs(tools: Sequence[str] | None = None) -> Iterator[ToolDoc]:
    """按名字排序遍历工具。排序是为了让 prompt 逐字节稳定 —— 可复现的前提。

    `tools` 是**动作空间的过滤名单**（默认全部已注册工具）。
    ⚠ 它必须与 `agents.executor.build_namespace(toolset=...)` 用的是同一份名单，
    否则会出现「提示词列了、运行期没有」的静默不一致（见 executor 里 `QA_TOOLSET` 的说明）。
    """
    _ensure_loaded()
    from tools.registry import TOOL_REGISTRY

    wanted = sorted(TOOL_REGISTRY) if tools is None else [n for n in tools if n in TOOL_REGISTRY]
    for name in wanted:
        fn: Callable[..., Any] = TOOL_REGISTRY[name]
        # 装饰器把 ctx 从 `__signature__` 里摘掉了 —— 所以这里拿到的是模型的视角。
        sig: inspect.Signature = getattr(fn, "__signature__", None) or inspect.signature(fn)
        params = tuple((p.name, _render_param(p)) for p in sig.parameters.values())
        sig_text = "%s(%s)" % (
            name,
            ", ".join(frag for _, frag in params),
        ) if params else "%s()" % name
        yield ToolDoc(
            name=name,
            params=params,
            summary=first_paragraph(inspect.getdoc(fn)),
            signature=sig_text,
        )


def tool_docs(tools: Sequence[str] | None = None) -> tuple[ToolDoc, ...]:
    return tuple(iter_docs(tools))


def tool_names() -> tuple[str, ...]:
    """工具白名单 —— AST 静态检查的第 2 项直接用它（§13.3(2)）。"""
    _ensure_loaded()
    from tools.registry import tool_names as _names

    return _names()


def params_of(name: str) -> tuple[str, ...]:
    """某工具的参数名 —— AST 静态检查的第 3 项（参数名是否存在于 schema）。"""
    _ensure_loaded()
    from tools.registry import tool_parameters

    return tool_parameters(name)


def signature_line(name: str) -> str:
    for d in iter_docs():
        if d.name == name:
            return d.signature
    return "%s(...)" % name


def docs_text(
    *,
    tools: Sequence[str] | None = None,
    modules: tuple[str, ...] = (),
    heading: str | None = None,
) -> str:
    """渲染成 prompt 里那一段文本。

    `modules` 是可 import 的模块白名单 —— 写在这里是因为「能 import 什么」
    与「能调用什么」是同一类信息，分成两处必然漂移。
    `tools` 是动作空间的过滤名单（见 `iter_docs`）。
    """
    lines: list[str] = []
    if heading:
        lines.append(heading)
    for d in iter_docs(tools):
        lines.append("  " + d.line().replace("\n", "\n  "))
    if modules:
        lines.append("")
        lines.append("可 import 的模块（其它一律不可用，且不是必需的）：%s" % ", ".join(modules))
    return "\n".join(lines)
