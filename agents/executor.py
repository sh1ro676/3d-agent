#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agents/executor.py —— 执行一段**已经写好的**程序，取回答案与证据。

这个文件不认识 LLM。它只回答一个问题：
「把这段源码跑起来，它有没有按 `submit()` 契约交出答案？」

早期基线实现的执行入口就是这条路，上面有四个真实的坑，
这里每一条都改成一个**结构上不可能发生**的东西：

| 早期基线的做法 | 后果（本机实测） | 这里的做法 |
|---|---|---|
| 取答案靠命名空间里有没有 `final_result` 这个变量 | 缺失时给 `""` 然后**静默算错** | 认 `submit()` 调用；不调用 = 明确的 `no_submit` 失败 |
| `exec` 时给完整 `globals()`，`open` 可用 | 绝对路径被当转义字符（`\\3D`→`\\x03`）→ `EINVAL`，5 次重试全废 | 命名空间里**没有 `open`、没有 `__import__`（白名单）、没有 `ctx`** |
| `signal.alarm` 看门狗 | Windows 上 `AttributeError` | 后台定时器 + 异步异常注入（见 `_Watchdog`） |
| 异常原样回灌 traceback | 模型得自己猜怎么救 | 结构化 `ToolResult.error.code` + `recovery`，executor 只负责**定位到行号** |

「命名空间里没有 `ctx`」这一条值得单独说：`ToolContext` 手里握着整个 `SceneGraph`，
里面每个节点都有 `centroid_3d`。如果程序能读到 `ctx`，
那么「坐标只能来自工具返回值」就退化成一句提示词祈祷 ——
模型可以直接 `ctx.scene.node("chair_1").centroid_3d` 抄出真坐标。
**把 `ctx` 从命名空间里拿掉，信息约束才是真的**（§13.3(2)）。
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import io
import math
import threading
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = [
    "ALLOWED_MODULES",
    "SAFE_BUILTIN_NAMES",
    "ABSTAIN_TOKENS",
    "QA_TOOLSET",
    "META_TOOLS",
    "ProgramTimeout",
    "ProgramContractError",
    "Submission",
    "ExecOutcome",
    "UserProgramError",
    "build_namespace",
    "execute_program",
    "file_line_of",
]

#: 允许 `import` 的模块。全是纯计算，**没有** os / sys / subprocess / socket / pathlib。
#: 名单同时喂给提示词与 AST 静态检查 —— 三处（提示词、静态检查、运行期）共用同一个常量，
#: 否则「提示词说了能用、运行时却 ImportError」这种不一致一定会出现。
ALLOWED_MODULES: tuple[str, ...] = ("math", "statistics", "itertools", "functools", "collections", "re")

#: 暴露给程序的内置函数。★ **这是唯一的一份名单** —— `agents/synthesizer.py` 的静态检查
#: 直接 import 它。此前那边另有一份 `_BUILTIN_OK`、比这里宽 4 个名字，后果不是理论的：
#: `getattr(...)` **静态检查放行、运行期 `NameError`**，而失败被归因成"模型不会写程序"。
#: 这正是本文件 `QA_TOOLSET` 那段注释警告过的"三处不一致、后果静默"——只是发生在内置名单上。
#:
#: ## ⚠⚠ 这张白名单**不是逃逸边界**（2026-09-22 实测，零 API 零 GPU）
#:
#: 探针 `_tmp_sandbox_probe.py` 把程序真的交给 `execute_program` 跑，实测：
#:
#:     r = list_objects()
#:     r.__class__.__mro__                                  # ok
#:     (1).__class__.__mro__[1].__subclasses__()            # ok —— 484 个类可达
#:     s[i].__init__.__globals__["__builtins__"]["open"]    # ok —— 拿到真正的 open
#:
#: ⟹ **属性访问是语法，不受名字白名单约束**（`__class__` / `__mro__` / `__subclasses__` /
#: `__init__.__globals__` 全是属性）。所以"排除 `type` / `getattr` 就能绕不出去"这个判断是
#: **错的**：绕路根本不需要它们。本文件早先那句"每一条都能绕出沙箱"**结论对、归因错**。
#:
#: 那这份名单管什么？**管直路，不管逃逸**：
#:   ① 让 `open` / `eval` / `exec` / `compile` / `__import__`(真身) / `globals` / `locals` /
#:      `vars` / `dir` / `object` / `breakpoint` / `help` 这些**一眼可见的名字**不在命名空间里，
#:      于是"程序顺手读了个文件"这种事故不会**偶然**发生。成本为零，仍然值得。
#:   ② **真正的信息约束不在这里**，而在 `build_namespace()` **不放 `ctx`** ——
#:      「一切空间数值只能来自工具返回值」全靠那一条，与内置名单毫无关系。
#:   ③ 本项目的威胁模型 = **本机 + 自己的照片 + 非对抗输入**。整套沙箱是
#:      **测量完整性装置**（防误用、防偶然、防把 harness 缺陷记到模型账上），
#:      **不是对恶意程序的隔离边界**。要真隔离得换进程级方案（受限子进程 / 独立解释器）。
#:      **不要因为它叫"沙箱"就把它写成安全边界。**
#:
#: ⚠ `getattr` / `setattr` / `type` / `issubclass` **在**名单里（2026-09-22 加回）：
#: 它们被排除的原始理由是"能绕出沙箱"，而实测表明**绕路不需要它们** ⟹
#: 排除它们**没换来任何安全性**，却制造了一次**真实失败**
#: （`NameError: name 'getattr' is not defined`，本机实测复现），且**静态检查还放行**它。
#: 它们是普通 Python 惯用法；留着，harness 噪声更少。
SAFE_BUILTIN_NAMES: tuple[str, ...] = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float",
    "format", "frozenset", "int", "isinstance", "issubclass", "len", "list", "map",
    "max", "min", "pow", "print", "range", "repr", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "zip",
    # 允许写 try/except —— 「错误恢复」是本项目的观测点，不给异常类型就写不出来。
    "Exception", "ValueError", "KeyError", "IndexError", "TypeError",
    "ZeroDivisionError", "ArithmeticError", "AssertionError", "RuntimeError",
    # 反射/惯用法。见上方实测说明：它们**不增加**逃逸能力。
    "getattr", "setattr", "type",
)

#: 弃答的答案标记。程序写 `submit("unknown", ...)` 即判定为「如实说我答不了」。
#: 单独成为一种结局，是为了让「弃答」与「答错」在指标里能分开计数 ——
#: 否则「模型不会编数」和「模型编了个数」会混成同一个数字。
ABSTAIN_TOKENS: frozenset[str] = frozenset({"unknown", "abstain", "cannot_answer", "n/a"})

#: **问答臂的动作空间**（L1–L4：读取 / 查询 / 几何 / 视觉语义）。这一份名单必须三处一致：
#:   ① 提示词里列出的工具（`llm.schema.docs_text(tools=...)`）
#:   ② AST 静态检查放行的工具（`synthesizer.static_check(tools=...)`）
#:   ③ 运行期命名空间里真实存在的工具（`build_namespace(toolset=...)`）
#: 三者不一致的后果是**静默的**：提示词里列了但命名空间没有 →
#: 模型照着写 → `NameError` → 表现成「模型不会写程序」。
#: `tests/test_agent_memory_and_layering.py::TestToolsetAgreement` 直接把三处相等断言下来。
#: （旧文档里引用的是 `tests/test_agent_toolset.py` —— 那个文件从未存在过，
#:   真正的断言一直在上面那个类里。指向不存在的测试文件比没有指针更糟：
#:   它让人以为这条约束有测试守着，而实际找的时候找不到。）
#:
#: ⚠ `get_attributes` 在名单里、但**关掉视觉时它也在名单里** —— 这是有意的：
#: 消融靠 `ToolContext.vlm=None` 让它在运行期返回 `CAPABILITY_DISABLED`，
#: 而不是靠「把它从名单里摘掉」。两者效果不同：从名单里摘掉，模型根本不知道
#: 有这条路，于是「关掉视觉」变成「换了一个动作空间」，消融就不再干净了。
QA_TOOLSET: tuple[str, ...] = (
    "list_objects", "get_object", "find_object", "single_object",
    "find_nearest", "find_farthest", "query_relation",
    "get_3d_position", "get_3d_extent", "calculate_distance", "calculate_angle",
    "get_attributes",
)

#: **故意排除**在这条臂之外的工具（L5 场景级输出 + 自省）。
#: 理由有两条，都很实际：
#:   · `describe_scene` 会一次吐出「全部物体 + 两两关系」。给它，
#:     模型就不必做工具选择了 ——「工具选择是否准确」这个指标直接失去观测点。
#:   · `diagnose_failure` / `counterfactual` 是**分析工具**（给人和给实验看），
#:     不是回答空间问题的手段。让被测方调用测量器械，归因会绕回自己身上。
#: 它们仍然存在于仓库、仍然可单测，只是**不属于问答臂的动作空间**。
META_TOOLS: tuple[str, ...] = (
    "describe_scene", "summarize_scene", "diagnose_failure", "counterfactual",
)


# ============================================================================
# 1. 异常
# ============================================================================


class ProgramTimeout(Exception):
    """程序超时（由 `_Watchdog` 注入到执行线程）。"""


class ProgramContractError(Exception):
    """程序违反了 `submit()` 契约（没给证据 / 答案类型不对 / 答案不是有限数）。"""


class UserProgramError(Exception):
    """程序自身在运行期抛错。message 里带类型名，`lineno` 由 `file_line_of` 定位。"""


class _Submitted(BaseException):
    """`submit()` 成功后抛出的内部控制流异常 —— 用来**立刻结束程序**。

    为什么用异常而不是返回值：`submit()` 的签名是 `NoReturn`，
    程序里写在中间（例如循环里满足条件就提交）也必须立刻停下。
    让程序继续跑下去只会产生副作用，而答案已经定了。

    ★ 为什么继承 `BaseException` 而**不是** `Exception`：因为程序看得见 `Exception`。
    `SAFE_BUILTIN_NAMES` 把 `Exception` 一类交给了模型（"错误恢复"是本项目的观测点，
    不给异常类型就写不出 try/except），于是模型很自然会写出这种兜底：

        try:
            submit(calculate_distance(a, b), evidence=["..."])
        except Exception:
            submit("unknown", evidence=["算不出来"])

    如果 `_Submitted` 是 `Exception` 的子类，上面那个 `except` 会**接住我们自己的
    控制流信号**。本机实测的两种后果（`_tmp_probe_submit.py`，零 API）：

      · 形态①：算对的答案被兜底**改写成 `"unknown"`**，落盘 `abstained=true`
        —— 模型没有弃答，是**记录说它弃答了**，直接污染"弃答率"这个硬指标；
      · 形态②（`except` 里 `pass`）：程序跑到结尾，落进 `no_submit` 分支
        —— **已提交的答案被判成根本没提交**。

    继承 `BaseException` 就把它移出了 `except Exception` 的捕获范围，
    与 `KeyboardInterrupt` / `SystemExit` 是同一类设计。
    程序若用 `except BaseException` 仍能吞掉（拦不住，也不该拦），
    所以 `_SubmitBox` 与 `execute_program` 各留一道兜底，见那两处。
    """


# ============================================================================
# 2. 结果类型
# ============================================================================


@dataclass(frozen=True)
class Submission:
    """一次成功的 `submit()`。"""

    answer: Any
    answer_type: str | None
    target_ids: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    #: `target_ids` 里不存在于场景的 id。**不致命**（答案仍然有效），
    #: 但要如实记下来 —— 它直接驱动 Viewer 高亮，也是幻觉率的另一个观测点。
    unknown_targets: tuple[str, ...] = ()
    abstained: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "answer_type": self.answer_type,
            "target_ids": list(self.target_ids),
            "evidence": list(self.evidence),
            "unknown_targets": list(self.unknown_targets),
            "abstained": self.abstained,
            "n_evidence": len(self.evidence),
        }


@dataclass(frozen=True)
class ExecOutcome:
    """一次执行的全部结果。

    `stage` 是**失败归类**，取值固定，直接进实验记录的 `failure.stage`：

        ok | syntax | no_submit | contract | runtime | timeout

    它与 §16.3 的 `diagnose_failure` 是同一条线：先知道「哪一类失败」，
    再谈「为什么失败」。混成一个 `Exception` 字符串就再也统计不出来。
    """

    ok: bool
    stage: str
    submission: Submission | None = None
    message: str = ""
    lineno: int | None = None
    tool_calls: int = 0
    duration_ms: float = 0.0
    stdout: str = ""
    trace: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "stage": self.stage,
            "message": self.message,
            "lineno": self.lineno,
            "tool_calls": self.tool_calls,
            "duration_ms": round(self.duration_ms, 3),
            "submission": self.submission.to_dict() if self.submission else None,
            "stdout": self.stdout[-2000:],
            "n_trace": len(self.trace),
            "trace_errors": _error_codes(self.trace),
        }


def _error_codes(trace: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """trace 里各类错误码的出现次数 —— 指标 6「错误恢复率」的原料。"""
    counts: dict[str, int] = {}
    for row in trace:
        res = row.get("result") or {}
        err = res.get("error") or {}
        code = err.get("code")
        if code:
            counts[code] = counts.get(code, 0) + 1
    return counts


# ============================================================================
# 3. 看门狗
# ============================================================================


def _inject_async_exception(exc_type: type, ident: int) -> int:
    """把 `exc_type` 注入指定线程。返回 1 表示成功注入恰好一个线程。

    与早期实验臂运行器里那份（已随上游检出于 2026-09-20 移出）是**同一套机制的两处落地**，
    不是遗漏的重复：`agents/` 不许 import `evaluation/` —— evaluation 是**测量** agent 的器械，
    让被测方依赖测量方会把依赖方向反过来（那一层楼塌了，循环和它的实验一起崩）。
    """
    import ctypes

    try:
        if not ident:
            return 0
        ctypes.pythonapi.PyThreadState_SetAsyncExc.argtypes = [ctypes.c_ulong, ctypes.py_object]
        ctypes.pythonapi.PyThreadState_SetAsyncExc.restype = ctypes.c_int
        n = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(ident),
                                                      ctypes.py_object(exc_type))
        if n > 1:
            # CPython 文档要求：注入到多个线程时必须立刻用 NULL 回滚
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(ident), None)
            return 0
        return int(n)
    except Exception:                      # noqa: BLE001  注入失败要退化到"没超时"，不能崩
        return 0


class _Watchdog:
    """硬超时。**只走一条路径：后台定时器 + 异步异常注入。**

    为什么不用 `sys.settrace` 那条路（早期基线的看门狗用过两条）：
    那边是**为了兼容它自己的调用方式** —— 它在装 tracer 之后才 `alarm()`，
    顺路搭车几乎零成本。而我们自己写执行器，没必要为此付出
    「逐行回调」的代价（对纯 Python 程序是几十倍减速，而这台机器上
    程序里每次工具调用都要做点云运算，tracer 会显著放大等待）。

    **明确的降级**：异步异常要等目标线程回到字节码边界才投递。
    程序若卡在一次不返回的长 C 调用里（例如工具内部的大图推理），
    这里**不会**及时打断。Linux 上真信号能中断可中断的系统调用，本实现不能。
    写在这里，免得后来者把它当成 `signal.alarm` 的等价替换。

    残余竞态：`__exit__` 取消定时器与注入之间存在毫秒级窗口，
    注入可能落在 `exec` 之后的代码上。所以注入前**在锁内复查** `_armed`，
    并且 `__exit__` 先置 `_armed=False` 再 cancel。
    """

    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = float(timeout_s)
        self.fired = False
        self.armed = False
        self._target = threading.get_ident()
        self._lock = threading.RLock()
        self._timer: threading.Timer | None = None

    def __enter__(self) -> "_Watchdog":
        if self.timeout_s and self.timeout_s > 0:
            with self._lock:
                self.armed = True
            t = threading.Timer(self.timeout_s, self._fire)
            t.daemon = True
            self._timer = t
            t.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        _ = exc_info
        with self._lock:
            self.armed = False
            t, self._timer = self._timer, None
        if t is not None:
            t.cancel()

    def _fire(self) -> None:
        with self._lock:
            if not self.armed:
                return
            self.armed = False          # 只打一枪
            self.fired = True
        _inject_async_exception(ProgramTimeout, self._target)


# ============================================================================
# 4. 命名空间
# ============================================================================


def _guarded_import(name: str, globals_: Any = None, locals_: Any = None,
                    fromlist: Sequence[str] = (), level: int = 0) -> Any:
    """`import` 的替代品：只放行 `ALLOWED_MODULES`。

    为什么还要给 `__import__`：程序里写 `import math` 是很自然的动作，
    不给它就会得到 `ImportError: __import__ not found` —— 一个**看不懂的错误**。
    给一个会明确说「os 不在白名单」的版本，模型读报错就知道怎么改。
    """
    root = (name or "").split(".")[0]
    if root not in ALLOWED_MODULES:
        raise ImportError(
            "模块 %r 不在白名单内。可用：%s（也可以直接使用这些模块里的函数，无需 import）"
            % (name, ", ".join(ALLOWED_MODULES))
        )
    return importlib.import_module(name)


def _build_builtins() -> dict[str, Any]:
    import builtins as _b

    ns: dict[str, Any] = {n: getattr(_b, n) for n in SAFE_BUILTIN_NAMES}
    ns["__import__"] = _guarded_import
    ns["__build_class__"] = _b.__build_class__      # 允许程序自己定义 class（无害）
    ns["__name__"] = "__agent_program__"
    return ns


def build_namespace(ctx: Any, submit: Any,
                    *, toolset: Sequence[str] | None = None) -> dict[str, Any]:
    """组装程序的全局命名空间。**这是沙箱的边界，只有这一处**。

    放进去的：`toolset` 里的工具（`ctx` 已被 `partial` 绑死，模型看不见它）、
              白名单内置、白名单模块、`submit`。
    刻意不放的：`ctx`（含 SceneGraph → 真坐标）、`TOOL_REGISTRY`、L5 元工具（见 `META_TOOLS`）。

    `toolset` 里出现一个注册表里没有的名字会**直接抛 KeyError**：
    提示词与运行期不一致必须当场响，不能等到模型写出一个 NameError 才察觉。
    """
    import tools

    tools.load_tools()                      # 幂等；只导入零 GPU 依赖的工具模块
    from tools.registry import TOOL_REGISTRY

    wanted = tuple(QA_TOOLSET if toolset is None else toolset)
    missing = [n for n in wanted if n not in TOOL_REGISTRY]
    if missing:
        raise KeyError(
            "动作空间里有未注册的工具 %s。已注册：%s。"
            "（这通常意味着提示词列了一个运行期不存在的工具 —— 那种不一致会让模型"
            "写出 NameError 的程序，表现成「模型不会写程序」。）"
            % (missing, sorted(TOOL_REGISTRY))
        )

    ns: dict[str, Any] = {}
    for name in wanted:
        ns[name] = functools.partial(TOOL_REGISTRY[name], ctx)
    for mod in ALLOWED_MODULES:
        ns[mod] = importlib.import_module(mod)
    ns["submit"] = submit
    ns["__builtins__"] = _build_builtins()
    ns["__doc__"] = None
    ns["__package__"] = None
    ns["__file__"] = "<agent_program>"
    return ns


# ============================================================================
# 5. submit 契约
# ============================================================================


class _SubmitBox:
    """`submit()` 的实现。

    每一次拒绝都给出**可执行**的原因（程序在重试轮里只看到这串文字），
    因为「答案被拒」如果不可理解，重试就退化成瞎猜。
    """

    def __init__(self, answer_type: str | None, known_ids: Sequence[str],
                 source_label: str) -> None:
        self.answer_type = answer_type
        self.known_ids = set(known_ids)
        self.source_label = source_label
        self.submission: Submission | None = None

    def __call__(self, answer: Any = None, target_ids: Any = None,
                 evidence: Any = None) -> Any:
        # 已经交过答卷了。正常路径下第一次调用就抛出 `_Submitted` 把程序终止，
        # 所以走到这里只可能是**程序把控制流吞掉了**（`except Exception` 已由
        # `_Submitted` 继承 `BaseException` 挡住，剩下 `except BaseException` 这条）。
        # 此时**保留第一份答案**：它是程序正常算出来的那一份，
        # 而第二次 submit 通常写在兜底分支里、带着猜测的默认值。
        if self.submission is not None:
            raise _Submitted()

        if answer is None:
            raise ProgramContractError(
                "submit(answer=...) 不能是 None。没算出来请用 "
                'submit("unknown", evidence=["为什么答不了"])。'
            )

        ev = _normalize_list(evidence, "evidence")
        if not ev:
            raise ProgramContractError(
                "submit 必须给出至少 1 条 evidence（写清这个数来自哪个工具/哪条公式）。"
                '例如 submit(answer, evidence=["calculate_distance(sofa_1, table_1) = 1.234 m"])。'
                "§13.3(5) 的『答案可回溯』就落在这条约束上。"
            )

        abstained = isinstance(answer, str) and answer.strip().lower() in ABSTAIN_TOKENS
        coerced = answer if abstained else self._coerce(answer)
        targets = tuple(_normalize_list(target_ids, "target_ids"))
        unknown = tuple(t for t in targets if self.known_ids and t not in self.known_ids)

        self.submission = Submission(
            answer=coerced,
            answer_type=self.answer_type,
            target_ids=targets,
            evidence=tuple(ev),
            unknown_targets=unknown,
            abstained=abstained,
        )
        raise _Submitted()

    def _coerce(self, answer: Any) -> Any:
        if self.answer_type in (None, "", "str"):
            return answer
        if self.answer_type == "int":
            v = _to_number(answer, "int")
            if isinstance(answer, bool):
                raise ProgramContractError(
                    "答案要求 int，但程序交来 %r。bool 是 int 的子类，"
                    "静默当成 1/0 会把「判成了真假」伪装成「算出了一个数」。" % (answer,)
                )
            return int(v)
        if self.answer_type == "float":
            v = _to_number(answer, "float")
            if isinstance(answer, bool):
                raise ProgramContractError("答案要求 float，但程序交来 %r（bool）。" % (answer,))
            return float(v)
        if self.answer_type == "bool":
            if isinstance(answer, bool):
                return answer
            if isinstance(answer, str) and answer.strip().lower() in ("true", "false", "yes", "no"):
                return answer.strip().lower() in ("true", "yes")
            raise ProgramContractError("答案要求 bool，但程序交来 %r。" % (answer,))
        raise ProgramContractError("未知的 answer_type=%r" % (self.answer_type,))


def _to_number(answer: Any, kind: str) -> float:
    if isinstance(answer, bool):
        return float(answer)
    if isinstance(answer, (int, float)):
        v = float(answer)
    elif isinstance(answer, str):
        try:
            v = float(answer.strip())
        except ValueError:
            raise ProgramContractError(
                "答案要求 %s，但程序交来字符串 %r（无法解析成数）。"
                "数字要直接给数字，不要拼上单位或说明。" % (kind, answer)
            ) from None
    else:
        raise ProgramContractError("答案要求 %s，但程序交来 %r（%s）。"
                                   % (kind, answer, type(answer).__name__))
    if math.isnan(v) or math.isinf(v):
        # 这一条挡的是「几何算飞了」最典型的伪装：nan 传下去会一路活到最后，
        # 变成一个看着像数字的答案。
        raise ProgramContractError("答案 %r 不是有限数（nan/inf）—— 几何量算飞了。" % (v,))
    return v


def _normalize_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items: Sequence[Any] = [value]
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
    else:
        items = [value]
    out: list[str] = []
    for item in items:
        if item is None:
            continue
        text = item if isinstance(item, str) else str(item)
        text = text.strip()
        if text:
            out.append(text)
    if not out and value not in (None, "", [], (), set(), frozenset()):
        raise ProgramContractError("%s 不是字符串或字符串列表：%r" % (field_name, value))
    return out


# ============================================================================
# 6. 定位报错行
# ============================================================================


def file_line_of(exc: BaseException, source_label: str) -> int | None:
    """异常在**被测源码**里的行号。

    必须按 filename 过滤：traceback 的最后一帧通常落在 `tools/` 里的工具实现上，
    那是我们的代码，不是模型写的那一行。把工具内部的行号报给模型，
    它会在一个根本没有问题的地方反复改 —— 这是「定向重生成」失效的常见原因。
    """
    tb = exc.__traceback__
    found: int | None = None
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == source_label:
            found = tb.tb_lineno
        tb = tb.tb_next
    return found


# ============================================================================
# 7. 执行
# ============================================================================


def execute_program(
    source: str,
    ctx: Any,
    *,
    answer_type: str | None = None,
    timeout_s: float = 60.0,
    source_label: str = "<agent_program>",
    toolset: Sequence[str] | None = None,
) -> ExecOutcome:
    """跑一段程序，返回 `ExecOutcome`。**不抛异常**（除了我们自己的 bug）。

    「不抛异常」是有意的：调用方是循环层，它需要的是一个可归类的结果，
    而不是一条要 try/except 的失败路径。所有失败都落到 `stage` 上。
    """
    import time

    t0 = time.perf_counter()
    trace_start = len(getattr(ctx, "trace", []))

    def finish(outcome: ExecOutcome, stdout: str) -> ExecOutcome:
        trace_slice = tuple(getattr(ctx, "trace", [])[trace_start:])
        return ExecOutcome(
            ok=outcome.ok,
            stage=outcome.stage,
            submission=outcome.submission,
            message=outcome.message,
            lineno=outcome.lineno,
            tool_calls=len(trace_slice),
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            stdout=stdout,
            trace=trace_slice,
        )

    try:
        code = compile(source, source_label, "exec")
    except SyntaxError as exc:
        return finish(ExecOutcome(ok=False, stage="syntax",
                                  message="%s: %s" % (type(exc).__name__, exc.msg),
                                  lineno=exc.lineno), "")

    known_ids = list(getattr(getattr(ctx, "scene", None), "ids", lambda: [])())
    box = _SubmitBox(answer_type, known_ids, source_label)
    namespace = build_namespace(ctx, box, toolset=toolset)
    buf = io.StringIO()

    try:
        with _Watchdog(timeout_s), contextlib.redirect_stdout(buf):
            exec(code, namespace)          # noqa: S102 —— 沙箱边界见 build_namespace
    except _Submitted:
        return finish(ExecOutcome(ok=True, stage="ok", submission=box.submission), buf.getvalue())
    except ProgramTimeout:
        return finish(ExecOutcome(
            ok=False, stage="timeout",
            message="程序超过 %.1f s 未结束（疑似死循环或过大的循环体）。"
                    "请减少工具调用次数，不要在 Python 里遍历点或像素。" % timeout_s,
        ), buf.getvalue())
    except ProgramContractError as exc:
        return finish(ExecOutcome(ok=False, stage="contract", message=str(exc)), buf.getvalue())
    except RecursionError as exc:
        return finish(ExecOutcome(ok=False, stage="runtime",
                                  message="RecursionError: %s" % exc,
                                  lineno=file_line_of(exc, source_label)), buf.getvalue())
    except Exception as exc:               # noqa: BLE001  程序的任何异常都是"可归类的失败"
        return finish(ExecOutcome(
            ok=False, stage="runtime",
            message="%s: %s" % (type(exc).__name__, exc),
            lineno=file_line_of(exc, source_label),
        ), buf.getvalue())

    # 跑到这里说明程序正常结束了但没有 submit —— 这正是「静默给 0 分」那条路。
    #
    # ⚠ 但"没走到 submit"有两种，必须分开：
    #   ① 程序**一次都没调用** `submit()` —— 真·没交答卷，判失败；
    #   ② 程序**调用过**，但控制流被它自己的 `except BaseException` 吞掉，
    #      于是程序继续跑到结尾、从 `exec` 正常返回。
    #   形态②下 `box.submission` 是有值的（本机实测）。已经算出来的答案
    #   不该因为程序多写了一个兜底 `except` 就被丢掉 —— 所以这里先认它。
    if box.submission is not None:
        return finish(ExecOutcome(
            ok=True, stage="ok", submission=box.submission,
            message="程序提交答案后吞掉了控制流信号（submit 已生效，采用第一份答案）。",
        ), buf.getvalue())

    return finish(ExecOutcome(
        ok=False, stage="no_submit",
        message="程序执行完毕但没有调用 submit()。答案只能用 submit(answer, evidence=[...]) 交出 —— "
                "靠『命名空间里有 final_result 就算成功、没有就静默给空串』取答案是错的，"
                "所以这里明确判失败。",
    ), buf.getvalue())
