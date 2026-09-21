"""`ToolResult` 信封 —— 全项目的共同语言。契约见方案文档 §13.3(1)。

五个工具层（L1–L5）、三个角色，**返回值全部是它**。因此这个文件的口径一旦定下来，
整条链路上「成功 / 失败 / 证据 / 耗时」四件事的表达方式就统一了。

三处设计上的「堵死」，逐条对应早期基线实现的一个具体缺陷：

1. **`ok` 是显式布尔，不是靠「有没有 value」推断。**
   早期基线靠命名空间里有没有 `final_result` 这个变量来取答案，
   缺失时静默给 `""` 然后算错 —— 不报错。这里 `__post_init__` 强制：
   失败必须携带 `ToolError`，成功不许携带 —— 构造不出「静默失败」这种对象。

2. **错误码是枚举，且每个码自带 `recovery`。**
   早期基线把原始 traceback 回灌给模型，模型得自己猜「这错该怎么救」。
   这里 `ErrorCode` → `Recovery` 的映射是写死在代码里的，
   模型直接读 `error.recovery` 就知道下一步该 `retry_query` 还是 `abstain`。

3. **`evidence` 是结构化几何证据。**
   §11 设计原则 2 要求「每个工具的返回值必须是几何证据，不是自然语言」。
   证据字段形如 `{"formula": "||a-b||_2", "inputs": {...}, "centroid_a_m": [...]}`，
   答案因此可回溯到质心与公式 —— 这是「输出可审计」的实现基础。

`CAPABILITY_DISABLED` 单独说一句：它是**消融臂的机制**，不是异常处理。
关闭某能力后工具返回的是一个程序必须处理的结构化事件，于是「关掉视觉语义」
是一次接口开关而不是一份改写过的 prompt —— 这是消融可复现的关键（§13.3(7)）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4

from tools.version import TOOLS_VERSION

__all__ = [
    "ErrorCode",
    "Recovery",
    "ToolError",
    "ToolMeta",
    "ToolResult",
    "ENVELOPE_VERSION",
    "new_call_id",
    "timer",
]

#: 信封自身的格式版本，与工具库版本独立演进。
ENVELOPE_VERSION = "1.0"


# ----------------------------------------------------------------------------
# 错误码与恢复策略
# ----------------------------------------------------------------------------


class ErrorCode(str, Enum):
    """六个错误码，一一对应 §13.3(1) 的表。

    用 `str` 混入是为了让它能直接 JSON 序列化 / 出现在 prompt 里而不必额外转换。
    """

    #: object_id 不存在 —— **幻觉捕获点**：只能用 list_objects 返回过的 id。
    NOT_IN_SCENE = "NOT_IN_SCENE"
    #: 查询无匹配（图里没有「门」这个物体）。
    NOT_FOUND = "NOT_FOUND"
    #: 候选多个但接口要求单个。
    AMBIGUOUS = "AMBIGUOUS"
    #: 几何退化：点数不足、共线、零长度、包围盒为零体积。
    DEGENERATE = "DEGENERATE"
    #: 检测分低于阈值 —— 不是错误，是「不确定」，禁止四舍五入成确定值。
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    #: 该能力在本次实验臂被关闭（消融开关）。
    CAPABILITY_DISABLED = "CAPABILITY_DISABLED"


class Recovery(str, Enum):
    """程序收到某个错误码后**应当**采取的应对（§13.3(1) 表的最右列）。

    把恢复策略也枚举化，是为了让 LLM 不必猜「出错了该怎么办」——
    它直接读 `ToolResult.error.recovery` 就能选下一步。

    这些值是喂给 LLM 的，所以同时进 `to_dict()` 的输出。
    """

    #: 调 list_objects 拿到合法 id 再试。
    READ_SCENE = "read_scene"
    #: 换查询词 / 放宽阈值 / 换类别名。
    RETRY_QUERY = "retry_query"
    #: 追加空间约束把多个候选收成一个（最近的 / 左边的）。
    ADD_CONSTRAINT = "add_constraint"
    #: 换 anchor：bbox 中心 → 掩码质心。
    CHANGE_ANCHOR = "change_anchor"
    #: 上报不确定，禁止取整成确定值。
    REPORT_UNCERTAIN = "report_uncertain"
    #: 改用几何替代（例如用 calculate_distance 代替视觉判断）。
    USE_GEOMETRY = "use_geometry"
    #: 声明无法回答本题。
    ABSTAIN = "abstain"


#: 错误码 → 允许的恢复动作。顺序即推荐优先级。
_RECOVERY: Mapping[ErrorCode, tuple[Recovery, ...]] = {
    ErrorCode.NOT_IN_SCENE: (Recovery.READ_SCENE,),
    ErrorCode.NOT_FOUND: (Recovery.RETRY_QUERY, Recovery.ABSTAIN),
    ErrorCode.AMBIGUOUS: (Recovery.ADD_CONSTRAINT,),
    ErrorCode.DEGENERATE: (Recovery.CHANGE_ANCHOR, Recovery.USE_GEOMETRY),
    ErrorCode.LOW_CONFIDENCE: (Recovery.REPORT_UNCERTAIN, Recovery.USE_GEOMETRY),
    ErrorCode.CAPABILITY_DISABLED: (Recovery.USE_GEOMETRY, Recovery.ABSTAIN),
}


def _jsonable(obj: Any) -> Any:
    """尽力把任意对象变成可 JSON 序列化的形式。

    刻意不 import numpy：`result.py` 是所有工具的共同依赖，
    它越轻，单独测试一个工具时导入的负担越小。用 duck typing 认 numpy 标量/数组。
    """
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):          # numpy 数组 / torch 张量
        return _jsonable(obj.tolist())
    if hasattr(obj, "item"):            # numpy 标量
        return _jsonable(obj.item())
    return repr(obj)


# ----------------------------------------------------------------------------
# 信封本体
# ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolError:
    """结构化的失败描述。`message` 是给人看的，`context` 是给程序用的。"""

    code: ErrorCode
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)

    @property
    def recovery(self) -> tuple[Recovery, ...]:
        """该错误码允许的恢复动作，按推荐优先级排列。"""
        return _RECOVERY[self.code]


@dataclass(frozen=True, slots=True)
class ToolMeta:
    """每次工具调用的元信息 —— trace 的最小单元（§16.3 指标 8/9/10 的数据来源）。"""

    tool: str
    version: str = TOOLS_VERSION
    tool_call_id: str = field(default_factory=lambda: new_call_id())
    latency_ms: float = 0.0
    cached: bool = False
    gpu_peak_mb: float | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """所有工具的返回类型。

    不要直接构造它，用 `success()` / `failure()` —— 它们会替你填好 meta 的默认字段。
    """

    ok: bool
    value: Any | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    error: ToolError | None = None
    meta: ToolMeta | None = None

    def __post_init__(self) -> None:
        # 这两条不变量就是「堵死静默失败」的地方。
        # 构造不出「失败但没有 error」的对象，「缺失即静默算错」就不可能发生。
        if self.ok and self.error is not None:
            raise ValueError("ok=True 的结果不应携带 error")
        if not self.ok and self.error is None:
            raise ValueError(
                "ok=False 必须携带 error —— 否则就是静默失败"
                "（engine.py:292-303 缺失 final_result 时给 \"\" 然后算错）"
            )

    # -- 构造 -----------------------------------------------------------------

    @classmethod
    def success(
        cls,
        value: Any = None,
        *,
        evidence: Mapping[str, Any] | None = None,
        tool: str = "unknown",
        latency_ms: float = 0.0,
        cached: bool = False,
        gpu_peak_mb: float | None = None,
        version: str = TOOLS_VERSION,
    ) -> "ToolResult":
        return cls(
            ok=True,
            value=value,
            evidence=dict(evidence or {}),
            error=None,
            meta=ToolMeta(
                tool=tool,
                version=version,
                latency_ms=latency_ms,
                cached=cached,
                gpu_peak_mb=gpu_peak_mb,
            ),
        )

    @classmethod
    def failure(
        cls,
        code: ErrorCode,
        message: str,
        *,
        context: Mapping[str, Any] | None = None,
        evidence: Mapping[str, Any] | None = None,
        tool: str = "unknown",
        latency_ms: float = 0.0,
        version: str = TOOLS_VERSION,
    ) -> "ToolResult":
        return cls(
            ok=False,
            value=None,
            evidence=dict(evidence or {}),
            error=ToolError(code=code, message=message, context=dict(context or {})),
            meta=ToolMeta(tool=tool, version=version, latency_ms=latency_ms),
        )

    @classmethod
    def disabled(
        cls,
        tool: str,
        capability: str,
        *,
        hint: str = "",
        latency_ms: float = 0.0,
    ) -> "ToolResult":
        """消融开关关闭某能力时的标准返回（§13.3(7)）。

        单独做一个工厂，是为了让「这个失败是消融造成的」在 trace 里一眼可辨 ——
        否则它会和真正的错误混在一起，统计不出消融的影响。
        """
        ctx: dict[str, Any] = {"capability": capability}
        if hint:
            ctx["hint"] = hint
        return cls.failure(
            ErrorCode.CAPABILITY_DISABLED,
            f"能力 {capability!r} 在本实验臂被关闭",
            context=ctx,
            tool=tool,
            latency_ms=latency_ms,
        )

    @classmethod
    def not_in_scene(
        cls,
        tool: str,
        object_id: str,
        known: Any = None,
        *,
        latency_ms: float = 0.0,
    ) -> "ToolResult":
        """幻觉捕获点的标准返回：object_id 不在场景里。

        `known` 填上合法 id 列表，模型据此改写程序；这比只报「不存在」有用得多。
        """
        ctx: dict[str, Any] = {"object_id": object_id}
        if known is not None:
            ctx["known_ids"] = list(known)
        return cls.failure(
            ErrorCode.NOT_IN_SCENE,
            f"场景中不存在 object_id={object_id!r}",
            context=ctx,
            tool=tool,
            latency_ms=latency_ms,
        )

    # -- 便利 -----------------------------------------------------------------

    def __bool__(self) -> bool:
        """允许 `if res:` —— 但**不要**用它来取值，显式读 `res.value` 更清楚。"""
        return self.ok

    def unwrap(self) -> Any:
        """成功时返回 value，失败时抛异常。

        给测试与确定性代码用。**LLM 生成的程序不应该看到这个**：
        程序必须自己处理 `ok=False`（那才是错误恢复能力的观测点）。
        """
        if not self.ok:
            assert self.error is not None  # __post_init__ 已保证
            raise ToolFailure(self.error)
        return self.value

    def to_dict(self) -> dict[str, Any]:
        """落 trace / 写 JSONL 的形态（§16.3 的指标 2/3/4 直接从这里算）。"""
        err: dict[str, Any] | None = None
        if self.error is not None:
            err = {
                "code": self.error.code.value,
                "message": self.error.message,
                "recovery": [r.value for r in self.error.recovery],
                "context": _jsonable(self.error.context),
            }
        meta: dict[str, Any] | None = None
        if self.meta is not None:
            meta = {
                "tool": self.meta.tool,
                "version": self.meta.version,
                "tool_call_id": self.meta.tool_call_id,
                "latency_ms": round(self.meta.latency_ms, 3),
                "cached": self.meta.cached,
                "gpu_peak_mb": self.meta.gpu_peak_mb,
                "timestamp": self.meta.timestamp,
            }
        return {
            "ok": self.ok,
            "value": _jsonable(self.value),
            "evidence": _jsonable(self.evidence),
            "error": err,
            "meta": meta,
        }


class ToolFailure(RuntimeError):
    """`unwrap()` 在失败时抛出的异常。"""

    def __init__(self, error: ToolError) -> None:
        super().__init__(f"[{error.code.value}] {error.message}")
        self.error = error


def new_call_id() -> str:
    """短一点的调用 id —— trace 里一眼能读，不必是完整 UUID。"""
    return f"c{uuid4().hex[:12]}"


class timer:
    """`with timer() as t: ...` 然后读 `t.elapsed_ms`。

    每个工具都要填 `ToolMeta.latency_ms`，手写 `perf_counter` 差值容易漏。
    """

    __slots__ = ("_t0", "elapsed_ms")

    def __init__(self) -> None:
        self._t0 = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> "timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        _ = exc_info
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
