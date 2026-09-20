"""工具注册表与运行时上下文 —— 工具的「外部层」基础设施。契约见 §13.3(4)(6)。

**三层职责划分**（这个划分决定了项目好不好测、好不好做消融）：

    relations.py          纯数学，吃 Node 返回 RelationVerdict。无 ctx、无 ToolResult。
    registry.py（本文件）  运行时容器：注入 scene / vlm / viewer，统一计时、trace、能力开关。
    geometry.py 等工具      直线代码：校验 id → 调数学 → 包 evidence → 返回 ToolResult。

工具函数一律写成 `(ctx, ...) -> ToolResult`，用 `@tool("名字")` 装饰。
装饰器负责四件事，工具函数里一行都不用写：
    ① 计时并填 `ToolMeta.latency_ms`
    ② 能力开关闸门（`CAPABILITY_DISABLED`）
    ③ 把内部的 `ToolAbort` / `MissingGeometry` 翻译成 `ToolResult`
    ④ 往 `ctx.trace` 追加一条记录（§16.3 指标 2/3/4 的数据来源）

**异常策略**（有意为之，不是偷懒）：
    • `ToolAbort`        → 业务性失败，翻成 ToolResult（模型看得见、能恢复）
    • `MissingGeometry`  → 翻成 `DEGENERATE` + 「换 anchor」建议
    • `ToolArgumentError`→ **不捕获**，让它冒泡。参数值域写错是「程序写错了」，
      不是「场景有问题」——它必须在失败诊断里被归类到「程序」而不是「工具」，
      否则报告里的失败归因会失真（§16.3 的 diagnose_failure）。
    • 其它任何异常       → 同样冒泡。工具内部的 bug 必须暴露，不能静默变成 ToolResult。
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterator, Mapping

from scene_graph.relations import MissingGeometry
from scene_graph.schema import SceneGraph
from tools.result import ErrorCode, ToolResult, timer
from tools.version import TOOLS_VERSION

__all__ = [
    "ToolAbort",
    "ToolArgumentError",
    "ToolContext",
    "TOOL_REGISTRY",
    "tool",
    "iter_tools",
    "tool_names",
]


class ToolAbort(Exception):
    """工具内部用来短路并返回一个 `ToolResult` 的控制流异常。

    用它是因为「校验失败就 return」在深层辅助函数里做不到 —— 辅助函数得把
    错误一路 return 上来，每个调用点都要判断。用异常做控制流，工具主体就能
    保持直线：`scene = _need_scene(...)`，后面不用管失败分支。
    """

    def __init__(self, result: ToolResult) -> None:
        super().__init__(result.error.message if result.error else "aborted")
        self.result = result


class ToolArgumentError(ValueError):
    """参数值域非法（例如 `anchor="foo"`）。

    **故意不被装饰器捕获**：它代表「程序写错了」，必须在失败诊断里归到「程序」这一类，
    而不是被当成工具失败 —— 后者会让「工具调用成功率」这个指标说谎。
    参数**名**写错则更早就会在 AST 静态检查阶段被拦下（§13.3(2)）。
    """


@dataclass
class ToolContext:
    """一次会话里所有工具的共享状态。

    刻意用可变 dataclass（不是 frozen）：trace 要往里追加。
    但 `scene` 本身是 frozen 的 `SceneGraph` —— 工具**不能**改场景，
    这保证了「工具调用是只读的」这条不变量。
    """

    #: 当前场景。为 None 时所有需要场景的工具返回 NOT_FOUND。
    scene: SceneGraph | None = None
    #: 角色② 的实现（`llm/vlm.py` 的 describe）。None = 该能力被关闭 → CAPABILITY_DISABLED。
    vlm: Any | None = None
    #: 角色④ 的 Viewer 句柄（Demo 阶段）。
    viewer: Any | None = None
    #: image_id → 图像对象（PIL / ndarray）。L1 感知工具用。
    images: dict[str, Any] = field(default_factory=dict)
    #: 其他实验臂开关（`action_space` / `render` / `planner` …）。
    flags: dict[str, Any] = field(default_factory=dict)
    #: 工具调用轨迹。每条是 `{tool, args, result}`，直接可算指标 2/3/4。
    trace: list[dict[str, Any]] = field(default_factory=list)
    #: 是否记录 trace。批量评测时打开，单次调试可关。
    record_trace: bool = True

    # -- 能力开关 --------------------------------------------------------------

    def enabled(self, capability: str) -> bool:
        """能力是否可用。

        `"vlm"` 直接看 `vlm is None` —— 这样「关掉视觉语义」只需要传 `vlm=None`，
        不需要同时记得改 flags，少一个能忘的地方（§13.3(7) 的开关表）。
        """
        if capability == "vlm":
            return self.vlm is not None
        return bool(self.flags.get(capability, True))

    # -- 轨迹 ------------------------------------------------------------------

    def record(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        result: ToolResult,
    ) -> None:
        if not self.record_trace:
            return
        self.trace.append(
            {
                "tool": tool_name,
                "args": {k: v for k, v in args.items()},
                "result": result.to_dict(),
            }
        )

    def trace_jsonl(self) -> Iterator[str]:
        """trace 的 JSONL 形态 —— 每题一个文件，答辩时可逐行回放。"""
        import json

        for row in self.trace:
            yield json.dumps(row, ensure_ascii=False)

    def reset_trace(self) -> None:
        self.trace.clear()


# ----------------------------------------------------------------------------
# 装饰器
# ----------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Callable[..., ToolResult]] = {}


def _translate(exc: Exception, tool_name: str) -> ToolResult:
    """把内部异常翻译成对模型有意义的结果。见模块 docstring 的异常策略。"""
    if isinstance(exc, ToolAbort):
        return exc.result
    if isinstance(exc, MissingGeometry):
        return ToolResult.failure(
            ErrorCode.DEGENERATE,
            str(exc),
            context={
                "reason": "missing_geometry_field",
                # 提示必须**具体到可执行**：只说「数据缺了」模型还是会重试同一条路。
                # 这里的正确方向是换到只依赖质心的锚点 —— 注意它与失败时的 anchor 相反。
                "hint": "改用 anchor='centroid'（默认）——它只依赖掩码质心；"
                        "若必须用包围盒，则要在构建场景图时补上 bbox_3d",
            },
            tool=tool_name,
        )
    # ToolArgumentError 与其它异常一律冒泡 —— 见模块 docstring。
    raise exc


def tool(
    name: str,
    *,
    capability: str | None = None,
    version: str = TOOLS_VERSION,
) -> Callable[[Callable[..., ToolResult]], Callable[..., ToolResult]]:
    """把一个 `(ctx, ...) -> ToolResult` 函数注册成工具。

    `capability` 非空时，该能力被关闭就直接返回 `CAPABILITY_DISABLED` ——
    于是消融是「一次配置」，而不是「一份改写过的 prompt」（§13.3(7)）。
    """

    def decorator(fn: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
        original_sig = inspect.signature(fn)
        # 暴露给 LLM 的签名里没有 ctx —— 那是我们的注入，不是模型的参数。
        public_params = list(original_sig.parameters.values())[1:]

        @functools.wraps(fn)
        def wrapper(ctx: ToolContext, *args: Any, **kwargs: Any) -> ToolResult:
            if capability is not None and not ctx.enabled(capability):
                result = ToolResult.disabled(name, capability)
                ctx.record(name, _bind(original_sig, args, kwargs), result)
                return result

            with timer() as t:
                try:
                    result = fn(ctx, *args, **kwargs)
                except Exception as exc:  # noqa: BLE001 —— _translate 决定是否放行
                    result = _translate(exc, name)

            result = _stamp(result, name, t.elapsed_ms, version)
            ctx.record(name, _bind(original_sig, args, kwargs), result)
            return result

        wrapper.__signature__ = original_sig.replace(parameters=public_params)  # type: ignore[attr-defined]
        wrapper.__tool_name__ = name  # type: ignore[attr-defined]
        TOOL_REGISTRY[name] = wrapper
        return wrapper

    return decorator


def _bind(sig: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """把实参绑到参数名上，且**跳过 ctx** —— trace 里记的是模型的调用意图。"""
    bound = sig.bind_partial(None, *args, **kwargs)
    bound.arguments.pop(next(iter(sig.parameters)), None)
    return {k: _simple(v) for k, v in bound.arguments.items()}


def _simple(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return list(v)
    return repr(v)


def _stamp(result: ToolResult, name: str, latency_ms: float, version: str) -> ToolResult:
    """补上工具名/版本/耗时。ToolResult 是 frozen 的，所以是重建而非赋值。"""
    if result.meta is None:
        return result
    return replace(result, meta=replace(result.meta, tool=name, version=version, latency_ms=latency_ms))


# ----------------------------------------------------------------------------
# 自省（供 llm/schema.py 渲染工具文档，以及 AST 静态检查的白名单）
# ----------------------------------------------------------------------------


def iter_tools() -> Iterator[tuple[str, Callable[..., ToolResult]]]:
    for name in sorted(TOOL_REGISTRY):
        yield name, TOOL_REGISTRY[name]


def tool_names() -> tuple[str, ...]:
    """工具名白名单 —— AST 静态检查用（§13.3(2) 的第 2 项）。"""
    return tuple(sorted(TOOL_REGISTRY))


def tool_parameters(name: str) -> tuple[str, ...]:
    """某工具的公开参数名 —— AST 静态检查用（第 3 项：参数名是否存在于 schema）。"""
    fn = TOOL_REGISTRY[name]
    sig: inspect.Signature = fn.__signature__  # type: ignore[attr-defined]
    return tuple(p for p in sig.parameters)
