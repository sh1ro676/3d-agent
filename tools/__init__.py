"""工具层 —— 暴露给 LLM 的「API」，以及全项目共用的返回信封。

契约见方案文档 §11（工具分层 L1–L5）与 §13.3（三角色接口契约）。

注意：本目录下还有一个 `build_doc_html.py`，它是方案文档的 HTML 构建脚本，
不属于工具库成员，只是碰巧住在同一层。导入 `tools` 不会执行它。

设计约束（§13.3(6) 三条禁令的一部分）：
    • 工具返回值一律是 `tools.result.ToolResult`，不返回裸值、不返回自然语言。
    • 工具可报错，且错误码来自枚举 —— Agent 的错误恢复能力必须可枚举、可统计。

**为什么要 `load_tools()` 而不是在 `__init__` 里自动导入**：
    本文件会被 `import tools.result` 触发，而 L1 感知工具（`perception.py`）将来要
    `import torch`。如果在 `__init__` 里自动导入全部子模块，那么连「只想用信封」的
    单元测试都得先加载 torch —— 在 8GB 机器上这是几秒钟的代价，而且会让
    「关系/工具可单测」这个卖点变得名不副实。所以纯几何工具由 `load_tools()` 显式装载，
    感知类工具由 agent 层按需导入。
"""

from __future__ import annotations

__all__ = ["load_tools", "loaded_tools"]


def load_tools() -> tuple[str, ...]:
    """导入全部**无 GPU 依赖**的工具模块并返回注册的工具名。

    感知类（L1，需要 GroundingDINO / SAM2 / UniDepth）不在这里 —— 见模块 docstring。
    L5（`scene_report`）在**这里** —— 它只对 `SceneGraph` 做序列化与图运算，
    不加载任何权重，正是「最好测的一层」。
    L4（`attributes`）也在**这里** —— 它只调用 `ctx.vlm`，自己不认识 torch；
    视觉端点与模型名来自环境变量，模块导入本身零成本。
    """
    from tools import attributes as _attributes  # noqa: F401
    from tools import geometry as _geometry  # noqa: F401
    from tools import scene_report as _scene_report  # noqa: F401
    from tools import spatial as _spatial  # noqa: F401
    from tools.registry import tool_names

    return tool_names()


def loaded_tools() -> tuple[str, ...]:
    """当前已注册的工具名（不额外导入任何东西）。"""
    from tools.registry import tool_names

    return tool_names()
