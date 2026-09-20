"""工具层的公共校验 —— 「外部层」职责的第 ① 项（校验存在性）。

独立成一个文件的原因：`get_3d_position`、`calculate_distance`、`query_relation`、
`find_nearest` …… 每个工具的第一段都是「解析场景、校验 id」。写五遍就是五份会漂移的
副本，而且 `NOT_IN_SCENE` 的 `context` 形状会各不相同 —— 于是模型要面对不一致的错误结构，
「读 error 换策略」这件事就变得不可靠。

所有校验失败都通过 `ToolAbort` 短路，工具函数主体因此能保持直线（见 registry 的说明）。
"""

from __future__ import annotations

from scene_graph.schema import Node, SceneGraph
from tools.registry import ToolAbort, ToolContext
from tools.result import ErrorCode, ToolResult

__all__ = ["need_scene", "need_node", "need_objects"]


def need_scene(ctx: ToolContext, scene_id: str | None = None) -> SceneGraph:
    """拿到当前场景；没有就明确告诉模型「该先做什么」。"""
    scene = ctx.scene
    if scene is None:
        raise ToolAbort(
            ToolResult.failure(
                ErrorCode.NOT_FOUND,
                "当前会话没有已加载的场景",
                context={
                    "hint": "先调用 build_scene_graph(image_id) 或 load_scene_graph(scene_id)",
                },
                tool="<scene>",
            )
        )
    if scene_id is not None and scene_id != scene.scene_id:
        raise ToolAbort(
            ToolResult.failure(
                ErrorCode.NOT_FOUND,
                f"找不到 scene_id={scene_id!r}",
                context={"current_scene_id": scene.scene_id, "hint": "省略 scene_id 即使用当前场景"},
                tool="<scene>",
            )
        )
    return scene


def need_node(scene: SceneGraph, object_id: str, tool_name: str) -> Node:
    """按 id 取节点，失败即 `NOT_IN_SCENE` —— **这就是幻觉捕获点**。

    `context["known_ids"]` 会带上全部合法 id：模型据此改写程序，
    比只回一句「不存在」有用得多，也让「幻觉率」这个指标可被直接统计。
    """
    try:
        return scene.node(object_id)
    except KeyError:
        raise ToolAbort(ToolResult.not_in_scene(tool_name, object_id, known=scene.ids())) from None


def need_objects(scene: SceneGraph, label: str, tool_name: str) -> list[Node]:
    """按类别取节点列表；一个都没有就 `NOT_FOUND`（图里确实没这东西）。

    注意与 `NOT_IN_SCENE` 的区别：`NOT_IN_SCENE` = id 是编的（幻觉）；
    `NOT_FOUND` = 类别查不到（世界的事实，不是模型的错）。
    两者对应**完全不同**的恢复动作（`read_scene` vs `retry_query`），必须分开。
    """
    nodes = scene.by_label(label)
    if not nodes:
        raise ToolAbort(
            ToolResult.failure(
                ErrorCode.NOT_FOUND,
                f"场景中没有 label 为 {label!r} 的物体",
                context={
                    "label": label,
                    "available_labels": sorted(scene.label_counts()),
                    "hint": "换一个查询词，或放宽检测阈值后重建场景图",
                },
                tool=tool_name,
            )
        )
    return nodes
