"""L3 几何工具 —— 外部层。契约见 §11.1（L3 表）与 §13.3(4)。

本文件里**没有数学**：距离、角度、关系判定全在 `scene_graph/relations.py`，
那里可以脱离场景图和 GPU 单测。这里只做三件事：

    ① 校验 object_id  → `NOT_IN_SCENE`（幻觉捕获点）
    ② 组装 evidence   → 答案可回溯到质心与公式
    ③ 翻译退化情形    → `MissingGeometry` → `DEGENERATE` + 「换 anchor」建议

Phase 5 待补（本批未做，先留位以免忘）：
    • `calibrate_scale` —— 现在只返回因子、不改场景；真正应用校正需要重新构建
      `centroid_3d`，属于 builder 的职责。因为它同时是 `3D Localization Error`
      这个指标的来源，单独做完再并进来更有把握。
"""

from __future__ import annotations

import math

from scene_graph.relations import RELATIONS_METHOD, MissingGeometry, distance_m
from scene_graph.schema import Node
from tools.guards import need_node, need_scene
from tools.registry import ToolArgumentError, tool
from tools.result import ErrorCode, ToolResult

__all__ = ["get_3d_position", "get_3d_extent", "calculate_distance", "calculate_angle"]

#: `get_3d_position` 的锚点选项。`bbox_center` 存在的主要用途是**消融**：
#: 实测两者在 9 个物体上差均值 83 mm / 最大 208 mm（§17），
#: 保留这个开关是为了让那组数字随时可复现，而不是只能引用一次。
_ANCHORS = ("centroid", "bbox_center")


def _position_of(node: Node, anchor: str) -> tuple[list[float], str]:
    if anchor == "centroid":
        return list(node.centroid_3d), "mask_median"
    if anchor == "bbox_center":
        bb = node.bbox_3d
        if bb is None:
            # 数据缺失 → DEGENERATE（装饰器会翻译），恢复动作是「换 anchor」。
            # 不要用 ToolArgumentError：那代表「程序写错」，会被归到失败诊断的
            # 「程序」类而不是「工具/数据」类，污染报告的归因（§16.3）。
            raise MissingGeometry(f"节点 {node.id!r} 没有 bbox_3d，无法用 anchor='bbox_center'")
        return [(bb.min[i] + bb.max[i]) / 2.0 for i in range(3)], "bbox_center"
    # anchor 值本身非法 = 程序写错了参数值 → 冒泡（不被装饰器捕获）。
    raise ToolArgumentError(f"anchor 必须是 {_ANCHORS} 之一，收到 {anchor!r}")


@tool("get_3d_position")
def get_3d_position(
    ctx,
    scene_id: str | None = None,
    object_id: str = "",
    anchor: str = "centroid",
) -> ToolResult:
    """物体的相机系三维坐标（米）。`res.value` 是 **`[x, y, z]`**（相机系米制；`anchor` 可切到 `'bbox_center'`）。

    ★ 这是全项目最核心的一个工具 —— VADAR 拿到的点云里本来就有 x/y，
    但它只把 z 当 depth 用（`predefined_modules.py:375`、`:395`），
    于是「二维查询」这个限制是自找的。升级到真三维不需要新模型、不需要新显存。
    """
    scene = need_scene(ctx, scene_id)
    node = need_node(scene, object_id, "get_3d_position")
    pos, source = _position_of(node, anchor)

    return ToolResult.success(
        value=pos,
        evidence={
            "method": RELATIONS_METHOD,
            "anchor": source,
            "centroid_m": pos,
            "n_points": node.n_points,
            "bbox_2d": list(node.bbox_2d) if node.bbox_2d else None,
            "score": node.score,
        },
    )


@tool("get_3d_extent")
def get_3d_extent(ctx, scene_id: str | None = None, object_id: str = "") -> ToolResult:
    """物体的真实三维尺寸 (w, h, l)，米。`res.value` 是 **`{w, h, l}`**（米，轴对齐包围盒的三边长，**不是 OBB 也不是体积**）。

    ★ VADAR 的对应物是 `get_2D_object_size`，它算的是 `2D 像素 × depth` ——
    整个式子里**没有焦距**，量纲都不成立（§5 结论 2）。这里的尺寸来自点云，
    是真正的米制量，因此可以拿来和真实物体比对、也可以拿来做尺度校正的锚。
    """
    scene = need_scene(ctx, scene_id)
    node = need_node(scene, object_id, "get_3d_extent")

    w, h, l = node.extent_3d
    return ToolResult.success(
        value={"w": w, "h": h, "l": l},
        evidence={
            "method": RELATIONS_METHOD,
            "extent_m": [w, h, l],
            "bbox_3d_min": list(node.bbox_3d.min) if node.bbox_3d else None,
            "bbox_3d_max": list(node.bbox_3d.max) if node.bbox_3d else None,
            "n_points": node.n_points,
        },
    )


@tool("calculate_distance")
def calculate_distance(
    ctx,
    scene_id: str | None = None,
    a: str = "",
    b: str = "",
) -> ToolResult:
    """两个物体之间的三维欧氏距离（米）。`res.value` 是 **float**（米）。

    ★ 与 VADAR 的差别很具体：VADAR 只有 `depth(image, bbox)` 这个单点深度，
    问「A 离 B 多远」只能 `|depth_a - depth_b|`。两个横向错开但等深的物体
    会得到 **0** —— 见 `test_euclidean_not_depth_difference`。
    在同一张图里，这种「错开」恰恰是最常见的情况。
    """
    scene = need_scene(ctx, scene_id)
    na = need_node(scene, a, "calculate_distance")
    nb = need_node(scene, b, "calculate_distance")

    verdict = distance_m(na, nb)
    return ToolResult.success(
        value=float(verdict.value),
        evidence={
            "method": verdict.method,
            "formula": "||centroid_a - centroid_b||_2",
            "anchor": "mask_median",
            **{k: v for k, v in verdict.metric.items()},
        },
    )


@tool("calculate_angle")
def calculate_angle(
    ctx,
    scene_id: str | None = None,
    a: str = "",
    b: str = "",
    c: str = "",
) -> ToolResult:
    """三点夹角（度），顶点是 **b**。即 ∠(a, b, c)。`res.value` 是 **float**（度）。

    纯空间量，任何 VLM 都很难给准，而几何一步到位。
    """
    scene = need_scene(ctx, scene_id)
    na = need_node(scene, a, "calculate_angle")
    nb = need_node(scene, b, "calculate_angle")
    nc = need_node(scene, c, "calculate_angle")

    ba = [na.centroid_3d[i] - nb.centroid_3d[i] for i in range(3)]
    bc = [nc.centroid_3d[i] - nb.centroid_3d[i] for i in range(3)]
    la = math.sqrt(sum(v * v for v in ba))
    lc = math.sqrt(sum(v * v for v in bc))

    if la <= 1e-9 or lc <= 1e-9:
        # 退化不是「程序错」，是几何数据不足 → DEGENERATE（可恢复：换 anchor / 换点）。
        # ⚠️ 必须传 ErrorCode 枚举而不是字符串：ErrorCode 是 str-mixin 枚举，
        # 但 hash 基于成员名，`_RECOVERY["DEGENERATE"]` 会 KeyError。
        return ToolResult.failure(
            ErrorCode.DEGENERATE,
            f"向量长度为零（|BA|={la:.3e}, |BC|={lc:.3e}），无法定义夹角",
            context={"leg_a_m": la, "leg_c_m": lc, "hint": "三点中至少两点重合，换一个顶点或换锚点"},
            tool="calculate_angle",
        )

    cos = sum(ba[i] * bc[i] for i in range(3)) / (la * lc)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    return ToolResult.success(
        value=angle,
        evidence={
            "method": RELATIONS_METHOD,
            "formula": "acos( (BA · BC) / (|BA| |BC|) )",
            "vertex": nb.id,
            "angle_deg": angle,
            "leg_a_m": la,
            "leg_c_m": lc,
        },
    )
