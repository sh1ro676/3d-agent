"""L2 场景查询 + L3 空间关系工具 —— 外部层。契约见 §11.1。

**这里给坐标，`scene_hint` 不给** —— 这两件事不矛盾，而是同一条设计的两面（§13.3(2)）：
    • `synthesize()` 的输入 `scene_hint` 只有 `{"chair": 4, "door": 1}` 与图像尺寸，
      于是模型**想抄坐标也无从抄起**；
    • 但 `list_objects()` 的返回值是**工具返回值**，它当然含坐标 —— 这正是
      「坐标的唯一来源」所指的那个来源。
把坐标从提示词里拿掉、只在工具返回值里给，等于把「不要编坐标」从提示词祈祷
变成了信息约束。

工具清单（对应 §11.1 的 L2/L3 表）：
    list_objects / get_object          场景读取
    find_object / single_object        按类别查（后者可能 AMBIGUOUS）
    find_nearest / find_farthest       组合查询
    query_relation                     全部 11 种关系的统一入口
"""

from __future__ import annotations

import inspect
from typing import Any

from scene_graph.relations import RELATIONS, distance_m
from scene_graph.schema import Node
from tools.guards import need_node, need_objects, need_scene
from tools.registry import ToolArgumentError, tool
from tools.result import ErrorCode, ToolResult

__all__ = [
    "list_objects",
    "get_object",
    "find_object",
    "single_object",
    "find_nearest",
    "find_farthest",
    "query_relation",
]


def _brief(node: Node) -> dict[str, Any]:
    """节点的紧凑摘要 —— 进工具返回值。

    刻意**不用** pydantic 默认的 `model_dump()`：那会带上 `mask_ref`、`bbox_2d`、
    `frame` 等对推理无用的字段，20 个物体就能把观察文本撑到几千 token。
    观察必须压缩（§18 Phase 7 风险②），压缩点就在这里。
    """
    return {
        "object_id": node.id,
        "label": node.label,
        "score": round(node.score, 3),
        "centroid_m": [round(c, 4) for c in node.centroid_3d],
        "extent_m": {"w": round(node.extent_3d[0], 3),
                     "h": round(node.extent_3d[1], 3),
                     "l": round(node.extent_3d[2], 3)},
    }


@tool("list_objects")
def list_objects(
    ctx,
    scene_id: str | None = None,
    label: str | None = None,
    limit: int | None = None,
) -> ToolResult:
    """列出场景中的物体。`label` 为空则返回全部。`res.value` 是 **list**，每项 `{object_id, label, score, centroid_m:[x,y,z], extent_m:{w,h,l}}` —— **所有物体的质心与尺寸一次给全**，找极值/计数/求均值不必再逐个调工具。

    这是模型**唯一**能拿到合法 object_id 的入口 —— 所以 `NOT_IN_SCENE` 的
    `context["known_ids"]` 与这里的返回必须一致，否则「幻觉捕获点」就失效了。
    """
    scene = need_scene(ctx, scene_id)
    nodes = scene.by_label(label) if label else list(scene.nodes)
    total = len(nodes)
    if limit is not None:
        if limit < 0:
            raise ToolArgumentError(f"limit 必须 >= 0，收到 {limit!r}")
        nodes = nodes[:limit]

    return ToolResult.success(
        value=[_brief(n) for n in nodes],
        evidence={
            "scene_id": scene.scene_id,
            "filter_label": label,
            "returned": len(nodes),
            "total_in_scene": total,
            "total_objects": len(scene.nodes),
            "label_counts": scene.label_counts(),
        },
    )


@tool("get_object")
def get_object(ctx, scene_id: str | None = None, object_id: str = "") -> ToolResult:
    """取单个物体的完整信息（含语义属性、掩码引用与二维框）。`res.value` 是**单个 dict**：`{object_id, label, score, centroid_m, extent_m, attributes, mask_ref, frame, bbox_2d}`。⚠ `bbox_2d` 是 `[x1, y1, x2, y2]`，**图像像素系**（原点左上），与相机系米制的 `centroid_m` 不是同一套坐标 —— 「图像左半边/画面上方」用它。

    比 `list_objects` 多出 `attributes` / `mask_ref` / `bbox_2d` —— `attributes` 来自角色②，
    `mask_ref` 让程序知道掩码在哪（但**不要**让程序去读点云数组，见 §13.3(6) 禁令），
    `bbox_2d` 是「二维区域」类题目**唯一**的判据（`_brief` 刻意不带它，以免撑大观察文本）。
    """
    scene = need_scene(ctx, scene_id)
    node = need_node(scene, object_id, "get_object")

    detail = _brief(node)
    detail["attributes"] = dict(node.attributes)
    detail["mask_ref"] = node.mask_ref
    detail["frame"] = node.frame
    # `bbox_2d` 是图像像素系（原点左上），**不是**相机系 —— 「图像左半边有几个物体」
    # 这类题只有它能答。放在 `get_object` 这条单物体路径上而不是 `_brief` 里：
    # `list_objects` 一次返回 9~20 个物体，每个多 4 个像素数会把观察文本撑大近一倍。
    detail["bbox_2d"] = (None if node.bbox_2d is None
                         else [round(float(v), 2) for v in node.bbox_2d])
    return ToolResult.success(
        value=detail,
        evidence={"scene_id": scene.scene_id, "bbox_3d_min": None if node.bbox_3d is None else list(node.bbox_3d.min),
                  "bbox_3d_max": None if node.bbox_3d is None else list(node.bbox_3d.max)},
    )


@tool("find_object")
def find_object(ctx, scene_id: str | None = None, label: str = "") -> ToolResult:
    """按类别查物体，**可能返回多个**（`res.value` 与 `list_objects` 同形）。一个都没有 → `NOT_FOUND`。

    「返回 0 个」与「返回 N 个」是两种不同的世界事实，必须让模型分清：
    前者应当 `retry_query` 或答「图中无此物」，后者应当继续做组合（argmin / filter）。
    所以这里不把 0 个包装成空列表，而是明确报错。
    """
    scene = need_scene(ctx, scene_id)
    nodes = need_objects(scene, label, "find_object")
    return ToolResult.success(
        value=[_brief(n) for n in nodes],
        evidence={"scene_id": scene.scene_id, "label": label, "count": len(nodes)},
    )


@tool("single_object")
def single_object(ctx, scene_id: str | None = None, label: str = "") -> ToolResult:
    """按类别取**恰好一个**物体（`res.value` 与 `get_object` 同形）；匹配到多个就 `AMBIGUOUS`。

    这个工具是 `AMBIGUOUS` 错误码唯一的产生处，也是它存在的理由：
    「把那把椅子高亮一下」这种话语天然是多义的。返回的 `context["candidates"]`
    带上全部候选 id 与坐标，模型据此**追加空间约束**（最近的 / 左边的）
    再调一次 —— 这正是 §13.3(1) 表里 `AMBIGUOUS` 对应的恢复动作。
    """
    scene = need_scene(ctx, scene_id)
    nodes = need_objects(scene, label, "single_object")

    if len(nodes) > 1:
        return ToolResult.failure(
            ErrorCode.AMBIGUOUS,
            f"label={label!r} 匹配到 {len(nodes)} 个物体，无法确定指代",
            context={
                "label": label,
                "candidates": [_brief(n) for n in nodes],
                "hint": "追加空间约束（如 find_nearest / 用 object_id 明确指定），不要随机挑一个",
            },
           tool="single_object",
        )

    node = nodes[0]
    detail = _brief(node)
    detail["attributes"] = dict(node.attributes)
    return ToolResult.success(
        value=detail,
        evidence={"scene_id": scene.scene_id, "label": label, "count": 1},
    )


def _rank_by_distance(
    ctx,
    scene_id: str | None,
    anchor: str,
    label: str,
    k: int,
    tool_name: str,
    *,
    descending: bool,
) -> ToolResult:
    """`find_nearest` / `find_farthest` 的共同实现。

    注意这里**排除 anchor 自己**：否则问「哪把椅子离椅子最近」会返回它自己，
    而模型很可能就此给出一个看起来合理、实际无意义的答案。
    """
    if k < 1:
        raise ToolArgumentError(f"k 必须 >= 1，收到 {k!r}")

    scene = need_scene(ctx, scene_id)
    anchor_node = need_node(scene, anchor, tool_name)
    candidates = [n for n in need_objects(scene, label, tool_name) if n.id != anchor_node.id]

    if not candidates:
        return ToolResult.failure(
            ErrorCode.NOT_FOUND,
            f"label={label!r} 除 anchor 自身外没有其他物体可选",
            context={"anchor": anchor_node.id, "label": label,
                     "hint": "anchor 与候选集是同一类；换一个候选类别"},
            tool=tool_name,
        )

    ranked = sorted(
        ((float(distance_m(anchor_node, n).value), n) for n in candidates),
        key=lambda pair: pair[0],
        reverse=descending,
    )
    top = ranked[:k]

    return ToolResult.success(
        value=[
            {**_brief(n), "distance_m": round(d, 4)} for d, n in top
        ],
        evidence={
            "scene_id": scene.scene_id,
            "anchor": anchor_node.id,
            "anchor_centroid_m": [round(c, 4) for c in anchor_node.centroid_3d],
            "candidate_label": label,
            "n_candidates": len(candidates),
            "order": "descending" if descending else "ascending",
            "ranked": [{"object_id": n.id, "distance_m": round(d, 4)} for d, n in ranked],
        },
    )


@tool("find_nearest")
def find_nearest(
    ctx,
    scene_id: str | None = None,
    anchor: str = "",
    label: str = "",
    k: int = 1,
) -> ToolResult:
    """在 `label` 类物体中找出离 `anchor` 最近的 k 个（按距离升序）。`res.value` = `list_objects` 同形 + `distance_m`（米）。

    ★ 这是「JSON tool call 表达不了、Python 程序一行就够」的典型例子（§13.3(2)）。
    「哪把椅子离门最近」在程序里是 `min(chairs, key=dist)`；
    换成逐步 tool-calling，模型必须自己维护一个「已算过的距离」列表并比较 ——
    对 4B 级模型是显著负担，而且中间结果不落 trace 就没法核对。
    """
    return _rank_by_distance(ctx, scene_id, anchor, label, k, "find_nearest", descending=False)


@tool("find_farthest")
def find_farthest(
    ctx,
    scene_id: str | None = None,
    anchor: str = "",
    label: str = "",
    k: int = 1,
) -> ToolResult:
    """同 `find_nearest`，但 `res.value` 的 `distance_m` 降序（离得最远的在前）。"""
    return _rank_by_distance(ctx, scene_id, anchor, label, k, "find_farthest", descending=True)


# `query_relation` 的 tol 参数要落到不同关系函数的不同形参上 —— 它们的默认值语义不同
# （`near` 的 thresh 默认 1.0 m 是「远近阈值」，其余 tol 默认 0.05 m 是「判定死区」）。
# 所以只在调用方**显式**传 tol 时才覆盖，否则让各函数用自己的默认值。
_TOL_PARAM_CANDIDATES = ("tol", "tol_v", "thresh")


@tool("query_relation")
def query_relation(
    ctx,
    scene_id: str | None = None,
    relation: str = "left_of",
    a: str = "",
    b: str = "",
    tol: float | None = None,
) -> ToolResult:
    """查询两个物体之间的某一种关系。11 种关系的统一入口。`res.value` 是 **bool**（`distance` 给 float 米），**不是 dict**。⚠ `tol`（米）覆盖该关系自带的判定死区，**默认 0.05**（`near`/`far` 是 1.0）—— 差 40 mm 的一对在默认 `tol` 下判 **False**；计数类题目别拿默认值数，`tol=0` 才是严格比较。

    为什么要有统一入口，而不是 11 个工具：工具文档的长度直接决定 prompt 长度
    （VADAR 的 program prompt 实测已达 6965 字符）。把 11 个关系收成一个工具 +
    一个枚举参数，prompt 能短一大截，而模型需要记的东西也更少
    —— 参数取值列表比 11 个函数名容易记。
    """
    if relation not in RELATIONS:
        raise ToolArgumentError(
            f"未知关系 {relation!r}；可用值：{sorted(RELATIONS)}"
        )

    scene = need_scene(ctx, scene_id)
    na = need_node(scene, a, "query_relation")
    nb = need_node(scene, b, "query_relation")

    fn = RELATIONS[relation]
    sig = inspect.signature(fn)
    kwargs: dict[str, Any] = {}
    if tol is not None:
        for cand in _TOL_PARAM_CANDIDATES:
            if cand in sig.parameters:
                kwargs[cand] = tol
                break
    if "up" in sig.parameters:
        kwargs["up"] = scene.up_axis

    verdict = fn(na, nb, **kwargs)
    return ToolResult.success(
        value=bool(verdict.value) if verdict.is_bool else float(verdict.value),
        evidence={
            "method": verdict.method,
            "scene_id": scene.scene_id,
            "relation": relation,
            "a": na.id,
            "b": nb.id,
            **verdict.metric,
        },
    )
