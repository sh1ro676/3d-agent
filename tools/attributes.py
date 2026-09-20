#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L4 视觉语义工具 —— 角色②（`llm/vlm.py`）的**外部层**。

职责只有一条：把「一个 object_id + 想要哪些属性」翻译成一次受控的视觉调用，
并把结果包成 `ToolResult`。它自己做三件事，都在这一层才做得了：

1. **能力闸门。** 装饰器上写 `capability="vlm"`，于是 `ctx.vlm=None` 时本工具
   自动返回 `CAPABILITY_DISABLED`（`recovery = use_geometry / abstain`）。
   「关掉视觉语义」因此是**一次配置**，不是一份改写过的提示词 —— 这是消融可复现的关键（§13.3(7)）。

2. **把 `Node` 组装成 `Region`。** 裁剪用的 bbox 由这里从场景图取，
   **不进提示词**：角色② 的输入侧也不该出现坐标数字。模型看到的是「一张裁出来的小图，
   里面有一个椅子」，不是「bbox=(120, 340, 480, 610) 的物体」。

3. **上报不确定。** 置信度低于阈值、或取值落在闭集外时，返回 `LOW_CONFIDENCE`
   而不是把值当确定值给出去。**但值仍然放在 `context["attributes"]` 里** ——
   与 `AMBIGUOUS` 带回 `candidates` 同一套做法：让程序有机会自己判断，
   但绝不让「不确定」看起来像「确定」。

缓存
====
`Node.attributes` 是可写的 dict（`Node` 本身 frozen），所以命中缓存时直接返回、
不再调模型。**只缓存高置信度的结果** —— 把不确定的值写进场景图，
等于让一次猜测在后续所有问题里被当成事实，那比多花一次调用贵得多。

分层：本文件**不 import `llm`**（有一条 AST 静态检查钉着这件事）
============================================================
本工具是唯一会用到视觉后端的工具，但它的依赖方向仍然守规矩：

    from vision.semantics import ATTRS, DEFAULT_CONFIDENCE_THRESHOLD, Region, SemanticBackendError

词表、`Region`、失败类型都住在**中性契约层**（`vision/semantics.py`），
视觉后端本身则在运行时由 `ctx.vlm` **注入**（一个只需要有 `describe()` 的鸭子类型）。
于是「关掉视觉语义」这条消融臂是**真的少了一条依赖**，
而不是"把提示词改了一版"：工具层在 `ctx.vlm=None` 时照样构造得出来。
**注入是运行时的事，类型是编译期的事** —— 用不着为了前者在后者上开例外。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from vision.semantics import (
    ATTRS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    Region,
    SemanticBackendError,
)
from tools.guards import need_node, need_scene
from tools.registry import ToolArgumentError, tool
from tools.result import ErrorCode, ToolResult

__all__ = ["get_attributes"]


def _want_attrs(attrs: Any) -> list[str]:
    """规范化 `attrs` 参数：None → 全部；str → 单个；列表 → 去重保序。

    未知属性一律 `ToolArgumentError`（**故意让它冒泡**）：那是程序写错了，
    不是场景有问题，必须在失败诊断里归到「程序」而不是「工具」（见 registry 的异常策略）。
    """
    if attrs is None:
        return list(ATTRS)
    if isinstance(attrs, str):
        items: Sequence[Any] = [attrs]
    elif isinstance(attrs, (list, tuple, set, frozenset)):
        items = list(attrs)
    else:
        raise ToolArgumentError(
            "attrs 必须是字符串或字符串列表，收到 %r。可用值：%s" % (attrs, ", ".join(ATTRS))
        )

    out: list[str] = []
    for raw in items:
        name = str(raw).strip().lower()
        if not name:
            continue
        if name not in ATTRS:
            raise ToolArgumentError(
                "不支持的属性 %r。可用值：%s。"
                "（位置/朝向/远近/大小**不在其中** —— 那些一律用几何工具回答："
                "get_3d_position / calculate_distance / query_relation。）"
                % (name, ", ".join(ATTRS))
            )
        if name not in out:
            out.append(name)
    if not out:
        raise ToolArgumentError("attrs 解析后为空。可用值：%s" % ", ".join(ATTRS))
    return out


def _want_candidates(candidates: Any, want: Sequence[str]) -> dict[str, list[str]] | None:
    """规范化闭集参数。给了就必须是非空字符串列表 —— 空集合会静默废掉闭集约束。"""
    if not candidates:
        return None
    if not isinstance(candidates, Mapping):
        raise ToolArgumentError(
            "candidates 必须是 {属性名: [候选值, ...]} 这样的字典，收到 %r" % (candidates,)
        )
    out: dict[str, list[str]] = {}
    for key, values in candidates.items():
        name = str(key).strip().lower()
        if name not in ATTRS:
            raise ToolArgumentError("candidates 里有未知属性 %r（可用：%s）" % (name, ", ".join(ATTRS)))
        if name not in want:
            # 提了闭集却没要这个属性 —— 静默忽略会让模型以为自己约束上了。
            raise ToolArgumentError(
                "candidates 里给了 %r 的闭集，但 attrs 里没有要它。请把 %r 也加进 attrs。"
                % (name, name)
            )
        if isinstance(values, str):
            values = [values]
        clean = [str(v).strip() for v in values if str(v).strip()]
        if not clean:
            raise ToolArgumentError("candidates[%r] 是空的 —— 空闭集等于没有约束，请去掉它。" % name)
        out[name] = clean
    return out or None


def _image_for(ctx: Any, image_id: str) -> Any | None:
    """按 `image_id` 取图；取不到时若会话里只有一张图，就用那一张。

    单图场景下这个兜底很有用：调用方可能用文件名而不是 `image_id` 当键。
    多图场景则**不做猜测** —— 猜错会把 A 图的颜色安到 B 图的物体上，
    而那种错误在任何后续检查里都看不出来。
    """
    images: Mapping[str, Any] = getattr(ctx, "images", None) or {}
    if image_id and image_id in images:
        return images[image_id]
    if len(images) == 1:
        return next(iter(images.values()))
    return None


@tool("get_attributes", capability="vlm")
def get_attributes(
    ctx,
    scene_id: str | None = None,
    object_id: str = "",
    attrs: list[str] | None = None,
    candidates: dict | None = None,
) -> ToolResult:
    """读取某个物体在**视觉语义**上的属性（颜色 / 材质 / 纹理 / 状态 / 形状）。成功时 `res.value` 是 **list**，每项形如 `{"name": "color", "value": "beige", "confidence": 0.9}` —— 按 `name` 取，不要当成 `{"color": ...}` 这种按属性名索引的 dict。置信度低或落在闭集外时整条返回 `LOW_CONFIDENCE`，值仍在 `context["attributes"]` 里。

    只回答「它是什么样」，**不回答「它在哪」** —— 位置、朝向、远近、大小一律用几何工具
    （`get_3d_position` / `calculate_distance` / `query_relation`）。
    本能力在实验臂里可以被关掉，届时会返回 `CAPABILITY_DISABLED`。

    ⚠ 返回值形状**必须写在第一段**，因为 `llm/schema.py::first_paragraph` 只把第一段
    渲染进提示词 —— 写在下面的设计说明模型一个字也看不到。
    （2026-09-18 实测：写了形状之前的第一次真跑，模型把 `value` 猜成按属性名索引的 dict，
    于是把一个已经算对的 `beige` 丢成了弃答。）
    """
    scene = need_scene(ctx, scene_id)
    node = need_node(scene, object_id, "get_attributes")
    want = _want_attrs(attrs)
    closed = _want_candidates(candidates, want)

    # -- 缓存：只认高置信度的旧值 ------------------------------------------------
    if closed is None:
        cached = {k: v for k, v in node.attributes.items() if k in want}
        if len(cached) == len(want):
            return ToolResult.success(
                value=[{"name": k, "value": cached[k], "confidence": 1.0,
                        "source": "scene_graph_cache", "in_closed_set": True, "raw": ""}
                       for k in want],
                evidence={"scene_id": scene.scene_id, "object_id": node.id, "label": node.label,
                          "attrs": list(want), "cached": True, "source": "scene_graph_cache"},
                tool="get_attributes",
                cached=True,
            )

    # -- 取图 -------------------------------------------------------------------
    image_id = str(getattr(scene, "image_id", "") or "")
    image = _image_for(ctx, image_id)
    if image is None:
        return ToolResult.failure(
            ErrorCode.NOT_FOUND,
            "本次会话没有可用的图像，无法读取视觉属性",
            context={
                "scene_id": scene.scene_id,
                "image_id": image_id,
                "loaded_image_keys": sorted((getattr(ctx, "images", None) or {}).keys()),
                "hint": "构建 ToolContext 时把图像放进 ctx.images[image_id]；"
                        "或者改用几何工具回答（视觉属性无法从场景图推出）。",
            },
            tool="get_attributes",
        )

    region = Region.for_node(scene, node)
    vlm = ctx.vlm
    try:
        values = vlm.describe(image, region, want, closed)
    except SemanticBackendError as exc:
        # 后端不可用。**刻意复用 CAPABILITY_DISABLED 而不是新加第七个错误码**：
        # §13.3(1) 的六个码是冻结契约，而这里对程序来说要做的动作与「能力被关掉」
        # 完全一样（改用几何 / 弃答）。两种情况靠 `context.reason` 区分，
        # 于是统计上仍然分得开（见 §13.3(7)：消融造成的失败必须可辨识）。
        return ToolResult.failure(
            ErrorCode.CAPABILITY_DISABLED,
            "视觉语义能力本次不可用：%s" % exc,
            context={
                "capability": "vlm",
                "reason": "backend_unavailable",
                "object_id": node.id,
                "hint": "改用几何工具回答本题；若题目问的就是颜色/材质，如实弃答。",
            },
            tool="get_attributes",
        )

    call = getattr(vlm, "last_call", None)
    report = call.to_dict() if call is not None else {}
    payload = [a.to_dict() for a in values]

    # ★ 判断「不确定」的主依据是**返回值本身**，不是 `last_call` 这个侧信道。
    #   理由：一个 duck-typed 的视觉后端可能根本不维护 `last_call`，
    #   而「低置信度必须上报」是这条链路上的一条**安全属性** ——
    #   安全属性不能依赖一个可选的旁路字段。`last_call` 只用来**补充**信息
    #   （例如后端自己额外标了什么），两者取并集。
    threshold = float(getattr(vlm, "threshold", DEFAULT_CONFIDENCE_THRESHOLD))
    low = list(dict.fromkeys(
        list(report.get("low_confidence") or ())
        + [a.name for a in values if not a.value or a.confidence < threshold]
    ))
    violations = list(dict.fromkeys(
        list(report.get("violations") or ())
        + [a.name for a in values if not a.in_closed_set]
    ))

    base_evidence: dict[str, Any] = {
        "scene_id": scene.scene_id,
        "object_id": node.id,
        "label": node.label,
        "attrs": list(want),
        "region": region.to_dict(),
        "source": "vlm",
        "threshold": threshold,
        "model": report.get("model") or getattr(getattr(vlm, "settings", None), "model", None),
        "usage": report.get("usage") or {},
        "elapsed_s": report.get("elapsed_s"),
        "cropped": report.get("cropped"),
        "values": {a.name: a.value for a in values if a.value},
        "confidences": {a.name: round(a.confidence, 4) for a in values},
        "closed_set": closed,
    }

    if violations or low:
        return ToolResult.failure(
            ErrorCode.LOW_CONFIDENCE,
            "视觉属性不确定：%s。**不要把不确定的值当成确定答案**。"
            % ("；".join(filter(None, [
                "取值不在给定闭集内：" + ", ".join(violations) if violations else "",
                "置信度低于阈值：" + ", ".join(low) if low else "",
            ]))),
            context={
                "object_id": node.id,
                # 值照样给 —— 让程序有机会自己取舍，但绝不让它「看起来像确定值」。
                "attributes": payload,
                "violations": violations,
                "low_confidence": low,
                "threshold": threshold,
                "hint": "可以据此如实弃答，或换一个更有区分度的属性/闭集再试一次。",
            },
            evidence=base_evidence,
            tool="get_attributes",
        )

    # -- 只缓存高置信度结果 ------------------------------------------------------
    for a in values:
        if a.value and a.in_closed_set:
            node.attributes[a.name] = a.value

    return ToolResult.success(
        value=payload,
        evidence=base_evidence,
        tool="get_attributes",
    )
