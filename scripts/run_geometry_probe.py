#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_geometry_probe.py —— 一档：几何误差的剂量-反应扫描 + 结构分解表。

它回答的问题
------------
`metrics.py` 只给「QA 答对没有」一个数。本脚本给的是**误差的结构**：

    哪一种错误，在**哪个**指标上，可见度是多少？

做法是往合成夹具里注入**已知的、可解释的**扰动，逐项看指标怎么响应。
这不是「跑一遍算个误差」—— 那是只报一个数；剂量-反应才能说明指标的**分辨力**：
同一份几何，`intrinsics_scale` 与 `depth_scale` 必须落在**不同**的指标格子里，
否则两个指标是重复的。

三层要分开报（这是本脚本最要紧的结构）
--------------------------------------
    ① 自洽性      predict = gt_visible ⟹ 所有误差**恰好 0**
                  不成立就说明尺子本身坏了，脚本以退出码 3 结束
    ② 固有偏差    gt_visible vs gt_box
                  可见性带来的、**任何方法都躲不掉**的部分（遮挡、只见部分表面）
    ③ 估计误差    pred vs gt_visible
                  真正要量的东西

外加一层 **②b 朝向**：轴对齐口径对斜放物体的**系统性高估**。
它不是误差，但会被误读成误差（「尺寸偏大」），所以单独报出来。

只报 ③ 不报 ②，读者会把「只看得见三个面」当成「算法不准」。
`--json-out` 里三层都在，人读表把 ①② 印在最前面。

口径
----
* 长度单位：表里是**毫米**（`mm`），JSON 里是**米**（`m`）—— 换算只发生一次，
  在打印函数里。混两套单位是这类报告最常见的读错来源。
* 质心误差恒定拆成 `total / lateral / depth` 三项。
  ⚠ `total² = lateral² + depth²`，**不是加法**（`geometry_metrics` 里有用例钉住）。
* 横纵向的判据来自相机系：x 向右、y 向下 ⟹ 横向 = 轴 0,1；纵深 = 轴 2。

零 API 成本、零 GPU、零联网：全部是解析渲染 + 纯 numpy。
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.builders.synthesize_geometry_probe import (  # noqa: E402
    PERTURBATIONS,
    Box3D,
    SyntheticScene,
    apply_perturbation,
    default_background_boxes,
    default_intrinsics,
    default_probe_boxes,
    default_scene_boxes,
    render_scene,
    visibility_report,
    visible_geometry,
)
from evaluation import geometry_metrics as gm  # noqa: E402

#: 扫描网格：`扰动名 -> 参数名 -> 取值序列`。
#: 参数取值按「工程上真实的量级」选，不是等距取点：
#:   * 内参/深度的缩放用 2%、5%、10%、25%、50%（5% 是本项目关系容差所在的量级）
#:   * 掩码用 1/2/4/8 像素（SAM2 边界误差实测在个位数像素）
#:   * 平移用 1/5/10/20 cm（20 cm 是「沙发变成 6.7 米」那种错误的量级）
SWEEPS: dict[str, tuple[str, tuple[float, ...]]] = {
    "intrinsics_scale": ("k", (1.02, 1.05, 1.10, 1.25, 1.50)),
    "depth_scale": ("k", (1.02, 1.05, 1.10, 1.25, 1.50)),
    "depth_noise": ("sigma_rel", (0.005, 0.01, 0.02, 0.05)),
    "mask_grow": ("n_px", (1.0, 2.0, 4.0, 8.0)),
    "mask_shrink": ("n_px", (1.0, 2.0, 4.0, 8.0)),
    "object_translate_x": ("dx_m", (0.01, 0.05, 0.10, 0.20)),
    "object_translate_z": ("dz_m", (0.01, 0.05, 0.10, 0.20)),
}

#: 关系准确率在哪些扰动下测（不必每个网格点都跑，关系是二次成本）。
RELATION_CASES: tuple[tuple[str, dict[str, float]], ...] = (
    ("none", {}),
    ("intrinsics_scale", {"k": 1.10}),
    ("depth_scale", {"k": 1.10}),
    ("mask_grow", {"n_px": 8.0}),
    ("object_translate_x", {"dx_m": 0.20}),
)


# ----------------------------------------------------------------------------
# 场景与真值
# ----------------------------------------------------------------------------


def _mm(x: float | None) -> str:
    """米 → 毫米字符串。`None` / `nan` 打成 `—`（**不是 0**）。"""
    if x is None or not np.isfinite(x):
        return "—"
    return "%.1f" % (1000.0 * x)


def _pred_geometry(scene: SyntheticScene, pts: np.ndarray,
                   masks: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {oid: visible_geometry(pts, masks[oid], scene.grid_hw) for oid in scene.masks}


def _per_object_errors(
    pred: dict[str, dict[str, Any]],
    gt: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, float]], list[dict[str, float]], list[str]]:
    """逐物体的 (质心误差行, 尺寸误差行, 拿不到几何的物体 id)。

    ⚠ 第三个返回值是**必须显式传递**的：`summarize` 的分母恒为 `n_expected`，
    而 `n_expected` 就是 `len(gt)`。一个物体「没算出几何」时它进 missing，
    **不能**悄悄从分母里消失 —— 那会把「3/9 失败」读成「6 个里还挺准」。
    """
    crows: list[dict[str, float]] = []
    erows: list[dict[str, float]] = []
    missing: list[str] = []
    for oid, g in gt.items():
        p = pred.get(oid) or {}
        if p.get("centroid_3d") is None or g.get("centroid_3d") is None:
            missing.append(oid)
            continue
        crows.append(gm.centroid_error(p["centroid_3d"], g["centroid_3d"]))
        erows.append(gm.extent_error(p["extent_3d"], g["extent_3d"]))
    return crows, erows, missing


def _summarize_errors(
    crows: Sequence[dict[str, float]],
    erows: Sequence[dict[str, float]],
    *,
    n_expected: int,
) -> dict[str, Any]:
    """把逐物体误差聚成一行 —— 横向/纵深分列，全部进同一个 dict。"""
    cs = gm.summarize(crows, n_expected=n_expected)
    es = gm.summarize(erows, n_expected=n_expected)
    return {
        "n_expected": n_expected,
        "n_measured": cs["n_measured"],
        "n_missing": cs["n_missing"],
        "centroid": {
            "total_m": cs.get("total_m", {}),
            "lateral_m": cs.get("lateral_m", {}),
            "depth_m": cs.get("depth_m", {}),
        },
        "extent": {
            "l1_m": es.get("l1_m", {}),
            "max_axis_m": es.get("max_axis_m", {}),
        },
    }


# ----------------------------------------------------------------------------
# 关系
# ----------------------------------------------------------------------------


def _node_from_geometry(object_id: str, label: str, centroid, extent):
    """用一组几何量造一个 `Node`，好喂给真实的关系层。"""
    from scene_graph.schema import BBox3D, Node

    c = np.asarray(centroid, dtype=np.float64)
    e = np.asarray(extent, dtype=np.float64)
    return Node(
        id=object_id,
        label=label,
        centroid_3d=tuple(float(v) for v in c),
        extent_3d=tuple(float(v) for v in e),
        bbox_3d=BBox3D(
            min=tuple(float(v) for v in (c - e / 2.0)),
            max=tuple(float(v) for v in (c + e / 2.0)),
        ),
    )


def _relation_set(geoms: dict[str, dict[str, Any]], labels: dict[str, str]) -> set[tuple[str, str, str]]:
    """把一组几何量变成 `(source, target, relation)` 集合 —— **用真实关系层**。

    不在本脚本里重写关系判据：那会给自己一把可以独立地错的尺子。
    """
    from scene_graph.relations import pairwise

    nodes = {
        oid: _node_from_geometry(oid, labels.get(oid, oid), g["centroid_3d"], g["extent_3d"])
        for oid, g in geoms.items()
        if g.get("centroid_3d") is not None
    }
    out: set[tuple[str, str, str]] = set()
    for a, b in itertools.combinations(sorted(nodes), 2):
        verdicts = pairwise(nodes[a], nodes[b])
        for rel, v in verdicts.items():
            if v.is_bool and v.value:
                out.add((a, b, rel))
    return out


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------


def run_self_consistency(scene: SyntheticScene) -> dict[str, Any]:
    """第 ① 层：无扰动时，误差必须**恰好**为 0。"""
    pts, masks = apply_perturbation(scene, "none")
    pred = _pred_geometry(scene, pts, masks)
    crows, erows, missing = _per_object_errors(pred, scene.gt_visible)
    worst_c = max((r["total_m"] for r in crows), default=0.0)
    worst_e = max((r["l1_m"] for r in erows), default=0.0)
    return {
        "n_objects": len(scene.gt_visible),
        "n_missing": len(missing),
        "max_centroid_total_m": worst_c,
        "max_extent_l1_m": worst_e,
        "all_zero": bool(worst_c == 0.0 and worst_e == 0.0 and not missing),
    }


def run_intrinsic_bias(scene: SyntheticScene) -> dict[str, Any]:
    """第 ② 层：`gt_visible` vs `gt_box` —— 可见性固有偏差。

    它**不是误差**，是口径的已知代价：物体的背面看不见，所以可测到的质心
    与包围盒天然不是盒子的那一份。谁都不该把这一层算到方法头上。
    """
    crows, erows, missing = _per_object_errors(scene.gt_visible, scene.gt_box)
    return {
        "centroid": _summarize_errors(crows, erows, n_expected=len(scene.gt_box)),
        "per_object": {
            oid: {
                "centroid_total_m": float(np.linalg.norm(
                    np.asarray(scene.gt_visible[oid]["centroid_3d"], dtype=np.float64)
                    - np.asarray(scene.gt_box[oid]["centroid_3d"], dtype=np.float64))),
                "extent_l1_m": float(np.abs(
                    np.asarray(scene.gt_visible[oid]["extent_3d"], dtype=np.float64)
                    - np.asarray(scene.gt_box[oid]["extent_3d"], dtype=np.float64)).sum()),
            }
            for oid in scene.gt_box
        },
        "n_missing": len(missing),
    }


def run_sweep(scene: SyntheticScene, kind: str, param: float) -> dict[str, Any]:
    """第 ③ 层：一个网格点上，pred vs gt_visible 的误差结构。

    ⚠ **聚合方式必须匹配扰动的语义。** 平移类扰动只动**一个**物体，
    此时全体统计的中位数**必然是 0**（其余 8 个没动），而 P90 也只有真值的 20%
    （9 个样本的 90 分位落在第 7~8 名之间插值）。

    第一版就是这样报的：表格显示 `object_translate_x 0.20 → 中位 0.0、P90 40.0 mm`，
    而真相是「那个物体偏了整整 200 mm」。它**看起来像指标失灵**，
    实际是「用全体统计去描述一个单物体扰动」。所以这里同时给出
    `summary`（全体）与 `target_summary`（被扰动的那一个），打印时按语义选。
    """
    pname, _ = SWEEPS[kind]
    kw: dict[str, Any] = {pname: param}
    target: str | None = None
    if kind.startswith("object_translate"):
        target = "sofa_1"
        kw["object_id"] = target

    pts, masks = apply_perturbation(scene, kind, **kw)
    pred = _pred_geometry(scene, pts, masks)
    crows, erows, missing = _per_object_errors(pred, scene.gt_visible)
    out: dict[str, Any] = {
        "kind": kind,
        "param_name": pname,
        "param": param,
        "summary": _summarize_errors(crows, erows, n_expected=len(scene.gt_visible)),
        "missing_objects": missing,
    }
    if target is not None:
        tc = [gm.centroid_error(pred[target]["centroid_3d"],
                                scene.gt_visible[target]["centroid_3d"])]
        te = [gm.extent_error(pred[target]["extent_3d"],
                              scene.gt_visible[target]["extent_3d"])]
        out["target_object"] = target
        out["target_summary"] = _summarize_errors(tc, te, n_expected=1)
    return out


def run_relation_cases(scene: SyntheticScene, labels: dict[str, str]) -> list[dict[str, Any]]:
    """关系准确率：同一份扰动下，关系层还能不能给对答案。

    §22 的实测是「关系对几何错误**不敏感**」（128/133 = 96% 一致）。
    这里把它变成一条可复算的对照：如果几何误差涨了十倍而 F1 基本不动，
    那说明「关系准确率」这个指标**看不见**这类错误 —— 它不能单独当护栏。
    """
    gt_set = _relation_set(scene.gt_visible, labels)
    rows: list[dict[str, Any]] = []
    for kind, kw in RELATION_CASES:
        if kind == "none":
            pts, masks = scene.points_chw, scene.masks
        else:
            pname, _ = SWEEPS[kind]
            call_kw = dict(kw)
            if kind.startswith("object_translate"):
                call_kw["object_id"] = "sofa_1"
            pts, masks = apply_perturbation(scene, kind, **call_kw)
        pred = _pred_geometry(scene, pts, masks)
        got_set = _relation_set(pred, labels)
        prf = gm.relation_prf(got_set, gt_set)
        rows.append({"kind": kind, "params": kw, "n_gt": len(gt_set), **prf})
    return rows


# ----------------------------------------------------------------------------
# 朝向层
# ----------------------------------------------------------------------------


#: 朝向层扫描的 yaw（度）。0 是参照 —— 等于「不加朝向」，此时全部轴对齐。
YAW_DEGREES: tuple[float, ...] = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)

#: 朝向层的盒子：`Lx = 1.00 / Ly = 0.75 / Lz = 0.70 m`。
#: **`Lx ≠ Lz` 是刻意的** —— 若相等，45° 处 x/z 的互换在数字上看不出来。
#: **偏轴**（x ≈ 0.9，不跨光轴）也是刻意的：轴上盒子的可见面只有正对那一个，
#: 它的轴对齐 x 跨度恰好等于 `Lx`，口径高估在它身上不显形。
ORIENTATION_BOX_MIN: tuple[float, float, float] = (0.40, -0.075, 2.65)
ORIENTATION_BOX_MAX: tuple[float, float, float] = (1.40, 0.675, 3.35)


def orientation_boxes(yaw_rad: float) -> list[Box3D]:
    """朝向层的场景：**一个盒子 + 背景（只为提供深度）**。

    为什么不用那 9 个物体的主场景：旋转一个物体会**改变遮挡关系**，
    于是「测得尺寸」的变化里同时含口径效应与可见性变化 —— 两条线缠在一起，
    曲线就不可归因了。宁可牺牲场景真实性换一条能解释的曲线：
    真实性由主场景负责，这一层只负责把口径量干净。
    """
    return [
        Box3D("probe_box", "box", ORIENTATION_BOX_MIN, ORIENTATION_BOX_MAX,
              yaw_rad=float(yaw_rad)),
        *default_background_boxes(),
    ]


def run_orientation_layer(
    *,
    intrinsics: np.ndarray,
    image_hw: tuple[int, int],
    degrees: Sequence[float] = YAW_DEGREES,
) -> list[dict[str, Any]]:
    """第 ②b 层：把盒子从 0° 转到 90°，看**实测尺寸**怎么变。

    三个量必须同时报，少一个就会读错：

        body      盒**自身轴**向边长 —— 物体真实尺寸，与朝向无关
        aabb      旋转后**轴对齐**跨度（解析）—— 口径上界
        measured  真实下游在**可见表面**上算出的轴对齐跨度

    于是 `measured − body` **不是**算法误差，而是
    「口径高估（≤ `aabb − body`，由朝向决定）＋ 可见性偏差」。
    这正是加朝向新增的可测量：没有朝向自由度时它 ≤ 0，一旦斜放就变正 ——
    而它以前只能被记在算法头上。

    每行都带 `self_consistency_max_m`：朝向打开后尺子自检**仍须为 0**，
    那是「本层数字可信」的前提。
    """
    rows: list[dict[str, Any]] = []
    for deg in degrees:
        yaw = float(np.radians(deg))
        scene = render_scene(orientation_boxes(yaw), intrinsics=intrinsics,
                             image_hw=image_hw)
        pred = _pred_geometry(scene, scene.points_chw, scene.masks)
        crows, erows, missing = _per_object_errors(pred, scene.gt_visible)
        worst_self = max([r["total_m"] for r in crows]
                         + [r["l1_m"] for r in erows], default=0.0)

        tgt = orientation_boxes(yaw)[0]
        oid = tgt.object_id
        body = np.asarray(tgt.extent, dtype=np.float64)
        aabb = np.asarray(tgt.aabb_extent, dtype=np.float64)
        meas = np.asarray(scene.gt_visible[oid]["extent_3d"], dtype=np.float64)
        shift = (np.asarray(scene.gt_visible[oid]["centroid_3d"], dtype=np.float64)
                 - np.asarray(scene.gt_box[oid]["centroid_3d"], dtype=np.float64))
        rows.append({
            "yaw_deg": float(deg),
            "object_id": oid,
            "n_visible_px": int(scene.meta["boxes"][0]["n_visible_px"]),
            "body_extent_m": [float(v) for v in body],
            "aabb_extent_m": [float(v) for v in aabb],
            "measured_extent_m": [float(v) for v in meas],
            #: 口径倍数 = 轴对齐跨度 ÷ 物体**自身** x 边长。
            #: =1.00 表示口径没引入偏差；>1 是膨胀，<1 是「转过去了，x 方向本来就短了」。
            #: ⚠ 它随朝向从 1.00 摆到 1.22 又摆到 0.70 —— **同一刚体**，读数变了 74%，
            #: 而物体一寸没动。这就是「轴对齐口径 ≠ 尺寸」的量化形式。
            "caliber_ratio_vs_body_x": float(aabb[0] / body[0]),
            #: 可见性捕获率 = 实测 ÷ 口径上界。**恒 ≤ 1**：只看得见一部分表面，
            #: 测得值不可能超过整个盒子的轴对齐跨度。
            "capture_ratio_x": float(meas[0] / aabb[0]),
            "centroid_shift_m": [float(v) for v in shift],
            "self_consistency_max_m": float(worst_self),
            "missing_objects": missing,
        })
    return rows


# ----------------------------------------------------------------------------
# 打印
# ----------------------------------------------------------------------------


def print_report(result: dict[str, Any]) -> None:
    vis = result["visibility"]
    sc = result["self_consistency"]
    # `--dry-run` 只有 ①，所以 ②③ 用 .get 取 —— 缺段时跳过而不是崩，
    # 这样干运行也能把「尺子检查」印出来（那正是干运行要看的东西）。
    ib = result.get("intrinsic_bias")

    print("=" * 96)
    print("一档：单目 3D 几何误差的剂量-反应扫描")
    print("=" * 96)
    print("场景：解析合成（射线-盒求交 + z-buffer），%d 个物体 + 背景" % len(vis["objects"]))
    print("渲染：%d×%d，fx=%.0f；全部为纯 numpy，零 API、零 GPU"
          % (vis["image_hw"][1], vis["image_hw"][0], result["intrinsics"][0][0]))
    print()

    # ---- 可见性体检（先看它，否则后面的误差数字不可信）----
    print("-" * 96)
    print("场景体检：每个物体是否出界 / 被遮挡（可见像素太少会让离散化主导误差）")
    print("-" * 96)
    print("%-12s %10s %11s %9s" % ("object", "px_alone", "px_in_scene", "vis_frac"))
    for o in vis["objects"]:
        flag = ""
        if o["off_screen"]:
            flag = "  <-- 出界！"
        elif o["visible_frac"] < 0.90:
            flag = "  <-- 部分被遮挡"
        print("%-12s %10d %11d %9.3f%s"
              % (o["object_id"], o["px_alone"], o["px_in_scene"], o["visible_frac"], flag))
    print("  最差可见率 %.3f ｜ 最小可见像素 %d ｜ 无命中像素 %d"
          % (vis["worst_visible_frac"], vis["min_px_in_scene"], vis["no_hit_px"]))
    print()

    # ---- 第 ① 层 ----
    print("-" * 96)
    print("[① 自洽性] 预测 = gt_visible ⟹ 所有误差必须**恰好**为 0（尺子检查）")
    print("-" * 96)
    print("  物体 %d 个 ｜ 拿不到几何 %d 个 ｜ 质心最大误差 %.3e m ｜ 尺寸最大误差 %.3e m"
          % (sc["n_objects"], sc["n_missing"], sc["max_centroid_total_m"], sc["max_extent_l1_m"]))
    print("  结论：%s" % ("通过 —— 尺子没错" if sc["all_zero"] else "**失败 —— 先修指标，别看下面的表**"))
    print()

    # ---- 第 ② 层 ----
    if ib is None:
        print("（--dry-run：略过 ② 固有偏差与 ③ 扫描）")
        print("=" * 96)
        return
    print("-" * 96)
    print("[② 固有偏差] gt_visible vs gt_box：可见性造成的、任何方法都躲不掉的部分")
    print("-" * 96)
    ic = ib["centroid"]
    print("  质心位移  中位 %s mm ｜ P90 %s mm"
          % (_mm(ic["centroid"]["total_m"].get("median")), _mm(ic["centroid"]["total_m"].get("p90"))))
    print("    ├ 横向  中位 %s mm" % _mm(ic["centroid"]["lateral_m"].get("median")))
    print("    └ 纵深  中位 %s mm" % _mm(ic["centroid"]["depth_m"].get("median")))
    print("  尺寸 L1   中位 %s mm" % _mm(ic["extent"]["l1_m"].get("median")))
    print("  ⚠ 这一层**不是误差**：物体背面看不见，可测到的质心与包围盒本来就不是盒子的那一份。")
    print()

    # ---- 第 ③ 层：剂量-反应 ----
    print("-" * 96)
    print("[③ 估计误差] 注入已知扰动后的响应 —— 单位 mm，中位数（括号内为 P90）")
    print("-" * 96)
    hdr = ("%-28s %7s %19s %19s %19s %16s %8s"
           % ("扰动", "参数", "质心 total", "质心 横向", "质心 纵深", "尺寸 L1", "缺几何"))
    print(hdr)
    print("-" * len(hdr))
    for kind, rows in result["sweeps"].items():
        for r in rows:
            # 平移类只动一个物体 ⟹ 全体统计的中位数必然是 0，用它会把
            # 「那个物体偏了 200 mm」印成「0.0」。按扰动的语义选聚合口径。
            single = r.get("target_object")
            s = r["target_summary"] if single else r["summary"]
            c = s["centroid"]
            e = s["extent"]
            name = "%s:%s" % (kind, single) if single else kind
            print("%-28s %7.3f %8s(%7s) %8s(%7s) %8s(%7s) %8s(%7s) %8d"
                  % (name, r["param"],
                     _mm(c["total_m"].get("median")), _mm(c["total_m"].get("p90")),
                     _mm(c["lateral_m"].get("median")), _mm(c["lateral_m"].get("p90")),
                     _mm(c["depth_m"].get("median")), _mm(c["depth_m"].get("p90")),
                     _mm(e["l1_m"].get("median")), _mm(e["l1_m"].get("p90")),
                     s["n_missing"]))
        print()
    print("  读法：**横向 vs 纵深两列要分开看**。同一份几何，不同扰动必须落在不同的格子里；")
    print("        两列一起变大说明该扰动是「整体尺度错」，只有一列变大才是「单轴错」。")
    print("  ⚠ 质心 total **不是** 横向 + 纵深，而是 sqrt(横向² + 纵深²)。")
    print("  ⚠ 标 `:sofa_1` 的行只统计**被平移的那一个物体**（其余 8 个没动，")
    print("     用全体统计会把 200 mm 印成 0.0 —— 那是聚合口径错，不是指标失灵）。")
    print()

    # ---- 关系 ----
    print("-" * 96)
    print("[③b 关系层] 同一份扰动下，关系判定的 precision / recall / F1")
    print("-" * 96)
    print("%-20s %8s %10s %10s %10s %10s" % ("扰动", "n_gt", "TP", "FP", "FN", "F1"))
    for r in result["relation_cases"]:
        print("%-20s %8.0f %10.0f %10.0f %10.0f %10.3f"
              % (r["kind"], r["n_gt"], r["tp"], r["fp"], r["fn"], r["f1"]))
    print("  读法：若几何误差涨了十倍而 F1 基本不动 ⟹ 关系准确率**看不见**这类错误，")
    print("        它不能单独当护栏（§22 的 128/133 = 96% 是同一现象）。")
    print()

    # ---- 第 ②b 层：朝向 ----
    orient = result.get("orientation")
    if orient:
        print("-" * 96)
        print("[②b 朝向层] 轴对齐口径对斜放物体的**系统性高估**（只转一个盒子）")
        print("-" * 96)
        hdr2 = ("%-5s %9s %9s %9s %11s %9s %9s %8s"
                % ("yaw", "body_x", "body_z", "aabb_x", "aabb/body", "meas_x", "capture", "px"))
        print(hdr2)
        print("-" * len(hdr2))
        for r in orient:
            b = r["body_extent_m"]
            a = r["aabb_extent_m"]
            m = r["measured_extent_m"]
            print("%-5s %9s %9s %9s %11.3f %9s %9.3f %8d"
                  % ("%.0f°" % r["yaw_deg"],
                     _mm(b[0]), _mm(b[2]), _mm(a[0]), r["caliber_ratio_vs_body_x"],
                     _mm(m[0]), r["capture_ratio_x"], r["n_visible_px"]))
        lo_a = min(r["aabb_extent_m"][0] for r in orient)
        hi_a = max(r["aabb_extent_m"][0] for r in orient)
        print("  单位 mm。`body_*` = 物体**自轴**边长（刚体，各朝向全同 —— 这才是尺寸）；")
        print("         `aabb_x` = 轴对齐跨度（下游 `robust_extent` 的口径）；`capture` = meas_x ÷ aabb_x。")
        print("  读法：**同一刚体**，口径上界从 %s 摆到 %s（%.2f×），而物体一寸没动 ⟹"
              % (_mm(lo_a), _mm(hi_a), hi_a / lo_a))
        print("         「轴对齐尺寸」这个读数**同时携带尺寸与朝向、两者不可分辨**。")
        print("         口径上界本身最高比自轴 x 边长膨胀 %+.0f%%（峰值 = √(Lx²+Lz²)，θ≈35°）；"
              % (100.0 * (max(r["caliber_ratio_vs_body_x"] for r in orient) - 1.0)))
        print("         实测读数还会被可见性再压一层（`capture` 恒 ≤ 1），最终落在两者之间。")
        print("         加朝向之前，这个「读数随朝向而变」的量在夹具里**根本不存在**。")
        worst = max((r["self_consistency_max_m"] for r in orient), default=0.0)
        print("  本层尺子自检：所有 yaw 下最大误差 %.3e m（必须为 0）" % worst)
        print()
    print("=" * 96)
    print("JSON 详情（含逐物体数字）见 --json-out 指定的文件。")
    print("=" * 96)


#: 默认视场：`fx = FOV_RATIO × width`。0.75 对应 320 宽的一张 67° HFoV。
#: 之所以按**宽度**定 fx，是为了让「改分辨率」不改视场 —— 否则降采样跑一次
#: 会把物体挤出画面，而它的表现是「几何误差突然变大」或「拿不到几何」。
#: 这个坑是测试逼出来的：`--width 160` 配 `fx=240` 时视场只剩 37°，
#: 自洽性直接变红。
FOV_RATIO = 0.75


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="一档：单目 3D 几何误差的剂量-反应扫描与结构分解。零 API 成本。")
    ap.add_argument("--json-out", default="reports/geometry_probe.json",
                    help="结果 JSON 落盘路径（米为单位）")
    ap.add_argument("--width", type=int, default=320, help="渲染宽度")
    ap.add_argument("--height", type=int, default=240, help="渲染高度")
    ap.add_argument("--fx", type=float, default=None,
                    help="焦距像素值；默认 %.2f×width（保持视场不随分辨率变）" % FOV_RATIO)
    ap.add_argument("--dry-run", action="store_true", help="只做体检与自洽性，不跑扫描")
    args = ap.parse_args(argv)

    image_hw = (args.height, args.width)
    fx = args.fx if args.fx is not None else FOV_RATIO * args.width
    intr = default_intrinsics(fx=fx, cx=args.width / 2.0, cy=args.height / 2.0)
    boxes = default_scene_boxes()

    print("渲染默认合成场景 ...", flush=True)
    scene = render_scene(boxes, intrinsics=intr, image_hw=image_hw)
    labels = {b.object_id: b.label for b in default_probe_boxes()}

    print("场景体检 ...", flush=True)
    vis = visibility_report(boxes, intrinsics=intr, image_hw=image_hw)

    print("第 ① 层：自洽性 ...", flush=True)
    sc = run_self_consistency(scene)

    result: dict[str, Any] = {
        "image_hw": [image_hw[0], image_hw[1]],
        "intrinsics": intr.tolist(),
        "scene_boxes": [
            {"object_id": b.object_id, "label": b.label, "min_xyz": list(b.min_xyz),
             "max_xyz": list(b.max_xyz), "is_background": b.is_background}
            for b in boxes
        ],
        "units": {"length_in_json": "m", "length_in_printed_table": "mm"},
        "perturbations_documented": PERTURBATIONS,
        "visibility": vis,
        "self_consistency": sc,
    }

    if not args.dry_run:
        print("第 ② 层：固有偏差 ...", flush=True)
        result["intrinsic_bias"] = run_intrinsic_bias(scene)

        print("第 ③ 层：剂量-反应扫描 ...", flush=True)
        sweeps: dict[str, list[dict[str, Any]]] = {}
        for kind, (_, values) in SWEEPS.items():
            rows = []
            for v in values:
                # 平移类的 0 无意义（等于不扰动），跳过以省一行噪声
                rows.append(run_sweep(scene, kind, float(v)))
            sweeps[kind] = rows
        result["sweeps"] = sweeps

        print("第 ③b：关系层 ...", flush=True)
        result["relation_cases"] = run_relation_cases(scene, labels)

        print("第 ②b：朝向层 ...", flush=True)
        result["orientation"] = run_orientation_layer(intrinsics=intr, image_hw=image_hw)
        result["orientation_scene"] = {
            "box_min": list(ORIENTATION_BOX_MIN),
            "box_max": list(ORIENTATION_BOX_MAX),
            "yaw_degrees": list(YAW_DEGREES),
            "note": ("单物体 + 背景：旋转会改变遮挡 ⟹ 多物体场景会把可见性变化"
                     "混进口径效应，曲线就不可归因"),
        }

    out = Path(args.json_out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("JSON 已写入 %s" % out, flush=True)
    print()

    print_report(result)

    if not sc["all_zero"]:
        print("\n退出码 3：自洽性检查失败 —— 指标本身有问题，任何下游结论都不成立。")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
