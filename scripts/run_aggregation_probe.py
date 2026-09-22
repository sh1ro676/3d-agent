#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_aggregation_probe.py —— 四档：**经真实 builder** 的聚合段探针。

它回答的问题
------------
`run_geometry_probe.py` 量的是**几何量本身**（直接调 `visible_geometry`，
绕开 build_scene_graph）。本脚本量的是**聚合段**：从 3D 盒子 → 合成感知桥
→ `build_scene_graph` 的 ①②③④⑤⑥ 全段 → 场景图。两者不是重复，因为
聚合段里有一批东西绕不过去就会漏测：

* `detect` 的 prompt 过滤（**漏一个类别 = 图里少一个物体**，不报错）
* `segment` 的框→掩码匹配
* `bbox_fallback`：掩码太空时退回检测框（真实照片里的**常态**）
* `no_valid_points`：掩码与框内都无有效点，物体**整个消失**

后两条此前从没在任何有真值的数据上被量过 —— `tests/test_builder.py` 的
fake 用的是解析常量深度，没有遮挡、没有背景、没有 nan 空洞，
两条路径都没有触发条件。

四档
----
    [① 基线]              全类别 prompt，走一遍完整真实链路
    [② 掩码抹空]          逐物体注入 ⟹ `bbox_fallback`，量**净代价**
    [③ 深度空洞]          逐物体注入 ⟹ `no_valid_points`，量节点/边损失与副作用
    [④ 双路对照]          builder 的 ④ 段 vs `visible_geometry`，**要求逐位相等**
    [⑤ prompt 召回]       默认 prompt（5 类）vs 补齐 prompt（9 类）
    [⑥ 内参守卫]          守卫必须**响**；并钉住 `intrinsics_source` 在本桥退化

口径（比数字重要）
------------------
* `[②]` 的 `delta_*` = 降级组的几何量 − **基线同物体**的几何量。
  这是降级的**净代价**：可见性固有偏差在两组里都有，相减即消掉。
  同一个物体在 `err_vs_gt_*` 里相对**盒中心**的误差则**含**可见性偏差，
  两项都报但名字不同 —— 把后者当降级代价读，会把「只看得见三个面」
  算到「掩码失败」头上。
* `[④]` 用 `==` 而不是 `approx`。两层走的是同一份点云、同一份掩码
  （`resample_mask_to` 在网格相同时返回同一块内存），所以「几乎相等」
  就说明其中一层多做了点什么 —— 那是缺陷，不是舍入。
* 长度单位：表里是**毫米**，JSON 里是**米**（换算只发生一次，在打印函数里）。

退出码
------
    0  一切正常
    3  双路对照不一致 —— **尺子坏了**（同 `run_geometry_probe.py` 的先例）
    4  注入了却没降级 —— 比结果不对更糟：报告会显示「一切正常」
    5  内参守卫没响 —— 「守卫从未生效」比「守卫失败」更难发现

零 API、零 GPU、零联网。整脚本秒级。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluation.geometry_metrics as gm  # noqa: E402
import vision.grounding as grounding  # noqa: E402
from dataset.builders.synthesize_geometry_probe import (  # noqa: E402
    Box3D,
    SyntheticScene,
    default_intrinsics,
    default_scene_boxes,
    visible_geometry,
)
from dataset.builders.synthetic_perception import (  # noqa: E402
    Degradation,
    SyntheticIntrinsicsError,
    SyntheticPerception,
)
from scene_graph.builder import BuildConfig, build_scene_graph  # noqa: E402
from scene_graph.relations import DEFAULT_TOL  # noqa: E402
from vision.geometry import (  # noqa: E402
    DEFAULT_K_P90,
    box_coverage,
    centroid_of,
    robust_extent,
    select_points,
)

IMAGE_HW = (240, 320)

#: `[②b]` 的检测框外扩网格，单位像素。0 = 框刚好是物体外接矩形（最乐观的一头）。
#: 24 px 在 320 宽上已是 7.5% 的画面宽度，超过真实 GroundingDINO 的典型偏大。
PAD_GRID: tuple[float, ...] = (0.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0)

#: 项目默认 prompt，来自 `vision/grounding.py`。**只有 5 类**。
DEFAULT_PROMPT = grounding.DEFAULT_PROMPT

#: 覆盖夹具全部 9 类的 prompt。
#: ⚠ 注意 `painting` —— 夹具里那个挂画叫 `painting`，而默认 prompt 写的是
#: `picture`。两个词都对，模型不会报错，只会少一个物体。这不是打字错误，
#: 是本次要量出来的东西之一（`[⑤]`）。
FULL_PROMPT = "sofa. chair. table. plant. lamp. monitor. shelf. cabinet. painting."

#: `[②]` 的口径声明 —— 随数字一起落盘，免得「净代价」被读成「绝对误差」。
BBOX_FALLBACK_SCOPE = (
    "delta_* = 降级组的几何量 − 基线**同物体**走 mask 路径的几何量，"
    "单位米。这是降级的**净代价**：可见性固有偏差（gt_visible vs gt_box）"
    "在两组里都存在，相减即消掉。"
    " err_vs_gt_* 是另一回事：它相对**盒中心**，**含**可见性偏差，"
    "不能当降级代价读 —— 两者都报，名字不同。"
)

#: `[③]` 的口径声明。
NO_VALID_POINTS_SCOPE = (
    "本组量的是「物体从场景图里消失」的后果：节点数、边数、"
    "以及**副作用**（其他物体是否被连带扰动、up_axis 是否变化）。"
    " 副作用必须单独报：nan 注入的区域是投影框（含外扩），"
    "它可能盖住别的物体的可见像素 —— 那会让「丢了一个物体」"
    "与「其他物体的数字也变了」混成一条无法归因的结论。"
)

#: `[⑥]` 的口径声明 —— 这一条是本次最容易被误用的地方。
INTRINSICS_SCOPE = (
    "合成桥里 `DepthField.intrinsics_source` **恒为 provided**，因为点云就是"
    "用 scene.intrinsics 逐像素反投影生成的，没有「模型相机头」这一环。"
    " ⟹ 在合成数据上 **不能用 intrinsics_source 判断内参有没有真的传进来**；"
    "那要靠本节的 6a/6b 两条守卫（不一致 / 缺失都抛）。"
    " 换句话说：`intrinsics_source == 'provided'` 在这里是结构性事实，"
    "不是一次需要被信任的声明 —— 拿它论证「内参条件生效」是循环论证。"
)

EXIT_OK = 0
EXIT_RULER_BROKEN = 3
EXIT_INJECTION_MISSED = 4
EXIT_GUARD_SILENT = 5


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------


def _mm(x: float | None) -> str:
    """米 → 毫米字符串。`None` / `nan` 打成 `—`（**不是 0**）。"""
    if x is None or not np.isfinite(x):
        return "—"
    return "%.1f" % (1000.0 * x)


def _k4(intrinsics: np.ndarray) -> tuple[float, float, float, float]:
    """`(3,3)` 内参 → `BuildConfig.known_intrinsics` 要的 `(fx, fy, cx, cy)`。"""
    intr = np.asarray(intrinsics, dtype=np.float64)
    return (
        float(intr[0, 0]),
        float(intr[1, 1]),
        float(intr[0, 2]),
        float(intr[1, 2]),
    )


def _run(
    boxes: Sequence[Box3D],
    intrinsics: np.ndarray,
    *,
    prompt: str = FULL_PROMPT,
    degrade: dict[str, Degradation] | None = None,
    known_intrinsics: tuple[float, float, float, float] | None = None,
    require_camera_K: bool = True,
    image_hw: tuple[int, int] = IMAGE_HW,
    scene_id: str = "agg_probe",
) -> tuple[SyntheticPerception, SyntheticScene, Any]:
    """渲染 → 造桥 → **走真实 `build_scene_graph`**。

    返回 `(桥, 夹具场景, BuildResult)`。夹具场景要交出来，因为
    `[④]` 的双路对照需要它 —— 那一路用的是**夹具原始点云**，
    而不是经桥之后的东西。
    """
    per = SyntheticPerception.from_boxes(
        boxes,
        intrinsics=intrinsics,
        image_hw=image_hw,
        degrade=degrade,
        require_camera_K=require_camera_K,
    )
    cfg = BuildConfig(
        prompt=prompt,
        known_intrinsics=known_intrinsics,
    )
    res = build_scene_graph(
        per.image,
        perception=per,
        scene_id=scene_id,
        image_id=scene_id,
        config=cfg,
    )
    return per, per.scene, res


def _expect_error(fn: Callable[[], Any]) -> dict[str, Any]:
    """跑 `fn`，记录它抛了什么。**没抛 = 不 ok**（守卫是装饰）。"""
    try:
        fn()
    except SyntheticIntrinsicsError as exc:
        return {"raised": "SyntheticIntrinsicsError", "ok": True, "message": str(exc)}
    except Exception as exc:  # noqa: BLE001 —— 抛了别的类型也不 ok
        return {
            "raised": type(exc).__name__,
            "ok": False,
            "message": f"抛的不是 SyntheticIntrinsicsError：{exc}",
        }
    return {"raised": None, "ok": False, "message": "没有抛 —— 守卫从未生效"}


def _nodes_of(res: Any) -> dict[str, Any]:
    return {n.id: n for n in res.scene.nodes}


def _edge_set(res: Any) -> set[tuple[str, str, str]]:
    return {(e.source, e.target, e.relation) for e in res.scene.edges}


# ----------------------------------------------------------------------------
# ① 基线
# ----------------------------------------------------------------------------


def run_baseline(
    boxes: Sequence[Box3D],
    intrinsics: np.ndarray,
    *,
    prompt: str = FULL_PROMPT,
) -> tuple[dict[str, Any], SyntheticScene]:
    _, scene, res = _run(
        boxes, intrinsics, prompt=prompt, known_intrinsics=_k4(intrinsics)
    )
    s = res.stats
    rec = {
        "prompt": prompt,
        "n_nodes": s["n_nodes"],
        "n_edges": s["n_edges"],
        "n_fallbacks": s["n_fallbacks"],
        "n_dropped": s["n_dropped"],
        "n_detections_raw": s["n_detections_raw"],
        "intrinsics_source": s["intrinsics_source"],
        "fov": s["fov"],
        "up_axis": s["up_axis"],
        "up_axis_reliable": s["up_axis_reliable"],
        "up_axis_tilt_deg": s["up_axis_tilt_deg"],
        "mask_box_coverage_mean": s["mask_box_coverage_mean"],
        "mask_box_coverage_min": s["mask_box_coverage_min"],
        "warnings": list(res.warnings),
        "perception_kind": s["perception"].get("kind"),
        "nodes": {
            n.id: {
                "label": n.label,
                "centroid_3d": [float(v) for v in n.centroid_3d],
                "extent_3d": [float(v) for v in n.extent_3d],
                "n_points": int(n.n_points) if n.n_points is not None else None,
                "centroid_source": n.centroid_source,
            }
            for n in res.scene.nodes
        },
        "edges": sorted([e.source, e.target, e.relation] for e in res.scene.edges),
    }
    return rec, scene


# ----------------------------------------------------------------------------
# ② 掩码抹空 ⟹ bbox_fallback
# ----------------------------------------------------------------------------


def run_bbox_fallback(
    boxes: Sequence[Box3D],
    intrinsics: np.ndarray,
    *,
    baseline: dict[str, Any],
    gt_box: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """逐物体把掩码抹空，量 `bbox_fallback` 的净代价。

    一次只注入一个物体：同时注入多个会让「哪一次降级贡献了多少」无法归因。
    """
    rows: list[dict[str, Any]] = []
    missed: list[str] = []
    for target in sorted(baseline["nodes"]):
        per, _, res = _run(
            boxes,
            intrinsics,
            degrade={target: Degradation(mask_empty=True, note="mask_empty")},
            known_intrinsics=_k4(intrinsics),
        )
        s = res.stats
        nodes = _nodes_of(res)
        node = nodes.get(target)
        fb = [f for f in s["fallbacks"] if f["object_id"] == target]
        applied = (
            node is not None
            and node.centroid_source == "bbox_fallback"
            and bool(fb)
        )
        if not applied:
            missed.append(target)
            continue

        base = baseline["nodes"][target]
        gt_c = gt_box[target]["centroid_3d"]
        pbox = per.projected_box(target)
        coverage = (
            float(box_coverage(per.scene.masks[target], pbox))
            if pbox is not None
            else None
        )
        rows.append(
            {
                "object_id": target,
                "label": node.label,
                "n_mask_points": int(fb[0]["n_mask_points"]),
                "n_box_points": int(fb[0]["n_box_points"]),
                "n_points_base": base["n_points"],
                # ★ 解读这张表的关键**自变量**：框里有多少是背景。
                #   它决定框内背景点能不能过半 ⟹ 能不能拉动逐轴中位数。
                "box_coverage": coverage,
                # ★ 净代价：与基线同物体相减
                "delta_m": gm.centroid_error(node.centroid_3d, base["centroid_3d"]),
                "delta_centroid_3d": [
                    float(node.centroid_3d[i] - base["centroid_3d"][i]) for i in range(3)
                ],
                "delta_extent_l1_m": gm.extent_error(
                    node.extent_3d, base["extent_3d"]
                )["l1_m"],
                # 相对盒中心 —— **含**可见性偏差，不可当降级代价
                "err_vs_gt_m": gm.centroid_error(node.centroid_3d, gt_c),
                "err_vs_gt_base_m": gm.centroid_error(base["centroid_3d"], gt_c),
            }
        )

    deltas = [r["delta_m"] for r in rows]
    summary = gm.summarize(deltas, n_expected=len(baseline["nodes"]))
    ext = [r["delta_extent_l1_m"] for r in rows]
    depth_deltas = [r["delta_centroid_3d"][2] for r in rows]
    farther = sum(1 for d in depth_deltas if d > 0.0)
    covs = [r["box_coverage"] for r in rows if r["box_coverage"] is not None]
    return {
        "scope": BBOX_FALLBACK_SCOPE,
        "n_objects_injected": len(baseline["nodes"]),
        "n_applied": len(rows),
        "missed": missed,
        "summary": {
            "n_expected": len(baseline["nodes"]),
            "n_measured": summary["n_measured"],
            "n_missing": summary["n_missing"],
            "total_m": summary.get("total_m", {}),
            "lateral_m": summary.get("lateral_m", {}),
            "depth_m": summary.get("depth_m", {}),
        },
        # ★ 尺寸代价单独一栏，而且必须**同时**给中位数与最大值。
        #   这次实测里中位数是 0.0 m、最大值是 7.09 m —— 只看中位数会得出
        #   「bbox_fallback 只影响质心、不影响尺寸」，而事实正好相反。
        #   这正是本项目反复遇到的那种静默形态：不报错、数字自洽、尾部炸飞。
        "extent_l1_m": {
            "median": float(np.median(ext)) if ext else None,
            "p90": float(np.percentile(ext, 90.0)) if ext else None,
            "max": float(np.max(ext)) if ext else None,
            "n_changed": int(sum(1 for v in ext if v > 0.0)),
            "scope": "extent_3d 的 L1 净代价（相对基线同物体），单位米。"
            "⚠ 中位数 0.0 与最大值 7.09 同时成立是**预期**的：半数物体的框内"
            "背景占比还不够高，`robust_extent` 的分位数没被拉动；"
            "而一旦拉动，拉的是**墙与地板**的跨度（x 方向可达十几米）。",
        },
        "box_coverage": {
            "median": float(np.median(covs)) if covs else None,
            "min": float(np.min(covs)) if covs else None,
            "max": float(np.max(covs)) if covs else None,
            "note": "框里有多少是**物体**（1 − 背景占比）。它是解读本档的关键自变量："
            "背景点占比过半才拉得动逐轴中位数与分位数。真实链路 Phase 0 实测 sofa "
            "的对应值是 46.2%，而本档中位数 0.97 ⟹ **本档落在最乐观的一头**，"
            "不能直接当真实照片上的预期。"
            " ⚠ 分母用的是未裁剪的框面积，框超出画面时读数**偏低**。",
        },
        "depth_direction": {
            "n_farther": farther,
            "n_nearer": len(depth_deltas) - farther,
            "note": "推论与实测**相反**，这条要留着：我原本预期「框内背景点的 z "
            "更大 ⟹ 降级把物体系统性推向远处」。实测是 9/9 个物体的**纵深中位数"
            "恰好不变**（差为精确 0）。原因是结构性的 —— 正对相机的盒子其可见面"
            "是一个平面，掩码内点的 z 分布集中在单点；框内多出来的背景像素虽然"
            "更远，但占比不到一半，**逐轴中位数取不到它们**。"
            " ⟹ 这条结论的适用边界是「覆盖率足够高」，**不是**"
            "「bbox_fallback 不影响纵深」。边界由 [②b 覆盖率扫描] 量出。",
        },
        "rows": rows,
    }


# ----------------------------------------------------------------------------
# ②b 覆盖率扫描 —— 把 [②] 的结论从「一个点」变成「一条边界」
# ----------------------------------------------------------------------------


def run_coverage_boundary(
    boxes: Sequence[Box3D],
    intrinsics: np.ndarray,
    *,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """把检测框**外扩 N 像素**，看覆盖率掉到多少时纵深中位数才开始被拉动。

    `[②]` 给出的「净代价全在横向、纵深恰好不变」是在**一个**覆盖率
    （框刚好是物体外接矩形、中位 0.91）上读到的。而真实链路里框几乎总比
    物体大一点，Phase 0 实测 sofa 的覆盖率只有 46.2%。所以那个结论的
    **适用边界在哪**，必须扫出来 —— 否则它会被当成「bbox_fallback 与纵深无关」
    这条更强的、错的命题。

    `box_pad_px` 是同时改变两个量的旋钮：框内点数变多、覆盖率变小。
    这两者本来就是同一件事的两面（框越大，背景点占比越高），
    所以用它扫出来的曲线可直接归因到「背景点占比」。
    """
    rows: list[dict[str, Any]] = []
    for pad in PAD_GRID:
        tot: list[float] = []
        lat: list[float] = []
        #: 非负的纵深误差（与 `[②]` 同口径，来自 `gm.centroid_error`）。
        dep_abs: list[float] = []
        #: ★ **带符号**的纵深变化（由质心直接相减）。符号信息只能从这里来
        #: —— 见下面那段关于 `abs` 的注释。
        dz_signed: list[float] = []
        ext: list[float] = []
        covs: list[float] = []
        for target in sorted(baseline["nodes"]):
            per, _, res = _run(
                boxes,
                intrinsics,
                degrade={
                    target: Degradation(
                        mask_empty=True,
                        box_pad_px=pad,
                        note=f"mask_empty+pad{pad:g}",
                    )
                },
                known_intrinsics=_k4(intrinsics),
            )
            node = _nodes_of(res).get(target)
            if node is None or node.centroid_source != "bbox_fallback":
                continue
            base = baseline["nodes"][target]
            d = gm.centroid_error(node.centroid_3d, base["centroid_3d"])
            tot.append(d["total_m"])
            lat.append(d["lateral_m"])
            dep_abs.append(d["depth_m"])
            dz_signed.append(float(node.centroid_3d[2] - base["centroid_3d"][2]))
            ext.append(gm.extent_error(node.extent_3d, base["extent_3d"])["l1_m"])
            pbox = per.projected_box(target)
            if pbox is not None:
                covs.append(float(box_coverage(per.scene.masks[target], pbox)))

        def _med(xs: list[float]) -> float | None:
            return float(np.median(xs)) if xs else None

        rows.append(
            {
                "box_pad_px": pad,
                "n_applied": len(tot),
                "coverage_median": _med(covs),
                "coverage_min": float(np.min(covs)) if covs else None,
                "median_total_m": _med(tot),
                "median_lateral_m": _med(lat),
                "median_depth_m": _med(dep_abs),
                "median_extent_l1_m": _med(ext),
                "max_extent_l1_m": float(np.max(ext)) if ext else None,
                # ⚠⚠ **判据必须作用在带符号量上。** `gm.centroid_error(...)["depth_m"]`
                # 是**非负**的误差量 —— 在它上面写 `abs(v) > 0` 等于什么都没做，
                # 而它看起来比 `> 0` 更严谨。本档第一版就是这么写的，
                # 抓到它的是下面那条「表里必须确实存在负向位移」的测试：
                # `n_depth_decreased` 恒为 0，于是断言变红。
                # 这正是本项目反复遇到的那种静默 —— **一个不改变任何行为的判据**，
                # 比没有判据更糟，因为它会让人以为那一维已经被照顾到了。
                "n_depth_changed": int(sum(1 for v in dz_signed if v != 0.0)),
                "n_depth_increased": int(sum(1 for v in dz_signed if v > 0.0)),
                "n_depth_decreased": int(sum(1 for v in dz_signed if v < 0.0)),
                "max_depth_m": float(np.max(np.abs(dz_signed))) if dz_signed else None,
            }
        )

    first_depth = next(
        (r["box_pad_px"] for r in rows if r["n_depth_changed"] > 0), None
    )
    return {
        "scope": "所有数字都是相对**基线同物体**的净代价（与 [②] 同口径），"
        "单位米；`box_pad_px` 只作用于被注入的那一个物体的投影框。"
        " `coverage_median` 是同一批物体在该 pad 下「框里有多少是物体」的中位数。"
        " ⚠ 分母用的是**未裁剪**的框面积（`box_coverage(box_area)` 不 clip），"
        "而 pad ≥ 16 时部分框已超出 320×240 画面 ⟹ 那两行的覆盖率读数**偏低**，"
        "即真实的背景占比比表里更小。这是本档唯一的已知偏差，写在这里而不是"
        "让读者自己发现。",
        "pads": list(PAD_GRID),
        "first_pad_with_depth_change": first_depth,
        "rows": rows,
    }


# ----------------------------------------------------------------------------
# ②c 尺寸失效机制 —— 把 robust_extent 的内部读数摊开
# ----------------------------------------------------------------------------


def run_extent_failure_mechanism(
    boxes: Sequence[Box3D],
    intrinsics: np.ndarray,
    *,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """`bbox_fallback` 为什么在覆盖率还很高时就把**尺寸**炸到米级。

    `[②]` 报出「尺寸 L1 中位 0.0、最大 7.09 m」。这个组合很怪，所以必须给出
    **机制**而不是只给数字 —— 否则它要么被当成 bug、要么被当成噪声忽略掉。

    做法是把 `robust_extent` 的内部读数（`n_rejected` / `radius_p90_m` /
    `reject_threshold_m`）在**同一份点云**上并排跑两条选择器（掩码 vs 投影框），
    于是差别只来自「选了哪些像素」。

    ⚠ 本档**只诊断，不改** `robust_extent`。它在它被设计的那条路径
    （掩码内点云）上表现正确 —— 实测 `n_rejected = 0` 正是「没有可剔的东西」
    的正确答案。要动它得先有一条能处理双峰分布的判据，那是另一件事。
    """
    rows: list[dict[str, Any]] = []
    for target in sorted(baseline["nodes"]):
        per, _, res = _run(
            boxes,
            intrinsics,
            degrade={target: Degradation(mask_empty=True, note="mask_empty")},
            known_intrinsics=_k4(intrinsics),
        )
        node = _nodes_of(res).get(target)
        if node is None or node.centroid_source != "bbox_fallback":
            continue

        pts_all = per.points_chw()
        rec: dict[str, Any] = {"object_id": target}
        for tag, sel in (
            ("mask", per.scene.masks[target]),
            ("box", per.box_selector_for(target)),
        ):
            p = select_points(pts_all, sel)
            extent, _lo, _hi, meta = robust_extent(p)
            centroid, _cmeta = centroid_of(p)
            rec[tag] = {
                "n_points": int(p.shape[1]),
                "extent_3d_m": [float(v) for v in extent],
                "z_span_m": float(p[2].max() - p[2].min()) if p.shape[1] else None,
                "centroid_z": float(centroid[2]) if centroid is not None else None,
                "n_inliers": int(meta["n_inliers"]),
                "n_rejected": int(meta["n_rejected"]),
                "radius_p90_m": float(meta["radius_p90_m"]),
                "reject_threshold_m": float(meta["reject_threshold_m"]),
            }
        rec["z_span_growth"] = (
            rec["box"]["z_span_m"] / rec["mask"]["z_span_m"]
            if rec["mask"]["z_span_m"]
            else None
        )
        rows.append(rec)

    rejected = [r["box"]["n_rejected"] for r in rows]
    growth = [r["z_span_growth"] for r in rows if r["z_span_growth"] is not None]
    return {
        "k_p90": float(DEFAULT_K_P90),
        "n_rows": len(rows),
        "box_n_rejected": {
            "min": int(np.min(rejected)) if rejected else None,
            "max": int(np.max(rejected)) if rejected else None,
            "n_zero": int(sum(1 for v in rejected if v == 0)),
        },
        "z_span_growth": {
            "min": float(np.min(growth)) if growth else None,
            "max": float(np.max(growth)) if growth else None,
            "n_undefined": int(sum(1 for r in rows if r["z_span_growth"] is None)),
            "note": "`z_span_growth` = box 路径的 z 跨度 ÷ mask 路径的 z 跨度。"
            "⚠ 对**可见面是一个平面**的物体（挂画、显示器，mask 侧 z 跨度恰为 0）"
            "它无定义，记为 null —— 这类物体任何 z 方向的读数都是纯误差，"
            "没有可比的基线。",
        },
        "note": "★ **机制**：`robust_extent` 的剔除判据 `r <= k_p90 × r_p90` 是"
        "按「离群点占少数、且半径远大于主体」设计的（见它自己的 docstring）。"
        "`bbox_fallback` 的框内点云打破了这个前提 —— 混进来的不是「少数远处点」，"
        "而是**与主体在 x/y 上重叠、只在 z 上分开的第二团**（前排物体的可见面）。"
        "第二团会**一起**把 `r_p90` 抬大，于是阈值落到第二团**之外**："
        "被判为离群、剔掉的是更远的**墙**，而第二团本身被当成了主体的一部分。"
        " ⟹ **它剔了，但没剔到该剔的东西**（实测 box 路径 `n_rejected` 在 "
        "%s–%s 之间，而 z 跨度照样从 0.22–0.62 m 涨到 2.0–4.3 m）。"
        " ⟹ 「`bbox_fallback` 只是质心精度下降」这个说法**不完整**："
        "它对质心的破坏是毫米~百毫米级（逐轴中位数抗双峰），"
        "对**尺寸**的破坏是**米级**（min/max 对双峰零抗性）——"
        "而 `extent_3d` 直接进 L5 场景报告，也是 `on` / `inside` 的判据。"
        " 修法方向（本轮不实现）：需要一条不依赖 `r_p90` 的双峰判据，"
        "例如按半径做 1D 聚类、或在 z 上用主峰宽度定阈值。"
        % (
            min(rejected) if rejected else "—",
            max(rejected) if rejected else "—",
        ),
        "rows": rows,
    }


# ----------------------------------------------------------------------------
# ③ 深度空洞 ⟹ no_valid_points
# ----------------------------------------------------------------------------


def run_no_valid_points(
    boxes: Sequence[Box3D],
    intrinsics: np.ndarray,
    *,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """逐物体把投影框内点云置 nan，量「物体消失」的后果与副作用。"""
    base_edges = {tuple(e) for e in baseline["edges"]}
    rows: list[dict[str, Any]] = []
    missed: list[str] = []

    for target in sorted(baseline["nodes"]):
        _, _, res = _run(
            boxes,
            intrinsics,
            degrade={
                target: Degradation(nan_depth_in_box=True, note="nan_depth_in_box")
            },
            known_intrinsics=_k4(intrinsics),
        )
        s = res.stats
        nodes = _nodes_of(res)
        dropped = [d for d in s["dropped"] if d.get("reason") == "no_valid_points"]

        # ⚠ `dropped` 的条目里只有 label（`Detection.as_dict()` 不带 object_id），
        #   所以「哪个物体没了」用节点集合来判断，label 只作交叉确认。
        gone = target not in nodes
        if not (gone and dropped):
            missed.append(target)
            continue

        # 副作用 1：其他物体的数字变了吗（nan 注入区域可能盖住它们）
        shifted = []
        for oid, node in sorted(nodes.items()):
            base = baseline["nodes"][oid]
            if tuple(node.centroid_3d) != tuple(base["centroid_3d"]):
                shifted.append(
                    {
                        "object_id": oid,
                        "delta_m": gm.centroid_error(node.centroid_3d, base["centroid_3d"]),
                    }
                )

        # 副作用 2：边损失必须**分类** —— 涉及目标 vs 不涉及
        now_edges = _edge_set(res)
        lost = sorted(base_edges - now_edges)
        lost_touching = [e for e in lost if target in (e[0], e[1])]
        lost_unrelated = [e for e in lost if target not in (e[0], e[1])]

        rows.append(
            {
                "object_id": target,
                "label": baseline["nodes"][target]["label"],
                "dropped_labels": [d.get("label") for d in dropped],
                "n_mask_points": dropped[0].get("n_mask_points"),
                "n_box_points": dropped[0].get("n_box_points"),
                "n_nodes_base": baseline["n_nodes"],
                "n_nodes_now": s["n_nodes"],
                "n_edges_base": baseline["n_edges"],
                "n_edges_now": s["n_edges"],
                "n_edges_lost": len(lost),
                "n_edges_lost_touching_target": len(lost_touching),
                "n_edges_lost_unrelated": len(lost_unrelated),
                "lost_edges_unrelated_sample": [list(e) for e in lost_unrelated[:8]],
                "n_others_shifted": len(shifted),
                "others_shifted": shifted[:8],
                "up_axis_base": baseline["up_axis"],
                "up_axis_now": s["up_axis"],
                "up_axis_changed": s["up_axis"] != baseline["up_axis"],
            }
        )

    # ---- 副作用汇总 ----
    all_side = [s["delta_m"]["total_m"] for r in rows for s in r["others_shifted"]]
    n_rows_with_side = sum(1 for r in rows if r["n_others_shifted"])
    n_above_tol = sum(1 for v in all_side if v > DEFAULT_TOL)
    return {
        "scope": NO_VALID_POINTS_SCOPE,
        "n_objects_injected": len(baseline["nodes"]),
        "n_applied": len(rows),
        "missed": missed,
        "side_effects": {
            "n_rows_with_side_effects": n_rows_with_side,
            "n_applied": len(rows),
            "n_disturbed_others_total": len(all_side),
            "max_other_delta_m": float(max(all_side)) if all_side else 0.0,
            "n_above_relation_tol": n_above_tol,
            "relation_tol_m": DEFAULT_TOL,
            "n_up_axis_changed": sum(1 for r in rows if r["up_axis_changed"]),
            "note": "★ **连带扰动是这条降级路径的内禀属性，不是本注入方式的缺陷。**"
            "触发 `no_valid_points` 要求「投影框内**所有**像素都没有有效深度」，"
            "而一个物体的投影框内必然包含其他物体的像素（覆盖率越高越必然）"
            " ⟹ 真实链路里这个降级不会只影响一个物体：它发生时，那一整块图像"
            "区域的深度都失效了（大面积反光、过曝、镜头沾污）。"
            " 实测 9 次注入里有 %d 次出现连带扰动，共扰动 %d 个物体，"
            "最大 %.1f mm；其中**超过关系容差 %.0f mm 的有 %d 次**"
            " ⟹ 足以把 near/far 这类布尔关系翻转，而场景图里不会留下任何痕迹。"
            " 这也是为什么 builder 对这种情况的选择是「剔除物体」而不是"
            "「补一个坐标」：补坐标会把一次**区域性**失效伪装成一个定位结论。"
            % (
                n_rows_with_side,
                len(all_side),
                1000.0 * (max(all_side) if all_side else 0.0),
                1000.0 * DEFAULT_TOL,
                n_above_tol,
            ),
        },
        "rows": rows,
    }


# ----------------------------------------------------------------------------
# ④ 双路对照
# ----------------------------------------------------------------------------


def run_dual_path(scene: SyntheticScene, baseline: dict[str, Any]) -> dict[str, Any]:
    """builder 的 ④ 段 vs `visible_geometry` —— **要求逐位相等**。

    builder 的 ④ 段（`resample_mask_to` → `select_points` → `centroid_of` →
    `robust_extent`）在概念上就是 `visible_geometry` 做的事，但它是**独立写的**。
    两条实现独立写出同样的数字，是「聚合段没有偷偷改数」的强证据；
    不一致则说明其中一条多做了点什么（例如偷偷 round、或者用错了网格）。

    这里用 `==` 不用 `approx`：同一份点云、同一份掩码
    （`resample_mask_to` 在网格相同时返回**同一块内存**），
    「几乎相等」本身就是缺陷信号。
    """
    mismatches: list[dict[str, Any]] = []
    checked = 0
    for oid, rec in sorted(baseline["nodes"].items()):
        direct = visible_geometry(scene.points_chw, scene.masks[oid], scene.grid_hw)
        checked += 1
        c_direct = tuple(float(v) for v in direct["centroid_3d"])
        c_builder = tuple(float(v) for v in rec["centroid_3d"])
        e_direct = tuple(float(v) for v in direct["extent_3d"])
        e_builder = tuple(float(v) for v in rec["extent_3d"])
        n_direct = int(direct["n_points"])
        n_builder = rec["n_points"]
        bad = []
        if c_direct != c_builder:
            bad.append("centroid_3d")
        if e_direct != e_builder:
            bad.append("extent_3d")
        if n_direct != n_builder:
            bad.append("n_points")
        if bad:
            mismatches.append(
                {
                    "object_id": oid,
                    "fields": bad,
                    "centroid_direct": list(c_direct),
                    "centroid_builder": list(c_builder),
                    "extent_direct": list(e_direct),
                    "extent_builder": list(e_builder),
                    "n_points_direct": n_direct,
                    "n_points_builder": n_builder,
                }
            )
    return {
        "n_checked": checked,
        "n_mismatch": len(mismatches),
        "criterion": "centroid_3d / extent_3d / n_points 三者全用 == 比较",
        "mismatches": mismatches,
    }


# ----------------------------------------------------------------------------
# ⑤ prompt 召回
# ----------------------------------------------------------------------------


def run_prompt_recall(
    boxes: Sequence[Box3D],
    intrinsics: np.ndarray,
) -> dict[str, Any]:
    """同一个场景、同一套感知，只换 prompt —— 少哪些物体、少多少条边。"""
    all_labels = sorted({b.label for b in boxes if not b.is_background})
    rec: dict[str, Any] = {"labels_in_fixture": all_labels, "variants": {}}
    for name, prompt in (("default", DEFAULT_PROMPT), ("full", FULL_PROMPT)):
        _, _, res = _run(
            boxes, intrinsics, prompt=prompt, known_intrinsics=_k4(intrinsics)
        )
        labels = sorted({n.label for n in res.scene.nodes})
        rec["variants"][name] = {
            "prompt": prompt,
            "n_nodes": res.stats["n_nodes"],
            "n_edges": res.stats["n_edges"],
            "labels_found": labels,
            "labels_missing": [lb for lb in all_labels if lb not in labels],
        }
    return rec


# ----------------------------------------------------------------------------
# ⑥ 内参守卫
# ----------------------------------------------------------------------------


def run_intrinsics_guards(
    boxes: Sequence[Box3D],
    intrinsics: np.ndarray,
) -> dict[str, Any]:
    """四条：守卫必须**响**，且 `intrinsics_source` 在本桥**退化**这件事要说清。"""
    k4 = _k4(intrinsics)
    out: dict[str, Any] = {"scope": INTRINSICS_SCOPE}

    # 6a 传了一份**错的** known_intrinsics ⟹ 必须响
    wrong = (k4[0] * 1.1, k4[1] * 1.1, k4[2] * 1.1, k4[3] * 1.1)
    out["6a_wrong_known_intrinsics"] = _expect_error(
        lambda: _run(
            boxes, intrinsics, known_intrinsics=wrong, require_camera_K=True
        )
    )
    out["6a_wrong_known_intrinsics"]["given"] = list(wrong)
    out["6a_wrong_known_intrinsics"]["truth"] = list(k4)

    # 6b 要求校验、但配置里没有 known_intrinsics ⟹ 必须响
    out["6b_require_camera_K_but_none_given"] = _expect_error(
        lambda: _run(
            boxes, intrinsics, known_intrinsics=None, require_camera_K=True
        )
    )

    # 6c 没要求校验、也没传 ⟹ **不抛**，但 intrinsics_source 仍是 provided
    #    ★ 这一条是警告，不是好消息：它说明在合成数据上
    #      `intrinsics_source` 无法用来判断内参有没有真的传过来。
    try:
        per, _, res = _run(
            boxes, intrinsics, known_intrinsics=None, require_camera_K=False
        )
        out["6c_no_known_intrinsics"] = {
            "raised": None,
            "ok": True,
            "intrinsics_source": res.stats["intrinsics_source"],
            "n_nodes": res.stats["n_nodes"],
            "lift_received_camera_K": per.lift_calls[-1] is not None,
            "note": "没传内参、lift 收到 None，而 intrinsics_source 依然写着 "
            "provided ⟹ 该字段在合成桥里是**结构性事实**，不能用来判断"
            "「内参条件是否生效」。",
        }
    except Exception as exc:  # noqa: BLE001
        out["6c_no_known_intrinsics"] = {
            "raised": type(exc).__name__,
            "ok": False,
            "message": str(exc),
        }

    # 6d provided 但视场不可信 —— builder 里那条**独立**的警告分支
    narrow = np.array(
        [[60.0, 0.0, 160.0], [0.0, 60.0, 120.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    _, _, res = _run(boxes, narrow, known_intrinsics=_k4(narrow))
    warns = [w for w in res.warnings if "视场不可信" in w]
    out["6d_implausible_fov_but_provided"] = {
        "hfov_deg": res.stats["fov"]["hfov_deg"],
        "plausible": res.stats["fov"]["plausible"],
        "intrinsics_source": res.stats["intrinsics_source"],
        "n_fov_warnings": len(warns),
        "warning": warns[0] if warns else None,
        "n_nodes": res.stats["n_nodes"],
    }
    return out


# ----------------------------------------------------------------------------
# 打印
# ----------------------------------------------------------------------------


def print_report(result: dict[str, Any]) -> None:
    b = result["baseline"]
    print("=" * 78)
    print("聚合段探针 —— 经**真实** build_scene_graph（零 API / 零 GPU）")
    print("=" * 78)
    print(
        "[① 基线] 节点 %d / 边 %d / 降级 %d / 丢弃 %d / raw 检测 %d"
        % (b["n_nodes"], b["n_edges"], b["n_fallbacks"], b["n_dropped"], b["n_detections_raw"])
    )
    print(
        "         fov %.2f° plausible=%s  intrinsics_source=%s  up_axis=%s(%s)"
        % (
            b["fov"]["hfov_deg"],
            b["fov"]["plausible"],
            b["intrinsics_source"],
            b["up_axis"],
            "可靠" if b["up_axis_reliable"] else "不可靠",
        )
    )
    print(
        "         掩码/框覆盖率 均值 %.4f 最小 %.4f；warnings %d 条"
        % (b["mask_box_coverage_mean"], b["mask_box_coverage_min"], len(b["warnings"]))
    )

    fb = result["bbox_fallback"]
    print()
    print("[② 掩码抹空 ⟹ bbox_fallback] 逐物体注入（单位 mm，中位数）")
    print("     注入并生效 %d/%d" % (fb["n_applied"], fb["n_objects_injected"]))
    s = fb["summary"]
    print(
        "     净代价  total %s  lateral %s  depth %s   （n=%d，键名取 gm._stats 的 median）"
        % (
            _mm(s["total_m"].get("median")),
            _mm(s["lateral_m"].get("median")),
            _mm(s["depth_m"].get("median")),
            s["n_measured"],
        )
    )
    print(
        "     净代价  P90    total %s  lateral %s  depth %s"
        % (
            _mm(s["total_m"].get("p90")),
            _mm(s["lateral_m"].get("p90")),
            _mm(s["depth_m"].get("p90")),
        )
    )
    d = fb["depth_direction"]
    print("     纵深方向：偏远处 %d 个 / 偏近处 %d 个" % (d["n_farther"], d["n_nearer"]))
    ex = fb["extent_l1_m"]
    print(
        "     ★ 尺寸 L1 净代价：中位 %s   P90 %s   最大 %s   （%d/%d 个物体变了）"
        % (
            _mm(ex["median"]),
            _mm(ex["p90"]),
            _mm(ex["max"]),
            ex["n_changed"],
            fb["n_objects_injected"],
        )
    )
    print("        —— 中位数与最大值必须同看：只看中位数会得出「不影响尺寸」，正好相反")
    bc = fb["box_coverage"]
    print("     框覆盖率（框里有多少是物体）中位 %.4f，最低 %.4f"
          % (bc["median"], bc["min"]))
    print("     %-12s %8s %9s %9s %9s %9s %9s"
          % ("物体", "框内点", "覆盖率", "净total", "净depth", "尺寸L1", "vs盒心"))
    for r in fb["rows"]:
        print(
            "     %-12s %8d %9.4f %9s %9s %9s %9s"
            % (
                r["object_id"],
                r["n_box_points"],
                r["box_coverage"] if r["box_coverage"] is not None else float("nan"),
                _mm(r["delta_m"]["total_m"]),
                _mm(r["delta_m"]["depth_m"]),
                _mm(r["delta_extent_l1_m"]),
                _mm(r["err_vs_gt_m"]["total_m"]),
            )
        )

    cb = result["coverage_boundary"]
    print()
    print("[②b 覆盖率扫描] 框外扩 N 像素 —— [②]「纵深不变」的适用边界（单位 mm，中位数）")
    print("     %5s %10s %10s %10s %10s %12s %10s"
          % ("pad", "覆盖率", "净total", "净lateral", "净depth", "净尺寸L1", "纵深变了"))
    for r in cb["rows"]:
        cov = r["coverage_median"]
        print(
            "     %5g %10s %10s %10s %10s %12s %7d/%d"
            % (
                r["box_pad_px"],
                ("%.4f" % cov) if cov is not None else "—",
                _mm(r["median_total_m"]),
                _mm(r["median_lateral_m"]),
                _mm(r["median_depth_m"]),
                _mm(r["median_extent_l1_m"]),
                r["n_depth_changed"],
                r["n_applied"],
            )
        )
    print(
        "     纵深首次变化的 pad = %s"
        % (
            cb["first_pad_with_depth_change"]
            if cb["first_pad_with_depth_change"] is not None
            else "在整个网格内都没变化"
        )
    )
    print("     注：pad ≥ 16 时部分框已超出 320×240，覆盖率分母未裁剪 ⟹ 那两行读数偏低")

    em = result["extent_failure_mechanism"]
    print()
    print("[②c 尺寸失效机制] robust_extent 的内部读数（同一份点云，只换选择器）")
    br, zg = em["box_n_rejected"], em["z_span_growth"]
    print(
        "     k_p90 = %g；box 路径 n_rejected ∈ [%d, %d]（其中 %d 个为 0）"
        % (em["k_p90"], br["min"], br["max"], br["n_zero"])
    )
    print(
        "     z 跨度涨幅 ∈ [%.1f×, %.1f×]（%d 个无定义 —— mask 侧可见面是平面）"
        % (zg["min"], zg["max"], zg["n_undefined"])
    )
    print("     %-12s %7s %7s %11s %15s %11s"
          % ("物体", "n_mask", "n_box", "剔除mask/box", "z跨度 mask→box", "涨幅"))
    for r in em["rows"]:
        mk, bx = r["mask"], r["box"]
        g = r["z_span_growth"]
        print(
            "     %-12s %7d %7d %5d/%-5d %6.3f → %6.3f %9s"
            % (
                r["object_id"],
                mk["n_points"],
                bx["n_points"],
                mk["n_rejected"],
                bx["n_rejected"],
                mk["z_span_m"],
                bx["z_span_m"],
                ("×%.1f" % g) if g else "—（平面）",
            )
        )
    print("        —— 剔了 ≠ 剔对了：阈值 `k_p90 × r_p90` 被「第二团」自己抬大，"
          "于是判为离群的是更远的墙，第二团被当成主体")

    nv = result["no_valid_points"]
    print()
    print("[③ 深度空洞 ⟹ no_valid_points] 逐物体注入（物体整个消失）")
    print("     注入并生效 %d/%d" % (nv["n_applied"], nv["n_objects_injected"]))
    print(
        "     %-12s %7s %7s %8s %8s %7s"
        % ("物体", "节点", "边", "涉及边", "无关边", "他人变动")
    )
    for r in nv["rows"]:
        print(
            "     %-12s %3d→%-3d %3d→%-3d %8d %8d %7d"
            % (
                r["object_id"],
                r["n_nodes_base"],
                r["n_nodes_now"],
                r["n_edges_base"],
                r["n_edges_now"],
                r["n_edges_lost_touching_target"],
                r["n_edges_lost_unrelated"],
                r["n_others_shifted"],
            )
        )
    n_axis = sum(1 for r in nv["rows"] if r["up_axis_changed"])
    print("     up_axis 变化次数 %d / %d" % (n_axis, len(nv["rows"])))
    se = nv["side_effects"]
    print(
        "     ★ 连带扰动：%d/%d 次注入波及他人，共 %d 个物体，最大 %s；"
        "超关系容差 %.0f mm 的 %d 次"
        % (
            se["n_rows_with_side_effects"],
            se["n_applied"],
            se["n_disturbed_others_total"],
            _mm(se["max_other_delta_m"]),
            1000.0 * se["relation_tol_m"],
            se["n_above_relation_tol"],
        )
    )
    print("        —— 这是该降级路径的**内禀属性**：「框内全无深度」必然波及框内的其他物体")

    dp = result["dual_path"]
    print()
    print("[④ 双路对照] builder ④ 段 vs visible_geometry —— 要求**逐位相等**")
    print("     检查 %d 个物体，不一致 %d 个 %s"
          % (dp["n_checked"], dp["n_mismatch"], "✓" if dp["n_mismatch"] == 0 else "*** ✗ ***"))
    for mm in dp["mismatches"]:
        print("     %s: %s" % (mm["object_id"], mm["fields"]))

    pr = result["prompt_recall"]
    print()
    print("[⑤ prompt 召回] 同一场景、同一套感知，只换 prompt")
    print("     夹具里的类别：%s" % (", ".join(pr["labels_in_fixture"])))
    for name, v in pr["variants"].items():
        print(
            "     %-8s 节点 %2d / 边 %3d  缺 %s"
            % (name, v["n_nodes"], v["n_edges"], v["labels_missing"] or "无")
        )

    gu = result["intrinsics_guards"]
    print()
    print("[⑥ 内参守卫] 期望「响」的两条 + 一条警告 + 一条独立分支")
    for key in ("6a_wrong_known_intrinsics", "6b_require_camera_K_but_none_given"):
        rec = gu[key]
        flag = "✓ 响了" if rec["ok"] else "*** ✗ 没响 ***"
        print("     %-38s %s (%s)" % (key, flag, rec["raised"]))
    c = gu["6c_no_known_intrinsics"]
    print(
        "     %-38s 不抛=%s，但 intrinsics_source=%s ⚠ 退化了"
        % ("6c_no_known_intrinsics", c["raised"] is None, c.get("intrinsics_source"))
    )
    f = gu["6d_implausible_fov_but_provided"]
    print(
        "     %-38s HFoV %.1f° plausible=%s 警告 %d 条"
        % ("6d_implausible_fov_but_provided", f["hfov_deg"], f["plausible"], f["n_fov_warnings"])
    )

    if result["problems"]:
        print()
        print("*** 有问题 ***")
        for p in result["problems"]:
            print("   [退出码 %d] %s" % (p["exit_code"], p["message"]))
    else:
        print()
        print("全部检查通过。")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="聚合段探针（经真实 build_scene_graph）")
    ap.add_argument(
        "--json-out",
        default=str(ROOT / "reports" / "aggregation_probe.json"),
        help="JSON 输出路径（默认 reports/aggregation_probe.json）",
    )
    ap.add_argument("--quiet", action="store_true", help="只写 JSON，不打印")
    args = ap.parse_args(argv)

    boxes = default_scene_boxes()
    intrinsics = default_intrinsics()

    baseline, scene = run_baseline(boxes, intrinsics)
    bbox_fb = run_bbox_fallback(
        boxes, intrinsics, baseline=baseline, gt_box=scene.gt_box
    )
    cov_bound = run_coverage_boundary(boxes, intrinsics, baseline=baseline)
    ext_mech = run_extent_failure_mechanism(boxes, intrinsics, baseline=baseline)
    no_pts = run_no_valid_points(boxes, intrinsics, baseline=baseline)
    dual = run_dual_path(scene, baseline)
    recall = run_prompt_recall(boxes, intrinsics)
    guards = run_intrinsics_guards(boxes, intrinsics)

    problems: list[dict[str, Any]] = []
    # 严重度排序：尺子坏了 > 守卫没响 > 注入没生效
    if dual["n_mismatch"]:
        problems.append(
            {
                "exit_code": EXIT_RULER_BROKEN,
                "message": "双路对照不一致 —— **尺子坏了**："
                "builder 的 ④ 段与 visible_geometry 给出不同的数字，"
                "两条实现里至少有一条多做了点什么。先查它，别急着看别的档。",
            }
        )
    for key in ("6a_wrong_known_intrinsics", "6b_require_camera_K_but_none_given"):
        if not guards[key]["ok"]:
            problems.append(
                {
                    "exit_code": EXIT_GUARD_SILENT,
                    "message": f"{key} 没有抛 —— 「守卫从未生效」比「守卫失败」更难发现。",
                }
            )
    if bbox_fb["missed"] or no_pts["missed"]:
        problems.append(
            {
                "exit_code": EXIT_INJECTION_MISSED,
                "message": "注入了却没降级，missed=%s（bbox_fallback）/ %s "
                "（no_valid_points）—— 报告会显示「一切正常」，"
                "而这正是最该防的静默。" % (bbox_fb["missed"], no_pts["missed"]),
            }
        )

    result: dict[str, Any] = {
        "config": {
            "image_hw": list(IMAGE_HW),
            "intrinsics": [list(row) for row in np.asarray(intrinsics).tolist()],
            "default_prompt": DEFAULT_PROMPT,
            "full_prompt": FULL_PROMPT,
            "n_boxes": len(boxes),
            "n_foreground": sum(1 for b in boxes if not b.is_background),
            "zero_api": True,
            "path": ["detect(prompt)", "segment(IoU 匹配)", "lift(夹具点云)",
                     "build_scene_graph ①②③④⑤⑥"],
        },
        "baseline": baseline,
        "bbox_fallback": bbox_fb,
        "coverage_boundary": cov_bound,
        "extent_failure_mechanism": ext_mech,
        "no_valid_points": no_pts,
        "dual_path": dual,
        "prompt_recall": recall,
        "intrinsics_guards": guards,
        "problems": problems,
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    if not args.quiet:
        print_report(result)
        print()
        print("JSON → %s" % out)

    return problems[0]["exit_code"] if problems else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
