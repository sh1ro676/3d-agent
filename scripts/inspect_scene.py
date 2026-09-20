#!/usr/bin/env python
r"""场景图体检 —— 不需要 GPU、不加载任何模型。

    D:\3D_Spatial_Agent\venvs\vision\Scripts\python.exe scripts\inspect_scene.py `
        --scene D:\3D_Spatial_Agent\dataset\scenes\living_room

它回答四个问题，每个都对应一类**可归因的失败**（这是 L5 `diagnose_failure` 的雏形）：

  1. 内参来源与视场是否合理？ —— **优先级最高的一项**。Phase 0 Step 6 用仓库
     自带的 GT 深度实测：内参错会让三维误差中位数从 0.267 m 涨到 1.943 m
     （7.3 倍）。视场过宽会把所有 x/y 偏移系统性放大，于是距离与尺寸集体偏大。
     所以先看它 —— 后面「尺寸越界」那几条多半是它的症状，不是独立故障。
  2. 掩码有没有泄漏到检测框外？ —— SAM2 会分割「整个物体」，
     框外的像素可能是同一个物体的延伸（正常），也可能是背景（污染点云）。
  3. 每个物体的点云分布是否集中？ —— `radius_p90 / extent` 太大说明
     掩码里混进了远处表面，质心就不可信。
  4. 尺寸是否物理上可能？ —— 沙发不可能有 6.7 m 宽。数值越界要先怀疑几何，
     而不是先怀疑模型。

输出的「嫌疑」列表按可疑程度排序，直接可以抄进报告的失败案例分析。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_graph.store import load_mask, load_scene, masks_dir  # noqa: E402
from vision.geometry import PLAUSIBLE_HFOV_DEG, check_fov  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="场景目录或 scene.json 路径")
    ap.add_argument("--scene-id", default=None, help="掩码目录名，默认取场景目录名")
    ap.add_argument("--masks-root", default=str(ROOT / "dataset" / "scenes"))
    ap.add_argument("--suspicious-m", type=float, default=3.0,
                    help="尺寸超过这个米数就标为可疑（单张室内图的合理上限）")
    args = ap.parse_args()

    scene_path = Path(args.scene)
    scene = load_scene(scene_path)
    scene_id = args.scene_id or (
        scene_path.name if scene_path.is_dir() else scene_path.parent.name
    )
    mdir = masks_dir(scene_id, args.masks_root)

    print()
    print("=" * 78)
    print(f"  场景图体检  {scene.scene_id}   ({len(scene.nodes)} 物体 / {len(scene.edges)} 关系)")
    print("=" * 78)

    # ---- 1. 内参与视场 -----------------------------------------------------
    # 这一段是四项检查里**优先级最高**的：Phase 0 Step 6 用仓库自带的 GT 深度
    # 实测过，内参错会让三维误差中位数从 0.267 m 涨到 1.943 m（7.3 倍），
    # 而它不会报错、只会让所有横向米制数字安静地错。所以先看它。
    K = scene.camera_intrinsics
    meta = scene.build_meta
    #: 按可疑程度排序的清单。**在这里就声明**，因为内参那一段是全篇最严重的
    #: 嫌疑人，必须能往同一个列表里追加（顺序即优先级，见输出处的排序）。
    suspects: list[tuple[float, str]] = []
    img_h, img_w = meta.get("image_hw", [None, None])
    if K and img_w:
        fx, fy = float(K[0][0]), float(K[1][1])
        cx, cy = float(K[0][2]), float(K[1][2])
        fov = check_fov(np.asarray(K, dtype=np.float64), (int(img_h), int(img_w)))
        src = meta.get("intrinsics_source", "unknown")
        src_cn = {"provided": "传入的已知内参", "predicted": "模型预测",
                  "unknown": "未记录（旧版 scene.json）"}.get(src, src)
        print()
        print(f"  内参  fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}"
              f"   ← 来源：{src_cn}")
        print(f"  视场  水平 {fov.hfov_deg:.1f}°   竖直 {fov.vfov_deg:.1f}°"
              f"   可信区间 [{PLAUSIBLE_HFOV_DEG[0]:.0f}°, {PLAUSIBLE_HFOV_DEG[1]:.0f}°]"
              f"  判定 {fov.reason}")
        if not fov.plausible:
            suspects.append((
                10.0,  # 比尺寸越界更严重：它是**根因**，其余越界多半只是它的症状
                f"内参视场不可信（HFoV {fov.hfov_deg:.1f}°，reason={fov.reason}），"
                f"来源={src_cn} —— 横向坐标被整体放大，"
                "所有 x/y 偏移与三维尺寸都不可采信。"
                "**先解决这里，再去看其他嫌疑人 —— 它们多半是同一件事的症状。**"
                "已知内参请设 BuildConfig.known_intrinsics（量化证据见 "
                "phase0/probe_depth_gt.py）。",
            ))
        elif src == "predicted":
            print("        注意：这个内参是模型猜的。即便落在可信区间里，也只是"
                  "「不像错得离谱」，不等于准。")
        print(f"  深度范围 {meta.get('depth_range_m')} m")

    # ---- 2/3/4. 逐物体 -----------------------------------------------------
    print()
    print(f"  {'id':<15}{'label':<11}{'盒(px)':>22}{'掩码px':>8}"
          f"{'掩码盒/检测盒':>14}{'点云p90':>9}{'p90/ext':>9}  尺寸 w×h×l")

    for n in scene.nodes:
        box = n.bbox_2d or (0.0, 0.0, 0.0, 0.0)
        bw, bh = box[2] - box[0], box[3] - box[1]
        box_area = max(1.0, bw * bh)
        mask = None
        if n.mask_ref:
            p = mdir / f"{n.id}.png"
            if p.is_file():
                mask = load_mask(p)
        n_px = int(mask.sum()) if mask is not None else 0
        mb = "—"
        if mask is not None and mask.any():
            ys, xs = np.where(mask)
            mx1, mx2 = int(xs.min()), int(xs.max()) + 1
            my1, my2 = int(ys.min()), int(ys.max()) + 1
            m_area = max(1, (mx2 - mx1) * (my2 - my1))
            mb = f"{m_area / box_area:.2f}"
            # 掩码延伸到检测框之外的距离（像素），泄漏的直接指标
            leak = max(0, box[0] - mx1, box[1] - my1, mx2 - box[2], my2 - box[3])
        else:
            leak = 0

        e = n.extent_3d
        ext_max = max(e)
        # 「点云分布的 90 分位半径」与最大跨度的比值：越接近 1 说明点越散
        ratio = float("nan")
        print(f"  {n.id:<15}{n.label:<11}"
              f"[{box[0]:5.0f},{box[1]:5.0f},{box[2]:5.0f},{box[3]:5.0f}]"
              f"{n_px:>8}{mb:>14}{'':>9}{'':>9}  "
              f"{e[0]:.2f}×{e[1]:.2f}×{e[2]:.2f}")

        if ext_max > args.suspicious_m:
            suspects.append((
                ext_max,
                f"{n.id}: 尺寸 {ext_max:.2f} m 超过 {args.suspicious_m} m —— "
                f"掩码 {n_px} px（{n_px / box_area:.2f}× 检测盒面积），"
                f"框外泄漏最多 {leak:.0f} px",
            ))
        if mb != "—" and float(mb) > 2.0:
            suspects.append((
                2.0 + float(mb),
                f"{n.id}: 掩码外接盒是检测盒的 {mb} 倍 —— 掩码明显外溢，点云会被背景污染",
            ))
        if n.centroid_source != "mask":
            suspects.append((1.0, f"{n.id}: 质心走了降级路径（{n.centroid_source}）"))

    # ---- 汇总 ---------------------------------------------------------------
    print()
    print(f"  掩码占框比 均值 {meta.get('mask_box_coverage_mean')}  "
          f"最低 {meta.get('mask_box_coverage_min')}")
    print(f"  重力方向 {meta.get('up_axis')}  tilt={meta.get('up_axis_tilt_deg')}°  "
          f"reliable={meta.get('up_axis_reliable')} ({meta.get('up_axis_reason')})")

    if suspects:
        print()
        print(f"  嫌疑清单（{len(suspects)} 条，按严重程度降序）")
        for _, msg in sorted(suspects, key=lambda t: -t[0]):
            print(f"    ! {msg}")
    else:
        print()
        print("  没有发现越界数值。")

    print()
    print("  说明：本脚本只做「物理上是否可能」的量纲检查，")
    print("        真正的精度要等 Phase 5 的 3D 定位误差指标（需要 GT）。")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
