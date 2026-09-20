#!/usr/bin/env python
r"""
Phase 0 / Step 5 -- 内参：预测 vs 真值，以及「传入已知内参」这条被忽略的能力。

## 为什么必须做这个探针

Phase 1c 第一次把三个模型串起来跑真实照片时，场景图里出现了 **6.70 m 宽的沙发**。
沙发不可能 6.7 m 宽，所以先做量纲检查，一路查到源头：

    预测内参  fx = 163.7  →  水平视场 125.8°
    真值内参  fx = 518.9  →  水平视场  63.3°      （vendor/UniDepth/assets/demo/intrinsics.npy）
    比值      0.316

125.8° 的水平视场不是任何常见相机的形态。而 `x = (u - cx) · z / fx` ——
fx 小 3.16 倍，所有**横向**坐标就大 3.16 倍。于是：

    尺寸按预测内参算        → 沙发 6.70 m（荒谬）
    尺寸按真值内参算        → 沙发 1.91 m（正常双人沙发）

这不是 `builder.py` 的 bug。`unidepthv2.py:330` 的 `_postprocess_intrinsics` 只在
发生了 padding/resize 时才改 K，而本图 640×480 = 307200 px 落在
`pixels_bounds = [200000, 600000]` 内、宽高比 1.333 落在 `ratio_bounds = [0.5, 2.5]` 内
—— **既没 padding 也没 resize**，拿到的就是模型的原始预测。

## 本探针要回答的三个问题

  A. 预测内参与真值差多少？（已经把上面那段算清楚，这里做成可复现的读数）
  B. `infer(rgb, camera=...)` 传已知内参到底会不会被模型采用？
     —— 源码 `unidepthv2.py:361-362` 显示 camera 只在**非 None 时**才被用来生成 rays，
     所以这是一条**真实存在但从未被走通**的路径（VADAR 也从没传过）。
  C. 传入真值内参后，同一个物体区域的三维尺寸是否变得物理上合理？

## 结论会改变什么

如果 B/C 成立，那么「内参来源」就是一个**独立于模型选型的精度杠杆**：
  • 已知内参（EXIF、标定、数据集自带 —— Omni3D-Bench 就带 GT 相机）
    → 应传进去，而不是让模型猜。
  • 未知内参 → 预测值必须带上「视场合理性」检查，并在报告里如实标注。
这正好是本项目的核心主张在一个新层面的复现：**能测的就不要猜。**

用法：
    D:\3D_Spatial_Agent\venvs\vision\Scripts\python.exe phase0\probe_intrinsics.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEMO = ROOT / "vendor" / "UniDepth" / "assets" / "demo"
MASK_DIR = ROOT / "dataset" / "scenes" / "living_room" / "masks"
RESULT: dict = {}


def hr(title: str) -> None:
    print()
    print("=" * 70)
    print("  " + title)
    print("=" * 70)


def pick_key(d: dict, *frags: str):
    for k in sorted(d.keys()):
        if any(f in str(k).lower() for f in frags):
            return k
    return None


def as_chw(t) -> np.ndarray:
    a = t.detach().float().cpu().numpy()
    if a.ndim == 4:
        a = a[0]
    return a


def fov_deg(f: float, n: int) -> float:
    return float(2 * np.degrees(np.arctan(n / (2 * f))))


def centre_and_extent(pts: np.ndarray, sel: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """掩码内点云的中位数质心 + 逐轴跨度（与 `vision/geometry.py` 同口径）。"""
    flat = sel.reshape(-1)
    patch = pts.reshape(3, -1)[:, flat]
    patch = patch[:, np.isfinite(patch).all(axis=0)]
    if patch.shape[1] == 0:
        return np.zeros(3), np.zeros(3), 0
    c = np.median(patch, axis=1)
    r = np.linalg.norm(patch - c[:, None], axis=0)
    r90 = float(np.percentile(r, 90))
    keep = r <= 1.5 * r90 if r90 > 1e-12 else np.ones(patch.shape[1], bool)
    inl = patch[:, keep]
    return c, inl.max(axis=1) - inl.min(axis=1), int(patch.shape[1])


def main() -> int:
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from PIL import Image

    from unidepth.models import UniDepthV2
    from unidepth.utils.camera import Pinhole

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = "lpiccinelli/unidepth-v2-vits14"
    image = Image.open(DEMO / "rgb.png").convert("RGB")
    W, H = image.size

    print()
    print("Phase 0 / Step 5 -- 内参探针")
    print(f"  image {DEMO / 'rgb.png'}  {W}x{H}   device {device}")

    gt = np.load(DEMO / "intrinsics.npy").astype(np.float64).reshape(3, 3)
    gfx, gfy, gcx, gcy = float(gt[0, 0]), float(gt[1, 1]), float(gt[0, 2]), float(gt[1, 2])

    t0 = time.time()
    model = UniDepthV2.from_pretrained(repo).to(device).eval()
    load_s = time.time() - t0

    rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(device)

    # ---- 预处理：真的没有 padding / resize 吗？-----------------------------
    hr("A.  预处理与内参：预测 vs 真值")
    from unidepth.models.unidepthv2.unidepthv2 import get_paddings, get_resize_factor

    sc = model.shape_constraints
    ratio_bounds = sc["ratio_bounds"]
    pixels_bounds = (sc["pixels_min"], sc["pixels_max"])
    paddings, (pad_H, pad_W) = get_paddings((H, W), ratio_bounds)
    resize_factor, (new_H, new_W) = get_resize_factor((pad_H, pad_W), pixels_bounds)

    print(f"  shape_constraints   pixels{bounds_join(pixels_bounds)}  "
          f"ratio{ratio_bounds}")
    print(f"  input {W}x{H} = {W*H} px  -> padded {pad_W}x{pad_H}  -> internal {new_W}x{new_H}")
    print(f"  paddings={paddings}   resize_factor={resize_factor:.6f}")
    if resize_factor == 1.0 and paddings == (0, 0, 0, 0):
        print("  [事实] 既没 padding 也没 resize —— 拿到的 K 就是模型的原始预测，")
        print("         不是被某种预处理算歪的。")
    RESULT["preprocess"] = {
        "image_hw": [H, W], "pixels": W * H,
        "pixels_bounds": list(pixels_bounds), "ratio_bounds": list(ratio_bounds),
        "paddings": list(paddings), "resize_factor": float(resize_factor),
        "internal_hw": [int(new_H), int(new_W)],
    }

    with torch.no_grad():
        t0 = time.time()
        out_pred = model.infer(rgb)
        ms_pred = (time.time() - t0) * 1000
    K_pred = out_pred["intrinsics"].float().cpu().numpy().reshape(3, 3)
    pfx, pfy, pcx, pcy = (float(K_pred[0, 0]), float(K_pred[1, 1]),
                          float(K_pred[0, 2]), float(K_pred[1, 2]))

    print()
    print(f"  {'':<6}{'fx':>10}{'fy':>10}{'cx':>10}{'cy':>10}{'hfov':>9}{'vfov':>9}")
    print(f"  {'GT':<6}{gfx:>10.1f}{gfy:>10.1f}{gcx:>10.1f}{gcy:>10.1f}"
          f"{fov_deg(gfx, W):>9.1f}{fov_deg(gfy, H):>9.1f}")
    print(f"  {'pred':<6}{pfx:>10.1f}{pfy:>10.1f}{pcx:>10.1f}{pcy:>10.1f}"
          f"{fov_deg(pfx, W):>9.1f}{fov_deg(pfy, H):>9.1f}")
    print()
    print(f"  pred/GT 比值：fx {pfx/gfx:.4f}   fy {pfy/gfy:.4f}")
    print(f"  -> 横向坐标（x 与 y）都被放大了 {gfx/pfx:.2f} 倍")
    print(f"  infer 耗时 {ms_pred:.0f} ms   load {load_s:.1f} s")
    RESULT["intrinsics"] = {
        "gt": [gfx, gfy, gcx, gcy], "pred": [pfx, pfy, pcx, pcy],
        "gt_hfov": fov_deg(gfx, W), "pred_hfov": fov_deg(pfx, W),
        "ratio_fx": pfx / gfx, "ratio_fy": pfy / gfy,
        "lateral_inflation": gfx / pfx,
        "infer_ms": round(ms_pred, 1), "load_s": round(load_s, 2),
    }

    # ---- B. 传入已知内参会被采用吗？----------------------------------------
    hr("B.  infer(rgb, camera=Pinhole(K=GT)) —— 这条路径存在吗？")
    K_t = torch.tensor(gt, dtype=torch.float32, device=device).unsqueeze(0)
    try:
        cam = Pinhole(K=K_t)
        with torch.no_grad():
            t0 = time.time()
            out_gt = model.infer(rgb, camera=cam)
            ms_gt = (time.time() - t0) * 1000
        K_out = out_gt["intrinsics"].float().cpu().numpy().reshape(3, 3)
        print(f"  传入 GT 内参后，模型回传的 intrinsics：")
        print(f"    fx={float(K_out[0,0]):.2f}  fy={float(K_out[1,1]):.2f}  "
              f"cx={float(K_out[0,2]):.2f}  cy={float(K_out[1,2]):.2f}")
        used = abs(float(K_out[0, 0]) - gfx) < 1.0
        print(f"  [{'PASS' if used else 'FAIL'}] 模型"
              f"{'采用了' if used else '没有采用'}传入的内参")
        print(f"  耗时 {ms_gt:.0f} ms")
        RESULT["camera_input"] = {
            "accepted": bool(used),
            "returned_K": [float(v) for v in K_out.reshape(-1)],
            "infer_ms": round(ms_gt, 1),
        }
    except Exception as e:                                       # noqa: BLE001
        print(f"  [FAIL] 传 camera 失败：{type(e).__name__}: {e}")
        RESULT["camera_input"] = {"accepted": False, "error": f"{type(e).__name__}: {e}"}
        out_gt = None

    # ---- C. 同一个物体区域，两种内参下的三维尺寸 -----------------------------
    hr("C.  物理合理性：同一个掩码区域在两种内参下的三维尺寸")
    mask_files = sorted(MASK_DIR.glob("*.png")) if MASK_DIR.is_dir() else []
    if not mask_files:
        print(f"  [skip] 找不到掩码（{MASK_DIR}）。先跑 scripts/build_scene.py。")
    elif out_gt is None:
        print("  [skip] 没有拿到 camera 条件化的输出。")
    else:
        print(f"  用 {len(mask_files)} 个已有掩码作为固定像素区域（与内参无关，"
              f"所以两种内参下比较的是同一片像素）")
        print()
        print(f"  {'object':<16}{'n_px':>7}"
              f"{'预测内参 w×h×l (m)':>26}{'真值内参 w×h×l (m)':>26}")
        rows = []
        for mf in mask_files:
            with Image.open(mf) as im:
                m = np.asarray(im.convert("1"), dtype=np.uint8).astype(bool)
            if m.shape != (H, W):
                continue
            cp, ep, np_ = centre_and_extent(as_chw(out_pred["points"]), m)
            cg, eg, ng = centre_and_extent(as_chw(out_gt["points"]), m)
            if np_ == 0:
                continue
            print(f"  {mf.stem:<16}{np_:>7}"
                  f"{ep[0]:>9.2f}×{ep[1]:>5.2f}×{ep[2]:>5.2f}"
                  f"{'':>4}{eg[0]:>9.2f}×{eg[1]:>5.2f}×{eg[2]:>5.2f}")
            rows.append({
                "object": mf.stem, "n_px": np_,
                "pred_centre_m": [round(float(v), 4) for v in cp],
                "gt_centre_m": [round(float(v), 4) for v in cg],
                "pred_extent_m": [round(float(v), 4) for v in ep],
                "gt_extent_m": [round(float(v), 4) for v in eg],
            })
        if rows:
            pmax = max(max(r["pred_extent_m"]) for r in rows)
            gmax = max(max(r["gt_extent_m"]) for r in rows)
            prange = max(r["pred_centre_m"][0] for r in rows) - min(r["pred_centre_m"][0] for r in rows)
            grange = max(r["gt_centre_m"][0] for r in rows) - min(r["gt_centre_m"][0] for r in rows)
            print()
            print(f"  最大物体跨度   预测 {pmax:.2f} m   真值内参 {gmax:.2f} m")
            print(f"  场景横向跨度   预测 {prange:.2f} m   真值内参 {grange:.2f} m")
            print()
            print("  怎么读：真值内参那边给出的尺寸应当落在常识范围内")
            print("  （沙发 1.5–2.5 m、茶几 0.8–1.5 m、挂画 0.5–1.5 m）。")
            print("  如果两边差 ~3 倍，那就说明**横向尺度被内参误差整体放大**，")
            print("  而 z（深度）基本不受影响 —— 因为 depth 就是 points 的 z 列，")
            print("  近轴方向上 x=y≈0，focal 误差在那里不产生作用。")
            RESULT["object_sizes"] = {
                "rows": rows,
                "pred_max_span_m": round(float(pmax), 4),
                "gt_max_span_m": round(float(gmax), 4),
                "pred_scene_width_m": round(float(prange), 4),
                "gt_scene_width_m": round(float(grange), 4),
            }

            # ---- depth 是否一致 -------------------------------------------------
            dp = as_chw(out_pred["points"])[:, 0, 0]
            dg = as_chw(out_gt["points"])[:, 0, 0]
            print()
            print(f"  图像中心像素的深度：预测 {dp[2]:.3f} m   真值内参 {dg[2]:.3f} m"
                  f"   （差 {abs(dp[2]-dg[2])*1000:.0f} mm）")
            RESULT["centre_depth"] = {
                "pred_m": round(float(dp[2]), 4), "gt_m": round(float(dg[2]), 4),
            }

    # ---- VERDICT -----------------------------------------------------------
    hr("VERDICT")
    r = RESULT.get("intrinsics", {})
    if r:
        print(f"  预测视场 {r['pred_hfov']:.1f}°  vs  真值 {r['gt_hfov']:.1f}°"
              f"   —— 横向尺度差 {r['lateral_inflation']:.2f} 倍")
    if RESULT.get("camera_input", {}).get("accepted"):
        print("  [PASS] `infer(camera=...)` 确实会采用传入的已知内参。")
        print("         -> 「内参来源」是一个独立于模型选型的精度杠杆：")
        print("            有 GT 相机（EXIF / 标定 / Omni3D-Bench）就该传进去，")
        print("            而不是让模型猜。这要进 Phase 1 的接口与实验臂。")
    else:
        print("  [INFO] 传入 camera 未被采用或报错，见上文。")
    print()
    print("  ⚠️ 这条发现同时修正了一个此前没被写清楚的判断：")
    print("     `builder.py` 里 `scale_factor` 那一个标量**修不了**内参错误 ——")
    print("     focal 错会让横向按 k 倍放大而 z 不动，是各向异性的，")
    print("     不是「整体缩放」。所以 `calibrate_scale` 的设计要重新讨论。")
    print()

    out_path = HERE / "probe_intrinsics_result.json"
    out_path.write_text(json.dumps(RESULT, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  结果写入 {out_path}")
    print()
    return 0


def bounds_join(b) -> str:
    return f"=[{b[0]}, {b[1]}]"


if __name__ == "__main__":
    raise SystemExit(main())
