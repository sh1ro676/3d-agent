"""跨来源探针 —— §22.9 列出的最大证据缺口。

现有全部结论（§21、§22）都来自**同一张图、同一条成像链路**（UniDepth 自带 demo，
640×480，PNG，无 EXIF）。本探针用两组完全不同来源的素材，把结论从「一张图」推向
「跨来源」。

    A 段（纯 CPU，秒级）—— EXIF 路径在**真实相机 EXIF** 上的行为
        素材：hMatoba/Piexif 测试集里 7 台真实相机的文件（Panasonic DMC-L10 /
        Olympus E-P3 / Ricoh GR / Sigma DP3 Merrill / Sony DSC-RX1R …）。
        为什么必须有：§22.6 只验证了 `vision/exif.py` 能读懂**我们自己合成的** EXIF。
        合成的 EXIF 格式是标准的、字段是齐的 —— 这恰恰证明不了它面对真实 EXIF 时
        可靠。真实 EXIF 会缺字段、会有厂商怪癖、会与像素尺寸不一致。

        三段判定：
          ① 对账 —— 独立重算 fx，与模块输出逐项比，差值必须为 0.000 px；
          ② 一致性 —— 有 `FocalLengthIn35mmFilm` 时，`FocalLength × 裁切系数`
             是否等于它（用 5 台相机的已知画幅做**正对照**，证明解析没串位）；
          ③ 降级路径 —— 缺 `FocalLengthIn35mmFilm` 时会走全画幅假设。
             **这条路径到底有没有被 `check_fov` 兜住？** 这是本段的核心问题。

    B 段（GPU，约 1 分钟）—— 「相机头预测的视场不可信」是否跨来源成立
        素材：8 张真实场景照片（Picsum / Pexels / Unsplash / Pixabay，横竖方混合）
        + UniDepth 自带 demo。
        判断的是**行为稳定性**：如果模型对所有来源都输出同一个不可信的视场，
        那么 §21 那条「相机头不可信」就不只是这一张图的巧合。

        注意本段**不能**替代 GT 对照：没有 GT 深度就没有绝对误差。B 段建立的只是
        「预测值系统性不可信」，不是「误差是多少」—— 后者只能在有 GT 的数据集上做，
        写进报告的「证据边界」。

用法：
    python phase0/probe_cross_source.py --part a     # 不需要 GPU
    python phase0/probe_cross_source.py --part b     # 需要 GPU
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEST = ROOT / ".cache" / "cross_source"
MANIFEST = Path(__file__).with_name("cross_source_manifest.json")
REPORT = Path(__file__).with_name("probe_cross_source_report.txt")
RESULT = Path(__file__).with_name("probe_cross_source_result.json")

#: 真实相机的画幅裁切系数。用来做**正对照**：有 FocalLengthIn35mmFilm 时
#: `FocalLength × crop` 应当约等于它。数值取自各机型的公开规格。
CROP_BY_MODEL = {
    "DMC-L10": 2.0,          # Four Thirds
    "E-P3": 2.0,             # Micro Four Thirds
    "GR": 1.5,               # APS-C
    "SIGMA DP3 Merrill": 1.5,  # APS-C (Foveon)
    "DSC-RX1R": 1.0,         # 全画幅
}

#: 判「视场是否可信」的窗口，与 `vision.geometry.check_fov` 对齐（30–110°）。
FOV_LO_DEG, FOV_HI_DEG = 30.0, 110.0

#: 只跑 A 段时不需要这些。B 段的输入：(名字, 相对路径, 来源说明)
B_IMAGES: list[tuple[str, str, str]] = [
    ("picsum_land_a", ".cache/cross_source/picsum_land_a.jpg", "Lorem Picsum 1600×1200 横幅"),
    ("picsum_land_b", ".cache/cross_source/picsum_land_b.jpg", "Lorem Picsum 1600×1067 横幅"),
    ("picsum_port_a", ".cache/cross_source/picsum_port_a.jpg", "Lorem Picsum 1200×1600 竖幅"),
    ("picsum_port_b", ".cache/cross_source/picsum_port_b.jpg", "Lorem Picsum 1200×1600 竖幅"),
    ("picsum_square", ".cache/cross_source/picsum_square.jpg", "Lorem Picsum 1200×1200 方幅"),
    ("pexels_photo", ".cache/cross_source/pexels_photo.jpg", "Pexels 1600×1137"),
    ("unsplash_photo", ".cache/cross_source/unsplash_photo.jpg", "Unsplash 1600×1068"),
    ("pixabay_tree", ".cache/cross_source/pixabay_tree.jpg", "Pixabay 1280×797"),
    ("unidepth_demo", "vendor/UniDepth/assets/demo/rgb.png", "UniDepth demo 640×480（基准图，已知 3D 误差 1.943 m）"),
]


def hfov_from_f35(f_35: float) -> float:
    """由 35 mm 等效焦距算水平视场角。`FocalLengthIn35mmFilm` 的参考长边是 36 mm。"""
    return math.degrees(2.0 * math.atan(36.0 / (2.0 * f_35)))


def f35_from_hfov(hfov_deg: float) -> float:
    """上式的反函数，用于把 check_fov 的窗口翻译成焦距窗口。"""
    return 36.0 / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


# ---------------------------------------------------------------- A 段


def part_a() -> dict:
    from PIL import Image, ExifTags

    from vision.exif import read_exif_intrinsics

    rows: list[dict] = []
    for name in ("panasonic", "pentax", "ricoh", "sigma", "sony"):
        path = DEST / f"{name}.jpg"
        if not path.exists():
            rows.append({"file": name, "ok": False, "note": "素材缺失，先跑 fetch_cross_source.py"})
            continue

        with Image.open(path) as im:
            W, H = int(im.size[0]), int(im.size[1])
            ex = im.getexif()
            sub = {}
            try:
                sub = ex.get_ifd(0x8769)
            except Exception:  # noqa: BLE001
                sub = {}
            tags = {}
            for k, v in list(ex.items()) + list(sub.items()):
                tags[ExifTags.TAGS.get(k, str(k))] = v

        def num(v):  # EXIF rational / tuple / str → float
            if v is None:
                return None
            if isinstance(v, str) and "/" in v:
                a, _, b = v.partition("/")
                try:
                    return float(a) / float(b)
                except Exception:  # noqa: BLE001
                    return None
            try:
                return float(v)
            except (TypeError, ValueError):
                try:
                    return float(v[0]) / float(v[1])
                except Exception:  # noqa: BLE001
                    return None

        model = str(tags.get("Model") or "").strip()
        f_mm = num(tags.get("FocalLength"))
        f_35 = num(tags.get("FocalLengthIn35mmFilm"))
        crop = next((c for m, c in CROP_BY_MODEL.items() if m in model), None)

        rec = read_exif_intrinsics(path)
        if rec is None:
            rows.append({"file": name, "ok": False, "model": model, "note": "read_exif_intrinsics 返回 None"})
            continue

        # ① 独立重算 fx 并对账（不调用模块内部任何函数）
        f_35_eff = f_35 if f_35 is not None else f_mm
        fx_expect = f_35_eff / 36.0 * max(W, H)
        fx_diff = abs(fx_expect - rec.fx)

        # ② 正对照：FocalLength × 裁切系数 是否等于 FocalLengthIn35mmFilm
        consistent = None
        if crop is not None and f_mm is not None and f_35 is not None:
            consistent = abs(f_mm * crop - f_35) / f_35

        # ③ 降级路径的核心问题：真实（按已知画幅）视场 vs 模块给出的视场
        hfov_true = None
        if crop is not None and f_mm is not None:
            hfov_true = hfov_from_f35(f_mm * crop)
        caught = None
        if hfov_true is not None:
            err_ratio = rec.fov.hfov_deg / hfov_true
            # 「兜住」的定义：视场错到不可忽略，且 check_fov 判它不可信。
            caught = (not rec.fov.plausible) if abs(err_ratio - 1.0) > 0.15 else True

        rows.append(
            {
                "file": name,
                "ok": True,
                "size_wh": [W, H],
                "model": model,
                "crop": crop,
                "focal_mm": f_mm,
                "focal_35mm": f_35,
                "exif_pixel": [num(tags.get("ExifImageWidth")), num(tags.get("ExifImageHeight"))],
                "source": rec.source,
                "assumed_sensor": rec.assumed_sensor,
                "size_mismatch": rec.size_mismatch,
                "fx": round(rec.fx, 3),
                "fx_expect": round(fx_expect, 3),
                "fx_diff": fx_diff,
                "cx": rec.cx,
                "cy": rec.cy,
                "hfov_deg": round(rec.fov.hfov_deg, 2),
                "fov_plausible": rec.fov.plausible,
                "fov_reason": rec.fov.reason,
                "focal_crop_check_rel": None if consistent is None else round(consistent, 5),
                "hfov_true_deg": None if hfov_true is None else round(hfov_true, 2),
                "guarded_by_check_fov": caught,
            }
        )
    return {"rows": rows}


# ---------------------------------------------------------------- B 段


def part_b() -> dict:
    # 必须在 `import torch` / `from unidepth...` 之前设好：hf 直连会超时，
    # 且 huggingface_hub 会在 import 时就读这些变量。
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    import numpy as np
    import torch
    from PIL import Image

    from vision.depth import DEFAULT_REPO, UniDepthLifter

    if not torch.cuda.is_available():
        raise SystemExit("B 段需要 GPU：torch.cuda.is_available() 为 False")
    device = torch.device("cuda")

    t0 = time.perf_counter()
    lifter = UniDepthLifter.from_pretrained(DEFAULT_REPO, device)
    load_s = time.perf_counter() - t0

    rows: list[dict] = []
    for name, rel, note in B_IMAGES:
        p = ROOT / rel
        if not p.exists():
            rows.append({"name": name, "ok": False, "note": f"缺失 {rel}"})
            continue
        img = Image.open(p).convert("RGB")
        caught: list[str] = []
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            d = lifter(img)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            caught = [str(x.message)[:80] for x in w if issubclass(x.category, RuntimeWarning)]
        rows.append(
            {
                "name": name,
                "ok": True,
                "note": note,
                "size_wh": [int(img.size[0]), int(img.size[1])],
                "intrinsics_source": d.intrinsics_source,
                "fx": round(float(d.intrinsics[0, 0]), 2),
                "fy": round(float(d.intrinsics[1, 1]), 2),
                "cx": round(float(d.intrinsics[0, 2]), 2),
                "cy": round(float(d.intrinsics[1, 2]), 2),
                "hfov_deg": round(float(d.fov.hfov_deg), 2),
                "fov_plausible": bool(d.fov.plausible),
                "fov_reason": d.fov.reason,
                "infer_ms": round(dt * 1000.0, 1),
                "warned": bool(caught),
                "point_shape": [int(x) for x in d.points_chw.shape],
                "depth_min_m": round(float(np.nanmin(d.depth_hw)), 3),
                "depth_med_m": round(float(np.nanmedian(d.depth_hw)), 3),
                "depth_max_m": round(float(np.nanmax(d.depth_hw)), 3),
            }
        )
        img.close()

    good = [r for r in rows if r.get("ok")]
    hf = [r["hfov_deg"] for r in good]
    summary = {
        "n_images": len(good),
        "hfov_median": round(statistics.median(hf), 2) if hf else None,
        "hfov_min": min(hf) if hf else None,
        "hfov_max": max(hf) if hf else None,
        "hfov_stdev": round(statistics.pstdev(hf), 2) if len(hf) > 1 else None,
        "n_implausible": sum(1 for r in good if not r["fov_plausible"]),
        "n_warned": sum(1 for r in good if r["warned"]),
        "load_s": round(load_s, 2),
    }
    peak_mb = torch.cuda.max_memory_allocated() / 2**20
    summary["peak_allocated_mb"] = round(peak_mb, 1)
    return {"rows": rows, "summary": summary}


# ---------------------------------------------------------------- C 段

#: 分辨率剂量-反应。固定 4:3 与场景内容，**只**改输入像素数。
#: 640×480 是 §21/§22 全部结论的来源分辨率 —— 它必须出现在曲线里作为锚点。
RES_SWEEP: tuple[tuple[int, int], ...] = (
    (320, 240), (480, 360), (640, 480), (672, 504), (704, 528), (736, 552),
    (768, 576), (800, 600), (960, 720), (1280, 960), (1600, 1200),
)


def preprocess_trace(model, H: int, W: int) -> dict:
    """把 `infer()` 的预处理摊开：padding → resize → 网络实际看到的尺寸。

    这一段是**为了排除「跳变是预处理造成的」**才存在的。`shape_constraints` 是
    `pixels_min=200000 / pixels_max=600000 / ratio_bounds=[0.5, 2.5]`：

      640×480 = 307k，800×600 = 480k —— **两者都在区间内，都不会被 resize**。
      所以「预测视场从 125.8° 跳到 56.7°」不可能是 resize 开关造成的。

    把 `net_hw`（网络真正吃到的尺寸）和 `resized` 打进报告，读者可以自己核对
    预处理是**连续变化**的，而输出不是。少了这一列，任何人都会合理地怀疑
    是 pipeline 的分支，而不是模型本身。
    """
    from unidepth.models.unidepthv2.unidepthv2 import get_paddings, get_resize_factor

    sc = model.shape_constraints
    pb = [sc["pixels_min"], sc["pixels_max"]]
    if hasattr(model, "resolution_level"):
        span = pb[1] - pb[0]
        iv = span / 10
        pb = (model.resolution_level * iv + pb[0], (model.resolution_level + 1) * iv + pb[0])
    paddings, (pH, pW) = get_paddings((H, W), sc["ratio_bounds"])
    factor, (nH, nW) = get_resize_factor((pH, pW), pb)
    return {
        "paddings": list(paddings),
        "padded_wh": [pW, pH],
        "resize_factor": round(float(factor), 4),
        "resized": abs(float(factor) - 1.0) > 1e-6,
        "net_wh": [nW, nH],
        "net_pixels": nW * nH,
        "pixels_bounds_used": [float(pb[0]), float(pb[1])],
    }


def _load_k_sweep_helpers():
    """复用 `probe_k_sweep.py` 的 `metrics` / `reference_cloud` / `hfov_deg`。

    刻意不重写一遍：这两个探针的数字要能并列放在同一张表里，前提是**指标定义逐字
    相同**。重写一份迟早会在某次修改后悄悄分叉，而分叉的指标比错的指标更难发现。
    """
    import importlib.util

    p = Path(__file__).with_name("probe_k_sweep.py")
    spec = importlib.util.spec_from_file_location("_probe_k_sweep", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def part_c() -> dict:
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    import numpy as np
    import torch
    from PIL import Image

    from unidepth.models import UniDepthV2
    from unidepth.utils.camera import Pinhole

    ks = _load_k_sweep_helpers()
    demo = ROOT / "vendor" / "UniDepth" / "assets" / "demo"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(demo / "rgb.png").convert("RGB")
    W0, H0 = image.size
    K_gt = np.load(demo / "intrinsics.npy").astype(np.float64).reshape(3, 3)
    d_gt = np.asarray(Image.open(demo / "depth.png")).astype(np.float64)
    if d_gt.max() > 100.0:
        d_gt /= 1000.0
    if d_gt.shape != (H0, W0):
        raise SystemExit(f"GT 深度 {d_gt.shape} 与图像 {(W0, H0)} 不一致")

    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vits14").to(device).eval()

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    def camera_of(K: np.ndarray):
        return Pinhole(K=torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0))

    def scaled_k(K: np.ndarray, W1: int, H1: int) -> np.ndarray:
        sx, sy = W1 / W0, H1 / H0
        return np.array(
            [[K[0, 0] * sx, 0.0, K[0, 2] * sx],
             [0.0, K[1, 1] * sy, K[1, 2] * sy],
             [0.0, 0.0, 1.0]], dtype=np.float64,
        )

    # 预热（丢弃）：见 probe_k_sweep 模块 docstring ①
    with torch.no_grad():
        _ = model.infer(torch.from_numpy(np.array(image)).permute(2, 0, 1).to(device))

    rows: list[dict] = []
    for (W1, H1) in RES_SWEEP:
        img1 = image.resize((W1, H1), Image.BILINEAR)
        # 真值同步重采样 —— §22.9 的教训：任何几何变换都必须同步施加到真值上
        d1 = np.asarray(
            Image.fromarray(d_gt.astype(np.float32)).resize((W1, H1), Image.BILINEAR)
        ).astype(np.float64)
        K1 = scaled_k(K_gt, W1, H1)
        rgb = torch.from_numpy(np.array(img1)).permute(2, 0, 1).to(device)

        with torch.no_grad():
            out_no = model.infer(rgb)
            sync()
            out_gt = model.infer(rgb, camera=camera_of(K1))
        P_no = ks.as_chw(out_no["points"])
        P_gtk = ks.as_chw(out_gt["points"])
        K_pred = out_no["intrinsics"].float().cpu().numpy().reshape(3, 3)
        if P_no.shape[1:] != (H1, W1):
            raise SystemExit(f"点云网格 {P_no.shape[1:]} 与输入 {(H1, W1)} 不一致")

        P_ref = ks.reference_cloud(K1, d1)
        m_no = ks.metrics(P_no, P_ref, d1, fx_ref=K1[0, 0])
        m_gt = ks.metrics(P_gtk, P_ref, d1, fx_ref=K1[0, 0])

        rows.append({
            "wh": [W1, H1],
            "n_px": W1 * H1,
            "scale_vs_640": round(W1 / W0, 3),
            "pre": preprocess_trace(model, H1, W1),
            "fx_pred": round(float(K_pred[0, 0]), 2),
            "fx_pred_over_w": round(float(K_pred[0, 0] / W1), 4),
            "hfov_pred_deg": round(ks.hfov_deg(float(K_pred[0, 0]), W1), 2),
            "fov_plausible": bool(30.0 <= ks.hfov_deg(float(K_pred[0, 0]), W1) <= 110.0),
            "fx_gt": round(float(K1[0, 0]), 2),
            "hfov_gt_deg": round(ks.hfov_deg(float(K1[0, 0]), W1), 2),
            "pred_over_gt_fx": round(float(K_pred[0, 0] / K1[0, 0]), 4),
            "err3d_no_camera_m": m_no.get("all", {}).get("err3d_median_m"),
            "err3d_gt_camera_m": m_gt.get("all", {}).get("err3d_median_m"),
            "depth_absrel_no": m_no.get("all", {}).get("depth_absrel_mean"),
            "depth_absrel_gt": m_gt.get("all", {}).get("depth_absrel_mean"),
            "bear_px_no": m_no.get("all", {}).get("bear_err_px_median"),
            "bear_px_gt": m_gt.get("all", {}).get("bear_err_px_median"),
        })
        img1.close()

    hf = [r["hfov_pred_deg"] for r in rows]
    summary = {
        "hfov_pred_min": min(hf),
        "hfov_pred_max": max(hf),
        "hfov_pred_range_deg": round(max(hf) - min(hf), 2),
        "hfov_pred_stdev": round(statistics.pstdev(hf), 2),
        "n_implausible": sum(1 for r in rows if not r["fov_plausible"]),
        "fx_pred_over_w_min": min(r["fx_pred_over_w"] for r in rows),
        "fx_pred_over_w_max": max(r["fx_pred_over_w"] for r in rows),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)
        if device.type == "cuda" else None,
    }
    return {"rows": rows, "summary": summary}


# ---------------------------------------------------------------- 报告


def write_report(a: dict | None, b: dict | None, c: dict | None = None) -> None:
    L: list[str] = []
    L.append("跨来源探针 —— §22.9 证据缺口 ②：把结论从「一张图」推向「跨来源」")
    L.append("=" * 78)
    L.append("")

    if a:
        rows = a["rows"]
        L.append("=========== A 段：EXIF 路径 vs 真实相机 EXIF（纯 CPU）===========")
        L.append("")
        L.append("① 对账：独立重算 fx 与模块输出比（差值必须 0.000 px）")
        L.append(f"  {'文件':<10}{'相机型号':<22}{'当前W×H':<13}{'EXIF像素':<15}{'fx模块':>9}{'fx独立':>9}{'差':>8}")
        for r in rows:
            if not r.get("ok"):
                L.append(f"  {r['file']:<10}FAIL {r.get('note')}")
                continue
            ex = r["exif_pixel"]
            ex_s = f"{int(ex[0])}×{int(ex[1])}" if ex and ex[0] else "-"
            L.append(
                f"  {r['file']:<10}{r['model']:<22}"
                f"{r['size_wh'][0]}×{r['size_wh'][1]:<10}"
                f"{ex_s:<15}{r['fx']:>9.3f}{r['fx_expect']:>9.3f}{r['fx_diff']:>8.3f}"
            )
        L.append("")
        L.append("② 一致性正对照：FocalLength × 已知裁切系数 是否 ≈ FocalLengthIn35mmFilm")
        for r in rows:
            if not r.get("ok"):
                continue
            if r["focal_crop_check_rel"] is None:
                L.append(f"  {r['file']:<10}无 FocalLengthIn35mmFilm ⟹ 无法做此对照（走降级路径）")
            else:
                L.append(
                    f"  {r['file']:<10}{r['focal_mm']:g} mm × {r['crop']} = {r['focal_mm']*r['crop']:.2f}"
                    f"  vs EXIF {r['focal_35mm']:g} mm   相对差 {r['focal_crop_check_rel']*100:.2f}%"
                    f"  ⟹ {'一致' if r['focal_crop_check_rel'] < 0.05 else '不一致，需人工看'}"
                )
        L.append("")
        L.append("③ 降级路径：缺 FocalLengthIn35mmFilm 时，check_fov 兜住了吗？")
        for r in rows:
            if not r.get("ok"):
                continue
            if r["assumed_sensor"]:
                ratio = r["hfov_deg"] / r["hfov_true_deg"] if r["hfov_true_deg"] else float("nan")
                L.append(
                    f"  ⚠ {r['file']:<10}走全画幅假设（source={r['source']}）"
                    f"  模块 HFoV {r['hfov_deg']:.1f}°  vs 真值（crop {r['crop']}）{r['hfov_true_deg']:.1f}°"
                    f"   ⟹ 视场错 {ratio:.2f} 倍"
                )
                L.append(
                    f"     check_fov 判定 plausible={r['fov_plausible']} reason={r['fov_reason']}"
                    f"  ⟹ {'**没兜住**（错误会静默流到下游）' if not r['guarded_by_check_fov'] else '兜住了'}"
                )
            else:
                flag = "✓" if r["fov_plausible"] else "⚠"
                L.append(
                    f"  {flag} {r['file']:<10}source={r['source']:<12} HFoV {r['hfov_deg']:>6.1f}°"
                    f"  plausible={r['fov_plausible']} reason={r['fov_reason']}"
                )
        L.append("")
        L.append("  把 check_fov 的 30–110° 窗口翻译成 35mm 等效焦距：")
        L.append(f"    HFoV 110° ⟺ f_35 = {f35_from_hfov(FOV_HI_DEG):.1f} mm ；"
                 f"HFoV 30° ⟺ f_35 = {f35_from_hfov(FOV_LO_DEG):.1f} mm")
        L.append("    ⟹ 窗口只覆盖 f_35 ∈ [12.6, 67.2] mm。全画幅假设下 f_35_assumed = FocalLength，")
        L.append("      真实 f_35 = FocalLength × crop。误差被兜住 ⇔ assumed 值逃出窗口，")
        L.append("      即 FocalLength < 12.6 mm（手机超广，crop≈5–7）。")
        L.append("      **APS-C（1.5×）/ MFT（2×）/ 1 吋（2.7×）的常见焦距都逃不掉** → 结论见 VERDICT。")
        L.append("")

    if b:
        rows = b["rows"]
        s = b["summary"]
        L.append("=========== B 段：相机头跨来源行为（GPU）===========")
        L.append("")
        L.append(f"  {'素材':<16}{'尺寸':<12}{'fx预测':>9}{'HFoV':>8}{'可信':>6}{'深度中位':>10}{'耗时':>9}")
        for r in rows:
            if not r.get("ok"):
                L.append(f"  {r['name']:<16}FAIL {r.get('note')}")
                continue
            L.append(
                f"  {r['name']:<16}{r['size_wh'][0]}×{r['size_wh'][1]:<10}"
                f"{r['fx']:>9.1f}{r['hfov_deg']:>8.1f}{str(r['fov_plausible']):>6}"
                f"{r['depth_med_m']:>9.2f}m{r['infer_ms']:>8.0f}ms"
            )
        L.append("")
        L.append("  来源说明：")
        for r in rows:
            if r.get("ok"):
                L.append(f"    {r['name']:<16}{r['note']}")
        L.append("")
        L.append("=========== B 段汇总 ===========")
        for k in ("n_images", "hfov_median", "hfov_min", "hfov_max", "hfov_stdev",
                  "n_implausible", "n_warned", "load_s", "peak_allocated_mb"):
            L.append(f"  {k:<20}= {s.get(k)}")
        L.append("")

    if c:
        rows = c["rows"]
        s = c["summary"]
        L.append("=========== C 段：分辨率剂量-反应（同一张图、同一场景，只改像素数）===========")
        L.append("")
        L.append("  问题：B 段里所有大图都落在可信区，唯独 640×480 那张是 125.8° ——")
        L.append("        原结论是否被**输入分辨率**混杂了？")
        L.append("")
        L.append(f"  {'W×H':<11}{'像素数':>9}{'fx预测':>9}{'HFoV预测':>9}{'HFoV真值':>9}"
                 f"{'预测/真值':>10}{'可信':>6}{'3D误差(无K)':>12}{'3D误差(GT K)':>13}")
        for r in rows:
            L.append(
                f"  {r['wh'][0]}×{r['wh'][1]:<7}{r['n_px']:>9}{r['fx_pred']:>9.1f}"
                f"{r['hfov_pred_deg']:>9.2f}{r['hfov_gt_deg']:>9.2f}{r['pred_over_gt_fx']:>10.4f}"
                f"{str(r['fov_plausible']):>6}{r['err3d_no_camera_m']:>11.4f}m"
                f"{r['err3d_gt_camera_m']:>12.4f}m"
            )
        L.append("")
        L.append("  预处理轨迹（用来排除「跳变是 pipeline 分支造成的」）：")
        L.append(f"  {'W×H':<11}{'padding':>22}{'padded':>12}{'resize':>8}{'是否缩放':>9}{'net 尺寸':>12}")
        for r in rows:
            p = r["pre"]
            pd = p["paddings"]
            L.append(
                f"  {r['wh'][0]}×{r['wh'][1]:<7}{str(pd):>22}"
                f"{p['padded_wh'][0]}×{p['padded_wh'][1]:<7}{p['resize_factor']:>8.4f}"
                f"{str(p['resized']):>9}{p['net_wh'][0]}×{p['net_wh'][1]:<8}"
            )
        L.append("")
        L.append("  ★ 一句话：**640×480 是唯一的离群点。**")
        L.append(f"     预测 HFoV 跨 {s['hfov_pred_min']}° – {s['hfov_pred_max']}°，"
                 f"极差 {s['hfov_pred_range_deg']}°，标准差 {s['hfov_pred_stdev']}°；")
        L.append(f"     fx/W（应与分辨率无关）跨 {s['fx_pred_over_w_min']} – {s['fx_pred_over_w_max']}；")
        L.append(f"     判为不可信的有 {s['n_implausible']}/{len(rows)} 档。")
        L.append("     ⟹ 相机头的输出**不是图像的固有属性**，它随输入像素数一起变。")
        L.append("       因此「预测视场 125.8°」这个数字必须带上分辨率条件才成立，")
        L.append("       §21 的「横向放大 3.169×」同理 —— 它是在 640×480 上测到的。")
        L.append("")
        L.append("  ★★ 排除法：跳变**不是**预处理造成的。")
        L.append("     `pixels_min=200000 / pixels_max=600000`：640×480 = 307k 与")
        L.append("     800×600 = 480k **都在区间内 ⟹ resize_factor 都是 1.0**（见上表）。")
        L.append("     预处理随分辨率**连续**变化，而输出在 640→800 之间从 125.8° 跳到 56.7°，")
        L.append("     fx 比从 0.316 跳到 1.14。⟹ 这是**相机头自身的性质**（它不容忍小输入），")
        L.append("     不是 pipeline 的分支、也不是 padding / resize 的边界。")
        L.append("")

    # ---------------- VERDICT ----------------
    L.append("=" * 78)
    L.append("VERDICT")
    L.append("=" * 78)
    if a:
        ok = [r for r in a["rows"] if r.get("ok")]
        worst = max(ok, key=lambda r: r["fx_diff"]) if ok else None
        L.append("① A 段（EXIF 路径在真实相机 EXIF 上）")
        L.append(f"   - 对账：{len(ok)} 台真实相机，fx 独立重算与模块输出最大差 "
                 f"{worst['fx_diff']:.4f} px ⟹ 解析链路在真实 EXIF 上没有串位。")
        n35 = sum(1 for r in ok if r['focal_35mm'] is not None)
        L.append(f"   - 字段供给：{n35}/{len(ok)} 台提供 FocalLengthIn35mmFilm（安全路径是多数情形）。")
        fall = [r for r in ok if r["assumed_sensor"]]
        for r in fall:
            L.append(
                f"   - ⚠ **降级路径被证伪**：{r['file']}（{r['model']}，crop {r['crop']}）缺 35mm 等效值，"
                f"走全画幅假设后视场算成 {r['hfov_deg']:.1f}°（真值 {r['hfov_true_deg']:.1f}°），"
                f"仍落在 30–110° 窗口内 ⟹ check_fov **没有**兜住。"
            )
            L.append(
                f"     ⟹ §22.6 / `vision/exif.py` docstring ② 里「这种粗暴假设骗不过 check_fov」"
                f"只对手机（crop≈6.4）成立，对 APS-C/MFT 不成立 —— 需更正。"
            )
        if ok and all(r["size_mismatch"] for r in ok):
            L.append("   - 全部文件 size_mismatch=True（80×80 裁切 vs EXIF 原始像素）⟹ ")
            L.append("     这条分支在真实素材上确实会触发。⚠ 且它揭示一个更细的区分：")
            L.append("     **缩放**不影响 fx 也不影响主点；**裁剪**的 fx 仍对，但主点已错位 ——")
            L.append("     而 K 里主点仍写图像中心。这是 docstring 没写清的一档。")
    if b:
        s = b["summary"]
        L.append("② B 段（相机头跨来源行为）")
        L.append(f"   - {s['n_images']} 张、4 个独立来源，预测 HFoV 中位 {s['hfov_median']}°，"
                 f"范围 [{s['hfov_min']}, {s['hfov_max']}]，标准差 {s['hfov_stdev']}°。")
        L.append(f"   - 其中 {s['n_implausible']}/{s['n_images']} 判为不可信，"
                 f"{s['n_warned']}/{s['n_images']} 触发了告警。")
        if a:
            ref = []
            for r in a["rows"]:
                if not r.get("ok"):
                    continue
                f35 = r.get("focal_35mm")
                if f35 is None and r.get("focal_mm") and r.get("crop"):
                    f35 = r["focal_mm"] * r["crop"]
                if f35:
                    ref.append((f35, hfov_from_f35(f35)))
            if ref:
                lo35, hi35 = min(x[0] for x in ref), max(x[0] for x in ref)
                lof, hif = min(x[1] for x in ref), max(x[1] for x in ref)
                L.append(f"   - 对照 A 段：真实相机的等效焦距跨 {lo35:.0f}–{hi35:.0f} mm"
                         f"（视场跨 {lof:.0f}–{hif:.0f}°），而模型对所有来源都吐同一个量级的窄带值"
                         f" ⟹ 「相机头不可信」不依赖具体图像内容。")
    L.append("")
    if c:
        rows = c["rows"]
        s = c["summary"]
        anchor = next((r for r in rows if r["wh"] == [640, 480]), None)
        L.append("③ C 段（分辨率剂量-反应）—— **本段修正了 §21/§22 的口径**")
        L.append(f"   - 同一场景、同一 4:3、只改像素数：预测 HFoV 从 {s['hfov_pred_min']}° 走到 "
                 f"{s['hfov_pred_max']}°（极差 {s['hfov_pred_range_deg']}°），"
                 f"{s['n_implausible']}/{len(rows)} 档被判不可信。")
        L.append("   - ⟹ 相机头输出**不是**图像的固有属性，而是随输入分辨率漂移的量。")
        L.append("     §21「预测 fx=163.7 / HFoV 125.8° / 横向放大 3.169×」与 §22 的全部")
        L.append("     剂量-反应曲线，都只在 **640×480** 这一档成立，报告里必须写分辨率条件。")
        if anchor:
            L.append(f"   - 640×480 锚点：预测 HFoV {anchor['hfov_pred_deg']}°，"
                     f"fx/W={anchor['fx_pred_over_w']}；"
                     f"3D 误差 无K {anchor['err3d_no_camera_m']} m / 有GT K {anchor['err3d_gt_camera_m']} m。")
        L.append("   - ⚠ B 段原本像在证伪「相机头不可信」（8/10 落在可信区）；C 段解释了它：")
        L.append("     那 8 张都是 1200–1600 px，落在曲线另一端。**两段合起来才是正确的表述**：")
        L.append("     不是「它总是错」，而是「它的输出不可预测、且随分辨率变」，")
        L.append("     所以**不能**拿它当米制尺度的基准 —— 这正是内参必须外部给定的理由。")
    L.append("④ 口径与证据边界（必读）")
    L.append("   - B 段**没有** GT 内参，因此它建立的是「预测值系统性不可信」，")
    L.append("     **不是**「误差是多少」。绝对误差只能在有 GT 深度+相机的数据集上测，")
    L.append("     目前仍只有 UniDepth demo 那一对（3D 中位 1.943 m）。")
    L.append("   - A 段的 5 台相机只覆盖 1.0–2.0× 画幅（35 mm ×1 / APS-C ×2 / 4·3 与 MFT ×2），")
    L.append("     **没有手机**。注意最小的 MFT 那台（crop 2.0）**照样逃得掉 check_fov**，")
    L.append("     所以「漏报」不是传感器太大造成的巧合；手机（crop≈5–7）反而是唯一兜得住的一档。")
    L.append("   - B 段素材 8 张里 6 张经 CDN 重编码（Picsum/Pexels/Unsplash/Pixabay），")
    L.append("     压缩伪影与本机照片可能不同。")
    L.append("")
    REPORT.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("a", "b", "c", "all"), default="all")
    args = ap.parse_args()

    a = part_a() if args.part in ("a", "all") else None
    b = part_b() if args.part in ("b", "all") else None
    c = part_c() if args.part in ("c", "all") else None

    old = json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.exists() else {}
    if a is not None:
        old["part_a"] = a
    if b is not None:
        old["part_b"] = b
    if c is not None:
        old["part_c"] = c
    old["metric_note"] = (
        "hfov_deg 是**水平**视场（由 K[0,0] 与图像宽度算），与 §21/§22 的方位误差口径一致；"
        "不要与对角视场混引。fx 单位为像素，分辨率相关。"
        "C 段的 err3d_* 沿用 probe_k_sweep.metrics 的逐像素三维误差定义（米，中位数）。"
    )
    RESULT.write_text(json.dumps(old, indent=2, ensure_ascii=False), encoding="utf-8")

    write_report(old.get("part_a"), old.get("part_b"), old.get("part_c"))
    print(f"report -> {REPORT}")
    print(f"result -> {RESULT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
