#!/usr/bin/env python
r"""Phase 0 / Step 6 -- 用仓库自带的 GT depth，把「内参来源」这件事从「像不像」
量化成「差多少」。

## 为什么在 Step 5 之后还要再做一次

Step 5（`probe_intrinsics.py`）已经测出：

    预测内参  fx = 163.7  →  水平视场 125.8°
    真值内参  fx = 518.9  →  水平视场  63.3°
    比值      0.316        →  横向坐标被放大 3.17 倍

但那是**用一个先验判断**（「沙发不可能 6.7 m 宽」）推出来的结论。先验判断
在答辩时是可以被质疑的：「也许这个房间真的很大呢？」。所以需要一条不依赖
常识的、可以逐像素验证的证据。

它就在仓库里：`vendor/UniDepth/assets/demo/depth.png` 是与 `rgb.png` **配对
的 GT 深度图**（单位毫米）。`vendor/UniDepth/scripts/demo.py` 里
`depth_gt = np.array(Image.open("assets/demo/depth.png")).astype(float) / 1000.0`
—— 官方 demo 自己就是这么用的。于是：

    P_gt = unproject(像素网格, GT内参 K) × GT深度      ← 完全已知，不含任何模型预测

有了 `P_gt`，两条推理路径的**三维误差**就可以逐像素算出来：
    A 路径  infer(rgb)                        ← 模型自己猜内参
    B 路径  infer(rgb, camera=Pinhole(K_gt))  ← 喂已知内参

## 顺带纠正一条被官方 README 掩盖的事实

`README.md:140` 的措辞是 "You can use ground truth intrinsics as input to the
model **as well**"（as well = 锦上添花）。但 `scripts/demo.py:14` 的官方 demo
**就是传了 camera 的**——也就是说，作者自己验证效果时用的从来不是纯 RGB 路径。
本探针量化这个差别到底有多大。

## 分区统计是刻意加的

如果误差纯粹是「内参错了导致横向拉伸」，那么它应当**随视场角增大而增大**：
画面正中（近轴）几乎不受影响，四角最严重。这个趋势能反过来验证诊断本身，
并且能解释一个容易被忽略的后果 —— **z 也会错**，因为 `depth` 就是
`radius × ray_z`，`ray_z` 在外围会因为方向场错误而偏离 cos(θ)。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEMO = ROOT / "vendor" / "UniDepth" / "assets" / "demo"
RESULT: dict = {}
LINES: list[str] = []

#: 视场「可信」区间（水平，度）。低于 30° 是长焦，高于 110° 是鱼眼 ——
#: 都不是普通手机/相机拍室内照片会出现的形态。见 `vision/geometry.py`。
PLAUSIBLE_HFOV = (30.0, 110.0)


def say(s: str = "") -> None:
    print(s)
    LINES.append(s)


def hr(title: str) -> None:
    say()
    say("=" * 74)
    say("  " + title)
    say("=" * 74)


def as_chw(t) -> np.ndarray:
    a = t.detach().float().cpu().numpy()
    if a.ndim == 4:
        a = a[0]
    return a.astype(np.float64)


def hfov_deg(f: float, w: int) -> float:
    return float(2 * np.degrees(np.arctan(w / (2.0 * f))))


def unproject(K: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """像素网格 → z=1 平面上的 (x, y)。

    用 0 基整数网格（`coords_grid` 的常见约定）。用 0.5 基只差半个像素，
    在 fx≈519、3 m 处的横向影响约 3 mm —— 相对本次要测的 3 倍误差可忽略，
    但这个约定必须写出来，否则它是一个无声的系统偏差。
    """
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    return (uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1]


def zone_radius(h: int, w: int) -> np.ndarray:
    """归一化到「半对角线 = 1」的像面半径。"""
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = np.hypot(uu - cx, vv - cy)
    return r / float(np.hypot(cx, cy))


def metrics(P: np.ndarray, P_gt: np.ndarray, d_gt: np.ndarray, valid: np.ndarray) -> dict:
    """逐像素三维误差。`P` 与 `P_gt` 同为 (3, H, W)，同一像素系。"""
    d_p = P[2]
    e3d = np.linalg.norm(P - P_gt, axis=0)
    n_gt = np.linalg.norm(P_gt, axis=0)

    def agg(m: np.ndarray) -> dict:
        if not m.any():
            return {}
        d_pred, d_ref = d_p[m], d_gt[m]
        rel_d = np.abs(d_pred - d_ref) / np.clip(d_ref, 1e-6, None)
        ratio = np.maximum(d_pred / np.clip(d_ref, 1e-6, None),
                           d_ref / np.clip(d_pred, 1e-6, None))
        ee, nn = e3d[m], n_gt[m]
        return {
            "n_px": int(m.sum()),
            "depth_absrel_mean": round(float(rel_d.mean()), 4),
            "depth_absrel_median": round(float(np.median(rel_d)), 4),
            "depth_delta1_25": round(float((ratio < 1.25).mean()), 4),
            "depth_rmse_m": round(float(np.sqrt((d_pred - d_ref) ** 2).mean()), 4),
            "err3d_median_m": round(float(np.median(ee)), 4),
            "err3d_p90_m": round(float(np.percentile(ee, 90)), 4),
            "err3d_rel_median": round(float(np.median(ee / np.clip(nn, 1e-6, None))), 4),
        }

    r = zone_radius(P.shape[1], P.shape[2])
    out = {"all": agg(valid), "inner_r50": agg(valid & (r <= 0.5)),
           "outer_r50": agg(valid & (r > 0.5))}
    return out


def main() -> int:
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    import torch
    from PIL import Image

    from unidepth.models import UniDepthV2
    from unidepth.utils.camera import Pinhole

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = "lpiccinelli/unidepth-v2-vits14"
    image = Image.open(DEMO / "rgb.png").convert("RGB")
    W, H = image.size

    say()
    say("Phase 0 / Step 6 -- GT 深度下的内参消融")
    say(f"  image  {DEMO / 'rgb.png'}  {W}x{H}   device {device}")

    K_gt = np.load(DEMO / "intrinsics.npy").astype(np.float64).reshape(3, 3)
    d_gt_raw = np.asarray(Image.open(DEMO / "depth.png")).astype(np.float64)
    say(f"  depth  {DEMO / 'depth.png'}  shape={d_gt_raw.shape}  "
        f"dtype={np.asarray(Image.open(DEMO / 'depth.png')).dtype}")
    d_gt = d_gt_raw.copy()
    if d_gt.max() > 100.0:            # 毫米
        d_gt /= 1000.0
    valid = np.isfinite(d_gt) & (d_gt > 0)
    say(f"  GT 深度范围 {np.nanmin(d_gt[valid]):.3f} – {np.nanmax(d_gt[valid]):.3f} m"
        f"   有效像素 {int(valid.sum())}/{d_gt.size} ({valid.mean()*100:.1f}%)")
    say(f"  GT 内参  fx={K_gt[0,0]:.1f} fy={K_gt[1,1]:.1f} "
        f"cx={K_gt[0,2]:.1f} cy={K_gt[1,2]:.1f}   HFoV={hfov_deg(K_gt[0,0], W):.1f}°")

    t0 = time.time()
    model = UniDepthV2.from_pretrained(repo).to(device).eval()
    load_s = time.time() - t0
    rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(device)

    # ---- 构造 P_gt（不含任何模型输出）---------------------------------------
    hr("A.  参照物 P_gt 的构造（全部来自已知量）")
    xg, yg = unproject(K_gt, H, W)
    with np.errstate(invalid="ignore"):
        P_gt = np.stack([xg * d_gt, yg * d_gt, d_gt], axis=0)
    say("  P_gt = unproject(像素网格, GT内参) × GT深度")
    say("  -> 它不含任何模型预测，因此可以当作逐像素的真值")
    say(f"  场景横向跨度（P_gt，有效像素的中位数两侧各 1%）"
        f"：{np.percentile(P_gt[0][valid], 1):.2f} – "
        f"{np.percentile(P_gt[0][valid], 99):.2f} m")

    # ---- 两条路径 -----------------------------------------------------------
    # 先做一次丢弃的预热。**这一步不是可选的**：第一次 `infer` 会带 cuDNN
    # autotune / kernel 选择，实测 476–560 ms，之后就落到几十 ms。
    # 不预热的话「第二条路径」总是显得比第一条快 5–7 倍，会被误读成
    # 「传 camera 让模型变快了」—— 一个纯粹由调用顺序造出来的假结论。
    def sync() -> None:
        """CUDA 是异步的：不同步的话测到的是核函数启动耗时，不是计算耗时。

        实测差异是量级级的 —— 不同步会读出 28 ms（第一次跑出来的数字），
        而 Phase 0 用同步方式测同一个模型是 789–1120 ms。
        """
        if device.type == "cuda":
            torch.cuda.synchronize()

    def timed(fn):
        sync()
        t0 = time.time()
        out = fn()
        sync()
        return out, (time.time() - t0) * 1000

    with torch.no_grad():
        _ = model.infer(rgb)                       # 丢弃的预热

    with torch.no_grad():
        out_pred, ms_pred = timed(lambda: model.infer(rgb))
        out_gt, ms_gt = timed(lambda: model.infer(rgb, camera=Pinhole(
            K=torch.tensor(K_gt, dtype=torch.float32, device=device).unsqueeze(0))))

    Pa = as_chw(out_pred["points"])
    Pb = as_chw(out_gt["points"])
    Ka = out_pred["intrinsics"].float().cpu().numpy().reshape(3, 3)
    Kb = out_gt["intrinsics"].float().cpu().numpy().reshape(3, 3)

    hr("B.  两条路径的逐像素三维误差（参照物 = P_gt）")
    say(f"  A 路径 infer(rgb)                        -> 模型回传 K 的 fx={Ka[0,0]:.1f}"
        f"  (HFoV {hfov_deg(Ka[0,0], W):.1f}°)   {ms_pred:.0f} ms")
    say(f"  B 路径 infer(rgb, camera=Pinhole(K_gt))  -> 模型回传 K 的 fx={Kb[0,0]:.1f}"
        f"  (HFoV {hfov_deg(Kb[0,0], W):.1f}°)   {ms_gt:.0f} ms")
    say(f"  （两条路径都在一次丢弃的预热之后、且用 `torch.cuda.synchronize()` "
        f"夹住计时，所以这个 ms 可比）")
    say()
    say("  注意 B 行：**即便把 GT 内参传进去，模型回传的 intrinsics 仍然是它自己")
    say("  那个相机头的输出（fx≈163.7）**。也就是说 camera 参数并没有改写 intrinsics")
    say("  这个键，它走的是另一条路：`unidepthv2.py:361-362` 把 camera 转成 rays")
    say("  喂进 decoder 当条件。所以「回传的 K」完全不能用来判断传进去的 K 有没有生效")
    say("  —— 只能看几何误差。这正是本节要做的。")

    m_a = metrics(Pa, P_gt, d_gt, valid)
    m_b = metrics(Pb, P_gt, d_gt, valid)

    def row(name: str, m: dict) -> str:
        a = m["all"]
        return (f"  {name:<22}{a['depth_absrel_mean']*100:>8.1f}%"
                f"{a['depth_delta1_25']*100:>9.1f}%"
                f"{a['depth_rmse_m']:>11.3f}"
                f"{a['err3d_median_m']:>12.3f}"
                f"{a['err3d_p90_m']:>12.3f}"
                f"{a['err3d_rel_median']*100:>11.1f}%")

    say()
    say(f"  {'路径':<22}{'深度ARel':>9}{'δ<1.25':>10}{'深度RMSE':>11}"
        f"{'3D误差中位':>12}{'3D误差p90':>12}{'3D相对中位':>12}")
    say(f"  {'-'*22}{'-'*9}{'-'*10}{'-'*11}{'-'*12}{'-'*12}{'-'*12}")
    say(row("A 不给内参", m_a))
    say(row("B 给 GT 内参", m_b))

    hr("C.  分区：误差是否随视场角增大？（验证「方向场错了」这个诊断）")
    say(f"  {'路径':<14}{'区域':<12}{'像素数':>9}{'深度ARel':>10}"
        f"{'3D误差中位':>12}{'3D误差p90':>12}")
    say(f"  {'-'*14}{'-'*12}{'-'*9}{'-'*10}{'-'*12}{'-'*12}")
    for label, m in (("A 不给内参", m_a), ("B 给 GT 内参", m_b)):
        for zname, zlabel in (("inner_r50", "中心 r<0.5"),
                              ("outer_r50", "外围 r>0.5")):
            z = m[zname]
            say(f"  {label:<14}{zlabel:<12}{z['n_px']:>9}"
                f"{z['depth_absrel_mean']*100:>9.1f}%"
                f"{z['err3d_median_m']:>12.3f}{z['err3d_p90_m']:>12.3f}")
        ia, oa = m["inner_r50"], m["outer_r50"]
        ratio = (oa["err3d_median_m"] / ia["err3d_median_m"]) if ia["err3d_median_m"] > 0 else float("nan")
        say(f"  {'':<14}{'外围/中心':<12}{'':>9}{'':>10}{ratio:>11.2f}×")
        say()

    say("  怎么读：如果 A 路径的「外围/中心」比值明显大于 B 路径，那就说明 A 的误差")
    say("  主要是**方向场**在外围偏离真值造成的 —— 也就是内参错误在几何上的签名，")
    say("  而不是「模型整体不准」。B 路径把这个比值压回接近 1，说明给对内参是有效的。")

    # ---- 横向尺度的直接读数 ---------------------------------------------------
    hr("D.  横向尺度：一个独立于误差统计的直读")
    infl = float(K_gt[0, 0] / Ka[0, 0])
    say(f"  A 路径内参 fx = {Ka[0,0]:.1f}  →  HFoV {hfov_deg(Ka[0,0], W):.1f}°")
    say(f"  GT     内参 fx = {K_gt[0,0]:.1f}  →  HFoV {hfov_deg(K_gt[0,0], W):.1f}°")
    say(f"  横向放大倍数 = GT_fx / A_fx = {infl:.3f}")
    say()
    say(f"  A 路径点云的 X 跨度（1–99 百分位）"
        f"：{np.percentile(Pa[0][valid],1):.2f} – {np.percentile(Pa[0][valid],99):.2f} m")
    say(f"  GT      点云的 X 跨度（1–99 百分位）"
        f"：{np.percentile(P_gt[0][valid],1):.2f} – {np.percentile(P_gt[0][valid],99):.2f} m")
    say(f"  B 路径点云的 X 跨度（1–99 百分位）"
        f"：{np.percentile(Pb[0][valid],1):.2f} – {np.percentile(Pb[0][valid],99):.2f} m")

    plausible = PLAUSIBLE_HFOV[0] <= hfov_deg(Ka[0, 0], W) <= PLAUSIBLE_HFOV[1]

    hr("VERDICT")
    say(f"  A 路径（不给内参）：深度 ARel {m_a['all']['depth_absrel_mean']*100:.1f}%，"
        f"三维误差中位 {m_a['all']['err3d_median_m']*100:.0f} cm，"
        f"δ<1.25 {m_a['all']['depth_delta1_25']*100:.1f}%")
    say(f"  B 路径（给 GT 内参）：深度 ARel {m_b['all']['depth_absrel_mean']*100:.1f}%，"
        f"三维误差中位 {m_b['all']['err3d_median_m']*100:.0f} cm，"
        f"δ<1.25 {m_b['all']['depth_delta1_25']*100:.1f}%")
    say()
    if m_b["all"]["err3d_median_m"] > 0:
        say(f"  给对相机后，三维误差中位数降到 A 路径的 "
            f"{m_b['all']['err3d_median_m']/m_a['all']['err3d_median_m']*100:.1f}%")
    say(f"  A 路径预测视场 {hfov_deg(Ka[0,0], W):.1f}° "
        f"{'在' if plausible else '不在'}可信区间 "
        f"[{PLAUSIBLE_HFOV[0]:.0f}°, {PLAUSIBLE_HFOV[1]:.0f}°]")
    say()
    say("  ⟹ 三条要进方案文档的结论：")
    say("     ① 「内参来源」是独立于模型选型的精度杠杆，量级远超换模型带来的差别。")
    say("     ② 有已知内参就该传（EXIF / 标定 / Omni3D-Bench 自带 GT 相机）；")
    say("        没有则必须做视场合理性检查并在报告里标注，不能让数字悄悄流到下游。")
    say("     ③ 模型回传的 `intrinsics` 不能用来判断 camera 是否生效 ——")
    say("        它是独立预测头，即便 camera 生效也照样回错值（B 行 fx 仍是 163.7）。")
    say("        所以 `DepthField.intrinsics` 在传入已知内参时必须记**实际使用的那一份**。")

    RESULT.update({
        "gt": {"fx": float(K_gt[0, 0]), "fy": float(K_gt[1, 1]),
               "cx": float(K_gt[0, 2]), "cy": float(K_gt[1, 2]),
               "hfov_deg": hfov_deg(K_gt[0, 0], W)},
        "depth_gt_valid_px": int(valid.sum()),
        "depth_gt_range_m": [round(float(np.nanmin(d_gt[valid])), 4),
                             round(float(np.nanmax(d_gt[valid])), 4)],
        "returned_K": {
            "path_a_fx": float(Ka[0, 0]), "path_a_hfov_deg": hfov_deg(Ka[0, 0], W),
            "path_b_fx": float(Kb[0, 0]), "path_b_hfov_deg": hfov_deg(Kb[0, 0], W),
        },
        "metrics": {"path_a_no_camera": m_a, "path_b_gt_camera": m_b},
        "lateral_inflation": infl,
        "plausible_hfov": list(PLAUSIBLE_HFOV),
        "path_a_hfov_plausible": bool(plausible),
        "timings_ms": {"path_a": round(ms_pred, 1), "path_b": round(ms_gt, 1)},
        "load_s": round(load_s, 2),
    })

    (HERE / "probe_depth_gt_result.json").write_text(
        json.dumps(RESULT, indent=2, ensure_ascii=False), encoding="utf-8")
    (HERE / "probe_depth_gt_report.txt").write_text("\n".join(LINES) + "\n",
                                                    encoding="utf-8")
    say()
    say(f"  结果写入 {HERE / 'probe_depth_gt_result.json'}")
    say(f"  报告写入 {HERE / 'probe_depth_gt_report.txt'}")
    say()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
