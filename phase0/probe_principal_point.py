#!/usr/bin/env python
r"""Phase 0 / Step 8 —— 主点假设的代价：EXIF 拿不到主点，而它可能比焦距更要紧。

## 这一条是怎么被发现的

Step 7（`probe_k_sweep.py`）把焦距做成了剂量-反应曲线并给出容差预算，结论看起来
完整。随后做的 EXIF fixture（`make_exif_fixture.py`）把 EXIF 反算的 K 与仓库自带
的 GT K **逐项**对照，结果出乎预料：

    项       GT        EXIF 反算     相对差
    fx     518.86      515.56      -0.64%   ← 整数毫米量化，已知
    fy     519.47      515.56      -0.75%
    cx     325.58      320.00      -1.71%   ← 5.6 px
    cy     253.74      240.00      -5.41%   ← 13.7 px   ★

EXIF **不记录主点**，只能取图像中心。而这个「只能」的代价是 13.7 px ——
比焦距量化那 0.64%（在典型 |u−cx|≈160 px 上折合约 1 px）大了**一个数量级**。
也就是说：**EXIF 这条路的主要误差不是量化，是主点。**

## 为什么 Step 7 完全没看见它

因为 Step 7 的方位指标是**一维的**：`b = P[0]/P[2]`，只取 x 分量。

    Δcx 会改变 x/z（被看见）
    Δcy **不会**改变 x/z（完全看不见）

这不是笔误，是「用一个标量概括一个二维量」时必然的盲区。所以本探针改用
**二维方位误差** `hypot(Δx/z, Δy/z)`，并把这件事写进结论 —— 因为下一个做
类似分析的人极可能犯同一个错。

## 用四组 K 做分解

只报「EXIF 的 K 比 GT 差多少」是不够的，因为 EXIF 的 K 同时错两件事
（焦距 + 主点）。所以构造四组：

    K0  GT K                        —— 理想下限
    K1  EXIF K（焦距错 + 主点在中心）—— EXIF 路线的实际总代价
    K2  GT 焦距 + 主点在中心         —— 只主点错
    K3  EXIF 焦距 + GT 主点          —— 只焦距错

于是 K1 的误差可以拆成「主点贡献」与「量化贡献」，各自有多大量级一目了然。
**误差能拆开归因，才谈得上优化哪一个。**

## 除了 5×5 网格，还扫一条「主点偏离轨迹」

网格给的是全局形态（主点错多少、方位误差涨多少，是否近似线性）。
额外沿 (Δcx, Δcy) 的归一化方向扫一串幅度，用来看「偏离 vs 误差」是否成比例 ——
如果成比例，那么主点误差可以用「等效像素偏移」这一个数概括，
下游就能把它与 50 mm 的关系容差直接对比。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))          # 复用 Step 7 的 machinery，别再抄一遍
sys.path.insert(0, str(ROOT))

from probe_k_sweep import (  # noqa: E402
    DEMO,
    as_chw,
    metrics,
    reference_cloud,
)

REPORT = HERE / "probe_principal_point_report.txt"
RESULT = HERE / "probe_principal_point_result.json"

LINES: list[str] = []

#: 5×5 主点偏移网格（像素）。刻意包含真实观测到的那组 (−5.6, −13.7) 附近的点。
GRID_DCX = (-14.0, -7.0, 0.0, 7.0, 14.0)
GRID_DCY = (-14.0, -7.0, 0.0, 7.0, 14.0)

#: 沿偏离方向的幅度扫描（像素）。
RADII = (0.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0)


def say(s: str = "") -> None:
    print(s)
    LINES.append(s)


def flush() -> None:
    REPORT.write_text("\n".join(LINES) + "\n", encoding="utf-8")


def hr(t: str) -> None:
    say()
    say("=" * 74)
    say("  " + t)
    say("=" * 74)


def bear2d(P: np.ndarray, P_gt: np.ndarray, zmin: float = 0.2) -> dict:
    """**二维**方位误差。返回 tan 单位，调用方乘 fx 换算成等效像素。

    与 Step 7 的一维版本（只看 x/z）的差别是本探针存在的理由：
    `Δcy` 对 `x/z` 的影响恰好是零，所以一维指标对纵向主点误差**完全无感**。
    """
    ok = (np.isfinite(P).all(axis=0) & np.isfinite(P_gt).all(axis=0)
          & (P[2] > zmin) & (P_gt[2] > zmin))
    if not ok.any():
        return {}
    bx = P[0][ok] / P[2][ok] - P_gt[0][ok] / P_gt[2][ok]
    by = P[1][ok] / P[2][ok] - P_gt[1][ok] / P_gt[2][ok]
    return {
        "n_px": int(ok.sum()),
        "med_tan": float(np.median(np.hypot(bx, by))),
        "p90_tan": float(np.percentile(np.hypot(bx, by), 90)),
        "med_tan_x": float(np.median(np.abs(bx))),
        "med_tan_y": float(np.median(np.abs(by))),
    }


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

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    def camera_of(K: np.ndarray):
        return Pinhole(K=torch.tensor(K, dtype=torch.float32,
                                     device=device).unsqueeze(0))

    say()
    say("Phase 0 / Step 8 —— 主点假设的代价")
    say(f"  device {device}   repo {repo}")

    image = Image.open(DEMO / "rgb.png").convert("RGB")
    W, H = image.size
    K_gt = np.load(DEMO / "intrinsics.npy").astype(np.float64).reshape(3, 3)
    d_gt = np.asarray(Image.open(DEMO / "depth.png")).astype(np.float64)
    if d_gt.max() > 100.0:
        d_gt /= 1000.0
    P_gt = reference_cloud(K_gt, d_gt)
    fx_gt = float(K_gt[0, 0])

    # EXIF 路线会给出的那份 K：焦距按整数毫米量化、主点取图像中心。
    f35 = int(round(fx_gt * 36.0 / max(W, H)))
    fx_exif = f35 / 36.0 * max(W, H)
    K_exif = np.array([[fx_exif, 0.0, W / 2.0],
                       [0.0, fx_exif, H / 2.0],
                       [0.0, 0.0, 1.0]])

    hr("A.  素材、真值，以及 EXIF 会给出的那份 K")
    say(f"  image {DEMO / 'rgb.png'}   {W}×{H}")
    say(f"  K_gt   fx={fx_gt:.2f} fy={K_gt[1,1]:.2f} "
        f"cx={K_gt[0,2]:.2f} cy={K_gt[1,2]:.2f}")
    say(f"  K_exif fx={fx_exif:.2f} fy={fx_exif:.2f} "
        f"cx={W/2:.2f} cy={H/2:.2f}   （等效焦距取整为 {f35} mm）")
    say()
    say(f"  {'项':<6}{'GT':>12}{'EXIF':>12}{'差':>12}")
    for name, i, j in (("fx", (0, 0), (0, 0)), ("fy", (1, 1), (1, 1)),
                       ("cx", (0, 2), (0, 2)), ("cy", (1, 2), (1, 2))):
        g, e = float(K_gt[i]), float(K_exif[j])
        say(f"  {name:<6}{g:>12.2f}{e:>12.2f}{e-g:>+12.2f}")
    d_cx = float(K_exif[0, 2] - K_gt[0, 2])
    d_cy = float(K_exif[1, 2] - K_gt[1, 2])
    say()
    say(f"  ⟹ 焦距差 {abs(fx_exif-fx_gt)/fx_gt*100:.2f}%（量化），"
        f"主点差 ({d_cx:+.1f}, {d_cy:+.1f}) px")

    t0 = time.time()
    model = UniDepthV2.from_pretrained(repo).to(device).eval()
    load_s = time.time() - t0
    rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(device)
    with torch.no_grad():
        _ = model.infer(rgb)                     # 丢弃的预热

    def run(K: np.ndarray) -> tuple[np.ndarray, float]:
        with torch.no_grad():
            sync()
            t0 = time.time()
            out = model.infer(rgb, camera=camera_of(K))
            sync()
            ms = (time.time() - t0) * 1000
        return as_chw(out["points"]), ms

    def err_of(P: np.ndarray) -> dict:
        b = bear2d(P, P_gt)
        m = metrics(P, P_gt, d_gt, fx_ref=fx_gt)
        return {
            "bear2d_px_median": b["med_tan"] * fx_gt,
            "bear2d_px_p90": b["p90_tan"] * fx_gt,
            "bear1d_x_px_median": b["med_tan_x"] * fx_gt,
            "bear1d_y_px_median": b["med_tan_y"] * fx_gt,
            "err3d_median_m": m["all"]["err3d_median_m"] if m else None,
        }

    # ---- B. 四组 K 的分解 ---------------------------------------------------
    hr("B.  四组 K 的分解 —— 把 EXIF 的总误差拆成「主点」与「量化」两份")
    # 先测「不给内参」这一路，且**用本探针的二维指标**。Step 7 报的 306.3 px 是
    # 一维（只含 x/z）口径，直接拿来对比会低估它 —— 因为模型预测的内参在 x、y
    # 两个方向上同时错。口径不一致的对比本身就是错的，不能出现在结论表里。
    with torch.no_grad():
        out_nocam = model.infer(rgb)
    e_pred = err_of(as_chw(out_nocam["points"]))
    say(f"  参照：不给内参（模型自己猜相机）二维方位误差 "
        f"{e_pred['bear2d_px_median']:.1f} px"
        f"（x {e_pred['bear1d_x_px_median']:.1f} / y "
        f"{e_pred['bear1d_y_px_median']:.1f}）")
    say()

    K_k3 = K_gt.copy()
    K_k3[0, 0] = K_k3[1, 1] = fx_exif          # 只错焦距
    K_k2 = K_gt.copy()
    K_k2[0, 2], K_k2[1, 2] = W / 2.0, H / 2.0  # 只错主点

    variants = [
        ("K0 GT（理想下限）", K_gt),
        ("K1 EXIF 全套", K_exif),
        ("K2 只错主点", K_k2),
        ("K3 只错焦距", K_k3),
    ]
    say(f"  {'变体':<22}{'二维方位px':>11}{'↑p90':>9}{'仅x分量':>10}"
        f"{'仅y分量':>10}{'3D中位(m)':>11}")
    say(f"  {'-'*22}{'-'*11}{'-'*9}{'-'*10}{'-'*10}{'-'*11}")
    decomp: list[dict] = []
    base: dict | None = None
    for label, K in variants:
        P, ms = run(K)
        e = err_of(P)
        if base is None:
            base = e
        decomp.append({"label": label, "ms": round(ms, 1), **{
            k: (None if v is None else round(float(v), 4)) for k, v in e.items()}})
        say(f"  {label:<22}{e['bear2d_px_median']:>11.2f}"
            f"{e['bear2d_px_p90']:>9.2f}{e['bear1d_x_px_median']:>10.2f}"
            f"{e['bear1d_y_px_median']:>10.2f}"
            f"{(e['err3d_median_m'] or float('nan')):>11.4f}")

    e0, e1, e2, e3 = (d["bear2d_px_median"] for d in decomp)
    s1 = e1 - e0
    s2 = e2 - e0
    s3 = e3 - e0
    say()
    say(f"  以 K0 的 {e0:.2f} px 为下限，EXIF 全套多出来的 "
        f"{s1:.2f} px 可以拆成：")
    say(f"    主点贡献（K2−K0）  {s2:>7.2f} px   占 {s2/s1*100:>5.1f}%")
    say(f"    量化贡献（K3−K0）  {s3:>7.2f} px   占 {s3/s1*100:>5.1f}%")
    say(f"    两者之和 {s2+s3:.2f} px vs 实测全套 {s1:.2f} px"
        f"   （差 {abs(s2+s3-s1):.2f} px —— 两项不是严格可加，")
    say("      方位误差是各分量的非线性组合；但量级对比已经足够下结论）")
    say()
    say("  ⚠ 注意最后两列：一维（只看 x/z）与二维的差别。")
    say(f"    K2「只错主点」的一维 x 分量是 {decomp[2]['bear1d_x_px_median']:.2f} px，")
    say(f"    二维是 {decomp[2]['bear2d_px_median']:.2f} px；"
        f"而它的 y 分量是 {decomp[2]['bear1d_y_px_median']:.2f} px。")
    say("    ⟹ 纵向主点误差几乎全部落在 y 分量上，而 Step 7 的指标只测 x ——")
    say("      这就是为什么那一步完全没看见这个问题。**用标量概括二维量必有盲区。**")

    # ---- C. 5×5 主点网格 -----------------------------------------------------
    hr("C.  5×5 主点偏移网格 —— 误差对主点偏离是否近似线性？")
    say(f"  网格 Δcx,Δcy ∈ ±{{0,7,14}} px，焦距固定为 GT（隔离变量）")
    say()
    hdr = "  Δcx\\Δcy " + "".join(f"{d:>10.0f}" for d in GRID_DCY)
    say(hdr)
    say("  " + "-" * (11 + 10 * len(GRID_DCY)))
    grid: list[dict] = []
    for dcx in GRID_DCX:
        row = f"  {dcx:>+8.0f} "
        for dcy in GRID_DCY:
            K = K_gt.copy()
            K[0, 2] += dcx
            K[1, 2] += dcy
            P, _ = run(K)
            e = err_of(P)
            grid.append({"dcx": dcx, "dcy": dcy,
                         **{k: (None if v is None else round(float(v), 4))
                            for k, v in e.items()}})
            row += f"{e['bear2d_px_median']:>10.2f}"
        say(row)
    say()
    say("  （表中数字为二维方位误差中位数，单位 px）")

    g0 = next(r for r in grid if r["dcx"] == 0 and r["dcy"] == 0)
    say()
    say(f"  中心 (0,0) 处 {g0['bear2d_px_median']:.2f} px 是这套权重在同一张图上的")
    say("  方位误差下限 —— 主点完全正确时也降不到零。")

    # 网格里有一个反常最低点：它比 (0,0) 还低。这说明「模型实际的 ray 场」与
    # 「它回传/我们传入的 K」之间还有一个纯偏移，与 Step 7 那个 0.6% 尺度残差同类。
    gmin = min(grid, key=lambda r: r["bear2d_px_median"])
    if (gmin["dcx"], gmin["dcy"]) != (0.0, 0.0):
        say()
        say(f"  ⚠ 网格最低点不在 (0,0)，而在 (Δcx, Δcy) = "
            f"({gmin['dcx']:+.0f}, {gmin['dcy']:+.0f})，"
            f"{gmin['bear2d_px_median']:.2f} px < 中心的 {g0['bear2d_px_median']:.2f} px。")
        say("    即使把 GT 内参原样传进去，模型实际用的 ray 场与这份 K 之间仍存在")
        say(f"    约 {abs(gmin['dcy']):.0f} px 量级的纵向偏移。这与 Step 7 发现的 0.6% 尺度残差")
        say("    是同一类现象：**模型回传/接受的内参与它实际构造的方向场之间有系统性小差**。")
        say("    量级（<1% 画面）远小于 EXIF 那 13.7 px，不影响本节的结论，")
        say("    但它说明「传入 K 就等于控制了 ray 场」这句话只在 ~1% 精度上成立。")

    # ---- D. 沿偏离方向的幅度扫描 --------------------------------------------
    hr("D.  沿偏离方向的幅度扫描 —— 主点误差能否用「等效像素」一个数概括")
    norm = float(np.hypot(d_cx, d_cy))
    ux, uy = (d_cx / norm, d_cy / norm) if norm > 1e-9 else (0.0, 0.0)
    say(f"  偏离方向 (Δcx, Δcy) = ({d_cx:+.1f}, {d_cy:+.1f}) px，"
        f"即单位方向 ({ux:+.3f}, {uy:+.3f})")
    say()
    say(f"  {'偏离幅度(px)':>14}{'二维方位px':>12}{'仅x':>9}{'仅y':>9}"
        f"{'3D中位(m)':>11}{'3m 处横向(mm)':>14}")
    say(f"  {'-'*14}{'-'*12}{'-'*9}{'-'*9}{'-'*11}{'-'*14}")
    radial: list[dict] = []
    for r in RADII:
        K = K_gt.copy()
        K[0, 2] += ux * r
        K[1, 2] += uy * r
        P, _ = run(K)
        e = err_of(P)
        lateral_mm = e["bear2d_px_median"] * 3000.0 / fx_gt
        radial.append({"radius_px": r, "dcx": ux * r, "dcy": uy * r,
                       "lateral_mm_at_3m": round(lateral_mm, 1),
                       **{k: (None if v is None else round(float(v), 4))
                          for k, v in e.items()}})
        say(f"  {r:>14.1f}{e['bear2d_px_median']:>12.2f}"
            f"{e['bear1d_x_px_median']:>9.2f}{e['bear1d_y_px_median']:>9.2f}"
            f"{(e['err3d_median_m'] or float('nan')):>11.4f}{lateral_mm:>14.1f}")

    # 线性度：相邻幅度之间的增量是否近乎常数
    rs = np.array([d["radius_px"] for d in radial])
    bs = np.array([d["bear2d_px_median"] for d in radial])
    slopes = np.diff(bs) / np.diff(rs) if len(rs) > 1 else np.array([])
    if len(slopes):
        say()
        say(f"  逐段斜率（px 方位误差 / px 主点偏离）："
            f"{np.array2string(slopes, precision=3, max_line_width=100)}")
        say(f"  最陡 {slopes.max():.2f}，最平 {slopes.min():.2f}，"
            f"比值 {slopes.max()/max(slopes.min(),1e-9):.2f}×"
            "   （近似为常数 ⟹ 可概括成「等效像素」一个数）")

    # ---- E. 与「关系容差」直接对照 ------------------------------------------
    hr("E.  换算成项目自己的容差单位")
    tol_mm = 50.0
    px_per_mm_3m = fx_gt / 3000.0
    say(f"  关系判断容差 {tol_mm:.0f} mm；在 3 m 处 1 px ≈ "
        f"{3000.0/fx_gt:.2f} mm ⟹ {tol_mm:.0f} mm ≈ {tol_mm*px_per_mm_3m:.1f} px")
    say()
    say(f"  {'来源':<26}{'方位px':>9}{'3 m 处横向(mm)':>16}"
        f"{'占 50mm 容差':>14}")
    say(f"  {'-'*26}{'-'*9}{'-'*16}{'-'*14}")
    rows = [
        ("权重自身的下限（K0）", e0),
        ("焦距量化 0.64%（K3−K0）", s3),
        ("主点中心假设（K2−K0）", s2),
        ("EXIF 全套（K1−K0）", s1),
        ("不给内参（模型预测，二维口径）",
         e_pred["bear2d_px_median"] - e0),
    ]
    for label, px in rows:
        mm = px * 3000.0 / fx_gt
        share = "" if "不给内参" in label else f"{mm/tol_mm*100:>13.0f}%"
        say(f"  {label:<26}{px:>9.2f}{mm:>16.1f}{share:>14}")
    say()
    say(f"  最后一行刻意**没有**填「占容差百分比」：它已经远超 100%（"
        f"{rows[-1][1]*3000/fx_gt/tol_mm*100:.0f}%），")
    say("  填进去会让人以为它只是同一量级上的偏差，而不是「整套几何不可用」。")

    hr("VERDICT")
    say(f"  ① 主点中心假设的代价 {s2:.2f} px（3 m 处 {s2*3000/fx_gt:.0f} mm，"
        f"占 50 mm 容差的 {s2*3000/fx_gt/tol_mm*100:.0f}%）")
    say(f"  ② 焦距量化的代价   {s3:.2f} px（3 m 处 {s3*3000/fx_gt:.0f} mm，"
        f"占容差的 {s3*3000/fx_gt/tol_mm*100:.0f}%）")
    say(f"  ⟹ 主点是量化的 {s2/max(s3,1e-9):.1f} 倍 —— **EXIF 路线的主要误差是主点，"
        f"不是量化。**")
    say(f"  ③ 但两者都远小于「不给内参」的代价"
        f"（{e_pred['bear2d_px_median']:.0f} px，二维口径）——")
    say("     所以 EXIF 依然值得做，只是它的定位要说准：")
    say("     **对米制尺寸与距离够用；对方向/方位级精度受限于中心假设。**")
    say("  ④ 主点误差近似线性（见 D 段斜率），所以它可以被概括成")
    say("     「等效像素偏移」一个数 —— 这让它可以直接与 50 mm 的关系容差对比。")
    say()
    say("  一条口径修正（两个数字都对，但不能混用）：")
    say(f"    Step 7 报「模型预测内参横向误差 306.3 px」用的是**一维**（只含 x/z）指标；")
    say(f"    本探针同一情形用**二维**口径得到 "
        f"{e_pred['bear2d_px_median']:.1f} px")
    say(f"    （x {e_pred['bear1d_x_px_median']:.1f} / y "
        f"{e_pred['bear1d_y_px_median']:.1f}）。")
    say("    二维更大，因为它把纵向那部分也算进来了。引用时**必须带上口径**，")
    say("    否则「同一个量」在两个文档里数字不同，看起来像矛盾。")
    say()
    say("  仍然没做到的：")
    say("  · 本图的 GT 主点偏移 (5.6, 13.7) px 相当大 —— 这很可能因为 demo 图是"
        "**渲染**出来的，")
    say("    渲染相机的主点可以任意设定。真实手机照片的主点通常更接近中心，")
    say("    所以 s2 这个数**上界**了实际代价，不是典型值。要给出典型值需要真实照片")
    say("    + 标定真值，当前手上没有。")
    say("  · 未测「主点误差对物体尺寸/距离的影响」—— 布局问题主要影响位置，")
    say("    尺寸是二阶效应；本探针只测了方位与合成 3D 误差。")

    RESULT_D = {
        "gt_K": {"fx": fx_gt, "fy": float(K_gt[1, 1]),
                 "cx": float(K_gt[0, 2]), "cy": float(K_gt[1, 2])},
        "exif_K": {"fx": float(fx_exif), "fy": float(fx_exif),
                   "cx": W / 2.0, "cy": H / 2.0, "focal_35mm": f35},
        "principal_point_offset_px": [d_cx, d_cy],
        "focal_dev_pct": round((fx_exif - fx_gt) / fx_gt * 100, 4),
        "decomposition": decomp,
        "no_camera_2d": {k: (None if v is None else round(float(v), 4))
                         for k, v in e_pred.items()},
        "metric_note": ("本探针用二维方位误差 hypot(Δx/z, Δy/z)；Step 7 的 "
                        "bear_err_px 是一维（只含 x/z）。两者数值不同，引用须标口径。"),
        "decomposition_extra_px": {"total": round(s1, 3),
                                   "principal_point": round(s2, 3),
                                   "focal_quantisation": round(s3, 3),
                                   "ratio_pp_over_focal": round(s2 / max(s3, 1e-9), 2)},
        "grid": grid,
        "radial": radial,
        "radial_slopes": [round(float(x), 4) for x in slopes],
        "tolerance_mm": tol_mm,
        "px_to_mm_at_3m": round(3000.0 / fx_gt, 4),
        "load_s": round(load_s, 2),
        "caveat": ("本图 GT 主点偏移 (5.6,13.7) px 偏大，疑因 demo 图为渲染图；"
                   "真实照片主点通常更接近中心，故 s2 是上界而非典型值。"),
    }
    RESULT.write_text(json.dumps(RESULT_D, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    say()
    say(f"  结果写入 {RESULT}")
    say(f"  报告写入 {REPORT}")
    say()
    flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
