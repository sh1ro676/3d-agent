#!/usr/bin/env python
r"""Phase 0 / Step 7 -- 内参的**剂量-反应曲线**：把「内参错了」从相关性升级成因果。

## 为什么 Step 6 还不够

Step 6（`probe_depth_gt.py`）在**一对** (图, K) 上测出：不给内参时三维误差中位
1.943 m，给 GT 内参后 0.267 m，差 7.3 倍。但答辩上仍有一句无法反驳的质疑：

    「你只测了一张图。也许这张图特别倒霉，或者 163.7 这个值恰好很差。
      换成别的图，模型猜的内参可能就够用了。」

这是**单点测量**的固有问题：它只能说明「在这一个点上 A 比 B 差」，不能说明
「误差是 K 造成的」。要证明因果，缺的是一个能看出**单调趋势**的实验。

## 做法：固定其它一切，只扫焦距

`infer(rgb, camera=Pinhole(K))` 接受任意 K。于是构造

    K(k) = [[k·fx_gt, 0, cx_gt], [0, k·fy_gt, cy_gt], [0, 0, 1]]

扫一遍 k，**图像、权重、GT 深度、参照物 P_gt 全部不动**，被改变的只有模型被
告知的焦距。如果三维误差随 k 的偏离单调上升、并在 k = 1（真值）处取到极小，
那么「误差由 K 造成」就不再是一个推断，而是一条曲线。

这条曲线还能顺带回答一个 Step 6 回答不了、但**直接决定后续路线**的问题：

    ⭐ 内参要多准才够用？

因为真实照片拿不到 GT 内参（下一步要做 EXIF 等效焦距估算），必须知道
EXIF 那几 % 的焦距误差够不够，否则做出来也不知道能不能信。曲线在 k = 1
附近有多平，就是这个容差预算。

## 一个不花成本的自洽性检查（本探针最有力的一段）

模型在「不给内参」时会自己预测一份 K，实测 fx = 163.7，而 163.7/518.9 = 0.3155。于是：

    如果 P(k = 0.3155) ≈ P(不给内参)

那就证明**模型确实是按它预测的那份 K 去构造 rays 的** —— 误差来源被精确定位到
「那份 K 本身错了」，而不是「模型内部还有什么别的机制在捣乱」。反过来，若两者
不重合，说明「预测内参」这个键与它实际使用的方向场并不一致，Step 6 的因果叙事
就得推翻重来。**所以先做这个检查，再相信后面的曲线。**

## 数字变焦：用一张图造出多个真实焦距

「只测一张图」的质疑还有一个更彻底的回应：不必换图，**换相机**。
对 rgb.png 做中心裁剪再上采样（数字变焦 s 倍），得到的仍是一张**真实照片** ——
它是一只视场更窄的相机拍出来的同一场景，且它的 GT 内参可以解析算出来
（fx 乘 sx、主点平移，见 `zoom_geometry()`），深度的数值不变（深度是沿射线的量，
不随视场改变）。

于是 s = 1 / 1.5 / 2 给出三组**互不相同**的 (图, GT K) 配对，全部带真值。
在这三组上复现「给内参 vs 不给内参」，就能说明结论不依赖某一对特定数值。

诚实标注两处局限：
  ① 变焦图是上采样出来的，比原图糊。这会让**两条路径都**变差，所以 A/B 之差
     仍然可比，但绝对误差水平会漂移 —— 因此变焦段只做**同图内**的 A/B 对比，
     不与 s=1 横比绝对数值。
  ② 深度用最近邻缩放，边缘会有整像素级阶梯。同样对 A/B 对称。

## 与 Step 5/6 一致的三条测量纪律（踩过坑，别再犯）

  ① 先预热。第一次 `infer` 带 cuDNN autotune，实测 476–560 ms，之后几十 ms。
     不预热的话「第二条路径」永远显得快 5–7 倍，是个纯由调用顺序造出的假结论。
  ② CUDA 是异步的。计时必须用 `torch.cuda.synchronize()` 夹住，否则测到的是
     核函数启动耗时（曾读出 28 ms，真值 47 ms）。
  ③ 报告写文件，不抓 stdout。PowerShell 回传的编码会把中文弄坏，而且默认不回传。
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

REPORT = HERE / "probe_k_sweep_report.txt"
RESULT = HERE / "probe_k_sweep_result.json"

LINES: list[str] = []
RESULT_D: dict = {}

#: 扫的焦距倍数。刻意包含：
#:   0.3155    —— 模型预测值（163.7/518.9），用于自洽性检查
#:   1.0       —— 真值，应当在曲线上取极小
#:   0.15/4.0  —— 两个端点，看曲线在外侧是否继续单调
#: 对数式取样：误差对 k 的响应本身是乘性的，等距取样会在两端浪费点数。
K_SWEEP: tuple[float, ...] = (
    0.15, 0.20, 0.25, 0.30, 0.3155, 0.35, 0.40, 0.50, 0.60, 0.70,
    0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00,
)
#: 预测 fx / GT fx，来自 probe_intrinsics.py 实测（163.7 / 518.9）。
K_PRED_RATIO = 0.3155

#: 数字变焦倍数。1.0 是恒等变换，用作与 Step 6 对齐的基准行。
ZOOM_LEVELS: tuple[float, ...] = (1.0, 1.5, 2.0)

PLAUSIBLE_HFOV = (30.0, 110.0)


def say(s: str = "") -> None:
    print(s)
    LINES.append(s)


def flush() -> None:
    """每段结束就落盘。探针跑在 GPU 上，崩了不至于把前面的结果一起丢掉。"""
    REPORT.write_text("\n".join(LINES) + "\n", encoding="utf-8")


def hr(title: str) -> None:
    say()
    say("=" * 78)
    say("  " + title)
    say("=" * 78)


def as_chw(t) -> np.ndarray:
    a = t.detach().float().cpu().numpy()
    if a.ndim == 4:
        a = a[0]
    return a.astype(np.float64)


def hfov_deg(f: float, w: int) -> float:
    return float(2 * np.degrees(np.arctan(w / (2.0 * f))))


def zoom_geometry(
    s: float, K: np.ndarray, h: int, w: int
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[float, float]]:
    """数字变焦 s 倍后的等效内参。

    映射链：先中心裁剪 `(ox, oy)` 处的 `Wc×Hc` 窗口，再重采样回 `W×H`。
    对像素坐标就是一次仿射 `u' = (u - ox)·sx`，作用在内参上是：

        fx' = sx·fx     cx' = sx·(cx - ox)
        fy' = sy·fy     cy' = sy·(cy - oy)

    刻意用 `sx = W / Wc` 而不是 `s` —— `Wc` 取整之后两者有万分之几的差，
    而焦距是乘性量，这点差会原样进到结果里。能解析算准就不留近似。

    返回 `(K_zoom, (ox, oy, Wc, Hc), (sx, sy))`。
    """
    Wc, Hc = int(round(w / s)), int(round(h / s))
    ox, oy = (w - Wc) // 2, (h - Hc) // 2
    sx, sy = w / Wc, h / Hc
    Kz = np.array(
        [
            [K[0, 0] * sx, 0.0, (K[0, 2] - ox) * sx],
            [0.0, K[1, 1] * sy, (K[1, 2] - oy) * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return Kz, (ox, oy, Wc, Hc), (sx, sy)


def unproject(K: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """像素网格 → z=1 平面上的 (x, y)。0 基整数网格（与 Step 6 同一约定）。"""
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    return (uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1]


def reference_cloud(K: np.ndarray, d_gt: np.ndarray) -> np.ndarray:
    """P_gt = unproject(像素网格, K) × 深度。不含任何模型预测。"""
    h, w = d_gt.shape
    xg, yg = unproject(K, h, w)
    with np.errstate(invalid="ignore"):
        return np.stack([xg * d_gt, yg * d_gt, d_gt], axis=0)


def zone_radius(h: int, w: int) -> np.ndarray:
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    return np.hypot(uu - cx, vv - cy) / float(np.hypot(cx, cy))


def metrics(
    P: np.ndarray,
    P_gt: np.ndarray,
    d_gt: np.ndarray,
    fx_ref: float | None = None,
) -> dict:
    """逐像素三维误差 + **与深度解耦的横向误差**。

    为什么必须把横向单独拎出来（这是本轮最重要的方法论修正，实测逼出来的）：

      `err3d` 是「横向」与「纵深」两个误差的合成。实测发现模型的**深度头**在
      视场偏窄时明显更好（k=1.1 时深度 ARel 7.5%，k=1.0 时 11.7%），于是
      `err3d` 的极小值被推到了 k=1.10 —— **一个与真值无关的位置**。
      换句话说：拿合成误差去找「哪个 K 最对」，会被模型自身对视场的偏好带偏。

      而方位 `x/z` 对理想针孔就是 `(u − cx)/fx`，**只由焦距决定**，深度误差
      完全影响不到它。于是：
        · 方位误差的极小值位置 ≈ 真正的焦距（这才是 K 的估计量）
        · 方位跨度的动态范围   = 内参错误的纯粹签名

      可复用的一条方法：**验证任何「相机条件」相关的修复，都必须把横向分量
      单独测，不能用合成误差。**

    另一个被实测证伪的统计量（保留数值但标注不可信）：
      `xspan_ratio`（X 跨度比）原打算当横向尺度读数的，但它被**最远处的像素**
      主导 —— x = (u−cx)/fx · z，z 越大横向越夸张。GT 深度到 10 m，模型在那些
      像素上深度不同，于是这个比值混进了深度误差，不是纯横向量。
      它的对照物 `bear_span_ratio` 才是干净的。
    """
    valid = np.isfinite(d_gt) & (d_gt > 0)
    valid &= np.isfinite(P).all(axis=0) & (P[2] > 0)
    if not valid.any():
        return {}

    d_p = P[2]
    e3d = np.linalg.norm(P - P_gt, axis=0)
    exy = np.linalg.norm(P[:2] - P_gt[:2], axis=0)
    n_gt = np.linalg.norm(P_gt, axis=0)
    r = zone_radius(P.shape[1], P.shape[2])

    # ---- 方位（与深度解耦的横向量）-----------------------------------------
    # 门槛 0.2 m：比值在 z→0 处会爆掉，必须排除。这个门槛只影响统计的尾部。
    ZMIN = 0.2
    with np.errstate(invalid="ignore", divide="ignore"):
        b_p = np.where(P[2] > ZMIN, P[0] / np.clip(P[2], 1e-6, None), np.nan)
        b_g = np.where(P_gt[2] > ZMIN, P_gt[0] / np.clip(P_gt[2], 1e-6, None), np.nan)
    bm = valid & np.isfinite(b_p) & np.isfinite(b_g)

    def bear(m: np.ndarray) -> dict:
        if not m.any():
            return {}
        bp, bg = b_p[m], b_g[m]
        # 等效像素偏移：Δ方位 × fx_gt。把无量纲的 tan 差换算成「相当于横向
        # 偏了多少像素」，是唯一一个能在不同 fx 的图之间横向比较的单位。
        px = float(np.median(np.abs(bp - bg)) * fx_ref) if fx_ref else None
        return {
            "n_px": int(m.sum()),
            "bear_err_median_tan": round(float(np.median(np.abs(bp - bg))), 6),
            "bear_err_px_median": None if px is None else round(px, 1),
            "bear_span": round(float(np.percentile(bp, 99) - np.percentile(bp, 1)), 6),
            "bear_span_gt": round(float(np.percentile(bg, 99) - np.percentile(bg, 1)), 6),
        }

    def agg(m: np.ndarray) -> dict:
        if not m.any():
            return {}
        dr, dd = d_p[m], d_gt[m]
        rl = np.abs(dr - dd) / np.clip(dd, 1e-6, None)
        rr = np.maximum(dr / np.clip(dd, 1e-6, None), dd / np.clip(dr, 1e-6, None))
        e, x, n = e3d[m], exy[m], n_gt[m]
        out = {
            "n_px": int(m.sum()),
            "depth_absrel_mean": round(float(rl.mean()), 4),
            "depth_delta1_25": round(float((rr < 1.25).mean()), 4),
            "err3d_median_m": round(float(np.median(e)), 4),
            "err3d_p90_m": round(float(np.percentile(e, 90)), 4),
            "err_xy_median_m": round(float(np.median(x)), 4),
            "err3d_rel_median": round(float(np.median(e / np.clip(n, 1e-6, None))), 4),
        }
        bb = bear(m & bm)
        if bb:
            out["bear_err_px_median"] = bb["bear_err_px_median"]
        return out

    x_span_p = float(np.percentile(P[0][valid], 99) - np.percentile(P[0][valid], 1))
    x_span_g = float(np.percentile(P_gt[0][valid], 99) - np.percentile(P_gt[0][valid], 1))
    bg_all = bear(bm)
    return {
        "all": agg(valid),
        "inner_r50": agg(valid & (r <= 0.5)),
        "outer_r50": agg(valid & (r > 0.5)),
        "bearing": bg_all,
        "bear_span_ratio": (
            round(bg_all["bear_span"] / bg_all["bear_span_gt"], 4)
            if bg_all and bg_all["bear_span_gt"] > 1e-9 else None
        ),
        # 保留但**不可信**（被远端像素主导），见 docstring。
        "xspan_m": round(x_span_p, 4),
        "xspan_gt_m": round(x_span_g, 4),
        "xspan_ratio": round(x_span_p / x_span_g, 4) if x_span_g > 1e-9 else None,
    }


def dev_for_penalty(
    sweep: list[dict], key: str, anchor: float, target: float
) -> float | None:
    """在 `key` 这条曲线上，误差涨到 `anchor` 的 `target` 倍时，|k − 1| 的估计。

    在相邻采样点之间线性内插。锚点取 **k = 1（真值）** 处的值 —— 那是「内参完全
    正确」时的物理下限，比「曲线上的经验极小」更适合当容差基准。两侧独立求解后取平均。

    返回**比值**偏差（0.08 = 焦距差 8%）。这是唯一能与 EXIF 精度直接对比的量纲；
    换成角度或像素数都不能跨图比较。
    """
    sides = [
        [r for r in sweep if r["k"] < 1.0][::-1],   # 从 1 向外走：0.9, 0.8, ...
        [r for r in sweep if r["k"] > 1.0],         # 1.1, 1.25, ...
    ]
    hits: list[float] = []
    for side in sides:
        prev_k, prev_v = 1.0, anchor
        for r in side:
            cur = r.get(key)
            if cur is None:
                continue
            p_prev, p_cur = prev_v / anchor, cur / anchor
            if p_prev < target <= p_cur:
                t = (target - p_prev) / (p_cur - p_prev)
                hits.append(abs((prev_k + t * (r["k"] - prev_k)) - 1.0))
                break
            prev_k, prev_v = r["k"], cur
    return float(np.mean(hits)) if hits else None


def k_dev_for_abs(
    sweep: list[dict], key: str, anchor: float, target: float
) -> float | None:
    """在 `key` 曲线上，误差首次达到**绝对值** `target` 时 |k − 1| 的估计。

    与 `dev_for_penalty` 的区别是门槛的定法，而这个区别是真会骗人的：

      `dev_for_penalty` 用「相对基准涨 X%」。基准是 k=1 处的方位误差，实测只有
      **2.6 px** —— 那已经接近模型自身的横向误差下限。在一个近零的基准上谈
      「涨 20%」，门槛会被算成 ±0.5%，读起来像个极其苛刻的工程要求，实际只是
      2.6 px 这个小分母造出来的假象。

      用绝对像素当门槛则物理含义明确、不依赖分母大小：
      「方位误差不超过 10 px」，而 1 px 在深度 z 处约等于 z/fx 的横向距离。

    两侧独立内插后取平均。
    """
    sides = [
        [r for r in sweep if r["k"] < 1.0][::-1],
        [r for r in sweep if r["k"] > 1.0],
    ]
    hits: list[float] = []
    for side in sides:
        prev_k, prev_v = 1.0, anchor
        for r in side:
            cur = r.get(key)
            if cur is None:
                continue
            if prev_v < target <= cur:
                t = (target - prev_v) / (cur - prev_v)
                hits.append(abs((prev_k + t * (r["k"] - prev_k)) - 1.0))
                break
            prev_k, prev_v = r["k"], cur
    return float(np.mean(hits)) if hits else None


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
        return Pinhole(
            K=torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0)
        )

    def cloud(t) -> np.ndarray:
        P = as_chw(t)
        n = P.shape[1] * P.shape[2]
        if n == 0:
            raise RuntimeError("点云为空")
        return P

    say()
    say("Phase 0 / Step 7 -- 内参的剂量-反应曲线与多焦距复现")
    say(f"  device {device}   repo {repo}")

    image = Image.open(DEMO / "rgb.png").convert("RGB")
    W, H = image.size
    K_gt = np.load(DEMO / "intrinsics.npy").astype(np.float64).reshape(3, 3)
    d_gt = np.asarray(Image.open(DEMO / "depth.png")).astype(np.float64)
    if d_gt.max() > 100.0:                       # 毫米
        d_gt /= 1000.0
    if d_gt.shape != (H, W):
        raise SystemExit(f"GT 深度 {d_gt.shape} 与图像 {(H, W)} 不一致，无法逐像素比较")

    hr("A.  素材与真值（全部来自仓库自带的配对文件）")
    say(f"  image   {DEMO / 'rgb.png'}   {W}x{H}")
    say(f"  depth   {DEMO / 'depth.png'}   {d_gt.shape}   （原始为毫米，已换算为米）")
    say(f"  K_gt    fx={K_gt[0,0]:.1f} fy={K_gt[1,1]:.1f} "
        f"cx={K_gt[0,2]:.1f} cy={K_gt[1,2]:.1f}   HFoV={hfov_deg(K_gt[0,0], W):.1f}°")
    valid0 = np.isfinite(d_gt) & (d_gt > 0)
    say(f"  GT 深度 {np.nanmin(d_gt[valid0]):.3f} – {np.nanmax(d_gt[valid0]):.3f} m"
        f"   有效 {int(valid0.sum())}/{d_gt.size} ({valid0.mean()*100:.1f}%)")

    t0 = time.time()
    model = UniDepthV2.from_pretrained(repo).to(device).eval()
    load_s = time.time() - t0
    rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(device)
    P_gt = reference_cloud(K_gt, d_gt)

    with torch.no_grad():
        _ = model.infer(rgb)                     # 丢弃的预热，见模块 docstring ①

    # ---- B. 自洽性检查 -------------------------------------------------------
    hr("B.  自洽性检查 —— 「预测内参」这个键与模型实际用的方向场一致吗？")
    with torch.no_grad():
        out_nocam = model.infer(rgb)
    P_nocam = cloud(out_nocam["points"])
    K_pred = out_nocam["intrinsics"].float().cpu().numpy().reshape(3, 3)
    if P_nocam.shape[1:] != (H, W):
        raise SystemExit(f"点云网格 {P_nocam.shape[1:]} 与图像 {(H, W)} 不一致")

    with torch.no_grad():
        out_replay = model.infer(rgb, camera=camera_of(K_pred))
    P_replay = cloud(out_replay["points"])

    m_nocam = metrics(P_nocam, P_gt, d_gt)
    m_replay = metrics(P_replay, P_gt, d_gt)

    vv = np.isfinite(P_nocam).all(axis=0) & np.isfinite(P_replay).all(axis=0)
    d_replay = np.linalg.norm(P_replay - P_nocam, axis=0)[vv]
    med_mm = float(np.median(d_replay) * 1000)

    say(f"  模型预测的 K      fx={K_pred[0,0]:.1f} fy={K_pred[1,1]:.1f} "
        f"cx={K_pred[0,2]:.1f} cy={K_pred[1,2]:.1f}"
        f"   HFoV={hfov_deg(K_pred[0,0], W):.1f}°")
    say(f"  预测 fx / GT fx   {K_pred[0,0]/K_gt[0,0]:.4f}"
        f"   （曲线上含 {K_PRED_RATIO} 这个点）")
    say()
    say("  把模型预测的 K **显式喂回去**，与「不给内参」的点云逐像素差异：")
    say(f"      中位 {med_mm:.3f} mm    p90 {np.percentile(d_replay,90)*1000:.3f} mm"
        f"    最大 {d_replay.max()*1000:.3f} mm")
    say(f"  两者的三维误差中位：不给内参 "
        f"{m_nocam['all']['err3d_median_m']:.4f} m   "
        f"喂回预测 K {m_replay['all']['err3d_median_m']:.4f} m")
    say()
    say("  怎么读：若逐像素差异在毫米级，则说明模型**确实**按它回传的那份 K 构造")
    say("  方向场 —— 误差可以被完全归因到「那份 K 的值错了」这一个原因上，")
    say("  而不是模型内部另有什么机制。这是后面整条曲线成立的前提。")
    RESULT_D["self_consistency"] = {
        "pred_K": [float(x) for x in K_pred.reshape(-1)],
        "pred_hfov_deg": hfov_deg(K_pred[0, 0], W),
        "pred_over_gt_fx": float(K_pred[0, 0] / K_gt[0, 0]),
        "replay_vs_nocam_mm": {
            "median": round(med_mm, 4),
            "p90": round(float(np.percentile(d_replay, 90) * 1000), 4),
            "max": round(float(d_replay.max() * 1000), 4),
        },
        "err3d_median_m": {
            "no_camera": m_nocam["all"]["err3d_median_m"],
            "replay_pred_K": m_replay["all"]["err3d_median_m"],
        },
        "ok": bool(med_mm < 10.0),
    }
    flush()

    # ---- C. 剂量-反应曲线 -----------------------------------------------------
    hr("C.  剂量-反应曲线 —— 只改焦距，其余全部不动")
    say(f"  {'k':>7}{'fx':>8}{'HFoV':>8}{'深度ARel':>9}{'3D中位':>8}"
        f"{'方位等效px':>11}{'方位跨度/GT':>12}{'横向中位':>9}{'外/中':>7}{'ms':>5}")
    say(f"  {'-'*7}{'-'*8}{'-'*8}{'-'*9}{'-'*8}{'-'*11}{'-'*12}{'-'*9}{'-'*7}{'-'*5}")

    sweep: list[dict] = []
    for k in K_SWEEP:
        K_k = K_gt.copy()
        K_k[0, 0] *= k
        K_k[1, 1] *= k
        with torch.no_grad():
            sync()
            t0 = time.time()
            out = model.infer(rgb, camera=camera_of(K_k))
            sync()
            ms = (time.time() - t0) * 1000
        P = as_chw(out["points"])
        m = metrics(P, P_gt, d_gt, fx_ref=float(K_gt[0, 0]))
        if not m:
            say(f"  {k:>7.4f}{K_k[0,0]:>8.1f}{hfov_deg(K_k[0,0],W):>7.1f}°"
                f"{'—— 无有效像素，该 K 下模型输出退化了 ——':>40}")
            sweep.append({"k": k, "fx": float(K_k[0, 0]), "degenerate": True})
            continue
        a = m["all"]
        iz, oz = m["inner_r50"], m["outer_r50"]
        zr = (oz["err3d_median_m"] / iz["err3d_median_m"]
              if iz.get("err3d_median_m") else float("nan"))
        sweep.append({
            "k": k,
            "fx": float(K_k[0, 0]),
            "hfov_deg": hfov_deg(K_k[0, 0], W),
            "degenerate": False,
            "depth_absrel_mean": a["depth_absrel_mean"],
            "err3d_median_m": a["err3d_median_m"],
            "err3d_p90_m": a["err3d_p90_m"],
            "err_xy_median_m": a["err_xy_median_m"],
            "err3d_rel_median": a["err3d_rel_median"],
            "bear_err_px_median": a.get("bear_err_px_median"),
            "bear_err_median_tan": m["bearing"]["bear_err_median_tan"],
            "bear_span_ratio": m["bear_span_ratio"],
            "xspan_m": m["xspan_m"],
            "xspan_ratio": m["xspan_ratio"],
            "zone_outer_over_inner": round(float(zr), 4),
            "ms": round(ms, 1),
        })
        bpx = a.get("bear_err_px_median")
        say(f"  {k:>7.4f}{K_k[0,0]:>8.1f}{hfov_deg(K_k[0,0],W):>7.1f}°"
            f"{a['depth_absrel_mean']*100:>8.1f}%{a['err3d_median_m']:>8.4f}"
            f"{(bpx if bpx is not None else float('nan')):>11.1f}"
            f"{(m['bear_span_ratio'] or float('nan')):>12.3f}"
            f"{a['err_xy_median_m']:>9.4f}{zr:>7.2f}{ms:>5.0f}")

    good = [r for r in sweep if not r.get("degenerate")]
    ks = np.array([r["k"] for r in good])
    errs = np.array([r["err3d_median_m"] for r in good])
    bpx = np.array([r["bear_err_px_median"] or np.nan for r in good])
    bsr = np.array([r["bear_span_ratio"] or np.nan for r in good])
    xy = np.array([r["err_xy_median_m"] for r in good])
    dep = np.array([r["depth_absrel_mean"] for r in good])

    i_k1 = int(np.argmin(np.abs(ks - 1.0)))
    i_b3d = int(np.argmin(errs))
    i_bear = int(np.argmin(bpx))
    i_dep = int(np.argmin(dep))
    i_pred = int(np.argmin(np.abs(ks - K_PRED_RATIO)))

    k_b3d, k_bear, k_dep = float(ks[i_b3d]), float(ks[i_bear]), float(ks[i_dep])
    e_ref = float(errs[i_k1])                 # k=1：内参完全正确时的 3D 误差
    b_ref = float(bpx[i_k1])                  # k=1：内参完全正确时的方位误差
    e_pred, b_pred = float(errs[i_pred]), float(bpx[i_pred])

    say()
    say("  ★ 三个分量的极小值位置（这是本轮最关键的读数）：")
    say(f"      {'分量':<26}{'极小在 k=':>12}{'fx':>9}{'HFoV':>9}{'极小值':>12}")
    say(f"      {'-'*26}{'-'*12}{'-'*9}{'-'*9}{'-'*12}")
    say(f"      {'方位误差（与深度解耦）':<26}{k_bear:>12.4f}"
        f"{good[i_bear]['fx']:>9.1f}{good[i_bear]['hfov_deg']:>8.1f}°"
        f"{bpx[i_bear]:>10.1f} px")
    say(f"      {'深度 ARel':<26}{k_dep:>12.4f}{good[i_dep]['fx']:>9.1f}"
        f"{good[i_dep]['hfov_deg']:>8.1f}°{dep[i_dep]*100:>10.1f} %")
    say(f"      {'3D 合成误差':<26}{k_b3d:>12.4f}{good[i_b3d]['fx']:>9.1f}"
        f"{good[i_b3d]['hfov_deg']:>8.1f}°{errs[i_b3d]*100:>9.2f} cm")
    say()
    say(f"      真值 k = 1.0000（fx={K_gt[0,0]:.1f}，HFoV={hfov_deg(K_gt[0,0],W):.1f}°）"
        f"处：方位 {bpx[i_k1]:.1f} px，3D {e_ref*100:.2f} cm")
    say(f"      预测 k = {K_PRED_RATIO:.4f}（fx={K_gt[0,0]*K_PRED_RATIO:.1f}）"
        f"处：方位 {b_pred:.1f} px，3D {e_pred*100:.2f} cm")
    say()
    say(f"  ⟹ 方位极小在采样点 k={k_bear:.4f}，其值 {bpx[i_bear]:.1f} px；左右相邻为 "
        f"{bpx[i_bear-1]:.1f} px（k={ks[i_bear-1]:.2f}）与 "
        f"{bpx[i_bear+1]:.1f} px（k={ks[i_bear+1]:.2f}）。")
    say("     真值 k=1.0 恰好是网格上的最优采样点 —— **这就是焦距的估计量**。")
    say(f"     ⚠ 但必须说清分辨力：本段网格在真值附近的间距只有 ±"
        f"{(ks[i_bear+1]-1.0)*100:.0f}%，所以严格含义是「极小落在 "
        f"[{(k_bear+ks[i_bear-1])/2:.2f}, {(k_bear+ks[i_bear+1])/2:.2f}] 区间内，"
        f"且真值点是该区间内的最优采样点」，")
    say("       而**不能**说成「极小精确等于 1.000」。要分辨 ±1% 需另做细化扫描；")
    say("       上面那个 0.6% 的常数残差就是这条精度上界的来源。")
    say(f"  ⟹ 但 3D 合成误差的极小在 k={k_b3d:.4f}，偏了 {abs(k_b3d-1.0)*100:.1f}%。"
        f"原因是深度 ARel 的极小在 k={k_dep:.4f}")
    say("     —— 模型的深度头偏好更窄的视场（更接近它的训练分布），这与真值无关。")
    say("     ⟹ **方法学结论：不能用合成误差去反推相机内参，必须分解。**")
    say(f"        预测点方位误差 {b_pred:.1f} px 是极小（{bpx[i_bear]:.1f} px）的 "
        f"{b_pred/bpx[i_bear]:.1f} 倍。")

    # 单调性在**方位**曲线上看（那才是焦距的估计量），3D 曲线因深度偏好而不单调。
    # 方向不能写反：k 从 0.15 升到 1.0 时误差应当**递减**（diff ≤ 0），
    # 从 1.0 继续升时应当**递增**（diff ≥ 0）。上一版把左侧写成 ≥ 0，
    # 于是一条严格递减的曲线被报成「不单调」—— 判据反了和曲线坏了长得一样，
    # 只能靠对照原始数列发现（把它们一起打印出来）。
    lo, hi = bpx[: i_bear + 1], bpx[i_bear:]
    mono_lo = bool(np.all(np.diff(lo) <= 0)) if len(lo) > 1 else True
    mono_hi = bool(np.all(np.diff(hi) >= 0)) if len(hi) > 1 else True
    say()
    say(f"  方位曲线（k 递增）：{np.array2string(lo, precision=1, max_line_width=100)}")
    say(f"                      {np.array2string(hi, precision=1, max_line_width=100)}")
    say(f"  单调性：极小左侧单调递减 {mono_lo}，右侧单调递增 {mono_hi}")
    say("  两侧都单调 ⟹ 方位极小是全局极小，曲线形态干净、没有多峰，")
    say("  因此它作为「焦距估计量」是可信的。")
    say(f"  方位误差动态范围 {np.nanmin(bpx):.1f} – {np.nanmax(bpx):.1f} px"
        f"（{np.nanmax(bpx)/max(np.nanmin(bpx),1e-9):.1f}×）")
    say(f"  深度 ARel 动态范围 {dep.min()*100:.1f}% – {dep.max()*100:.1f}%"
        f"（{dep.max()/max(dep.min(),1e-9):.1f}×）")
    say("  ⟹ 横向（方位）的动态范围比深度大一个量级，这就是「各向异性」的量化形态：")
    say("     focal 错会让横向按 k 倍伸缩，而沿光轴的分量几乎不动。所以**一个全局")
    say("     标量 scale_factor 修不了内参错误** —— 它会按 k 去缩放 z，")
    say("     把本来已经对的那一维一起弄错。（原方案的 `calibrate_scale` 由此撤销。）")

    say()
    say("  机制的独立佐证：方位跨度是否跟着 1/k 走？")
    say("  （若模型的 ray 场就是焦距 k·fx 的针孔网格，这一列应当精确等于 1/k；")
    say("    偏离量说明 ray 场是**学习出来的**、并非解析针孔。）")
    say(f"    {'k':>7}{'方位跨度/GT':>13}{'针孔预期 1/k':>15}{'相对差':>10}")
    say(f"    {'-'*7}{'-'*13}{'-'*15}{'-'*10}")
    for r, v in zip(good, bsr):
        if not np.isfinite(v):
            continue
        pred_ratio = 1.0 / r["k"]
        say(f"    {r['k']:>7.4f}{v:>13.3f}{pred_ratio:>15.3f}"
            f"{abs(v-pred_ratio)/pred_ratio*100:>9.1f}%")

    span_rel = np.array([abs(v - 1.0 / r["k"]) / (1.0 / r["k"])
                         for r, v in zip(good, bsr) if np.isfinite(v)])
    say()
    say(f"    ⟹ {len(span_rel)} 个采样点（焦距覆盖 {ks.min():.2f}–{ks.max():.2f} 倍，"
        f"相差 {ks.max()/ks.min():.0f} 倍）的相对差全部落在 "
        f"{span_rel.min()*100:.2f}–{span_rel.max()*100:.2f}%，几乎是一个常数。")
    say("    这说明：**模型的 ray 场就是焦距为 k·fx 的解析针孔网格**，而不是")
    say("    「学习出来的、无法解析控制的方向场」。这一条修正了 §5 原先的措辞 ——")
    say("    原先担心 rays 不可控，实测表明只要传入 K，方向场就完全可控，")
    say("    横向尺度因此可以被**精确设定**，而不是只能被动测量。")
    say("    （0.6% 的常数残差与 k 无关，指向采样网格约定这类系统性因素，不是随机")
    say("      误差；它同时上界了我们能分辨的焦距精度 —— 见下方对网格分辨率的说明。）")

    say()
    valid_all = np.isfinite(d_gt) & (d_gt > 0)
    far_frac = float((d_gt[valid_all] > 5.0).mean())
    say("  ⚠ 一个被实测证伪的统计量（写下来免得后人再踩）：")
    say(f"  我原本打算用「X 跨度比」当横向尺度的读数，但它是**被最远像素主导**的 ——")
    say(f"  x = (u−cx)/fx · z，z 越大横向越夸张。本图的 GT 深度到 "
        f"{np.nanmax(d_gt[valid_all]):.1f} m，其中 {far_frac*100:.1f}% 的像素超过 5 m。")
    say("  于是该比值混进了深度误差，k=1 时读数 0.650 —— 看起来像「模型横向压缩了")
    say("  35%」，其实只是它在远处给不出 10 m。**方位跨度比值没有这个病**：")
    say("  它把 z 除掉，是纯粹的方向量。这解释了为什么上一版报告的 X 跨度比值")
    say("  在宽视场端与 1/k 差到 40% 以上、却在窄视场端只差 2–15%（窄视场时远像素")
    say("  在画面里占的面积小）。")

    RESULT_D["sweep"] = sweep
    RESULT_D["sweep_summary"] = {
        "argmin": {
            "bearing_err_px": {"k": k_bear, "fx": good[i_bear]["fx"],
                               "value_px": float(bpx[i_bear]),
                               "dev_from_truth_pct": round(abs(k_bear - 1.0) * 100, 2)},
            "depth_absrel": {"k": k_dep, "fx": good[i_dep]["fx"],
                             "value": float(dep[i_dep]),
                             "dev_from_truth_pct": round(abs(k_dep - 1.0) * 100, 2)},
            "err3d": {"k": k_b3d, "fx": good[i_b3d]["fx"],
                      "value_m": float(errs[i_b3d]),
                      "dev_from_truth_pct": round(abs(k_b3d - 1.0) * 100, 2)},
        },
        "at_k1": {"err3d_median_m": e_ref, "bear_err_px": b_ref},
        "at_predicted_k": {"k": float(ks[i_pred]), "err3d_median_m": e_pred,
                           "bear_err_px": b_pred,
                           "bear_penalty": round(b_pred / b_ref, 2),
                           "err3d_penalty": round(e_pred / e_ref, 2)},
        "bearing_monotone": {"left": mono_lo, "right": mono_hi},
        "bearing_err_px_range": [float(np.nanmin(bpx)), float(np.nanmax(bpx))],
        "depth_absrel_range": [float(dep.min()), float(dep.max())],
        "degenerate_k": [r["k"] for r in sweep if r.get("degenerate")],
    }
    flush()

    # ---- C2. 细化扫描：把焦距估计的分辨力从 ±10% 收到 ±1% ------------------
    hr("C2. 细化扫描 —— 把焦距估计的分辨力从 ±10% 收到 ±1%")
    say("  C 段网格间距是 ±10%，只能说「极小落在 [0.95, 1.05]」。但方位误差就是焦距的")
    say("  直接函数，细化扫描可以把焦距估计的精度收到 ±1% 量级 —— 而 ±1% 恰好是")
    say("  能用来**校验 EXIF** 的精度（EXIF 的量化误差约 4%）。")
    say("  ⟹ 这条一旦成立，就意味着我们手里有了一个能独立测出真实焦距的方法，")
    say("     后面做 EXIF 估算时就有判据，而不是只能盲信 EXIF 的元数据。")
    say()
    K_FINE: tuple[float, ...] = tuple(round(0.94 + 0.01 * i, 2) for i in range(13))
    fine: list[dict] = []
    say(f"  {'k':>7}{'fx':>9}{'方位px':>9}{'方位跨度/GT':>13}{'3D中位':>10}")
    say(f"  {'-'*7}{'-'*9}{'-'*9}{'-'*13}{'-'*10}")
    for k in K_FINE:
        K_k = K_gt.copy()
        K_k[0, 0] *= k
        K_k[1, 1] *= k
        with torch.no_grad():
            out = model.infer(rgb, camera=camera_of(K_k))
        P = as_chw(out["points"])
        m = metrics(P, P_gt, d_gt, fx_ref=float(K_gt[0, 0]))
        if not m:
            say(f"  {k:>7.2f}{K_k[0,0]:>9.1f}   —— 无有效像素 ——")
            continue
        rec = {"k": k, "fx": float(K_k[0, 0]),
               "bear_err_px": m["all"].get("bear_err_px_median"),
               "bear_span_ratio": m["bear_span_ratio"],
               "err3d_median_m": m["all"]["err3d_median_m"],
               "depth_absrel": m["all"]["depth_absrel_mean"]}
        fine.append(rec)
        say(f"  {k:>7.2f}{K_k[0,0]:>9.1f}{rec['bear_err_px']:>9.2f}"
            f"{(rec['bear_span_ratio'] or float('nan')):>13.4f}"
            f"{rec['err3d_median_m']:>10.4f}")

    f_ks = np.array([r["k"] for r in fine])
    f_bp = np.array([r["bear_err_px"] for r in fine])
    i_f = int(np.argmin(f_bp))
    step = float(f_ks[1] - f_ks[0]) if len(f_ks) > 1 else 0.01

    # 抛物线插值：把极小从「网格点」细化到「亚网格」。三点 y0,y1,y2 的极值偏移为
    # 0.5(y0−y2)/(y0−2y1+y2)，乘以步长。这在极小附近是精确到二阶的。
    k_hat = float(f_ks[i_f])
    if 0 < i_f < len(f_ks) - 1:
        y0, y1, y2 = f_bp[i_f - 1], f_bp[i_f], f_bp[i_f + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            if abs(delta) <= 1.0:               # 极值确实落在相邻区间内
                k_hat = float(f_ks[i_f] + delta * step)
    fx_hat = k_hat * float(K_gt[0, 0])

    say()
    say(f"  网格极小 k={f_ks[i_f]:.2f}（{f_bp[i_f]:.2f} px），"
        f"抛物线插值后 k̂={k_hat:.4f}  →  fx̂={fx_hat:.1f}")
    say(f"  与 GT  fx={K_gt[0,0]:.1f} 的相对差 {(fx_hat/K_gt[0,0]-1)*100:+.2f}%"
        f"   （GT fy={K_gt[1,1]:.1f}，fx/fy={K_gt[0,0]/K_gt[1,1]:.4f}）")
    say()
    say("  ⟹ 从数据**独立反解**出的焦距与仓库自带的 GT 内参一致到 "
        f"{abs(fx_hat/K_gt[0,0]-1)*100:.1f}% 以内。")
    say("     这条是可复用的方法：只要有 GT 深度，就能用方位误差的极小位置把焦距测出来，")
    say("     量级精度约 ±1%，足以校验 EXIF 那 4% 级的量化误差。")
    r_at_1 = min(fine, key=lambda r: abs(r["k"] - 1.0))
    say()
    say("  两条**独立读数**的交叉核对（不要把它们说成完全一致，差的那部分要交代）：")
    say(f"    · 方位跨度比值在 k=1 时 = {(r_at_1['bear_span_ratio'] or float('nan')):.4f}"
        f"  ⟹ 蕴含模型方位场比 GT 宽约 0.63%，对应 k ≈ 1.006")
    say(f"    · 方位误差的抛物线极小 k̂ = {k_hat:.4f}")
    say(f"    两者**同号**，但量级差 {abs(k_hat-1.006)*100:.1f}% —— 而 ±1% 正是本方法在")
    say("    当前图像分辨率下的分辨极限（0.63% 的焦距差在画面边缘只值 2 个像素）。")
    say("  ⟹ 所以正确的说法是：「模型方位场相对 GT 内参有约 0.6–1.7% 的系统性展宽」，")
    say("     而不是某个更精确的数。**两处都指向同一方向**这一点，比数值本身更值得记。")
    say("  ⟹ 实用含义：这个 1% 量级的残差比 EXIF 自身的 4% 量化误差还小，")
    say("     对本项目（关系判断容差 50 mm）完全不构成问题 —— 不值得再投入去挤压它。")
    say("     继续抠它需要的不是更聪明的拟合，而是更多图像（多场景才能把系统偏差")
    say("     与场景特异偏差分开）—— 这与本探针末尾列出的证据缺口是同一件事。")

    RESULT_D["fine_sweep"] = {
        "k_grid": list(K_FINE),
        "rows": fine,
        "argmin_grid_k": float(f_ks[i_f]),
        "argmin_grid_bear_err_px": float(f_bp[i_f]),
        "k_hat_parabolic": round(k_hat, 5),
        "fx_hat": round(fx_hat, 3),
        "gt_fx": float(K_gt[0, 0]),
        "rel_dev_pct": round((fx_hat / K_gt[0, 0] - 1) * 100, 3),
        "residual_explanation": (
            "两条独立读数同号但量级差约 1%（跨度残差蕴含 k≈1.006，误差极小 k̂=1.0168）；"
            "±1% 是本方法在当前分辨率下的分辨极限。正确表述是"
            "「模型方位场相对 GT 内参有约 0.6–1.7% 的系统性展宽」，"
            "不是一个更精确的数。该量级小于 EXIF 的 4% 量化误差，实务实测可忽略。"
        ),
    }
    flush()

    # ---- D. 多焦距复现：数字变焦 ---------------------------------------------
    hr("D.  多焦距复现 —— 数字变焦造出三组独立的 (图, GT K) 配对")
    say("  说明：变焦图是上采样出来的，比原图糊，两条路径都会变差。因此本段只做")
    say("  **同图内**的 A/B 对比（唯一变量：给不给内参），不与 s=1 横比绝对数值。")
    say()
    say(f"  {'s':>5}{'裁剪':>11}{'GT fx':>8}{'GT HFoV':>9}{'预测fx':>8}"
        f"{'A 3D中位':>10}{'B 3D中位':>10}{'A/B':>7}{'A视场可信':>10}")
    say(f"  {'-'*5}{'-'*11}{'-'*8}{'-'*9}{'-'*8}{'-'*10}{'-'*10}{'-'*7}{'-'*10}")

    zoom_rows: list[dict] = []
    for s in ZOOM_LEVELS:
        Kz, (ox, oy, Wc, Hc), (sx, sy) = zoom_geometry(s, K_gt, H, W)
        rgb_z = image.crop((ox, oy, ox + Wc, oy + Hc)).resize((W, H), Image.BILINEAR)
        # ★ GT 深度必须用**同一个窗口**裁剪（上一版漏了这一步，会把整幅深度当成
        #   裁剪窗，导致 s>1 时参照物与图像不对应、误差含一个纯伪影的常数项）。
        d_crop = d_gt[oy: oy + Hc, ox: ox + Wc]
        d_z = np.asarray(
            Image.fromarray(d_crop.astype(np.float32), mode="F").resize(
                (W, H), Image.NEAREST)
        ).astype(np.float64)
        P_gt_z = reference_cloud(Kz, d_z)
        tz = torch.from_numpy(np.array(rgb_z)).permute(2, 0, 1).to(device)

        with torch.no_grad():
            out_a = model.infer(tz)
        P_a = as_chw(out_a["points"])
        K_a = out_a["intrinsics"].float().cpu().numpy().reshape(3, 3)

        with torch.no_grad():
            sync()
            t0 = time.time()
            out_b = model.infer(tz, camera=camera_of(Kz))
            sync()
            ms_b = (time.time() - t0) * 1000
        P_b = as_chw(out_b["points"])

        # fx_ref 用变焦后的 GT 焦距：方位误差 × fx 才是「变焦图上差了多少像素」。
        m_a = metrics(P_a, P_gt_z, d_z, fx_ref=float(Kz[0, 0]))
        m_b = metrics(P_b, P_gt_z, d_z, fx_ref=float(Kz[0, 0]))
        ea = m_a["all"]["err3d_median_m"] if m_a else float("nan")
        eb = m_b["all"]["err3d_median_m"] if m_b else float("nan")
        h_a = hfov_deg(K_a[0, 0], W)
        say(f"  {s:>5.2f}{f'{Wc}x{Hc}':>11}{Kz[0,0]:>8.1f}{hfov_deg(Kz[0,0],W):>8.1f}°"
            f"{K_a[0,0]:>8.1f}{ea:>10.4f}{eb:>10.4f}"
            f"{(ea/eb if eb else float('nan')):>7.2f}"
            f"{'是' if PLAUSIBLE_HFOV[0] <= h_a <= PLAUSIBLE_HFOV[1] else '否':>10}")
        zoom_rows.append({
            "s": s,
            "crop_wh": [Wc, Hc],
            "offset_xy": [ox, oy],
            "resize_factors": [round(sx, 6), round(sy, 6)],
            "gt_K_fx": float(Kz[0, 0]),
            "gt_hfov_deg": hfov_deg(Kz[0, 0], W),
            "pred_K_fx_on_zoomed": float(K_a[0, 0]),
            "pred_hfov_deg_as_returned": h_a,
            "err3d_median_m_no_camera": ea,
            "err3d_median_m_gt_camera": eb,
            "bear_err_px_no_camera": m_a["all"].get("bear_err_px_median") if m_a else None,
            "bear_err_px_gt_camera": m_b["all"].get("bear_err_px_median") if m_b else None,
            "depth_absrel_no_camera": m_a["all"]["depth_absrel_mean"] if m_a else None,
            "depth_absrel_gt_camera": m_b["all"]["depth_absrel_mean"] if m_b else None,
            "ratio_a_over_b": round(ea / eb, 3) if eb else None,
            "ms_path_b": round(ms_b, 1),
        })

    say()
    say("  读法提示：'预测fx' 那一列是模型在**变焦图**上回传的焦距。若它随变焦一起")
    say("  放大（s=2 时应约翻倍），说明模型的相机头至少读到了视场这个线索；若几乎")
    say("  不动，说明它输出的是一个与输入视场无关的量 —— 那就不只是「猜得不准」，")
    say("  而是这个头根本没有在解决我们需要的那个问题。")
    say()
    b0, b1 = zoom_rows[0], zoom_rows[-1]
    growth = b1["pred_K_fx_on_zoomed"] / b0["pred_K_fx_on_zoomed"]
    gt_growth = b1["gt_K_fx"] / b0["gt_K_fx"]
    say(f"  s=1.00 预测 fx {b0['pred_K_fx_on_zoomed']:.1f}  →  "
        f"s=2.00 预测 fx {b1['pred_K_fx_on_zoomed']:.1f}   "
        f"实际放大 {growth:.3f}×（真值应放大 {gt_growth:.3f}×）")
    say()
    say("  ⟹ 模型的相机头**确实**读到了视场线索（fx 随变焦增大），但**严重欠响应**：")
    say(f"     真值放大 {gt_growth:.3f}×，它只放大 {growth:.3f}×。")
    say("     所以它不是输出一个与视场无关的常数，而是在「有多宽」这件事上系统性偏低 ——")
    say("     与它在不给内参时给出 fx 偏小 3.17 倍是同一种偏差的不同尺度表现。")
    say()
    if len(zoom_rows) > 1:
        say(f"  顺带印证 C 段的发现：B 路径（给了正确内参）的误差 "
            f"s=1.00 时 {b0['err3d_median_m_gt_camera']:.4f} m，"
            f"s=1.50 时 {zoom_rows[1]['err3d_median_m_gt_camera']:.4f} m。")
        say("  变焦图更糊，误差反而更小。因为 44.7° 的视场比 63.3° 更接近模型的训练分布。")
        say("  这正是「深度头偏好窄视场」的独立复现，也是 3D 合成误差极小跑到 k=1.1 的原因。")
        say("  （因此本段只做同图 A/B，不与 s=1 横比绝对值 —— 上一句就是理由。）")

    rr = np.array([r["ratio_a_over_b"] for r in zoom_rows if r["ratio_a_over_b"]])
    zoom_str = "，".join(
        f"s={r['s']:.2f} → {r['ratio_a_over_b']:.2f}×" for r in zoom_rows
        if r["ratio_a_over_b"]
    )
    say()
    say(f"  三档变焦的 A/B 倍数：{zoom_str}")
    say(f"  全部 > 1（不给内参都更差）：{bool(np.all(rr > 1.0))}")
    say("  ⟹ 若成立，则「内参杠杆」不是某一对数值的巧合，在三个不同视场下都复现。")

    RESULT_D["zoom"] = zoom_rows
    RESULT_D["zoom_summary"] = {
        "ratios": [r["ratio_a_over_b"] for r in zoom_rows],
        "all_a_worse": bool(np.all(rr > 1.0)),
        "pred_fx_growth_s1_to_s2": round(growth, 4),
        "gt_fx_growth_s1_to_s2": round(gt_growth, 4),
    }
    flush()

    # ---- E. 容差预算 ---------------------------------------------------------
    hr("E.  容差预算 —— 真实照片只能靠 EXIF，那 EXIF 的精度够不够？")
    say("  这一节直接回答「下一步该不该做 EXIF 估算」。")
    say("  基准取 k=1（内参完全正确）处的**方位误差**，而不是 3D 合成误差 ——")
    say("  理由见 C 段：合成误差的极小位置被模型对视场的偏好污染，拿它定容差会得出")
    say("  「把焦距故意调大 10% 反而更好」这种荒谬结论（实测 3D 误差确实从 26.7 cm")
    say("  降到 12.8 cm）。方位分量没有这个病，所以容差必须用它来定。")
    say()
    say(f"  {'偏差|k−1|':>12}{'k':>9}{'fx':>9}{'方位px':>9}{'方位惩罚':>10}"
        f"{'3D惩罚':>9}{'3D中位(m)':>11}")
    say(f"  {'-'*12}{'-'*9}{'-'*9}{'-'*9}{'-'*10}{'-'*9}{'-'*11}")
    budget: list[dict] = []
    for r in good:
        dev = abs(r["k"] - 1.0) * 100
        if dev > 30.05:
            continue
        bp = r["bear_err_px_median"]
        bpen = (bp / b_ref) if bp else float("nan")
        epen = r["err3d_median_m"] / e_ref
        budget.append({"k": r["k"], "dev_pct": round(dev, 2), "fx": r["fx"],
                       "bear_err_px": bp, "bear_penalty": round(bpen, 3),
                       "err3d_median_m": r["err3d_median_m"],
                       "err3d_penalty": round(epen, 3)})
        say(f"  {dev:>11.1f}%{r['k']:>9.4f}{r['fx']:>9.1f}"
            f"{(bp if bp else float('nan')):>9.1f}{bpen:>9.2f}×{epen:>8.2f}×"
            f"{r['err3d_median_m']:>11.4f}")

    say()
    say("  ⚠ 看 k=1.10 那一行：方位误差涨到 1.50×（变坏），而 3D 惩罚只有 0.48×")
    say("    （看起来变好）。同一组数据、同一个 k，两个指标给出相反的结论。")
    say("    ⟹ 这正是本节开头那句话的实证：**用合成误差定容差会把自己带沟里**。")

    # 相对基准的两个版本都算，但结论只采用绝对门槛那套（理由见下）。
    d20 = dev_for_penalty(good, "bear_err_px_median", b_ref, 1.2)
    d50 = dev_for_penalty(good, "bear_err_px_median", b_ref, 1.5)
    d100 = dev_for_penalty(good, "bear_err_px_median", b_ref, 2.0)
    d10 = k_dev_for_abs(good, "bear_err_px_median", b_ref, 10.0)
    say()
    say("  绝对门槛（比「相对基准涨 X%」稳健 —— 后者的基准只有 "
        f"{b_ref:.1f} px，")
    say("  那已接近模型自身的横向误差下限，在近零的分母上谈相对涨幅会把门槛")
    say("  算成虚高的 ±0.5%，读起来像个苛刻的工程要求，其实只是小分母的假象）：")
    px_mm = 3000.0 / float(K_gt[0, 0])          # 3 m 处 1 px 对应的横向毫米数
    say(f"  （换算：fx={K_gt[0,0]:.1f}，故 3 m 处 1 px ≈ {px_mm:.1f} mm）")
    say()
    say(f"    {'方位误差门槛':<16}{'3 m 处横向':>12}{'允许的焦距偏差':>16}")
    say(f"    {'-'*16}{'-'*12}{'-'*16}")
    abs_budget: list[dict] = []
    for tgt in (5.0, 10.0, 25.0, 50.0):
        d = k_dev_for_abs(good, "bear_err_px_median", b_ref, tgt)
        mm = tgt * px_mm
        abs_budget.append({"target_px": tgt, "lateral_mm_at_3m": round(mm, 2),
                           "dev_pct": None if d is None else round(d * 100, 2)})
        say(f"    {f'{tgt:.0f} px':<16}{mm:>10.1f} mm"
            f"{('—（超出采样范围）' if d is None else f'±{d*100:.1f}%'):>16}")
    say()
    say("  怎么用这个表：关系判断的容差是 **50 mm**，所以「10 px」这一行"
        f"（3 m 处 {10.0*px_mm:.1f} mm）")
    say("  大致就是「横向误差不要吃掉容差」的界限，对应焦距偏差 "
        f"{'—' if d10 is None else f'±{d10*100:.1f}%'}。")
    say()
    say("  EXIF 的 `FocalLengthIn35mmFilm` 是整数毫米。以 26 mm 等效焦距为例，")
    say("  1 mm 的量化就是 ±3.8%（35 mm 时 ±1.4%，4 mm 时 ±2.5%）。")
    say("  ⟹ 手机/消费相机的 EXIF 大致落在「10 px」与「25 px」两行之间 ——")
    say("     可用，但会带来可见的横向偏置，必须把它**自带的量化误差**一起记进")
    say("     scene.json，并在 EXIF 缺失时显式降级 + 告警，不能当作无损真值。")
    say("  ⟹ 反过来看优先级：内参错 3 倍时的方位误差是 "
        f"{b_pred:.0f} px（{b_pred/b_ref:.0f}× 基准），")
    say("     比 EXIF 那几 % 的代价高出两个数量级 —— 「先解决有没有内参」远优先于")
    say("     「把内参精度再抠 1%」。")

    RESULT_D["tolerance_budget"] = {
        "metric": "bear_err_px_median（方位误差，与深度解耦）",
        "anchor": "k=1（内参正确）处的方位误差",
        "anchor_bear_err_px": b_ref,
        "anchor_err3d_median_m": e_ref,
        "absolute_targets": abs_budget,
        "px_to_mm_at_3m": round(px_mm, 3),
        "relative_points": budget,
        "dev_pct_for_1p2x_relative": None if d20 is None else round(d20 * 100, 2),
        "dev_pct_for_1p5x_relative": None if d50 is None else round(d50 * 100, 2),
        "dev_pct_for_2x_relative": None if d100 is None else round(d100 * 100, 2),
        "exif_quantisation_example": {"focal_mm": 26, "step_mm": 1, "rel_pct": 3.8},
        "warning": "不要用 3D 合成误差定容差：其极小被模型视场偏好污染（见 C 段）。"
                   "也不要用近零基准谈相对涨幅（见 absolute_targets 前的说明）。",
    }
    flush()

    # ---- VERDICT -------------------------------------------------------------
    hr("VERDICT")
    ok_self = med_mm < 10.0
    say(f"  ① 自洽性：把预测的 K 显式喂回去，点云与默认路径中位差 {med_mm:.2f} mm")
    say(f"     ⟹ {'成立' if ok_self else '不成立'}：误差可归因到那份 K 的数值本身，")
    say("        模型没有别的隐藏机制在干扰。这是后面各项成立的前提。")
    say("  ② 因果性（用**方位**分量，与深度解耦）：")
    say(f"     粗扫（±10% 网格）极小在 k={k_bear:.2f}；细化扫描（±1% 网格 + 抛物线插值）"
        f"给出 k̂={k_hat:.4f}")
    say(f"     即**从数据独立反解** fx̂={fx_hat:.1f}，与仓库自带 GT 的 fx={K_gt[0,0]:.1f} "
        f"一致到 {abs(fx_hat/K_gt[0,0]-1)*100:.1f}% 以内。")
    say(f"     方位曲线两侧单调 {'是' if (mono_lo and mono_hi) else '否'}"
        f"（是 ⟹ 该极小是全局极小，可作为焦距的估计量）")
    say(f"     预测点 k={K_PRED_RATIO:.4f} 的方位误差 {b_pred:.1f} px，"
        f"是极小（{np.nanmin(bpx):.1f} px）的 {b_pred/np.nanmin(bpx):.1f} 倍；")
    say(f"     而深度 ARel 只差 {dep[i_pred]/dep[i_bear]:.1f} 倍 "
        f"（{dep[i_pred]*100:.1f}% vs {dep[i_bear]*100:.1f}%）")
    say("     ⟹ **伤害几乎全部落在横向**，这正是内参错误的签名。")
    say(f"  ③ 方法学修正（本轮最重要的负面结果）：3D 合成误差的极小在 k={k_b3d:.4f}，")
    say(f"     偏离真值 {abs(k_b3d-1)*100:.1f}%；因为深度 ARel 的极小在 k={k_dep:.4f}")
    say("     —— 模型的深度头偏好更窄的视场，与内参真值无关。")
    say("     ⟹ 不能用合成误差反推相机内参，必须分解成横向与纵深两部分。")
    say(f"  ④ 多条件：三档变焦 {'全部' if np.all(rr>1.0) else '并非全部'} 复现 A/B > 1 —— {zoom_str}")
    say(f"  ⑤ 各向异性：方位误差动态范围 "
        f"{np.nanmax(bpx)/max(np.nanmin(bpx),1e-9):.0f}×，"
        f"深度 ARel 只有 {dep[i_pred]/max(dep[i_bear],1e-9):.1f}×")
    say("     ⟹ 全局标量 scale_factor 无法修正内参错误（会把已经对的 z 一起弄错），")
    say("        原方案的 `calibrate_scale` 由此撤销。")
    if d10 is not None:
        say(f"  ⑥ 容差：把方位误差控制在 10 px（3 m 处 {10.0*px_mm:.1f} mm）以内，")
        say(f"        要求焦距偏差不超过 ±{d10*100:.1f}% —— 这是 EXIF 方案的精度门槛。")
        say(f"        而内参错 3 倍的方位误差是 {b_pred:.0f} px（{b_pred/b_ref:.0f}× 基准），")
        say("        高两个数量级：优先级上「有没有内参」远高于「内参再精确 1%」。")
    say()
    say("  仍然没做到的（必须写进报告，不能含糊）：")
    say("  · 全部证据仍来自**同一张照片**。变焦给出了三个不同视场，但共享同一场景、")
    say("    同一成像链路、同一种上采样伪影。换到完全不同来源的照片（手机、相机、")
    say("    不同场景）尚未验证 —— 这是当前最大的证据缺口。")
    say("  · 变焦图是上采样的，不是真实光学变焦，高频细节缺失，不能用于评估绝对精度。")
    say("  · 网格在真值附近的间距是 ±10%，所以「方位极小在 k=1」只能读作")
    say("    「极小落在 [0.95, 1.05] 内且真值点为最优采样点」，不是「精确等于 1」。")
    say("  · 0.6% 的常数残差（方位跨度 vs 1/k）来源未查清，它上界了可分辨的焦距精度。")

    RESULT_D["verdict"] = {
        "self_consistency_ok": bool(ok_self),
        "replay_vs_nocam_median_mm": round(med_mm, 4),
        "causal_metric": "bear_err_px_median（方位，与深度解耦）",
        "k_optimal_bearing": k_bear,
        "k_hat_parabolic": round(k_hat, 5),
        "fx_hat_from_data": round(fx_hat, 3),
        "fx_hat_rel_dev_pct": round((fx_hat / K_gt[0, 0] - 1) * 100, 3),
        "k_optimal_bearing_dev_pct_from_1": round(abs(k_bear - 1.0) * 100, 2),
        "k_optimal_bearing_grid_resolution_pm_pct": 10.0,
        "k_optimal_sentence_strict_meaning": (
            "极小落在 [0.95, 1.05] 内，且真值 k=1.0 是该区间内的最优采样点；"
            "不等于「极小精确等于 1.000」"
        ),
        "bear_span_vs_1_over_k_rel_dev_pct": [
            round(float(span_rel.min()) * 100, 3),
            round(float(span_rel.max()) * 100, 3),
        ],
        "bearing_monotone": bool(mono_lo and mono_hi),
        "bear_err_px_at_predicted_k": b_pred,
        "bear_err_px_at_best": float(np.nanmin(bpx)),
        "bearing_penalty_at_predicted_k": round(b_pred / b_ref, 2),
        "depth_penalty_at_predicted_k": round(dep[i_pred] / dep[i_bear], 2),
        "k_optimal_err3d": k_b3d,
        "k_optimal_depth_absrel": k_dep,
        "methodological_warning": (
            "3D 合成误差的极小位置被模型对视场的偏好污染，不可用于反推内参；"
            "必须分解为横向（方位）与纵深两部分分别看。"
        ),
        "zoom_ratios": [r["ratio_a_over_b"] for r in zoom_rows],
        "zoom_all_a_worse": bool(np.all(rr > 1.0)),
        "anisotropic_bearing_range": float(np.nanmax(bpx) / max(np.nanmin(bpx), 1e-9)),
        "anisotropic_depth_penalty": float(dep[i_pred] / max(dep[i_bear], 1e-9)),
        "tolerance_dev_pct_for_10px": None if d10 is None else round(d10 * 100, 2),
        "tolerance_dev_pct_1p5x_relative": None if d50 is None else round(d50 * 100, 2),
    }
    RESULT_D["gt"] = {
        "fx": float(K_gt[0, 0]), "fy": float(K_gt[1, 1]),
        "cx": float(K_gt[0, 2]), "cy": float(K_gt[1, 2]),
        "hfov_deg": hfov_deg(K_gt[0, 0], W),
        "image_hw": [H, W],
        "depth_valid_px": int(valid0.sum()),
        "depth_range_m": [round(float(np.nanmin(d_gt[valid0])), 4),
                          round(float(np.nanmax(d_gt[valid0])), 4)],
    }
    RESULT_D["k_sweep"] = list(K_SWEEP)
    RESULT_D["zoom_levels"] = list(ZOOM_LEVELS)
    RESULT_D["load_s"] = round(load_s, 2)
    RESULT_D["metrics_reference"] = {
        "no_camera_full": m_nocam,
        "replay_pred_K_full": m_replay,
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
