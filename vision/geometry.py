"""点云 → 几何量：掩码质心、稳健尺寸、包围盒、重力方向估计。

**纯 NumPy，不 import torch，不 import scene_graph。** 只吃 ndarray，只吐 ndarray / 元组。
（与 `scene_graph.schema` 的转换在 `builder.py` 里做 —— 依赖方向必须是
「表示层依赖感知层的输出」，反过来会让这个文件没法独立测试。）

它存在的理由是 Phase 0 的一条**实测裁决**（§20 Step 0.5b）：

    「三维中心取检测框内像素的中位数」会把背景算进去。
    9 个物体上对比「框内中位数 vs SAM2 掩码质心」：
    均值 83 mm、最大 208 mm，而关系判定容差是 50 mm。

208 mm 是容差的 4 倍以上。所以「用掩码」不是优化，是**正确性要求** ——
把它写成可单测的纯函数（`tests/test_vision_geometry.py`），
比在业务代码里写五行注释可靠得多。

另一条实测规律同样写进了实现：**框越大、背景越多、误差越大**。
sofa（最大框）背景占 53.8%、质心偏 208 mm；墙上小挂画背景近乎为零、只偏 1–40 mm。
所以 `builder.py` **没有**做「小框走框、大框走掩码」这类优化 ——
一律走掩码，让误差来源单一化，可归因。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

__all__ = [
    "DEFAULT_MIN_POINTS",
    "DEFAULT_K_P90",
    "PLAUSIBLE_HFOV_DEG",
    "UpAxisEstimate",
    "FovCheck",
    "check_fov",
    "fov_deg",
    "intrinsics_matrix",
    "resample_mask_to",
    "box_selector",
    "valid_point_selector",
    "select_points",
    "centroid_of",
    "robust_extent",
    "box_coverage",
    "clip_box_xyxy",
    "box_area",
    "iou_xyxy",
    "estimate_up_axis",
]

#: 一个物体的掩码里至少要有这么多**有效**点，其质心才可采信。
#: 30 这个数偏保守：640×480 的点云里一个正常物体有 1e3–1e5 个点，
#: 低于 30 意味着掩码基本是空的（检测框打在了天空/白墙上），此时退回检测框
#: 比硬用几十个点估计质心更稳。
DEFAULT_MIN_POINTS = 30

#: 离群点剔除阈值 = `k × 半径的 90 分位`。
#:
#: 为什么不用教科书上的 MAD（中位数绝对偏差）：MAD 对**紧凑而扁平**的点集
#: 会刚好切掉四角。均匀圆盘上 `3.5 × 1.4826 × MAD ≈ 0.95 R` —— 只差 5% 就
#: 开始裁真实极值；而挂画、屏幕、镜子恰好就是这种形状。1.5 × r_p90 则稳定地
#: 落在 `1.3–1.4 R` 之外，**可证明不裁**（见 `robust_extent` 的推导）。
#: 而掩码泄漏进来的背景点通常落在物体尺寸的 1.5 倍以上，仍会被稳稳切掉。
DEFAULT_K_P90 = 1.5


# ----------------------------------------------------------------------------
# 像素网格换算
# ----------------------------------------------------------------------------


def resample_mask_to(mask: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """把掩码按**最近邻**重采样到另一个网格。

    为什么必须有这个函数：掩码来自 SAM2 的 `post_process_masks`，尺寸是**输入图像**
    (HI, WI)；而点云来自 UniDepth，尺寸是**模型网格** (HP, WP)。两者不一定相等 ——
    `unidepthv2.py:282-336` 会 padding→resize→裁回。直接 `mask.reshape` 或
    `np.ix_` 都不对，必须做一次显式采样。

    用最近邻而不是双线性：掩码是布尔量，双线性会造出「0.4 个物体」这种中间值，
    再阈值化等于偷偷换了判据。最近邻的语义是「这个点云像素落在哪个图像像素上」，
    正是我们要问的问题。

    采样用像素中心对齐（`(i + 0.5) * HI / HP`），不是角点对齐 ——
    否则会整体偏移半个像素，在 640→320 这种 2 倍降采样下表现为系统性的半格错位。
    """
    if mask.ndim != 2:
        raise ValueError(f"mask 必须是 2D，收到 shape={mask.shape}")
    hi, wi = mask.shape
    ho, wo = int(out_hw[0]), int(out_hw[1])
    if (hi, wi) == (ho, wo):
        return mask.astype(bool, copy=False)

    rows = np.clip(((np.arange(ho) + 0.5) * hi / ho).astype(np.int64), 0, hi - 1)
    cols = np.clip(((np.arange(wo) + 0.5) * wi / wo).astype(np.int64), 0, wi - 1)
    return mask[np.ix_(rows, cols)].astype(bool, copy=False)


def box_selector(
    box_xyxy: Sequence[float],
    grid_hw: tuple[int, int],
    image_hw: tuple[int, int],
) -> np.ndarray:
    """把**图像像素系**的一个框变成**点云网格**上的 `(H, W)` bool 选择器。

    只在「掩码失效、退回检测框」这条降级路径上用到 —— 但正因为它是降级路径，
    更要写对：一条会静默偏移的降级路径比直接失败更难查。

    它住在这里而不是 `scene_graph/builder.py`，理由是**分层**：
    builder 为了拿 `DEFAULT_PROMPT` 会 import `vision.grounding`，而那一路会
    拉起 torch。点云工具要能在无 GPU 环境下跑、要能脱离 GPU 单测，所以它们
    只能依赖这个零 torch 的模块。同源实现放在这里，双方都不会各写一份。
    """
    hp, wp = int(grid_hw[0]), int(grid_hw[1])
    hi, wi = int(image_hw[0]), int(image_hw[1])
    x1, y1, x2, y2 = clip_box_xyxy(box_xyxy, image_hw)
    gx1 = int(np.clip(np.floor(x1 * wp / wi), 0, wp))
    gx2 = int(np.clip(np.ceil(x2 * wp / wi), 0, wp))
    gy1 = int(np.clip(np.floor(y1 * hp / hi), 0, hp))
    gy2 = int(np.clip(np.ceil(y2 * hp / hi), 0, hp))
    sel = np.zeros((hp, wp), dtype=bool)
    if gx2 > gx1 and gy2 > gy1:
        sel[gy1:gy2, gx1:gx2] = True
    return sel


# ----------------------------------------------------------------------------
# 点集选择
# ----------------------------------------------------------------------------


def valid_point_selector(points_chw: np.ndarray) -> np.ndarray:
    """哪些像素的点云可用（有限且在相机前方）。

    判据是 `z > 0` 而不是 `z != 0`：单目深度在天空、镜面、极远处会吐出
    `inf` / `nan` / 负值，这些点不能参与中位数 —— 一个 `inf` 就能把中位数拉飞。
    """
    if points_chw.ndim != 3 or points_chw.shape[0] != 3:
        raise ValueError(f"points 必须是 (3,H,W)，收到 shape={points_chw.shape}")
    finite = np.isfinite(points_chw).all(axis=0)
    return finite & (points_chw[2] > 0.0)


def select_points(
    points_chw: np.ndarray,
    selector: np.ndarray,
) -> np.ndarray:
    """按 `selector`（(H*W,) 或 (H,W) 的 bool）取出点，返回 `(3, N)` float64。

    同时**强制剔除无效点**，调用方不必自己 remember 这一步 ——
    忘记一次就得到一个被 `inf` 污染的中位数，而它不会报错，只会安静地错。
    """
    sel = np.asarray(selector).reshape(-1)
    if sel.dtype != bool:
        sel = sel > 0
    flat_valid = valid_point_selector(points_chw).reshape(-1)
    if sel.shape != flat_valid.shape:
        raise ValueError(
            f"selector 长度 {sel.shape[0]} 与点云像素数 {flat_valid.shape[0]} 不一致"
        )
    idx = sel & flat_valid
    pts = points_chw.reshape(3, -1)[:, idx]
    return np.ascontiguousarray(pts, dtype=np.float64)


# ----------------------------------------------------------------------------
# 质心
# ----------------------------------------------------------------------------


def centroid_of(
    pts: np.ndarray,
    *,
    reduce: Literal["median", "mean", "trimmed"] = "median",
) -> tuple[np.ndarray | None, dict]:
    """`(3,N)` 点集 → 质心 + 不确定度度量。

    默认**逐轴中位数**而不是均值。这不是审美选择：掩码几乎总会在边缘扫到几个
    远景像素，均值会被它们拖着走数百毫米，中位数不会。Phase 0 的探针用的也是中位数，
    保持同一口径，历史数字才可比。

    返回值第二项是证据（进 `Node` 或 `evidence`），核心是 `radius_p50` / `radius_p90`
    —— 「到这个中心的距离的中位数 / 90 分位」。`radius_p90` 可以直接当成
    **质心不确定度的上界**用：它就是掩码里最远那 10% 的点有多远。
    报告里的「3D 定位误差」需要它作分母，否则 208 mm 这种数字没有参照。
    """
    if pts.shape[1] == 0:
        return None, {"n_points": 0}
    n = int(pts.shape[1])

    if reduce == "mean":
        c = pts.mean(axis=1)
    elif reduce == "trimmed":
        # 逐轴去掉上下 10% 再取均值：比中位数用上更多样本，又不像均值那样怕离群。
        lo = np.percentile(pts, 10.0, axis=1)
        hi = np.percentile(pts, 90.0, axis=1)
        keep = ((pts >= lo[:, None]) & (pts <= hi[:, None])).all(axis=0)
        c = pts[:, keep].mean(axis=1) if keep.any() else pts.mean(axis=1)
    else:  # median
        c = np.median(pts, axis=1)

    r = np.linalg.norm(pts - c[:, None], axis=0)
    meta = {
        "n_points": n,
        "reduce": reduce,
        "radius_p50_m": float(np.percentile(r, 50.0)),
        "radius_p90_m": float(np.percentile(r, 90.0)),
        "radius_max_m": float(r.max()),
    }
    return c, meta


# ----------------------------------------------------------------------------
# 尺寸与包围盒
# ----------------------------------------------------------------------------


def robust_extent(
    pts: np.ndarray,
    *,
    k_p90: float = DEFAULT_K_P90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """`(3,N)` 点集 → `(extent, bbox_min, bbox_max, meta)`。

    先按「到质心的距离」剔除离群点，再在**内点**上取逐轴 min/max。

    为什么不直接 min/max：掩码泄漏的那几个远处背景点会把某个轴的跨度直接翻倍。
    为什么不用分位数（比如 2%–98%）：那会**系统性**削掉真实极值 ——
    一个 5000 点的掩码裁掉 100 个前尾点，可能就把物体的真实边界削没了。
    剔除准则必须**不对称**：真实极值有几百个点，离群点只有几个。

    判据 `r <= k_p90 × r_p90`（`r_p90` = 半径的 90 分位），而不是 MAD：

        对均匀圆盘，`r_p90 ≈ 0.95 R`，阈值 `≈ 1.42 R > R` —— **不裁**；
        对扁平矩形（挂画/屏幕），`r_p90 ≈ 0.9 R_max`，阈值 `≈ 1.35 R_max` —— **不裁**；
        而掩码泄漏的背景点通常在物体最大半径的 1.5 倍以外 —— **裁掉**。

    相比之下 MAD 对圆盘只到 `0.95 R`，是「刚好在边界上」；挂画这类扁平物体
    四角会不会被削，取决于长宽比，是不可靠的。这条换成可证明的判据不是洁癖：
    外表尺寸直接进 L5 的场景报告，也是 `on`/`inside` 的判据。
    """
    if pts.shape[1] == 0:
        z = np.zeros(3, dtype=np.float64)
        return z, z, z, {"n_points": 0, "n_inliers": 0, "n_rejected": 0}

    c = np.median(pts, axis=1)
    r = np.linalg.norm(pts - c[:, None], axis=0)
    r90 = float(np.percentile(r, 90.0))
    thr = k_p90 * r90
    if thr <= 1e-12:
        # 所有点重合（退化成一个点）—— 没有可剔除的东西。
        keep = np.ones(pts.shape[1], dtype=bool)
    else:
        keep = r <= thr

    inl = pts[:, keep] if keep.any() else pts
    lo = inl.min(axis=1)
    hi = inl.max(axis=1)
    extent = hi - lo
    meta = {
        "n_points": int(pts.shape[1]),
        "n_inliers": int(inl.shape[1]),
        "n_rejected": int(pts.shape[1] - inl.shape[1]),
        "k_p90": float(k_p90),
        "radius_p90_m": r90,
        "reject_threshold_m": float(thr),
        "centre_used_m": [float(v) for v in c],
    }
    return extent, lo, hi, meta


# ----------------------------------------------------------------------------
# 2D 框工具
# ----------------------------------------------------------------------------


def clip_box_xyxy(
    box: Sequence[float],
    image_hw: tuple[int, int],
) -> tuple[float, float, float, float]:
    """把框裁进图像范围，并保证 `x2 >= x1`、`y2 >= y1`。"""
    hi, wi = int(image_hw[0]), int(image_hw[1])
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return (
        min(max(x1, 0.0), float(wi)),
        min(max(y1, 0.0), float(hi)),
        min(max(x2, 0.0), float(wi)),
        min(max(y2, 0.0), float(hi)),
    )


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = (float(v) for v in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    union = box_area(a) + box_area(b) - inter
    return float(inter / union) if union > 0.0 else 0.0


def box_coverage(mask: np.ndarray, box_xyxy: Sequence[float]) -> float:
    """掩码像素数 / 框面积。用来量化「框里有多少是背景」。

    它是 `1 - 背景占比`：Phase 0 实测 sofa 的这个值是 46.2%（背景 53.8%），
    而墙上小挂画接近 100%。这个数字进 `build_meta`，是失败诊断
    （L5 `diagnose_failure`）判断「该物体为什么定位不准」的一手依据。
    """
    x1, y1, x2, y2 = (int(round(float(v))) for v in box_xyxy)
    h, w = mask.shape
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    a = box_area((x1, y1, x2, y2))
    if a <= 0.0:
        return 0.0
    return float(mask[y1:y2, x1:x2].sum()) / a


# ----------------------------------------------------------------------------
# 内参与视场
# ----------------------------------------------------------------------------

#: 水平视场的「可信区间」（度）。
#:
#: 30° 以下属于长焦，110° 以上属于鱼眼 / 超广角 —— 都不是手机或普通相机拍
#: 室内照片的正常形态。定成**区间**而不是一个等值，是因为这个检查的用途是
#: 兜住**量级错误**（Phase 0 Step 6 实测过 125.8° vs 真值 63.3°、横向尺度差
#: 3.17 倍这种），不是做标定。区间留宽一点，真正的超广角照片才不会被误判。
PLAUSIBLE_HFOV_DEG: tuple[float, float] = (30.0, 110.0)


def fov_deg(focal_px: float, n_px: float) -> float:
    """像素焦距 → 该方向上的视场角（度）。`focal_px <= 0` 时返回 NaN。"""
    f = float(focal_px)
    if f <= 0.0:
        return float("nan")
    return float(2.0 * np.degrees(np.arctan(float(n_px) / (2.0 * f))))


def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """`(fx, fy, cx, cy)` → 3×3 内参。`BuildConfig.known_intrinsics` 用它。"""
    return np.array(
        [[float(fx), 0.0, float(cx)],
         [0.0, float(fy), float(cy)],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class FovCheck:
    """一次视场合理性检查的结果。`reason` 会原样进 `build_meta` 与警告文本。"""

    hfov_deg: float
    vfov_deg: float
    plausible: bool
    reason: str

    def as_dict(self) -> dict[str, float | bool | str]:
        return {
            "hfov_deg": round(self.hfov_deg, 2),
            "vfov_deg": round(self.vfov_deg, 2),
            "plausible": self.plausible,
            "reason": self.reason,
        }


def check_fov(
    K: np.ndarray,
    image_hw: tuple[int, int],
    bounds: tuple[float, float] = PLAUSIBLE_HFOV_DEG,
) -> FovCheck:
    """内参 + 图像尺寸 → `(水平视场, 垂直视场, 是否可信, 原因)`。

    **为什么这个检查必须有**：UniDepth V2 的相机头在实测里对
    `assets/demo/rgb.png` 给出 `fx = 163.7`，即 125.8° 的水平视场，
    真值 63.3° —— 横向坐标被放大 3.17 倍，于是沙发量出 6.70 m 宽。
    这个错误**不会抛异常**，只会让下游所有米制数字安静地错。

    它是本项目的核心主张在一个新层面的复现：**能测的就不要猜**。
    已知内参就传进去；只能预测时，至少要知道预测值有多不可信，
    并把这个判断落到 `build_meta` 与警告里，而不是留在人脑里。
    """
    Ka = np.asarray(K, dtype=np.float64)
    if Ka.shape != (3, 3):
        raise ValueError(f"内参必须是 3×3，收到 {Ka.shape}")
    hi, wi = float(image_hw[0]), float(image_hw[1])
    h = fov_deg(Ka[0, 0], wi)
    v = fov_deg(Ka[1, 1], hi)
    if not (np.isfinite(h) and np.isfinite(v)):
        return FovCheck(h, v, False, "non_positive_focal")
    lo, up = float(bounds[0]), float(bounds[1])
    if h < lo or h > up:
        return FovCheck(h, v, False, "hfov_out_of_range")
    return FovCheck(h, v, True, "ok")


# ----------------------------------------------------------------------------
# 重力方向估计
# ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UpAxisEstimate:
    """`up_axis` 的估计结果。

    单张图下重力方向是**估出来的**，这是整张场景图最脆弱的一环
    （§12.3 步骤 6、§18 Phase 5 风险②）。所以这里不只回一个字符串，
    而是连**它有多可信**一起回 —— `tilt_deg` 与 `reliable` 都要进 `build_meta`。

    为什么 `axis` 只能是 `"-y"` / `"+y"` 这类轴对齐字符串：
    真实估计出的法线通常是斜的，但 `relations.py` 的 `UpAxis` 只支持轴对齐 ——
    这是**有意为之**：把一个斜法线投影回 y 轴再声称精度更高，是假精度。
    倾斜超过阈值时正确的做法是标记 `reliable=False` 并让报告讨论它，
    而不是偷偷用斜法线算出一个看起来更准的 `above`。
    """

    #: 可直接喂 `scene_graph.relations.UpAxis.parse`。
    axis: str
    #: 拟合地面法线与 `axis` 的夹角（度）。0 = 完美水平。
    tilt_deg: float
    reliable: bool
    n_points: int
    normal: tuple[float, float, float] | None
    reason: str


def estimate_up_axis(
    points_chw: np.ndarray,
    *,
    band: float = 0.2,
    min_points: int = 200,
    max_tilt_deg: float = 25.0,
    max_fit_points: int = 20000,
) -> UpAxisEstimate:
    """用画面**下部**的点云拟合一个平面，把它的法线当作重力方向。

    取画面底部 `band` 比例的行：室内照片的下沿几乎总是地板或桌面，两者都近似水平。
    对点集做一次 SVD，最小奇异向量就是平面法线（残差最小的方向）。

    三种判据、三种 reason：
      • 点太少（`too_few_points`）—— 下部是天空/白墙，点云无效
      • 拟合出的面不水平（`band_not_horizontal`）—— 下沿是一面竖直的墙，
        法线会落在水平方向，此时「上」根本无从谈起
      • `ok`

    ⚠️ 已知局限（要写进报告）：画面下沿若是**斜面**（楼梯、投影幕布下的斜台），
    或者相机有 roll，这个方法会给出错误的 up。它不解决 roll，只是**检测** tilt。
    """
    hp, wp = points_chw.shape[1], points_chw.shape[2]
    y0 = int(max(0.0, (1.0 - band)) * hp)
    sel = np.zeros((hp, wp), dtype=bool)
    sel[y0:, :] = True
    pts = select_points(points_chw, sel)
    n = int(pts.shape[1])

    if n < min_points:
        return UpAxisEstimate(
            axis="-y", tilt_deg=float("nan"), reliable=False, n_points=n,
            normal=None, reason="too_few_points",
        )

    if n > max_fit_points:
        step = int(np.ceil(n / max_fit_points))
        pts = pts[:, ::step]

    c = pts.mean(axis=1, keepdims=True)
    centred = (pts - c).T  # (N, 3)
    try:
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return UpAxisEstimate(
            axis="-y", tilt_deg=float("nan"), reliable=False, n_points=n,
            normal=None, reason="svd_failed",
        )
    nrm = vt[-1]
    norm = float(np.linalg.norm(nrm))
    if norm <= 1e-12:
        return UpAxisEstimate(
            axis="-y", tilt_deg=float("nan"), reliable=False, n_points=n,
            normal=None, reason="degenerate_normal",
        )
    nrm = nrm / norm

    # 相机系 y 向下 ⟹ 「上」是 -y。把法线定向到朝上的那一侧，
    # 这样夹角才是一个有意义的量（否则它会随机是 θ 或 180°-θ）。
    if nrm[1] > 0.0:
        nrm = -nrm

    cos_tilt = float(np.clip(-nrm[1], -1.0, 1.0))
    tilt_deg = float(np.degrees(np.arccos(cos_tilt)))
    reliable = tilt_deg <= max_tilt_deg

    return UpAxisEstimate(
        axis="-y",
        tilt_deg=tilt_deg,
        reliable=reliable,
        n_points=n,
        normal=(float(nrm[0]), float(nrm[1]), float(nrm[2])),
        reason="ok" if reliable else "band_not_horizontal",
    )
