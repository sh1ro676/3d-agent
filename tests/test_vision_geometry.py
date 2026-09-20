"""`vision/geometry.py` 的单元测试 —— 纯 NumPy，零 GPU、零模型、零网络。

这组测试的存在意义，是把 Phase 0 的两条**实测裁决**钉在代码上：

    ① 质心必须用掩码（框内中位数偏均值 83 mm / 最大 208 mm，容差只有 50 mm）
    ② 掩码与点云的分辨率必须显式换算（不能假设相等）

第 ② 条尤其需要测试：写错了不会报错，只是让所有质心系统性偏移 ——
那种 bug 靠肉眼审查是查不出来的，只能靠「同一物体在不同网格下质心应当不变」
这样的不变量来抓。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.geometry import (  # noqa: E402
    PLAUSIBLE_HFOV_DEG,
    box_area,
    box_coverage,
    centroid_of,
    check_fov,
    clip_box_xyxy,
    estimate_up_axis,
    fov_deg,
    intrinsics_matrix,
    iou_xyxy,
    resample_mask_to,
    robust_extent,
    select_points,
    valid_point_selector,
)


# ----------------------------------------------------------------------------
# 造点云
# ----------------------------------------------------------------------------


def pinhole_cloud(
    depth_hw: np.ndarray,
    *,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
) -> np.ndarray:
    """由深度图生成解析点云：`x=(u-cx)z/fx, y=(v-cy)z/fy, z=depth`。

    相机系 y 向下 —— 与 `scene_graph/schema.py` 的约定一致。
    默认把主点放在像素网格的**正中**（`W/2 - 0.5`），于是对称区域的质心恰好是 0，
    断言可以写成精确值而不是「大概」。
    """
    h, w = depth_hw.shape
    fx = float(w if fx is None else fx)
    fy = float(w if fy is None else fy)
    cx = (w / 2 - 0.5) if cx is None else cx
    cy = (h / 2 - 0.5) if cy is None else cy
    u = np.broadcast_to(np.arange(w, dtype=np.float64), (h, w))
    v = np.broadcast_to(np.arange(h, dtype=np.float64)[:, None], (h, w))
    z = depth_hw.astype(np.float64)
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z])


# ----------------------------------------------------------------------------
# 掩码重采样
# ----------------------------------------------------------------------------


class TestResampleMask:
    def test_same_shape_is_identity(self):
        m = np.zeros((4, 4), dtype=bool)
        m[1, 2] = True
        out = resample_mask_to(m, (4, 4))
        assert out.dtype == bool
        assert np.array_equal(out, m)

    def test_pixel_centre_alignment(self):
        """采样点必须落在像素中心，不是角点。

        角点对齐会整体偏半格，在 2× 降采样下表现为系统性的半格错位 ——
        这正是让所有质心悄悄偏移 1 cm 的那种 bug。
        """
        hi = np.zeros((4, 4), dtype=bool)
        hi[0:2, 0:2] = True              # 左上 2×2
        out = resample_mask_to(hi, (2, 2))
        # 中心对齐时采样行/列是 1 和 3（(i+0.5)*4/2），落在左上象限的只有 (1,1)
        assert out[0, 0] and not out[0, 1] and not out[1, 0] and not out[1, 1]

    def test_upsample_blocks(self):
        lo = np.zeros((2, 2), dtype=bool)
        lo[1, 0] = True
        out = resample_mask_to(lo, (4, 4))
        assert out.shape == (4, 4)
        # 单点放大后应是一个连通块，而不是全线命中
        assert 0 < out.sum() < 16

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            resample_mask_to(np.zeros((1, 4, 4), dtype=bool), (4, 4))


# ----------------------------------------------------------------------------
# 点集选择
# ----------------------------------------------------------------------------


class TestSelection:
    def test_valid_selector_drops_nan_and_nonpositive_z(self):
        pts = np.zeros((3, 2, 2))
        pts[2] = 1.0
        pts[2, 0, 0] = np.nan
        pts[2, 0, 1] = 0.0
        pts[2, 1, 0] = -1.0
        sel = valid_point_selector(pts)
        assert sel.tolist() == [[False, False], [False, True]]

    def test_select_points_filters_invalid(self):
        pts = np.zeros((3, 2, 2))
        pts[2] = 1.0
        pts[2, 0, 0] = np.inf
        got = select_points(pts, np.ones((2, 2), dtype=bool))
        # 4 个像素里 1 个无效，应只剩 3 个点
        assert got.shape == (3, 3)
        assert np.isfinite(got).all()

    def test_select_points_length_mismatch(self):
        pts = np.zeros((3, 2, 2))
        with pytest.raises(ValueError):
            select_points(pts, np.ones((3, 3), dtype=bool))

    def test_rejects_bad_points_shape(self):
        with pytest.raises(ValueError):
            valid_point_selector(np.zeros((2, 4, 4)))


# ----------------------------------------------------------------------------
# 质心与尺寸
# ----------------------------------------------------------------------------


class TestCentroid:
    def test_median_beats_mean_under_outliers(self):
        """这是「用中位数」这条决定的**存在证明**。

        385 个物体点 + 5 个远处背景点（物体距离的 30 倍）。中位数精确落在真值上，
        均值被这 5 个点拖出 0.7 m —— 量与 Phase 0 实测的「框内混背景 → 偏 208 mm」
        同源，只是这里把比例放大了以便断言。
        """
        grid = np.stack(
            np.meshgrid(
                np.linspace(-0.25, 0.25, 11),
                np.linspace(-0.15, 0.15, 7),
                np.linspace(2.0, 2.2, 5),
            ),
            axis=-1,
        ).reshape(-1, 3).T
        outliers = np.tile(np.array([[5.0], [5.0], [60.0]]), (1, 5))
        pts = np.concatenate([grid, outliers], axis=1)

        med, meta = centroid_of(pts)
        mean, _ = centroid_of(pts, reduce="mean")
        assert med is not None and mean is not None
        # 中位数逐轴精确命中真值（x/y/z 各自的中位数恰好落在网格中心）
        assert np.allclose(med, [0.0, 0.0, 2.1], atol=1e-9)
        assert abs(float(mean[2]) - 2.1) > 0.5
        assert meta["reduce"] == "median"
        assert meta["n_points"] == 390
        assert meta["radius_p50_m"] > 0.0

    def test_empty_returns_none(self):
        c, meta = centroid_of(np.zeros((3, 0)))
        assert c is None and meta["n_points"] == 0

    def test_trimmed_reduce_runs(self):
        pts = np.random.default_rng(0).normal(size=(3, 100))
        c, meta = centroid_of(pts, reduce="trimmed")
        assert c is not None and meta["reduce"] == "trimmed"


class TestRobustExtent:
    def test_extent_survives_outliers(self):
        """外表尺寸必须由真实几何决定，不被掩码泄漏的远景点撑大。

        构造一个 0.5 × 0.3 × 0.2 m 的点簇 + 5 个 30 倍远处的点：
        期望 extent 仍是 (0.5, 0.3, 0.2)，且离群点被明确计数。
        """
        grid = np.stack(
            np.meshgrid(
                np.linspace(-0.25, 0.25, 11),
                np.linspace(-0.15, 0.15, 7),
                np.linspace(2.0, 2.2, 5),
            ),
            axis=-1,
        ).reshape(-1, 3).T
        outliers = np.tile(np.array([[5.0], [5.0], [60.0]]), (1, 5))
        pts = np.concatenate([grid, outliers], axis=1)

        extent, lo, hi, meta = robust_extent(pts)
        assert np.allclose(extent, [0.5, 0.3, 0.2], atol=1e-9)
        assert np.allclose(lo, [-0.25, -0.15, 2.0], atol=1e-9)
        assert np.allclose(hi, [0.25, 0.15, 2.2], atol=1e-9)
        assert meta["n_rejected"] == 5
        assert meta["n_inliers"] == 385

    def test_flat_object_not_clipped(self):
        """扁平物体（挂画、屏幕、镜子）的真实边界不能被剔除准则削掉。

        这是把判据从 MAD 换成「1.5 × 半径 90 分位」的直接原因：
        MAD 对紧凑而扁平的二维点集只到 `≈0.95 R`，正好卡在边界上，
        四角会不会被削取决于长宽比 —— 不可靠。挂画正是最常见的扁平物体。
        """
        x = np.linspace(-0.3, 0.3, 30)
        y = np.linspace(-0.15, 0.15, 15)
        xx, yy = np.meshgrid(x, y)
        pts = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, 2.0)])
        extent, lo, hi, meta = robust_extent(pts)
        assert np.allclose(extent, [0.6, 0.3, 0.0], atol=1e-9)
        assert meta["n_rejected"] == 0

    def test_all_identical_points_degenerate(self):
        pts = np.tile(np.array([[1.0], [2.0], [3.0]]), (1, 50))
        extent, lo, hi, meta = robust_extent(pts)
        assert np.allclose(extent, 0.0)
        assert np.allclose(lo, hi)
        assert meta["n_points"] == 50
        assert meta["n_rejected"] == 0        # 全都重合，没有可剔除的东西

    def test_empty(self):
        extent, _, _, meta = robust_extent(np.zeros((3, 0)))
        assert np.allclose(extent, 0.0) and meta["n_points"] == 0


# ----------------------------------------------------------------------------
# 2D 框工具
# ----------------------------------------------------------------------------


class TestBoxTools:
    def test_clip_and_order(self):
        assert clip_box_xyxy((-5, 10, 300, 200), (160, 200)) == (0.0, 10.0, 200.0, 160.0)
        # 反序输入也要被理顺
        assert clip_box_xyxy((50, 60, 10, 20), (160, 200)) == (10.0, 20.0, 50.0, 60.0)

    def test_area_iou(self):
        assert box_area((0, 0, 10, 10)) == 100.0
        assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
        assert iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
        # 半重叠：交 50，并 150
        assert iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)

    def test_box_coverage_measures_background_fraction(self):
        mask = np.ones((10, 10), dtype=bool)
        assert box_coverage(mask, (0, 0, 5, 5)) == pytest.approx(1.0)
        half = np.zeros((10, 10), dtype=bool)
        half[0:5, 0:5] = True
        assert box_coverage(half, (0, 0, 10, 10)) == pytest.approx(0.25)
        assert box_coverage(half, (0, 0, 0, 0)) == 0.0


# ----------------------------------------------------------------------------
# 重力方向
# ----------------------------------------------------------------------------


class TestUpAxis:
    def test_too_few_points(self):
        est = estimate_up_axis(np.zeros((3, 8, 8)))
        assert est.reason == "too_few_points" and not est.reliable
        assert est.axis == "-y"          # 退化时仍给一个可用的默认值

    def test_fronto_parallel_band_is_not_horizontal(self):
        """画面下沿若是正对相机的**竖直**面（墙/屏幕），它不是地板。

        此时法线落在水平方向 → 必须判为不可靠，而不是硬算出一个 up。
        这个判据很重要：把墙当地板会**整体翻转** above/below。
        """
        depth = np.full((160, 200), 2.0)
        cloud = pinhole_cloud(depth)
        est = estimate_up_axis(cloud)
        assert not est.reliable
        assert est.reason == "band_not_horizontal"
        assert est.tilt_deg > 60.0

    def test_floor_band_is_reliable(self):
        """下沿是一片水平地面（y = 常数）时应给出 tilt ≈ 0。

        注意深度必须随行变化：若深度恒定，下沿的点只张开 x 一个方向，
        点集退化成一条线，SVD 的零空间不唯一 —— 那是**构造错误**，不是被测行为。
        """
        h, w = 160, 200
        depth = 2.0 + 0.01 * np.arange(h, dtype=np.float64)[:, None] * np.ones((1, w))
        cloud = pinhole_cloud(depth)
        floor_start = int(h * 0.8)
        cloud[1, floor_start:, :] = 1.0        # 下部整体压到同一高度 → 水平面

        est = estimate_up_axis(cloud)
        assert est.reliable and est.reason == "ok"
        assert est.tilt_deg == pytest.approx(0.0, abs=1.0)
        assert est.axis == "-y"
        assert est.n_points >= 200
        # 法线朝上 = 相机系 -y 方向
        assert est.normal is not None and est.normal[1] < 0

    def test_small_tilt_is_reported_not_hidden(self):
        """倾斜 10° 仍判为可靠，但 `tilt_deg` 必须如实报出来 —— 不隐藏、不四舍五入。

        这条测试守的是「诚实声明」：法线是斜的这件事必须能被下游读到，
        因为 `relations.UpAxis` 只能表达轴对齐，斜的那部分只能靠这个数字传递。
        """
        h, w = 160, 200
        cloud = np.zeros((3, h, w))
        rows = np.arange(h, dtype=np.float64)[:, None]
        band = np.arange(h) >= int(h * 0.8)

        # 带内构造一个法线相对 (0,-1,0) 倾斜 10° 的平面：y = -tan10 · z + c
        # （z 必须有跨度，否则平面欠定）
        z = np.broadcast_to(2.0 + 0.01 * rows, (h, w))
        zband = z[band]
        cloud[0][band] = np.broadcast_to((np.arange(w) - w / 2) * 0.01, (h, w))[band]
        cloud[1][band] = -np.tan(np.radians(10.0)) * zband + 1.0
        cloud[2][band] = zband
        cloud[2][~band] = 2.0                  # 带外给有效值；估计器只看带内

        est = estimate_up_axis(cloud)
        assert est.tilt_deg == pytest.approx(10.0, abs=1.5)
        assert est.reliable
        assert est.reason == "ok"


# ----------------------------------------------------------------------------
# 内参与视场
# ----------------------------------------------------------------------------


class TestFov:
    """`check_fov` / `fov_deg` / `intrinsics_matrix`。

    守的是 Phase 0 Step 6 的发现：UniDepth V2 的相机头对 `assets/demo/rgb.png`
    给出 fx=163.7（HFoV 125.8°），真值 fx=518.9（63.3°），横向坐标被放大
    3.17 倍，沙发量出 6.70 m。这个错误不抛异常，所以必须有测试证明
    「它会被拦下来」。
    """

    #: Phase 0 实测的直接回放，用真实数字而不是编的。
    DEMO = (640, 480)
    GT_FX, GT_FY, GT_CX, GT_CY = 518.86, 519.47, 325.58, 253.74
    PRED_FX, PRED_FY, PRED_CX, PRED_CY = 163.74, 163.42, 322.00, 248.11

    def test_fov_deg_matches_known_camera(self):
        # 50 mm 等效（fx≈519 于 640 宽）的水平视场是 63.3°
        assert fov_deg(518.86, 640) == pytest.approx(63.33, abs=0.05)
        # 90° 视场对应 fx = 宽/2
        assert fov_deg(320.0, 640) == pytest.approx(90.0, abs=1e-6)

    def test_fov_deg_rejects_non_positive_focal(self):
        assert np.isnan(fov_deg(0.0, 640))
        assert np.isnan(fov_deg(-10.0, 640))

    def test_intrinsics_matrix_layout(self):
        K = intrinsics_matrix(500.0, 501.0, 320.0, 240.0)
        assert K.shape == (3, 3)
        assert K[0, 0] == 500.0 and K[1, 1] == 501.0
        assert K[0, 2] == 320.0 and K[1, 2] == 240.0
        assert K[2, 2] == 1.0 and K[0, 1] == 0.0 and K[1, 0] == 0.0

    def test_gt_camera_is_plausible(self):
        K = intrinsics_matrix(self.GT_FX, self.GT_FY, self.GT_CX, self.GT_CY)
        fov = check_fov(K, (480, 640))
        assert fov.plausible is True
        assert fov.reason == "ok"
        assert fov.hfov_deg == pytest.approx(63.33, abs=0.05)
        assert fov.vfov_deg == pytest.approx(49.6, abs=0.1)

    def test_predicted_camera_is_rejected(self):
        """实测的预测内参必须被判为不可信 —— 这是整个检查存在的理由。"""
        K = intrinsics_matrix(self.PRED_FX, self.PRED_FY, self.PRED_CX, self.PRED_CY)
        fov = check_fov(K, (480, 640))
        assert fov.plausible is False
        assert fov.reason == "hfov_out_of_range"
        assert fov.hfov_deg == pytest.approx(125.8, abs=0.2)

    def test_lateral_inflation_is_the_ratio_of_focals(self):
        """横向放大倍数 = 真值 fx / 预测 fx，与实测的 3.169 一致。

        这不是复述公式，而是把「视场错 ⟹ 横向尺度错」这条因果链钉住：
        `x = (u - cx)·z / fx`，z 不变、fx 小 3.17 倍 ⟹ x 大 3.17 倍。
        """
        assert self.GT_FX / self.PRED_FX == pytest.approx(3.169, abs=0.005)

    def test_bounds_are_configurable(self):
        K = intrinsics_matrix(self.PRED_FX, self.PRED_FY, 320.0, 240.0)
        assert check_fov(K, (480, 640)).plausible is False
        assert check_fov(K, (480, 640), bounds=(30.0, 130.0)).plausible is True

    def test_non_positive_focal_reported_separately(self):
        """fx<=0 的原因码要能与「视场越界」区分开 —— 两者的修法不同。"""
        fov = check_fov(intrinsics_matrix(0.0, 500.0, 320.0, 240.0), (480, 640))
        assert fov.plausible is False
        assert fov.reason == "non_positive_focal"

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            check_fov(np.eye(4), (480, 640))

    def test_results_are_json_serialisable(self):
        """`as_dict` 要能直接进 `build_meta` 落盘 —— 不能带 numpy 标量。"""
        import json

        K = intrinsics_matrix(self.GT_FX, self.GT_FY, self.GT_CX, self.GT_CY)
        d = check_fov(K, (480, 640)).as_dict()
        assert json.loads(json.dumps(d)) == d
        assert isinstance(d["plausible"], bool)
        assert isinstance(d["reason"], str)

    def test_plausible_band_is_a_sane_interval(self):
        lo, hi = PLAUSIBLE_HFOV_DEG
        assert 0 < lo < hi
        assert lo <= 63.3 <= hi          # 普通相机
        assert not (lo <= 125.8 <= hi)   # 实测的错误值
