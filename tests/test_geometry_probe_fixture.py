#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合成几何夹具的单测。

这组测试的重心有两处，都不在「代码能不能跑」：

**① 渲染物理必须精确可解析。** 一个正对相机的盒子，其可见质心必须**恰好**
落在它的近面上，可见像素数必须**恰好**等于投影面积（`fx·w/z × fy·h/z`）。
这两条是后续一切数字的地基 —— 渲染悄悄偏一点，误差分解表就整体偏一点，
而且不会有任何东西报错。所以这里断言的是**精确值**，不是「大约」。

**② 扰动必须只改它声称要改的那一维。** `intrinsics_scale` 只许动横向、
`object_translate_z` 只许动纵深、`object_translate_x` 只许动**指定物体**。
方向搞反的扰动会产出一条看着很漂亮的响应曲线，而它证明的是别的东西。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.builders.synthesize_geometry_probe import (  # noqa: E402
    PERTURBATIONS,
    Box3D,
    apply_perturbation,
    default_intrinsics,
    default_probe_boxes,
    default_scene_boxes,
    render_scene,
    visibility_report,
    visible_geometry,
)

HW = (120, 160)
INTR = np.array([[120.0, 0.0, 80.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]])


def _front_box() -> Box3D:
    """正对相机、中心在光轴上的 0.5×0.5×0.5 盒子，近面 z=2.0。"""
    return Box3D("box_1", "box", (-0.25, -0.25, 2.0), (0.25, 0.25, 2.5))


# ---------------------------------------------------------------------------
# 渲染物理
# ---------------------------------------------------------------------------


class TestRenderPhysics:
    def test_front_face_centroid_lands_exactly_on_the_near_plane(self):
        """正对相机的盒子，可见质心 = 近面中心（精确）。

        x/y 必须**恰好**是 0：任何偏差都意味着射线或主点算错了，
        而这种错误会均匀地污染后面所有物体的质心。
        """
        scene = render_scene([_front_box()], intrinsics=INTR, image_hw=HW)
        gv = scene.gt_visible["box_1"]
        assert gv["centroid_3d"][0] == pytest.approx(0.0)
        assert gv["centroid_3d"][1] == pytest.approx(0.0)
        assert gv["centroid_3d"][2] == pytest.approx(2.0)

    def test_visible_pixel_count_equals_projected_area(self):
        """像素数必须**恰好**等于 `(fx·w/z)·(fy·h/z)` —— 离散化在这里是精确的。"""
        scene = render_scene([_front_box()], intrinsics=INTR, image_hw=HW)
        # fx=120, w=0.5, z=2.0 → 30 px；fy 同 → 30 px；共 900
        assert scene.gt_visible["box_1"]["n_points"] == 30 * 30
        assert int(scene.masks["box_1"].sum()) == 900

    def test_visible_extent_shows_only_the_front_face(self):
        """可见的只是一个正面：z 跨度必须为 0，x/y 跨度是面的尺寸。

        这条是「GT 取可见表面」这个决定的直接后果，也是它与 `gt_box` 的分歧处：
        `gt_box` 的 z 跨度是 0.5，而可见面的是 0。
        """
        scene = render_scene([_front_box()], intrinsics=INTR, image_hw=HW)
        gv = scene.gt_visible["box_1"]["extent_3d"]
        gb = scene.gt_box["box_1"]["extent_3d"]
        assert gv[2] == pytest.approx(0.0)                 # 只看得见正面
        assert gv[0] == pytest.approx(0.5, abs=0.02)       # 面宽（像素中心采样差半格）
        assert gv[1] == pytest.approx(0.5, abs=0.02)
        assert gb[2] == pytest.approx(0.5)                 # 盒子本体的纵深是真实存在的

    def test_visible_and_box_centroids_differ_by_construction(self):
        """两层真值必须在**有意义的方向上**不同，否则这个分层是装饰。

        正面可见 ⟹ 可见质心在近面上、盒子质心在近面与远面中间。
        """
        scene = render_scene([_front_box()], intrinsics=INTR, image_hw=HW)
        assert scene.gt_visible["box_1"]["centroid_3d"][2] == pytest.approx(2.0)
        assert scene.gt_box["box_1"]["centroid_3d"][2] == pytest.approx(2.25)

    def test_camera_inside_box_uses_exit_point(self):
        """相机落在盒子内部时，第一个可见点是**出射点**而不是进入点。

        少写这个分支，盒子套住相机的那一帧会整片消失（`t_enter < 0`）。
        """
        box = Box3D("shell_1", "shell", (-0.3, -0.3, -0.3), (0.3, 0.3, 0.3))
        scene = render_scene([box], intrinsics=INTR, image_hw=HW)
        # 出射点在 +z 面上（z=0.3），不是 z=-0.3
        assert scene.gt_visible["shell_1"]["centroid_3d"][2] == pytest.approx(0.3)

    def test_unhit_pixels_are_nan_not_zero(self):
        """未命中像素必须是 nan，不是 0。

        0 会被下游当成「相机处的有效点」，让一个本该空着的像素
        把中位数拉飞；nan 会被 `valid_point_selector` 正确剔掉。
        """
        box = Box3D("small_1", "small", (-0.05, -0.05, 2.0), (0.05, 0.05, 2.1))
        scene = render_scene([box], intrinsics=INTR, image_hw=HW)
        pts = scene.points_chw
        hit = scene.masks["small_1"]
        assert np.isnan(pts[:, ~hit]).all()
        assert np.isfinite(pts[:, hit]).all()
        # 零值只可能出现在「相机原点」，而这个场景里没有任何点在那里
        assert not (pts[2][hit] == 0.0).any()

    def test_zbuffer_keeps_the_nearest_object(self):
        """近处的盒子遮住远处的：后者可见像素为 0。"""
        near = Box3D("near_1", "box", (-0.25, -0.25, 1.5), (0.25, 0.25, 2.0))
        far = Box3D("far_1", "box", (-0.25, -0.25, 3.0), (0.25, 0.25, 3.5))
        scene = render_scene([far, near], intrinsics=INTR, image_hw=HW)
        assert int(scene.masks["near_1"].sum()) > 0
        assert int(scene.masks["far_1"].sum()) == 0

    def test_empty_boxes_rejected(self):
        with pytest.raises(ValueError, match="boxes 为空"):
            render_scene([], intrinsics=INTR, image_hw=HW)

    def test_bad_intrinsics_rejected(self):
        with pytest.raises(ValueError, match="3×3"):
            render_scene([_front_box()], intrinsics=np.eye(2), image_hw=HW)


class TestBoxValidation:
    def test_inverted_bounds_rejected(self):
        """max 必须逐轴大于 min —— 退化的轴是「不可测」，不是「很薄的物体」。"""
        with pytest.raises(ValueError, match="逐轴大于"):
            Box3D("b", "b", (0.0, 0.0, 1.0), (1.0, 1.0, 1.0))

    def test_arity_checked(self):
        with pytest.raises(ValueError, match="3 元组"):
            Box3D("b", "b", (0.0, 0.0), (1.0, 1.0, 1.0))

    def test_centre_and_extent(self):
        b = Box3D("b", "b", (-1.0, -2.0, 3.0), (1.0, 0.0, 4.0))
        assert b.centre == pytest.approx(np.array([0.0, -1.0, 3.5]))
        assert b.extent == pytest.approx(np.array([2.0, 2.0, 1.0]))


# ---------------------------------------------------------------------------
# 场景体检
# ---------------------------------------------------------------------------


class TestVisibilityReport:
    def test_catches_object_out_of_frame(self):
        """体检必须抓出「跑出画面」的物体 —— 这正是第一版默认场景发生的事。

        判据是 `visible_frac`（可见像素 ÷ 单独渲染时的像素数）。
        完全出界的物体单独渲染也是 0 像素，所以另用 `off_screen` 标记。
        """
        # 一个远离光轴的盒子：在 z=2 处 x 偏了 10 m，远在 67° 视场之外
        stray = Box3D("stray_1", "box", (9.9, -0.25, 2.0), (10.1, 0.25, 2.5))
        rep = visibility_report([stray], intrinsics=INTR, image_hw=HW)
        assert rep["objects"][0]["off_screen"] is True
        assert rep["objects"][0]["px_alone"] == 0

    def test_reports_occlusion(self):
        """被完全遮挡的物体：单独渲染有像素、场景里没有 ⟹ visible_frac = 0。"""
        near = Box3D("near_1", "box", (-1.0, -1.0, 1.5), (1.0, 1.0, 2.0))
        far = Box3D("far_1", "box", (-0.25, -0.25, 3.0), (0.25, 0.25, 3.5))
        rep = visibility_report([near, far], intrinsics=INTR, image_hw=HW)
        by_id = {o["object_id"]: o for o in rep["objects"]}
        assert by_id["far_1"]["px_alone"] > 0
        assert by_id["far_1"]["px_in_scene"] == 0
        assert by_id["far_1"]["visible_frac"] == pytest.approx(0.0)

    def test_default_scene_passes_its_own_gate(self):
        """默认场景必须过自己设的门槛：无出界、可见像素足够、画面无空洞。

        这条守着的是「默认场景被改坏」——改一个坐标就会让某个物体跑出画面，
        而它的表现是「几何误差突然变大了」。
        """
        rep = visibility_report(default_scene_boxes(), intrinsics=default_intrinsics(),
                                image_hw=(240, 320))
        assert all(not o["off_screen"] for o in rep["objects"])
        assert rep["worst_visible_frac"] >= 0.70
        assert rep["min_px_in_scene"] >= 700
        # 有背景之后，画面里不该再有「没有任何表面」的方向
        assert rep["no_hit_px"] == 0

    def test_background_boxes_are_not_objects(self):
        """背景只提供深度：不出现在 masks，也不出现在两层真值里。"""
        scene = render_scene(default_scene_boxes(), intrinsics=default_intrinsics(),
                             image_hw=(120, 160))
        assert "__back_wall__" not in scene.masks
        assert "__floor__" not in scene.masks
        assert "__back_wall__" not in scene.gt_visible
        assert set(scene.masks) == {b.object_id for b in default_probe_boxes()}
        assert scene.meta["background_ids"] == ["__back_wall__", "__floor__"]

    def test_background_makes_every_pixel_covered(self):
        """背景的意义之一：每个像素都有深度，与真实照片一致。"""
        with_bg = render_scene(default_scene_boxes(), intrinsics=default_intrinsics(),
                               image_hw=(120, 160))
        without_bg = render_scene(default_probe_boxes(), intrinsics=default_intrinsics(),
                                  image_hw=(120, 160))
        assert with_bg.meta["no_hit_px"] == 0
        assert without_bg.meta["no_hit_px"] > 0
        # 物体本身可见性不受影响（背景在物体后面，不挡）
        for oid in with_bg.masks:
            assert int(with_bg.masks[oid].sum()) == int(without_bg.masks[oid].sum())


# ---------------------------------------------------------------------------
# 扰动：只许改它声称要改的那一维
# ---------------------------------------------------------------------------


class TestPerturbations:
    def setup_method(self):
        self.scene = render_scene([_front_box()], intrinsics=INTR, image_hw=HW)

    def test_none_is_identity(self):
        pts, masks = apply_perturbation(self.scene, "none")
        assert np.allclose(pts, self.scene.points_chw, equal_nan=True)
        assert np.array_equal(masks["box_1"], self.scene.masks["box_1"])

    def test_intrinsics_scale_moves_lateral_only(self):
        """内参缩放只改横向：z 分毫不动。

        这是 §22 那条机制的复现 —— 「模型自预测内参 vs 外部给定内参」
        造成的横向按比例缩放，而纵深**不遵守**这个关系。
        """
        pts, _ = apply_perturbation(self.scene, "intrinsics_scale", k=2.0)
        base = self.scene.points_chw
        hit = self.scene.masks["box_1"]
        assert np.allclose(pts[2][hit], base[2][hit])              # z 不变
        assert np.allclose(pts[0][hit], base[0][hit] * 2.0)        # x 翻倍
        assert np.allclose(pts[1][hit], base[1][hit] * 2.0)

    def test_depth_scale_moves_everything(self):
        pts, _ = apply_perturbation(self.scene, "depth_scale", k=1.5)
        base = self.scene.points_chw
        hit = self.scene.masks["box_1"]
        assert np.allclose(pts[2][hit], base[2][hit] * 1.5)

    def test_translate_x_touches_only_the_named_object(self):
        """整片平移会连背景一起挪，于是「物体相对背景动了」完全看不出来。"""
        two = render_scene(
            [_front_box(), Box3D("box_2", "box", (0.6, -0.25, 2.0), (1.1, 0.25, 2.5))],
            intrinsics=INTR, image_hw=HW)
        pts, _ = apply_perturbation(two, "object_translate_x", object_id="box_1", dx_m=0.3)
        m1 = two.masks["box_1"]
        m2 = two.masks["box_2"]
        assert np.allclose(pts[0][m1], two.points_chw[0][m1] + 0.3)
        assert np.allclose(pts[0][m2], two.points_chw[0][m2])      # 另一个物体不动

    def test_translate_z_changes_depth_only(self):
        pts, _ = apply_perturbation(self.scene, "object_translate_z", object_id="box_1", dz_m=0.5)
        hit = self.scene.masks["box_1"]
        base = self.scene.points_chw
        assert np.allclose(pts[2][hit], base[2][hit] + 0.5)
        assert np.allclose(pts[0][hit], base[0][hit])

    def test_grow_and_shrink_are_opposite(self):
        grown, gm = apply_perturbation(self.scene, "mask_grow", n_px=2)
        shrunk, sm = apply_perturbation(self.scene, "mask_shrink", n_px=2)
        base = int(self.scene.masks["box_1"].sum())
        assert int(gm["box_1"].sum()) > base
        assert int(sm["box_1"].sum()) < base

    def test_grow_changes_the_measured_extent(self):
        """膨胀会把背景点拉进来 ⟹ 尺寸被撑大。方向必须对得上。

        ⚠ 这条测试**必须有背景**（墙/地板）。没有背景时，膨胀出来的像素落在
        nan 上、被 `select_points` 剔掉，几何量纹丝不动 —— 第一版就是这样失败的，
        而原因不是扰动写错，是**场景缺了背景**。
        """
        scene = render_scene(default_scene_boxes(), intrinsics=default_intrinsics(),
                             image_hw=(240, 320))
        base = scene.gt_visible["sofa_1"]["extent_3d"]
        _, gm = apply_perturbation(scene, "mask_grow", n_px=4)
        gv = visible_geometry(scene.points_chw, gm["sofa_1"], scene.grid_hw)
        assert gv["n_points"] > scene.gt_visible["sofa_1"]["n_points"]
        assert gv["extent_3d"][0] > base[0]
        assert gv["extent_3d"][1] > base[1]

    def test_depth_noise_is_reproducible_with_seed(self):
        a, _ = apply_perturbation(self.scene, "depth_noise", sigma_rel=0.05, seed=7)
        b, _ = apply_perturbation(self.scene, "depth_noise", sigma_rel=0.05, seed=7)
        c, _ = apply_perturbation(self.scene, "depth_noise", sigma_rel=0.05, seed=8)
        assert np.allclose(a, b, equal_nan=True)
        assert not np.allclose(a, c, equal_nan=True)

    def test_unknown_perturbation_is_rejected(self):
        with pytest.raises(ValueError, match="未知扰动"):
            apply_perturbation(self.scene, "no_such_thing")

    def test_translate_requires_object_id(self):
        with pytest.raises(ValueError, match="需要 object_id"):
            apply_perturbation(self.scene, "object_translate_x", dx_m=0.1)

    def test_translate_rejects_unknown_object(self):
        with pytest.raises(ValueError, match="没有物体"):
            apply_perturbation(self.scene, "object_translate_x", object_id="nope", dx_m=0.1)

    def test_non_positive_scales_rejected(self):
        for kind in ("intrinsics_scale", "depth_scale"):
            with pytest.raises(ValueError, match="必须为正"):
                apply_perturbation(self.scene, kind, k=0.0)

    def test_every_perturbation_is_documented(self):
        """扰动清单里的每一项都必须能一句话说清它模拟现实里的什么。

        没有解释的扰动只能产出「无法解释的响应曲线」。
        """
        for kind, desc in PERTURBATIONS.items():
            assert desc.strip(), f"{kind} 缺少一句话说明"
            apply_perturbation(self.scene, kind, k=1.1, n_px=1, sigma_rel=0.01,
                               dx_m=0.01, dz_m=0.01, object_id="box_1")


# ---------------------------------------------------------------------------
# 自洽性
# ---------------------------------------------------------------------------


class TestSelfConsistency:
    def test_geometry_is_deterministic(self):
        """同一输入必须给出同一几何 —— 否则误差分解表不可重放。"""
        scene = render_scene([_front_box()], intrinsics=INTR, image_hw=HW)
        a = visible_geometry(scene.points_chw, scene.masks["box_1"], HW)
        b = visible_geometry(scene.points_chw, scene.masks["box_1"], HW)
        assert a["n_points"] == b["n_points"]
        assert np.allclose(a["centroid_3d"], b["centroid_3d"])
        assert np.allclose(a["extent_3d"], b["extent_3d"])

    def test_unperturbed_prediction_equals_gt_visible(self):
        """无扰动时「预测」与 `gt_visible` 必须逐位相同。

        这是整个探针的零点：它成立，才轮到「有扰动时的偏差」有意义。
        不成立就说明 GT 与预测用了两套口径。
        """
        scene = render_scene(default_scene_boxes(), intrinsics=default_intrinsics(),
                             image_hw=(240, 320))
        pts, masks = apply_perturbation(scene, "none")
        for oid, gt in scene.gt_visible.items():
            got = visible_geometry(pts, masks[oid], scene.grid_hw)
            assert got["n_points"] == gt["n_points"]
            assert np.allclose(got["centroid_3d"], gt["centroid_3d"])
            assert np.allclose(got["extent_3d"], gt["extent_3d"])


# ---------------------------------------------------------------------------
# 朝向：把「轴对齐口径」从背景噪声变成一个可测的量
# ---------------------------------------------------------------------------


def _offaxis_box(yaw_deg: float) -> Box3D:
    """偏轴的立方盒。**偏轴很关键**：它让侧面也可见，量级才接近真实照片。

    刻意不用轴上盒子（`_front_box`）：轴上盒子的可见面只有正对那一个，
    轴对齐跨度这件事在它身上几乎不显形（它的 z 跨度恰好是 0）。
    """
    return Box3D("box_1", "box", (0.35, 0.0, 2.05), (0.85, 0.5, 2.55),
                 yaw_rad=float(np.radians(yaw_deg)))


class TestOrientation:
    def test_zero_yaw_is_the_identity_rotation(self):
        """`yaw_rad = 0` 必须是严格的单位旋转 —— 它是「朝向可关闭」的凭证。"""
        box = _offaxis_box(0.0)
        assert np.array_equal(box.rotation, np.eye(3))
        assert box.aabb_extent == pytest.approx(box.extent, abs=1e-12)

    def test_zero_yaw_render_is_bit_identical(self):
        """`yaw_rad = 0` 的渲染必须与「不给朝向」**逐位一致**。

        加朝向是一个新自由度，不该顺手改掉旧结果 —— 否则此前所有数字
        都要重跑，而重跑后的差异无法归因到朝向。
        """
        with_yaw = render_scene([_offaxis_box(0.0)], intrinsics=INTR, image_hw=HW)
        without = render_scene(
            [Box3D("box_1", "box", (0.35, 0.0, 2.05), (0.85, 0.5, 2.55))],
            intrinsics=INTR, image_hw=HW,
        )
        assert np.array_equal(np.nan_to_num(with_yaw.points_chw),
                              np.nan_to_num(without.points_chw))
        assert bool((with_yaw.masks["box_1"] == without.masks["box_1"]).all())
        assert with_yaw.meta["n_rotated"] == 0
        assert without.meta["renderer"] == "analytic-obb-raycast"

    def test_centre_and_body_extent_are_invariant_under_yaw(self):
        """绕自身中心旋转 ⟹ 中心与自身边长都不动。

        中心动了 = 旋转参照点用错；边长动了 = 「尺寸」会随朝向漂移，
        而那种漂移看起来与真实的尺度误差一模一样。
        """
        ref = _offaxis_box(0.0)
        for deg in (0.0, 17.0, 45.0, 90.0, 123.0):
            box = _offaxis_box(deg)
            assert box.centre == pytest.approx(ref.centre, abs=0.0)
            assert np.array_equal(box.extent, ref.extent)

    def test_rotation_is_a_rigid_transform(self):
        """旋转必须保距保角：`RᵀR = I`、`det R = 1`。"""
        R = _offaxis_box(37.0).rotation
        assert R.T @ R == pytest.approx(np.eye(3))
        assert float(np.linalg.det(R)) == pytest.approx(1.0)

    def test_aabb_extent_matches_the_analytic_formula(self):
        """斜放的轴对齐跨度有闭式解：`x = Lx|cosθ| + Lz|sinθ|`、`y = Ly`、
        `z = Lx|sinθ| + Lz|cosθ|`。

        有了它，「测得尺寸偏大」里**属于口径的那一部分**可以解析算出来，
        不必再靠估计误差去解释。
        """
        Lx, Ly, Lz = 0.5, 0.5, 0.5
        for deg in (0.0, 15.0, 30.0, 45.0, 60.0, 90.0):
            th = float(np.radians(deg))
            want = np.array([Lx * abs(np.cos(th)) + Lz * abs(np.sin(th)),
                             Ly,
                             Lx * abs(np.sin(th)) + Lz * abs(np.cos(th))])
            assert _offaxis_box(deg).aabb_extent == pytest.approx(want)

    def test_yaw_90_swaps_x_and_z_in_the_aabb(self):
        """转 90° ⟹ 轴对齐跨度里的 x / z 互换，y 不动。"""
        flat = Box3D("b", "box", (-0.2, 0.0, 2.0), (0.3, 0.5, 2.8))       # Lx=.5 Ly=.5 Lz=.8
        turned = Box3D("b", "box", (-0.2, 0.0, 2.0), (0.3, 0.5, 2.8),
                       yaw_rad=float(np.radians(90.0)))
        a, b = flat.aabb_extent, turned.aabb_extent
        assert b == pytest.approx([a[2], a[1], a[0]])
        assert b[1] == pytest.approx(a[1])

    def test_aabb_is_never_smaller_than_the_body_extent(self):
        """轴对齐跨度 ⊇ 自身边长，等号只在 yaw = 0 / 90 / 180 / 270 成立。

        这是「斜放必然高估」的一般形式：拿轴对齐口径的数字去比物体真实尺寸，
        **必然**得到偏大 —— 与算法好坏无关。
        """
        for deg in (0.0, 5.0, 30.0, 45.0, 75.0, 89.0, 90.0):
            box = _offaxis_box(deg)
            assert (box.aabb_extent >= box.extent - 1e-12).all()
            if deg in (0.0, 90.0):
                assert box.aabb_extent == pytest.approx(box.extent)

    def test_all_corners_lie_inside_the_aabb(self):
        box = _offaxis_box(37.0)
        lo, hi = box.aabb()
        cs = box.corners()
        assert cs.shape == (8, 3)
        assert (cs >= lo - 1e-12).all()
        assert (cs <= hi + 1e-12).all()

    def test_axis_aligned_extent_overestimates_when_tilted(self):
        """**这条是加朝向的全部理由**：斜放 ⟹ 测得的轴对齐尺寸大于物体真实尺寸。

        实测（Lx = Lz = 0.5 的偏轴盒，fx = 120 / 160 px）：
        yaw=0 时测得 **0.496 ≤ 0.5**；yaw=45° 时测得 **0.688 > 0.5**，
        且不超过解析上界 **0.707**。
        没有朝向这一个自由度时，「尺寸偏大」只能被记在算法头上。
        """
        def measured_x(deg: float) -> float:
            sc = render_scene([_offaxis_box(deg)], intrinsics=INTR, image_hw=HW)
            return float(np.asarray(sc.gt_visible["box_1"]["extent_3d"])[0])

        body_x = float(_offaxis_box(0.0).extent[0])
        flat, tilted = measured_x(0.0), measured_x(45.0)
        upper = float(_offaxis_box(45.0).aabb_extent[0])

        assert flat <= body_x + 1e-9, "轴对齐时不该出现口径高估"
        assert tilted > body_x + 0.1, "斜放必须显出轴对齐口径的高估"
        assert tilted <= upper + 1e-9, "测得值不可能超过整个盒子的轴对齐跨度"

    def test_gt_box_exposes_both_extent_conventions(self):
        """两层口径都要拿得到 —— 只给一个，写报告的人就只能猜。

        `extent_3d`（自身轴）与 `aabb_extent`（轴对齐）在斜放时不同，
        而下游 `robust_extent` 用的是后者。
        """
        sc = render_scene([_offaxis_box(30.0)], intrinsics=INTR, image_hw=HW)
        gb = sc.gt_box["box_1"]
        th = np.radians(30.0)
        assert gb["extent_3d"] == pytest.approx([0.5, 0.5, 0.5])
        assert gb["aabb_extent"][0] == pytest.approx(0.5 * (np.cos(th) + np.sin(th)))
        assert gb["aabb_extent"][0] > gb["extent_3d"][0]
        assert gb["yaw_rad"] == pytest.approx(th)
        assert sc.meta["n_rotated"] == 1

    def test_self_consistency_holds_with_rotation(self):
        """朝向打开时，尺子自检仍必须为零。

        这是「加朝向没有破坏口径同源」的证据：`gt_visible` 与「预测」
        都走 `vision.geometry` 的同一份实现。
        """
        boxes = [
            Box3D(b.object_id, b.label, b.min_xyz, b.max_xyz, b.is_background,
                  yaw_rad=float(np.radians(23.0)))
            for b in default_scene_boxes()
        ]
        scene = render_scene(boxes, intrinsics=default_intrinsics(), image_hw=(240, 320))
        assert scene.meta["n_rotated"] > 0
        pts, masks = apply_perturbation(scene, "none")
        for oid, gt in scene.gt_visible.items():
            got = visible_geometry(pts, masks[oid], scene.grid_hw)
            assert got["n_points"] == gt["n_points"]
            assert np.allclose(got["centroid_3d"], gt["centroid_3d"])
            assert np.allclose(got["extent_3d"], gt["extent_3d"])
