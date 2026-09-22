#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合成感知桥（`dataset/builders/synthetic_perception.py`）的单测。

两组重心，都不在「代码能不能跑」：

**① 投影必须与 `_pixel_rays` 同一套坐标约定。** 夹具的 `_pixel_rays` 把像素中心
取在 `arange + 0.5`，所以投影回去时**不该**再补或减半个像素。偏半格不会报错，
只会让 `box_selector` 整体错位一格 —— 而 `bbox_fallback` 的全部数字都建立在这个框上。
这里的断言写成**精确值**（正对相机的盒子投影框是可解析算出的整数）。

**② 守卫必须响。** 内参不一致、`boxes` 与 `scene` 不同源、注入污染夹具 ——
这三条在真实链路里都是**静默**的（点云照样有、数字照样自洽）。所以本文件
把「不响」当成失败，而不是把「响了」当成加分。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.builders.synthesize_geometry_probe import Box3D, render_scene  # noqa: E402
from dataset.builders.synthetic_perception import (  # noqa: E402
    FIDELITY_NOTES,
    Degradation,
    SyntheticIntrinsicsError,
    SyntheticPerception,
    parse_prompt_labels,
    project_box_xyxy,
)
from scene_graph.builder import BuildConfig, build_scene_graph  # noqa: E402

HW = (120, 160)
INTR = np.array([[120.0, 0.0, 80.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]])

#: 正对相机、中心在光轴上的盒子：近面 z=2.0，远面 z=2.5。
#: 投影框的解析值 = `u ∈ [65, 95]`、`v ∈ [45, 75]`（见类文档里的算法）。
_FRONT_BOX_XYXY = (65.0, 45.0, 95.0, 75.0)


def one_box() -> list[Box3D]:
    return [Box3D("box_1", "box", (-0.25, -0.25, 2.0), (0.25, 0.25, 2.5))]


def two_boxes() -> list[Box3D]:
    """两把同深度、横向错开的盒子（第二个整体靠右 0.7 m）。"""
    return [
        Box3D("box_1", "box", (-0.25, -0.25, 2.0), (0.25, 0.25, 2.5)),
        Box3D("box_2", "box", (0.45, -0.25, 2.0), (0.95, 0.25, 2.5)),
    ]


def bridge(boxes=None, **kw) -> SyntheticPerception:
    """正对相机的单物体场景（除非显式给 `boxes`）。"""
    return SyntheticPerception.from_boxes(
        one_box() if boxes is None else boxes,
        intrinsics=INTR,
        image_hw=HW,
        **kw,
    )


def build(per: SyntheticPerception, *, prompt: str = "box.", **cfg):
    return build_scene_graph(
        per.image,
        perception=per,
        scene_id="t",
        image_id="t",
        config=BuildConfig(prompt=prompt, **cfg),
    )


# ---------------------------------------------------------------------------
# 标签解析
# ---------------------------------------------------------------------------


class TestParsePromptLabels:
    def test_splits_on_periods_and_lowercases(self):
        assert parse_prompt_labels("sofa. chair. table.") == {"sofa", "chair", "table"}

    def test_ignores_empty_segments_and_whitespace(self):
        """`"sofa . chair ."`（点号两侧带空格）必须解析成同一集合。

        `vision/grounding.py` 的坑 ① 说这种写法会被切错 token —— 那是**模型**
        的行为。本函数是**精确集合**语义，两条都要能解析出来，
        否则探针会把一个纯格式差异误报成「prompt 漏类别」。
        """
        assert parse_prompt_labels("sofa .  chair .") == {"sofa", "chair"}
        assert parse_prompt_labels("") == set()
        assert parse_prompt_labels("...") == set()

    def test_is_case_insensitive(self):
        assert parse_prompt_labels("Sofa. CHAIR.") == {"sofa", "chair"}


# ---------------------------------------------------------------------------
# 投影
# ---------------------------------------------------------------------------


class TestProjectBox:
    def test_front_box_projects_to_its_exact_analytic_box(self):
        """正对相机的盒子，投影框是**精确**的解析值。

        偏半个像素是最可能犯的错（`_pixel_rays` 用 `arange + 0.5` 当像素中心，
        很容易顺手再减一次），而它不会报错，只会让框整体挪一格。
        """
        pbox = project_box_xyxy(one_box()[0], intrinsics=INTR)
        assert pbox == _FRONT_BOX_XYXY

    def test_exactness_survives_at_a_different_focal_length(self):
        """换一套内参，投影框仍须精确 —— 钉住「上一格不是巧合」。"""
        intr = np.array([[240.0, 0.0, 80.0], [0.0, 240.0, 60.0], [0.0, 0.0, 1.0]])
        pbox = project_box_xyxy(one_box()[0], intrinsics=intr)
        assert pbox is not None
        x1, y1, x2, y2 = pbox
        # 极值都出现在**近面** z=2.0（|x|/z 在那里最大）
        assert x1 == pytest.approx(240.0 * -0.25 / 2.0 + 80.0)
        assert x2 == pytest.approx(240.0 * 0.25 / 2.0 + 80.0)
        assert y1 == pytest.approx(240.0 * -0.25 / 2.0 + 60.0)
        assert y2 == pytest.approx(240.0 * 0.25 / 2.0 + 60.0)

    def test_corner_behind_the_camera_yields_none_not_a_mirrored_box(self):
        """有角点在相机后方时返回 `None`。

        硬算透视投影会得到一个**镜像**的、看着还挺合理的框 —— 那种框
        会让整个降级路径的数字都错得无声无息。
        """
        crossing = Box3D("box_1", "box", (-0.25, -0.25, -1.0), (0.25, 0.25, 1.0))
        assert project_box_xyxy(crossing, intrinsics=INTR) is None

    def test_rejects_a_malformed_intrinsics_matrix(self):
        with pytest.raises(ValueError, match="3×3"):
            project_box_xyxy(one_box()[0], intrinsics=np.eye(4))


# ---------------------------------------------------------------------------
# 构造守卫
# ---------------------------------------------------------------------------


class TestConstructionGuards:
    def test_mismatched_boxes_against_the_scene_are_rejected(self):
        """用一份盒子渲染、又用另一份投影 —— 框和掩码会互相错位，两边都不报错。

        ⚠ 两个 id 集合**相同**，所以拦住它的是**坐标**比对那一关，
        而不是集合比对那一关。这里用坐标不一致来测，正是因为集合一致
        才是更常见的犯错方式（同一份清单被原地改了数字）。
        """
        scene = render_scene(one_box(), intrinsics=INTR, image_hw=HW)
        moved = [Box3D("box_1", "box", (-0.30, -0.30, 2.0), (0.30, 0.30, 2.5))]
        with pytest.raises(ValueError, match="不一致"):
            SyntheticPerception(scene, moved)

    def test_a_box_list_with_a_different_id_set_is_rejected(self):
        scene = render_scene(two_boxes(), intrinsics=INTR, image_hw=HW)
        with pytest.raises(ValueError, match="不是同一次渲染"):
            SyntheticPerception(scene, one_box())

    def test_a_matching_box_list_is_accepted(self):
        """★ 反向：同源的 `boxes`/`scene` 必须**不抛**。

        没有这一条，「**永远抛**」的实现也能让上面那两条测试通过 ——
        revert-check 会以「仍然绿」的形式把这种反转暴露出来（本文件第一次
        跑 R3 时就是这样：往 `if not (...)` 里塞 `False` 反而让条件恒真）。
        只测「该抛时抛」是一半的测试，「该不抛时不抛」是另一半。
        """
        scene = render_scene(one_box(), intrinsics=INTR, image_hw=HW)
        per = SyntheticPerception(scene, one_box())
        assert per.n_foreground == 1
        assert per.n_background == 0

    def test_background_boxes_are_excluded_from_the_origin_check(self):
        """背景盒子刻意不进 `scene.gt_box`，所以同源校验必须放过它们。

        漏了这一点，这条守卫会在**每一个**带背景的正常场景上误报 ——
        而误报的守卫会被人顺手关掉，连带关掉它真正要守的东西。
        """
        boxes = one_box() + [
            Box3D("__wall__", "wall", (-6.0, -6.0, 6.6), (6.0, 6.0, 6.8),
                  is_background=True)
        ]
        per = bridge(boxes)
        assert per.n_foreground == 1
        assert per.n_background == 1

    def test_unknown_degradation_target_is_rejected(self):
        """拼错 id 的后果是「注入没生效」，而报告里那一栏会显示一切正常。"""
        with pytest.raises(ValueError, match="degrade"):
            bridge(degrade={"box_9": Degradation(mask_empty=True)})

    def test_image_size_must_match_the_fixture_grid(self):
        from PIL import Image

        with pytest.raises(ValueError, match="不一致"):
            SyntheticPerception.from_boxes(
                one_box(), intrinsics=INTR, image_hw=HW,
                image=Image.new("RGB", (99, 99)),
            )

    def test_missing_label_is_rejected(self):
        scene = render_scene(one_box(), intrinsics=INTR, image_hw=HW)
        scene.meta["boxes"] = []  # 抹掉夹具携带的 label
        with pytest.raises(ValueError, match="没有 label"):
            SyntheticPerception(scene, one_box())


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------


class TestDetect:
    def test_returns_only_the_labels_the_prompt_asked_for(self):
        per = bridge(two_boxes())
        assert len(per.detect(per.image, "box.")) == 2
        assert per.detect(per.image, "chair.") == []
        assert per.detect(per.image, "chair. table.") == []

    def test_box_equals_the_projection(self):
        per = bridge()
        det = per.detect(per.image, "box.")[0]
        assert det.box_xyxy == _FRONT_BOX_XYXY
        assert det.label == "box"

    def test_thresholds_do_not_filter_and_the_score_is_fixed(self):
        """阈值被记录但**不参与筛选** —— 分数是固定值，用它筛是假动作。"""
        per = bridge()
        assert len(per.detect(per.image, "box.", box_threshold=0.99)) == 1
        assert per.detect(per.image, "box.")[0].score == per.score
        assert per.detect_calls[-1]["box_threshold"] == 0.30


# ---------------------------------------------------------------------------
# segment
# ---------------------------------------------------------------------------


class TestSegment:
    def test_matches_by_box_not_by_position_in_the_batch(self):
        """按框 IoU 匹配 —— 顺序反过来也必须拿到对应的掩码。

        如果实现成「按调用顺序返回」，只要 builder 哪天改了批次顺序，
        掩码就会静默地串到别的物体上。
        """
        per = bridge(two_boxes())
        b1 = per.projected_box("box_1")
        b2 = per.projected_box("box_2")
        masks = per.segment(per.image, [b2, b1])
        assert np.array_equal(masks[0], per.scene.masks["box_2"])
        assert np.array_equal(masks[1], per.scene.masks["box_1"])
        # 两个掩码必须真的不同，否则「匹配」这个断言等于没测
        assert not np.array_equal(masks[0], masks[1])

    def test_unmatched_box_yields_an_all_false_mask(self):
        """框匹配不到物体 = 「SAM2 在这个框里什么也没分出来」，这是**被测对象**。"""
        per = bridge(two_boxes())
        masks = per.segment(per.image, [(0.0, 0.0, 1.0, 1.0)])
        assert masks.shape == (1, HW[0], HW[1])
        assert not masks.any()
        assert per.segment_matches[-1] == [None]

    def test_mask_empty_makes_the_segment_blank(self):
        per = bridge(degrade={"box_1": Degradation(mask_empty=True)})
        masks = per.segment(per.image, [per.projected_box("box_1")])
        assert not masks.any()
        # 匹配本身仍然成功 —— 「匹配不到」与「匹配到了但分割失败」是两件事
        assert per.segment_matches[-1] == ["box_1"]

    def test_returned_mask_content_equals_the_fixture_mask(self):
        """返回的数组必然是拷贝（要堆成 `(N,H,W)`，形状决定的），
        但**内容**必须逐位等于夹具掩码。

        别把「不是视图」误当成「内容不同」：前者是 numpy 的必然，
        后者才是缺陷。
        """
        per = bridge()
        masks = per.segment(per.image, [per.projected_box("box_1")])
        assert np.array_equal(masks[0], per.scene.masks["box_1"])
        assert masks[0] is not per.scene.masks["box_1"]
        # 改返回值不该影响夹具 —— 这正是「拷贝」这个事实的保护作用
        masks[0][:] = False
        assert per.scene.masks["box_1"].any()


# ---------------------------------------------------------------------------
# lift 与两条守卫
# ---------------------------------------------------------------------------


class TestLift:
    def test_matching_camera_K_is_accepted_and_source_is_provided(self):
        per = bridge()
        field = per.lift(per.image, camera_K=INTR.copy())
        assert field.intrinsics_source == "provided"
        assert field.grid_hw == HW
        assert field.image_hw == HW
        assert np.array_equal(field.intrinsics, INTR)
        assert field.fov is None, "fov 故意留空，让 builder 只走一条 check_fov 路径"

    def test_a_different_camera_K_raises(self):
        """★ 守卫 ①：传进来的内参与点云赖以生成的内参不一致。

        真实链路拿不到这个校验（它不知道真值），但也不该因此在本桥上放行。
        内参错 3.17 倍在真实链路是**静默**的（§20.3）。
        """
        per = bridge()
        wrong = INTR.copy()
        wrong[0, 0] *= 1.1
        with pytest.raises(SyntheticIntrinsicsError, match="不是同一份"):
            per.lift(per.image, camera_K=wrong)

    def test_require_camera_K_raises_when_none_is_given(self):
        """★ 守卫 ②：配置写了 known_intrinsics，但那条信息没到达感知层。"""
        per = bridge(require_camera_K=True)
        with pytest.raises(SyntheticIntrinsicsError, match="没有到达感知层"):
            per.lift(per.image, camera_K=None)

    def test_without_the_requirement_a_missing_camera_K_is_allowed(self):
        """不要求校验时缺传是允许的 —— 但 `intrinsics_source` 依然是 `provided`。

        这不是好消息，是**警告**：该字段在合成桥里是结构性事实
        （点云就是用 `scene.intrinsics` 生成的），不能用来判断
        「内参条件是否生效」。探针的 `[⑥ 6c]` 把这一点单列出来。
        """
        per = bridge(require_camera_K=False)
        field = per.lift(per.image, camera_K=None)
        assert field.intrinsics_source == "provided"
        assert per.lift_calls[-1] is None

    def test_lift_records_the_camera_K_it_received(self):
        per = bridge(require_camera_K=True)
        per.lift(per.image, camera_K=INTR)
        assert per.lift_calls[-1] is not None
        assert np.array_equal(per.lift_calls[-1], INTR)


# ---------------------------------------------------------------------------
# 点云注入
# ---------------------------------------------------------------------------


class TestPointsInjection:
    def test_without_injection_the_points_are_shared(self):
        per = bridge()
        assert per.points_chw() is per.scene.points_chw

    def test_nan_injection_copies_and_never_pollutes_the_fixture(self):
        """★ 注入**不得**污染夹具：污染了会让后续所有测量一起偏，且无声。

        ⚠ `np.array_equal` 默认把 nan 当成不相等，所以这里必须 `equal_nan=True`
        —— 不加这一项，一个**正确**的实现也会被判失败（本文件第一版就踩了）。
        """
        per = bridge(degrade={"box_1": Degradation(nan_depth_in_box=True)})
        before = per.scene.points_chw.copy()
        pts = per.points_chw()
        assert np.isnan(pts).any()
        assert per.scene.points_chw is not pts
        assert np.array_equal(per.scene.points_chw, before, equal_nan=True)
        # 注入的不变量：nan 集合**只增不减**，且有效位置上逐位不变。
        # ⚠ 别写成「注入后不该有 nan」—— 夹具在**没有表面命中的方向**上本来就
        #   是 nan（单物体无背景时 `no_hit_px = 18300`）。这条踩过一次。
        nan_before = np.isnan(before)
        nan_after = np.isnan(pts)
        assert not (nan_before & ~nan_after).any(), "注入不该把 nan 变回有效点"
        keep = ~nan_before
        assert (keep & nan_after).any(), "注入没有把任何本来有效的点变成 nan ⟹ 没生效"
        still = keep & ~nan_after
        assert np.array_equal(pts[still], before[still])

    def test_the_hole_covers_the_object_s_own_visible_pixels(self):
        """注入区域是**投影框**（`box_selector` 会向外取整）⟹ 必然包含物体自己。

        这条就是 `no_valid_points` 不需要与 `mask_empty` 组合的原因：
        掩码内的点会一起变 nan。
        """
        per = bridge(degrade={"box_1": Degradation(nan_depth_in_box=True)})
        sel = per.box_selector_for("box_1")
        assert sel.any()
        assert (sel & per.scene.masks["box_1"]).any()
        pts = per.points_chw().reshape(3, -1)[:, sel.reshape(-1)]
        assert pts.size > 0 and np.isnan(pts).all()

    def test_box_pad_grows_the_projected_box(self):
        per = bridge(degrade={"box_1": Degradation(box_pad_px=4.0)})
        x1, y1, x2, y2 = per.projected_box("box_1")
        assert (x1, y1, x2, y2) == (61.0, 41.0, 99.0, 79.0)


# ---------------------------------------------------------------------------
# 两条降级路径：各由对应开关触发（集成级，走真实 builder）
# ---------------------------------------------------------------------------


class TestDegradationIsReal:
    def test_baseline_is_clean(self):
        """基线唯一允许出现的警告是「重力方向估计不可靠」。

        单物体 + 无背景时 `estimate_up_axis` 必然报 `too_few_points` ——
        这是**预期行为**（点太少），不是缺陷。所以断言写成「除了它之外一条
        都不许有」，而不是「没有警告」：后者会逼着场景加背景来凑，
        而背景会让 `no_hit_px = 0`，把「没有表面命中的方向」这个观测条件抹掉。
        """
        per = bridge()
        res = build(per)
        assert res.stats["n_nodes"] == 1
        assert res.stats["n_fallbacks"] == 0
        assert res.stats["n_dropped"] == 0
        assert res.scene.nodes[0].centroid_source == "mask"
        others = [w for w in res.warnings if "重力方向" not in w]
        assert others == []
        assert res.stats["up_axis_reliable"] is False
        assert res.stats["up_axis_reason"] == "too_few_points"

    def test_mask_empty_triggers_bbox_fallback(self):
        per = bridge(degrade={"box_1": Degradation(mask_empty=True)})
        res = build(per)
        assert res.stats["n_fallbacks"] == 1
        fb = res.stats["fallbacks"][0]
        assert fb["object_id"] == "box_1"
        assert fb["reason"] == "mask_too_small"
        assert fb["n_mask_points"] == 0
        assert fb["n_box_points"] > 0
        assert res.stats["n_dropped"] == 0, "掩码空但框内有点 ⟹ 不该被剔除"
        assert res.scene.nodes[0].centroid_source == "bbox_fallback"

    def test_nan_depth_triggers_no_valid_points(self):
        per = bridge(degrade={"box_1": Degradation(nan_depth_in_box=True)})
        res = build(per)
        assert res.stats["n_nodes"] == 0
        assert res.stats["n_fallbacks"] == 0, "掩码与框内都无点 ⟹ 不走 fallback"
        assert res.stats["n_dropped"] == 1
        assert res.stats["dropped"][0]["reason"] == "no_valid_points"
        assert any("已从场景图剔除" in w for w in res.warnings)

    def test_the_two_switches_do_not_impersonate_each_other(self):
        """`mask_empty` 不会开出 `no_valid_points`，反之亦然。

        两条路径的触发条件必须互不冒充 —— 否则「测的是哪条」这件事本身
        就说不清了，报告里那一栏也就没法归因。
        """
        a = build(bridge(degrade={"box_1": Degradation(mask_empty=True)}))
        assert a.stats["n_fallbacks"] == 1 and a.stats["n_dropped"] == 0
        b = build(bridge(degrade={"box_1": Degradation(nan_depth_in_box=True)}))
        assert b.stats["n_fallbacks"] == 0 and b.stats["n_dropped"] == 1


# ---------------------------------------------------------------------------
# 自证：落盘的场景必须能看出它来自合成真值
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_declares_the_fixture_origin_and_the_fidelity_limits(self):
        """一份由本桥建出的 `scene.json` 与真实照片建出的长得一样，
        而两者的可信范围差得很远 ⟹ `build_meta["perception"]` 必须自证来源。"""
        per = bridge()
        s = per.stats()
        assert s["kind"] == "SyntheticPerception"
        assert s["fidelity"] == list(FIDELITY_NOTES)
        assert FIDELITY_NOTES, "保真度声明不能是空的 —— 空声明比没有声明更容易被误读"
        assert s["lift_intrinsics_source"] == "provided"
        assert s["n_objects_in_fixture"] == 1

    def test_stats_reaches_build_meta(self):
        per = bridge()
        res = build(per)
        meta = res.stats["perception"]
        assert meta["kind"] == "SyntheticPerception"
        assert meta["degradations"] == {}
        assert meta["segment_matches"] == [["box_1"]]

    def test_stats_records_the_degradation_that_was_applied(self):
        per = bridge(degrade={"box_1": Degradation(mask_empty=True, box_pad_px=3.0,
                                                   note="unit-test")})
        d = per.stats()["degradations"]["box_1"]
        assert d == {
            "mask_empty": True,
            "box_pad_px": 3.0,
            "nan_depth_in_box": False,
            "note": "unit-test",
        }
