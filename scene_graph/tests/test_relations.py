"""`scene_graph.relations` 的单元测试 —— 不需要 GPU、不需要模型、不需要场景图。

这组测试的存在本身就是一条论据：**关系函数可脱离场景图单测**，
而早期基线没有这层 —— 它的「空间判断」是一次模型调用，
既不确定（同一个问题两次答案可能不同）、又需要 GPU、还没法写断言。

坐标系（相机系，米）：x 向右、**y 向下**、z 向前。y 向下是全组测试里最容易搞错的假设，
所以专门有一组用例把它钉住。
"""

from __future__ import annotations

import pytest

from scene_graph.relations import (
    DEFAULT_TOL,
    MissingGeometry,
    UpAxis,
    above,
    below,
    distance_m,
    far,
    front_of,
    behind,
    inside,
    left_of,
    near,
    on,
    pairwise,
    right_of,
)
from scene_graph.schema import BBox3D, Node


# ----------------------------------------------------------------------------
# 构造辅助
# ----------------------------------------------------------------------------


def mk(
    node_id: str,
    label: str = "thing",
    xyz: tuple[float, float, float] = (0.0, 0.0, 2.0),
    size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    *,
    with_bbox: bool = True,
    score: float = 0.8,
) -> Node:
    """造一个几何上自洽的节点：bbox_3d 由中心 ± 半尺寸得到。"""
    x, y, z = xyz
    w, h, l = size
    bbox = None
    if with_bbox:
        bbox = BBox3D(
            min=(x - w / 2, y - h / 2, z - l / 2),
            max=(x + w / 2, y + h / 2, z + l / 2),
        )
    return Node(
        id=node_id,
        label=label,
        score=score,
        bbox_2d=(10.0, 10.0, 50.0, 50.0),
        centroid_3d=(x, y, z),
        extent_3d=(w, h, l),
        bbox_3d=bbox,
    )


# ----------------------------------------------------------------------------
# 坐标系约定
# ----------------------------------------------------------------------------


class TestCoordinateConvention:
    def test_up_axis_default_is_negative_y(self):
        u = UpAxis.parse("-y")
        assert (u.index, u.sign) == (1, -1.0)
        # y 越小越高 → up_coord 越大
        assert u.of((0.0, 0.5, 2.0)) > u.of((0.0, 1.5, 2.0))

    def test_up_axis_horizontal_axes_exclude_the_up_axis(self):
        assert UpAxis.parse("-y").horizontal_axes() == (0, 2)

    def test_up_axis_rejects_unknown_axis(self):
        with pytest.raises(ValueError, match="无法解析"):
            UpAxis.parse("-w")

    def test_above_uses_height_not_raw_y(self):
        """y 向下：上方物体的 y 更小。这条错了，above/below 会整体翻转。"""
        high = mk("high", xyz=(0.0, 0.5, 2.0))   # y=0.5，更靠上
        low = mk("low", xyz=(0.0, 1.5, 2.0))     # y=1.5，更靠下
        assert bool(above(high, low))
        assert not bool(above(low, high))
        assert bool(below(low, high))

    def test_flipping_up_axis_flips_the_verdict(self):
        """把 up 反向，above 的结果必须反过来 —— 证明关系确实读了 up_axis。"""
        a = mk("a", xyz=(0.0, 0.5, 2.0))
        b = mk("b", xyz=(0.0, 1.5, 2.0))
        assert bool(above(a, b, up="-y"))
        assert not bool(above(a, b, up="+y"))


# ----------------------------------------------------------------------------
# 水平方向
# ----------------------------------------------------------------------------


class TestHorizontal:
    def test_left_of_compares_x(self):
        a = mk("a", xyz=(-1.0, 0.0, 2.0))
        b = mk("b", xyz=(1.0, 0.0, 2.0))
        assert bool(left_of(a, b))
        assert not bool(left_of(b, a))

    def test_left_and_right_are_mutually_exclusive_beyond_tolerance(self):
        a = mk("a", xyz=(0.0, 0.0, 2.0))
        b = mk("b", xyz=(1.0, 0.0, 2.0))
        assert bool(left_of(a, b))
        assert not bool(right_of(a, b))

    def test_tolerance_is_a_dead_zone_not_a_bias(self):
        """错开量小于 tol 时，两个方向都不成立 —— 这是「不猜」的体现。"""
        a = mk("a", xyz=(0.00, 0.0, 2.0))
        b = mk("b", xyz=(0.02, 0.0, 2.0))   # 只错开 2 cm < 5 cm
        assert not bool(left_of(a, b))
        assert not bool(right_of(a, b))

    def test_tolerance_boundary_is_exclusive(self):
        tol = 0.05
        a = mk("a", xyz=(0.00, 0.0, 2.0))
        b_exact = mk("b", xyz=(tol, 0.0, 2.0))
        b_just_over = mk("b", xyz=(tol + 1e-6, 0.0, 2.0))
        assert not bool(left_of(a, b_exact, tol))        # 恰好等于 tol → 不算
        assert bool(left_of(a, b_just_over, tol))

    def test_front_of_means_smaller_z(self):
        """相机系 z 向前：z 越小离相机越近。"""
        near_obj = mk("near", xyz=(0.0, 0.0, 1.0))
        far_obj = mk("far", xyz=(0.0, 0.0, 3.0))
        assert bool(front_of(near_obj, far_obj))
        assert bool(behind(far_obj, near_obj))
        assert not bool(front_of(far_obj, near_obj))

    def test_horizontal_relations_are_axis_independent(self):
        """left_of 只看 x，front_of 只看 z —— 另一轴的变化不该影响结论。"""
        a = mk("a", xyz=(-1.0, 0.0, 2.0))
        b_shifted = mk("b", xyz=(1.0, 5.0, 0.1))   # y、z 都变了
        assert bool(left_of(a, b_shifted))
        assert not bool(front_of(a, b_shifted))


# ----------------------------------------------------------------------------
# 距离
# ----------------------------------------------------------------------------


class TestDistance:
    def test_euclidean_not_depth_difference(self):
        """横向错开的两个等深物体，距离必须非零。

        早期基线的 depth 差会给出 0 —— 这正是「depth 相减」不成立的证明。
        """
        a = mk("a", xyz=(0.0, 0.0, 2.0))
        b = mk("b", xyz=(1.0, 0.0, 2.0))   # 同深度，横向 1 m
        v = distance_m(a, b)
        assert v.value == pytest.approx(1.0)
        assert v.metric["delta_z"] == pytest.approx(0.0)

    def test_3d_distance(self):
        a = mk("a", xyz=(0.0, 0.0, 0.0))
        b = mk("b", xyz=(1.0, 2.0, 2.0))
        assert distance_m(a, b).value == pytest.approx(3.0)

    def test_near_and_far_split_at_threshold(self):
        a = mk("a", xyz=(0.0, 0.0, 1.0))
        b = mk("b", xyz=(0.0, 0.0, 1.7))
        assert bool(near(a, b, 1.0))
        assert not bool(far(a, b, 1.0))
        assert bool(far(a, b, 0.5))
        assert not bool(near(a, b, 0.5))

    def test_distance_value_is_float_not_bool(self):
        v = distance_m(mk("a"), mk("b", xyz=(1.0, 0.0, 2.0)))
        assert not v.is_bool
        # bool() 仍然可用，但语义是「非零」——所以文档里提醒读 .value
        assert bool(v)

    def test_distance_is_symmetric(self):
        a = mk("a", xyz=(0.3, -0.2, 1.4))
        b = mk("b", xyz=(-0.9, 0.6, 3.1))
        assert distance_m(a, b).value == pytest.approx(distance_m(b, a).value)


# ----------------------------------------------------------------------------
# on / inside
# ----------------------------------------------------------------------------


class TestContactAndContainment:
    def test_on_when_resting_on_top(self):
        """杯子放在桌上：杯底 ≈ 桌顶，水平投影重叠。"""
        table = mk("table", xyz=(0.0, 0.5, 2.0), size=(1.2, 0.1, 1.2))
        cup = mk("cup", xyz=(0.0, 0.0, 2.0), size=(0.1, 0.1, 0.1))
        # 桌面 top: y = 0.5 - 0.05 = 0.45；杯底 bottom: y = 0.0 + 0.05 = 0.05
        # → 需要重新对齐，见下面的显式构造
        table = mk("table", xyz=(0.0, 0.50, 2.0), size=(1.2, 0.10, 1.2))
        cup = mk("cup", xyz=(0.0, 0.40, 2.0), size=(0.10, 0.10, 0.10))
        # 桌面最高（y 最小）= 0.45；杯底最低（y 最大）= 0.45 → gap_v = 0
        v = on(cup, table)
        assert v.value is True
        assert v.metric["gap_v_m"] == pytest.approx(0.0)

    def test_on_false_when_floating(self):
        table = mk("table", xyz=(0.0, 0.50, 2.0), size=(1.2, 0.10, 1.2))
        floating = mk("cup", xyz=(0.0, 0.10, 2.0), size=(0.10, 0.10, 0.10))
        # 杯底 = 0.15，桌顶 = 0.45 → 悬空 30 cm
        assert not bool(on(floating, table))

    def test_on_false_when_horizontally_offset(self):
        """竖直贴合但水平完全错开 —— 不算「放在上面」。"""
        table = mk("table", xyz=(0.0, 0.50, 2.0), size=(0.4, 0.10, 0.4))
        elsewhere = mk("cup", xyz=(3.0, 0.40, 2.0), size=(0.10, 0.10, 0.10))
        v = on(elsewhere, table)
        assert v.value is False

    def test_inside_contained_object(self):
        shelf = mk("shelf", xyz=(0.0, 0.0, 2.0), size=(1.0, 1.0, 1.0))
        book = mk("book", xyz=(0.0, 0.0, 2.0), size=(0.2, 0.2, 0.2))
        assert bool(inside(book, shelf))

    def test_inside_false_when_protruding(self):
        shelf = mk("shelf", xyz=(0.0, 0.0, 2.0), size=(1.0, 1.0, 1.0))
        big = mk("big", xyz=(0.0, 0.0, 2.0), size=(2.0, 2.0, 2.0))
        assert not bool(inside(big, shelf))

    def test_inside_prefers_the_smaller_object(self):
        """体积更大的那个不该被判为「在里面」。"""
        shelf = mk("shelf", xyz=(0.0, 0.0, 2.0), size=(1.0, 1.0, 1.0))
        book = mk("book", xyz=(0.0, 0.0, 2.0), size=(0.2, 0.2, 0.2))
        assert bool(inside(book, shelf))
        assert not bool(inside(shelf, book))

    def test_missing_bbox_raises_missing_geometry(self):
        """数据不足时抛 MissingGeometry，由外层翻成 DEGENERATE + 「换 anchor」建议。"""
        with_bbox = mk("a", xyz=(0.0, 0.0, 2.0))
        without = mk("b", xyz=(0.0, 0.0, 2.0), with_bbox=False)
        with pytest.raises(MissingGeometry, match="bbox_3d"):
            on(without, with_bbox)
        with pytest.raises(MissingGeometry):
            inside(without, with_bbox)

    def test_centroid_only_relations_still_work_without_bbox(self):
        """缺 bbox 不该让**整对**关系作废 —— 只有需要包围盒的那两项不可用。"""
        a = mk("a", xyz=(-1.0, 0.0, 2.0), with_bbox=False)
        b = mk("b", xyz=(1.0, 0.0, 2.0), with_bbox=False)
        assert bool(left_of(a, b))
        assert distance_m(a, b).value == pytest.approx(2.0)


# ----------------------------------------------------------------------------
# pairwise
# ----------------------------------------------------------------------------


class TestPairwise:
    def test_covers_all_eleven_relations(self):
        a = mk("a", xyz=(-1.0, 0.0, 2.0))
        b = mk("b", xyz=(1.0, 0.0, 2.0))
        assert set(pairwise(a, b)) == {
            "distance", "near", "far", "left_of", "right_of",
            "front_of", "behind", "above", "below", "on", "inside",
        }

    def test_skips_bbox_relations_when_bbox_absent(self):
        a = mk("a", with_bbox=False)
        b = mk("b", xyz=(1.0, 0.0, 2.0), with_bbox=False)
        keys = set(pairwise(a, b))
        assert "on" not in keys and "inside" not in keys
        assert "distance" in keys

    def test_verdicts_carry_evidence(self):
        a = mk("a", xyz=(-0.5, 0.0, 2.0))
        b = mk("b", xyz=(0.5, 0.0, 2.0))
        rels = pairwise(a, b)
        assert rels["left_of"].metric["delta_x"] == pytest.approx(-1.0)
        assert rels["distance"].metric["distance_m"] == pytest.approx(1.0)

    def test_as_edge_fields_is_serializable(self):
        import json

        a = mk("a", xyz=(-0.5, 0.0, 2.0))
        b = mk("b", xyz=(0.5, 0.0, 2.0))
        for verdict in pairwise(a, b).values():
            json.dumps(verdict.as_edge_fields())   # 不抛异常即通过


# ----------------------------------------------------------------------------
# 容差是关系判定的核心参数，不是装饰
# ----------------------------------------------------------------------------


class TestToleranceIsMeaningful:
    def test_relation_flips_when_tolerance_crosses_the_separation(self):
        a = mk("a", xyz=(0.0, 0.0, 2.0))
        b = mk("b", xyz=(0.10, 0.0, 2.0))   # 错开 10 cm
        assert not bool(left_of(a, b, tol=0.15))   # 容差 15 cm > 10 cm → 不判
        assert bool(left_of(a, b, tol=0.05))       # 容差 5 cm < 10 cm → 判
        assert DEFAULT_TOL == 0.05
