#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`evaluation/geometry_metrics.py` 的口径回归测试。

这组测试里有一条是**整个模块的立项理由**，列在最前面：

    test_lateral_and_depth_are_separable

它证明「只动横向」时纵深误差恒为 0、反之亦然 —— 也就是**分解真的在分解**。
如果这条不成立，模块给的就是一个换了名字的 3D 欧氏误差，
而实测（§22）已经证明那样会读反：「尺寸错 2.9 倍」与「纵深几乎没动」同时成立。

其余各组分别守住：零点自洽（尺子没错）、缺几何不许静默排除、
以及三个「看起来是 0 其实是没测」的口径陷阱。
"""

from __future__ import annotations

import math

import pytest

from evaluation import geometry_metrics as gm


# ---------------------------------------------------------------------------
# 立项理由：横向与纵深必须能分开
# ---------------------------------------------------------------------------


class TestDecompositionIsReal:
    def test_lateral_and_depth_are_separable(self):
        """只动横向 → 纵深误差为 0；只动纵深 → 横向误差为 0。

        这是模块存在的理由。不成立的话，本模块只是 3D 欧氏误差的改版。
        """
        gt = [0.0, 0.0, 2.0]

        # 只动横向（x）：纵深必须纹丝不动
        lat_only = gm.centroid_error([0.5, 0.0, 2.0], gt)
        assert lat_only["lateral_m"] == pytest.approx(0.5)
        assert lat_only["depth_m"] == pytest.approx(0.0)

        # 只动纵深（z）：横向必须纹丝不动
        dep_only = gm.centroid_error([0.0, 0.0, 2.5], gt)
        assert dep_only["lateral_m"] == pytest.approx(0.0)
        assert dep_only["depth_m"] == pytest.approx(0.5)

    def test_pythagoras_identity_holds(self):
        """`total² = lateral² + depth²` —— 是勾股，不是加法。"""
        err = gm.centroid_error([0.3, 0.4, 2.0], [0.0, 0.0, 1.0])
        assert err["lateral_m"] == pytest.approx(0.5)       # sqrt(0.09+0.16)
        assert err["depth_m"] == pytest.approx(1.0)
        assert err["total_m"] == pytest.approx(math.sqrt(0.25 + 1.0))
        assert gm.decomposition_residual(err) == pytest.approx(0.0, abs=1e-12)

    def test_total_is_not_the_sum_of_parts(self):
        """显式钉住「不是加法」——写成加法会得到一个更大的数，且不报错。

        ⚠ 例子必须**两个分量都非零**：`lateral=0, depth=1` 时
        `sqrt(0²+1²) = 0+1`，加法与勾股重合，这个例子没有区分度
        （第一版就写成了这样，测试才会红）。
        """
        err = gm.centroid_error([0.3, 0.4, 2.0], [0.0, 0.0, 1.0])
        assert err["lateral_m"] == pytest.approx(0.5)
        assert err["depth_m"] == pytest.approx(1.0)
        assert err["total_m"] == pytest.approx(math.sqrt(1.25))
        assert err["total_m"] < err["lateral_m"] + err["depth_m"]      # 1.118 < 1.5

    def test_signed_depth_keeps_direction(self):
        """有符号纵深：系统性变远与变近必须能分开。"""
        assert gm.centroid_error([0.0, 0.0, 3.0], [0.0, 0.0, 2.0])["signed_dz_m"] == pytest.approx(1.0)
        assert gm.centroid_error([0.0, 0.0, 1.0], [0.0, 0.0, 2.0])["signed_dz_m"] == pytest.approx(-1.0)
        # 而 depth_m（绝对值）两者相同
        a = gm.centroid_error([0.0, 0.0, 3.0], [0.0, 0.0, 2.0])
        b = gm.centroid_error([0.0, 0.0, 1.0], [0.0, 0.0, 2.0])
        assert a["depth_m"] == pytest.approx(b["depth_m"])


# ---------------------------------------------------------------------------
# 零点自洽：预测 = 真值 ⟹ 所有误差恰好为 0
# ---------------------------------------------------------------------------


class TestZeroPointSelfConsistency:
    def test_identical_inputs_give_exact_zero(self):
        """任何指标上线前都要先跑这一条。这不是形式主义：
        本项目已经遇到过一次「尺子错了但没人发现」。"""
        xyz = [1.234, -0.567, 3.891]
        err = gm.centroid_error(xyz, list(xyz))
        assert err["total_m"] == 0.0
        assert err["lateral_m"] == 0.0
        assert err["depth_m"] == 0.0

        ext = [0.9, 0.4, 1.2]
        eerr = gm.extent_error(ext, list(ext))
        assert eerr["l1_m"] == 0.0
        assert eerr["max_axis_m"] == 0.0
        assert eerr["abs_dx_m"] == 0.0 and eerr["abs_dy_m"] == 0.0 and eerr["abs_dz_m"] == 0.0

    def test_identical_point_clouds_give_exact_zero(self):
        pts = [[0.0, 1.0, 2.0], [0.0, 0.5, 1.0], [1.0, 1.5, 2.5]]
        out = gm.point_cloud_error(pts, pts)
        assert out["abs_rel"] == 0.0
        assert out["rmse_m"] == 0.0
        assert out["delta1"] == 1.0
        assert out["delta2"] == 1.0
        assert out["delta3"] == 1.0

    def test_is_self_consistent_flags_tampered_decomposition(self):
        """把一个自洽的行改坏（纵深写成加法），检查器必须抓到。"""
        good = gm.centroid_error([0.3, 0.4, 2.0], [0.0, 0.0, 1.0])
        assert gm.is_self_consistent([good]) == (True, [])

        bad = dict(good)
        bad["total_m"] = bad["lateral_m"] + bad["depth_m"]      # 故意写成加法
        ok, rows = gm.is_self_consistent([good, bad])
        assert ok is False
        assert rows == [1]

    def test_is_self_consistent_skips_non_centroid_rows(self):
        """不是质心误差的行（例如尺寸误差）应当被跳过，而不是被误判。"""
        ok, rows = gm.is_self_consistent([gm.extent_error([1, 1, 1], [1, 1, 1])])
        assert (ok, rows) == (True, [])


# ---------------------------------------------------------------------------
# 缺几何不许静默排除
# ---------------------------------------------------------------------------


class TestMissingGeometryIsNeverSilent:
    def test_centroid_error_rejects_none(self):
        with pytest.raises(TypeError, match="不接受 None"):
            gm.centroid_error(None, [0.0, 0.0, 1.0])
        with pytest.raises(TypeError, match="不接受 None"):
            gm.centroid_error([0.0, 0.0, 1.0], None)

    def test_extent_error_rejects_none(self):
        with pytest.raises(TypeError, match="不接受 None"):
            gm.extent_error(None, [1.0, 1.0, 1.0])

    def test_summarize_requires_n_expected(self):
        """没有 n_expected 就无法区分「测了 5 个」与「只测出 5 个」。"""
        with pytest.raises(TypeError):
            gm.summarize([gm.centroid_error([0, 0, 2], [0, 0, 1])])  # type: ignore[call-arg]

    def test_summarize_counts_missing_and_keeps_denominator(self):
        rows = [gm.centroid_error([0.1, 0.0, 2.0], [0.0, 0.0, 2.0]) for _ in range(5)]
        out = gm.summarize(rows, n_expected=8)
        assert out["n_expected"] == 8
        assert out["n_measured"] == 5
        assert out["n_missing"] == 3
        assert out["missing_rate"] == pytest.approx(3 / 8)

    def test_summarize_rejects_impossible_counts(self):
        with pytest.raises(ValueError, match="小于实际行数"):
            gm.summarize([gm.centroid_error([0, 0, 2], [0, 0, 1])], n_expected=0)

    def test_nan_is_not_silently_dropped_from_n(self):
        """`_stats` 的 `n` 只数有限值 —— 所以「全是 nan」必须表现为 n=0，而不是 0 误差。"""
        out = gm.summarize([{"total_m": float("nan")}], n_expected=1)
        assert out["total_m"]["n"] == 0.0
        assert math.isnan(out["total_m"]["median"])


# ---------------------------------------------------------------------------
# 尺寸误差
# ---------------------------------------------------------------------------


class TestExtentError:
    def test_axis_wise_and_l1(self):
        err = gm.extent_error([1.0, 2.0, 3.0], [1.5, 2.0, 3.5])
        assert err["abs_dx_m"] == pytest.approx(0.5)
        assert err["abs_dy_m"] == pytest.approx(0.0)
        assert err["abs_dz_m"] == pytest.approx(0.5)
        assert err["l1_m"] == pytest.approx(1.0)
        assert err["max_axis_m"] == pytest.approx(0.5)

    def test_signed_exposes_systematic_scaling(self):
        """「尺寸系统性放大 2.9×」这类结论靠有符号值；abs 版本看不出来。"""
        err = gm.extent_error([2.9, 1.0, 1.0], [1.0, 1.0, 1.0])
        assert err["signed_dx_m"] == pytest.approx(1.9)
        assert err["abs_dx_m"] == pytest.approx(1.9)

    def test_relative_is_opt_in(self):
        """默认不出相对误差 —— 薄板的近零轴会让它没有意义。"""
        assert "rel_l1" not in gm.extent_error([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
        out = gm.extent_error([2.0, 1.0, 1.0], [1.0, 1.0, 1.0], relative=True)
        assert out["rel_l1"] == pytest.approx(1.0 / 3.0)

    def test_negative_span_is_rejected(self):
        """负跨度只可能来自 min/max 传反，不能当成「尺寸为负」继续算。"""
        with pytest.raises(ValueError, match="跨度不能为负"):
            gm.extent_error([-1.0, 1.0, 1.0], [1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# 点云指标
# ---------------------------------------------------------------------------


class TestPointCloudError:
    def test_shape_mismatch_is_rejected_not_matched(self):
        """形状不同就报错，不做最近邻匹配 —— 那会引入没有真值可校的误差源。"""
        with pytest.raises(ValueError, match="形状必须一致"):
            gm.point_cloud_error([[0.0, 1.0], [0.0, 0.0], [1.0, 1.0]],
                                 [[0.0], [0.0], [1.0]])

    def test_must_be_3_by_n(self):
        with pytest.raises(ValueError, match="必须是 \\(3,N\\)"):
            gm.point_cloud_error([[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]])

    def test_empty_cloud_reports_n_zero_not_zero_error(self):
        """空点云不是「误差 0」，是「没有可测的量」。"""
        out = gm.point_cloud_error([[], [], []], [[], [], []])
        assert out["n"] == 0
        assert math.isnan(out["abs_rel"])
        assert math.isnan(out["rmse_m"])

    def test_delta_denominator_uses_gt(self):
        """δ 的分母是真值。反了会让「整体放大」的预测看起来更好。"""
        gt = [[0.0, 1.0], [0.0, 0.0], [1.0, 1.0]]        # 半径 1, 1
        pred = [[0.0, 3.0], [0.0, 0.0], [1.0, 3.0]]      # 半径 1, 3
        out = gm.point_cloud_error(pred, gt)
        # 第 1 点比值 1（命中），第 2 点比值 3（不命中 ⟹ δ1 应为 0.5）
        assert out["delta1"] == pytest.approx(0.5)

    def test_abs_rel_uses_radius_not_z(self):
        """用「到相机原点的距离」而不是 z：横向错位必须计入。"""
        gt = [[0.0], [0.0], [1.0]]
        pred = [[1.0], [0.0], [1.0]]      # 横向错 1 m，z 不变
        out = gm.point_cloud_error(pred, gt)
        assert out["abs_rel"] == pytest.approx(math.sqrt(2.0) - 1.0)   # 用 z 会得到 0


# ---------------------------------------------------------------------------
# 关系
# ---------------------------------------------------------------------------


class TestRelationPRF:
    def test_basic_counts(self):
        pred = [("a", "b", "left_of"), ("a", "b", "near")]
        gt = [("a", "b", "left_of"), ("b", "c", "above")]
        out = gm.relation_prf(pred, gt)
        assert out["tp"] == 1 and out["fp"] == 1 and out["fn"] == 1
        assert out["precision"] == pytest.approx(0.5)
        assert out["recall"] == pytest.approx(0.5)
        assert out["f1"] == pytest.approx(0.5)

    def test_empty_prediction_is_nan_not_zero(self):
        """「没有预测」与「预测全错」不是同一件事。"""
        out = gm.relation_prf([], [("a", "b", "near")])
        assert math.isnan(out["precision"])
        assert out["recall"] == 0.0

    def test_empty_gt_is_nan_not_perfect(self):
        out = gm.relation_prf([("a", "b", "near")], [])
        assert math.isnan(out["recall"])
        assert out["precision"] == 0.0

    def test_perfect_match(self):
        s = [("a", "b", "left_of"), ("c", "d", "above")]
        out = gm.relation_prf(s, list(s))
        assert out["f1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 入参校验
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_non_finite_is_rejected_loudly(self):
        """inf/nan 会无声污染所有下游统计 —— 必须在入口挡住。"""
        with pytest.raises(ValueError, match="inf/nan"):
            gm.centroid_error([0.0, 0.0, float("inf")], [0.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="inf/nan"):
            gm.centroid_error([0.0, float("nan"), 1.0], [0.0, 0.0, 1.0])

    def test_wrong_arity_is_rejected(self):
        with pytest.raises(ValueError, match="必须是 3 维"):
            gm.centroid_error([0.0, 0.0], [0.0, 0.0, 1.0])

    def test_summarize_stats_shape(self):
        rows = [
            gm.centroid_error([0.1, 0.0, 2.0], [0.0, 0.0, 2.0]),
            gm.centroid_error([0.3, 0.0, 2.0], [0.0, 0.0, 2.0]),
        ]
        out = gm.summarize(rows, n_expected=2)
        assert out["lateral_m"]["median"] == pytest.approx(0.2)
        assert out["lateral_m"]["max"] == pytest.approx(0.3)
        # np.percentile([0.1, 0.3], 90) = 0.1 + 0.9 × (0.3 − 0.1) = 0.28（线性插值）
        assert out["lateral_m"]["p90"] == pytest.approx(0.28)
