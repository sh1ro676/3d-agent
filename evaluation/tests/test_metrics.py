"""`evaluation/metrics.py` 的单元测试 —— 纯标准库，零 torch、零 GPU、零联网。

## 这个文件真正在守什么

不是一个公式，而是**一个口径**。子指标的每一处细节都能改出 1–3 个百分点的差别，
而论文表格只给一位小数 —— 口径错了根本看不出来。所以这里逐条钉住：

* `int` 题解析失败 = **记 0 分**（分母含它），不是「该题被排除」。
  这是原实现里最容易读反的一处（`xxx_n += 1` 在 try 之前）。
* float 的 MRA 分母是**全部** float 题，含解析失败的；解析失败时
  10 档阈值一档都不命中，但不从分母里消失。
* `str` 要按**答案取值**再切成 yes/no 与 multi-choice 两类 ——
  只按 `answer_type` 分会把两类合成一类。
* Total = 按题数加权的 micro 平均。这条**不是猜的**：见
  `test_paper_totals_are_reproduced`，它用论文自己 8 行数字验算，
  残差 ≤0.2pp 才算过。
"""

from __future__ import annotations

import pytest

from evaluation.metrics import (
    ACTUAL_CLASS_COUNTS,
    MRA_THRESHOLDS,
    PAPER_OMNI3D_LEADERBOARD,
    compute_metrics,
    total_micro,
    verify_total_aggregation,
)


def rec(at, gt, pred):
    return {"answer_type": at, "ground_truth": gt, "prediction": pred}


class TestAnswerTypeSplit:
    def test_yes_no_vs_multi_choice_use_the_answer_value_not_only_the_type(self):
        # 两题的 answer_type 都是 str，但一类是 yes/no，一类是多选 ——
        # 只按 answer_type 分组会把它们混在一起，指标就再也拆不开。
        r = compute_metrics([
            rec("str", "yes", "yes"),
            rec("str", "chair", "chair"),
        ])
        assert r["counts"]["yes_no"] == 1
        assert r["counts"]["multi_choice"] == 1
        assert r["submetrics"]["yes_no"] == 100.0
        assert r["submetrics"]["multi_choice"] == 100.0

    def test_unknown_answer_type_is_ignored(self):
        r = compute_metrics([rec("bool", True, "yes")])
        assert r["n_questions"] == 0
        assert all(v is None for v in r["submetrics"].values())

    def test_prediction_is_lowercased_for_yes_no(self):
        r = compute_metrics([rec("str", "yes", "YES")])
        assert r["submetrics"]["yes_no"] == 100.0


class TestNumericCount:
    def test_exact_match_only(self):
        r = compute_metrics([rec("int", "3", "3"), rec("int", "3", "4")])
        assert r["counts"]["numeric_count"] == 2
        assert r["submetrics"]["numeric_count"] == 50.0

    def test_unparseable_prediction_counts_as_wrong_not_excluded(self):
        """★ 核心口径。如果实现把它当成「排除」，准确率会变成 100%（错 1 倍）。"""
        r = compute_metrics([rec("int", "3", "3"), rec("int", "5", "many")])
        assert r["counts"]["numeric_count"] == 2      # 分母含它
        assert r["submetrics"]["numeric_count"] == 50.0
        assert r["unparseable_predictions"]["numeric_count"] == 1

    def test_decimal_string_is_not_an_int(self):
        # int("3.7") 抛异常 —— 这是原实现行为，不是 bug。
        # 如果这里改成 float() 再取整，numeric_count 的数字会明显变高。
        r = compute_metrics([rec("int", "3", "3.7")])
        assert r["submetrics"]["numeric_count"] == 0.0
        assert r["unparseable_predictions"]["numeric_count"] == 1


class TestMrA:
    def test_exact_float_hits_every_threshold(self):
        r = compute_metrics([rec("float", "2.0", "2.0")])
        assert r["submetrics"]["numeric_other_mra"] == 100.0

    def test_relative_error_picks_the_matching_thresholds(self):
        # |2.0-2.18|/2.0 = 0.09 → 命中 0.5..0.1 共 9 档，不命中 0.05
        r = compute_metrics([rec("float", "2.0", "2.18")])
        assert r["mra_hits"]["0.1"] == 1
        assert r["mra_hits"]["0.05"] == 0
        assert r["submetrics"]["numeric_other_mra"] == pytest.approx(90.0, abs=0.05)

    def test_unparseable_float_stays_in_denominator(self):
        """★ 另一处核心口径：分母是全部 float 题，不是「能解析的那些」。"""
        r = compute_metrics([rec("float", "2.0", "2.0"), rec("float", "3.0", "about three")])
        assert r["counts"]["numeric_other"] == 2
        # 命中 100 与 0 各一题 → 平均 50
        assert r["submetrics"]["numeric_other_mra"] == 50.0

    def test_zero_ground_truth_does_not_divide_by_zero(self):
        r = compute_metrics([rec("float", "0.0", "0.0")])
        assert r["submetrics"]["numeric_other_mra"] == 0.0

    def test_threshold_list_matches_vadar(self):
        assert MRA_THRESHOLDS == (0.5, 0.45, 0.40, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05)
        assert len(MRA_THRESHOLDS) == 10


class TestTotalMicro:
    def test_weighted_by_class_size(self):
        subs = {"numeric_count": 100.0, "numeric_other_mra": 0.0,
                "yes_no": 0.0, "multi_choice": 0.0}
        # 只有 count 类满分，且该类占 70/501
        assert total_micro(subs, ACTUAL_CLASS_COUNTS) == pytest.approx(
            100.0 * 70 / 501, abs=0.01)

    def test_missing_class_is_skipped_not_counted_as_zero(self):
        """某一类无题时，它的分母要退出，而不是按 0 分计入 ——
        否则小子集（例如只有 yes/no 题）的总分会被人为压低。"""
        subs = {"numeric_count": None, "numeric_other_mra": None,
                "yes_no": 100.0, "multi_choice": None}
        assert total_micro(subs, {"yes_no": 5}) == 100.0

    def test_all_none_returns_none(self):
        assert total_micro({"numeric_count": None}, {}) is None


class TestAggregationVerification:
    def test_paper_totals_are_reproduced(self):
        """★ 口径判据：用论文自己的 8 行数字验算 Total。

        这不是「拟合」—— 题数是从 parquet 数出来的（70/270/75/86），
        公式是写死的一条。8 行残差同时 ≤0.2pp，说明口径对；
        任何一行超了都说明公式形式错，而不是「舍入误差」。
        """
        r = verify_total_aggregation()
        assert r["verdict"] == "micro_by_question_count"
        assert r["residual_max_pp"] <= 0.2
        assert r["n_methods_same_run"] == 8
        assert r["counts"] == ACTUAL_CLASS_COUNTS

    def test_external_methods_are_excluded_and_flagged(self):
        """ViperGPT / VisProg 是别人跑的批次，题数分布不同。
        必须**被排除并留下痕迹** —— 悄悄丢掉会让「8 行」这个数字无从追溯。"""
        r = verify_total_aggregation()
        excluded = {d["method"] for d in r["excluded_non_same_run"]}
        assert excluded == {"ViperGPT", "VisProg"}
        for d in r["excluded_non_same_run"]:
            assert abs(d["residual_pp"]) > 2.0     # 确实对不上，所以排除是有理由的

    def test_weights_sum_to_one(self):
        r = verify_total_aggregation()
        # weights 落盘时四舍五入到 6 位，4 个值的累计舍入误差可达 4e-6 ——
        # 所以容差取 1e-5 而不是 1e-6（后者是在测浮点，不是在测口径）
        assert abs(sum(r["weights"].values()) - 1.0) < 1e-5

    def test_leaderboard_rows_are_well_formed(self):
        for name, row in PAPER_OMNI3D_LEADERBOARD.items():
            assert len(row) == 6, name
            assert 0 <= row[4] <= 100, name

    def test_deprecated_derivation_is_marked_deprecated(self):
        """旧的「最小二乘反推」函数会给出负权重，必须显式标记废弃，
        否则有人会引用里面的数字。"""
        from evaluation.metrics import derive_total_weights

        d = derive_total_weights()
        assert d["deprecated"] is True
        assert d["use_instead"] == "verify_total_aggregation"


class TestExactMatch:
    def test_exact_match_is_stricter_than_mra(self):
        """VADAR 的 results.txt 里那个 Accuracy 是逐题字符串相等；
        数值题几乎不可能命中。这里确认它**不会**被误当成 MRA。"""
        r = compute_metrics([rec("float", "2.0", "2.00")])
        assert r["exact_match_accuracy"] == 0.0
        assert r["submetrics"]["numeric_other_mra"] == 100.0
