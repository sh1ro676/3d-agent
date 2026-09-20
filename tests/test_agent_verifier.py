"""`agents/verifier.py` 的单测 —— 零模型成本、零 GPU。

这个文件最想钉住的一件事：

    **「写了证据」与「证据对得上」是两条不同的指标。**

`submit()` 只保证前者。而基线臂最典型的失败恰恰是后者 ——
程序每一步都"成功"、`evidence` 也填了，最终交出的数却**在整条 trace 里找不到出处**
（实测里 998.65 / 1048 这一类）。

所以这里一半的用例在设计"差点就通过"的情形：数只出现在证据字符串里、
数只出现在 id 里（`sofa_1` 里的 1）、数差一个数量级……
这些都必须被区分开，而不是笼统地判个"有证据就行"。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.executor import Submission  # noqa: E402
from agents.verifier import (  # noqa: E402
    LEVELS,
    Check,
    Verdict,
    collect_numbers,
    collect_pools,
    verify,
)
from scene_graph.schema import Node, SceneGraph  # noqa: E402


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


def row(tool: str, value=None, *, ok=True, evidence=None, args=None, error=None) -> dict:
    return {
        "tool": tool,
        "args": dict(args or {}),
        "result": {
            "ok": ok,
            "value": value,
            "evidence": dict(evidence or {}),
            "error": error,
            "meta": {"tool": tool},
        },
    }


def submission(answer, *, evidence=("calculate_distance(a, b) = 2.5 m",),
               targets=(), unknown=(), abstained=False) -> Submission:
    return Submission(
        answer=answer, answer_type=None,
        target_ids=tuple(targets), evidence=tuple(evidence),
        unknown_targets=tuple(unknown), abstained=abstained,
    )


@pytest.fixture
def scene() -> SceneGraph:
    return SceneGraph(
        scene_id="s", image_id="rgb.png", up_axis="-y",
        nodes=(Node(id="chair_1", label="chair", centroid_3d=(0.0, 0.0, 1.0)),
               Node(id="door_1", label="door", centroid_3d=(-1.0, 0.0, 1.4))),
    )


DISTANCE_TRACE = [row("calculate_distance", 2.5024,
                      evidence={"a": "chair_1", "b": "door_1", "distance_m": 2.5024})]


# ---------------------------------------------------------------------------
# 没有 submit 就没有答案
# ---------------------------------------------------------------------------


class TestNoSubmission:
    def test_none_is_unsupported_not_skipped(self):
        """「没有答案可校验」本身就是一个结论，不能变成"跳过校验"。"""
        v = verify(None, trace=DISTANCE_TRACE)
        assert v.level == "unsupported" and v.ok is False
        assert v.failed() == ("has_submission",)

    def test_empty_trace_with_a_submission_still_checks_the_rest(self):
        v = verify(submission(1.0), trace=[])
        assert "tool_was_used" in v.failed()


# ---------------------------------------------------------------------------
# 数值答案的出处
# ---------------------------------------------------------------------------


class TestNumericBacking:
    def test_answer_matching_a_tool_value_is_supported(self, scene):
        v = verify(submission(2.5024), trace=DISTANCE_TRACE, scene=scene)
        assert v.level == "supported" and v.matched_from == "value"
        assert v.ok is True

    def test_answer_only_in_evidence_is_supported_but_weaker(self):
        """只落在证据/参数里也算通过（对得上），但要如实记下来是弱一档。"""
        trace = [row("calculate_distance", None, evidence={"distance_m": 2.5024})]
        v = verify(submission(2.5024), trace=trace)
        assert v.level == "supported" and v.matched_from == "evidence"

    def test_rounded_evidence_still_matches(self):
        """证据里的数常被 `round(x, 4)` 过，容差必须吃得住。"""
        trace = [row("calculate_distance", 2.5024, evidence={"distance_m": 2.5024})]
        assert verify(submission(2.50243), trace=trace).level == "supported"

    def test_a_number_about_nowhere_is_unsupported(self):
        """★ 这就是 998.65 那一类 —— 答案是凭空出现的。"""
        v = verify(submission(998.65), trace=DISTANCE_TRACE)
        assert v.level == "unsupported"
        assert "numeric_backed" in v.failed(hard_only=True)

    def test_wrong_magnitude_is_not_forgiven(self):
        """差一个数量级不算"差不多" —— 容差是 1e-3 相对，不是 10%。"""
        assert verify(submission(25.024), trace=DISTANCE_TRACE).level == "unsupported"

    def test_the_tolerance_boundary_is_tight_by_design(self):
        """容差是 **1e-3 相对**，只为吸收 `round()`，不是为了宽容地「近似对」。

            2.5030 vs 2.5024  → 差 0.024% → **通过**（这是同一个量，只是被取整过）
            2.55   vs 2.5024  → 差 1.9%   → **不通过**（这是另一个答案）

        把边界写进测试，是因为「容差多少」直接决定误判率：
        放宽到 10% 会让"差不多"的答案蒙混过关，收紧到 1e-6 会把所有取整过的
        证据判成假阴性 —— 两个方向都会让这一层失去意义。
        """
        assert verify(submission(2.5030), trace=DISTANCE_TRACE).level == "supported"
        assert verify(submission(2.55), trace=DISTANCE_TRACE).level == "unsupported"

    def test_answer_exactly_zero_matches_zero(self):
        trace = [row("calculate_angle", 0.0)]
        assert verify(submission(0.0), trace=trace).level == "supported"

    def test_nan_answer_is_unsupported(self):
        v = verify(submission(float("nan")), trace=DISTANCE_TRACE)
        assert v.level == "unsupported" and "answer_finite" in v.failed(hard_only=True)

    def test_inf_answer_is_unsupported(self):
        assert verify(submission(float("inf")), trace=DISTANCE_TRACE).level == "unsupported"

    def test_bool_answer_is_not_treated_as_a_number(self):
        """`True == 1` —— 静默当成数值会让「判成了真假」伪装成「算出了一个数」。"""
        v = verify(submission(True), trace=[row("query_relation", 1.0)])
        assert "answer_finite" not in [c.name for c in v.checks]
        assert v.level == "supported"          # 走的是文本/布尔那条路


class TestIdDigitsAreNotEvidence:
    """★ `sofa_1` 里的 1 不是量 —— 不排除它，一个巧合的答案会被判成 supported。"""

    def test_number_1_does_not_match_a_sofa_1_id(self):
        trace = [row("list_objects", [{"object_id": "sofa_1", "label": "sofa"}]),
                 row("calculate_distance", 3.0)]
        v = verify(submission(1.0), trace=trace)
        assert v.level == "unsupported"

    def test_id_digits_are_stripped_before_extraction(self):
        values, evidence = collect_numbers([row("get_object", {"object_id": "chair_12"})])
        assert 12 not in values

    def test_real_numbers_next_to_ids_are_still_collected(self):
        values, _ = collect_numbers(
            [row("find_nearest", [{"object_id": "chair_1", "distance_m": 2.5}])])
        assert 2.5 in values and 1 not in values


# ---------------------------------------------------------------------------
# 工具调用与证据
# ---------------------------------------------------------------------------


class TestToolAndEvidence:
    def test_only_failed_tool_calls_is_unsupported(self):
        trace = [row("get_3d_position", None, ok=False,
                     error={"code": "NOT_IN_SCENE"})]
        v = verify(submission(1.0), trace=trace)
        assert "tool_was_used" in v.failed(hard_only=True)

    def test_a_failure_before_a_success_is_fine(self):
        trace = [row("get_3d_position", None, ok=False, error={"code": "NOT_IN_SCENE"}),
                 row("calculate_distance", 2.5024)]
        assert verify(submission(2.5024), trace=trace).level == "supported"

    def test_missing_evidence_is_unsupported(self):
        v = verify(submission(2.5024, evidence=()), trace=DISTANCE_TRACE)
        assert "has_evidence" in v.failed(hard_only=True)

    def test_evidence_count_is_reported(self):
        v = verify(submission(2.5024, evidence=("a", "b")), trace=DISTANCE_TRACE)
        chk = {c.name: c for c in v.checks}["has_evidence"]
        assert chk.ok and "2 条" in chk.detail


# ---------------------------------------------------------------------------
# 指认的 id
# ---------------------------------------------------------------------------


class TestTargets:
    def test_unknown_target_is_a_soft_failure(self, scene):
        """幻觉 id 不致命（答案本身仍有效），但必须降级到 weak 并说清楚。"""
        v = verify(submission(2.5024, targets=("chair_9",), unknown=("chair_9",)),
                   trace=DISTANCE_TRACE, scene=scene)
        assert v.level == "weak"
        assert v.ok is True                      # weak 仍然算"交出了答案"
        assert "targets_in_scene" in v.failed()

    def test_clean_targets_keep_it_supported(self, scene):
        v = verify(submission(2.5024, targets=("chair_1", "door_1")),
                   trace=DISTANCE_TRACE, scene=scene)
        assert v.level == "supported"

    def test_no_targets_is_not_a_failure_but_is_noted(self, scene):
        v = verify(submission(2.5024), trace=DISTANCE_TRACE, scene=scene)
        chk = {c.name: c for c in v.checks}["targets_in_scene"]
        assert chk.ok and "没有指认" in chk.detail

    def test_targets_missing_from_the_graph_are_checked_against_the_scene(self, scene):
        """`unknown_targets` 是提交时的快照；有场景时再核对一次真实图。"""
        v = verify(submission(2.5024, targets=("chair_9",)), trace=DISTANCE_TRACE, scene=scene)
        assert "targets_exist" in v.failed()

    def test_scene_check_is_skipped_without_a_scene(self):
        v = verify(submission(2.5024, targets=("chair_1",)), trace=DISTANCE_TRACE)
        assert "targets_exist" not in [c.name for c in v.checks]


# ---------------------------------------------------------------------------
# 弃答是一条独立的结论
# ---------------------------------------------------------------------------


class TestAbstain:
    def test_abstain_is_its_own_level(self):
        v = verify(submission("unknown", evidence=("图里没有 door",), abstained=True),
                   trace=[row("find_object", None, ok=False, error={"code": "NOT_FOUND"})])
        assert v.level == "abstained" and v.ok is True and v.supported is False

    def test_abstain_is_not_confused_with_a_wrong_answer(self):
        """「如实说答不了」与「编了个数」在指标上必须是两件事（§13.3(5)）。"""
        abstained = verify(submission("unknown", abstained=True),
                           trace=[row("find_object", 0.0)])
        wrong = verify(submission("unknown"), trace=[row("find_object", 0.0)])
        assert abstained.level == "abstained"
        assert wrong.level != "abstained"

    def test_abstain_does_not_require_a_tool_use(self):
        assert verify(submission("unknown", abstained=True), trace=[]).level == "abstained"


# ---------------------------------------------------------------------------
# 文本答案
# ---------------------------------------------------------------------------


class TestTextAnswers:
    def test_a_text_answer_present_in_the_trace_is_supported(self):
        trace = [row("single_object", {"object_id": "chair_1", "label": "chair"})]
        assert verify(submission("chair"), trace=trace).level == "supported"

    def test_a_text_answer_absent_from_the_trace_is_only_weak(self):
        """字符串比对容易有假阴性（大小写、归一化），不该因此判 unsupported。"""
        trace = [row("single_object", {"object_id": "chair_1", "label": "chair"})]
        v = verify(submission("sofa"), trace=trace)
        assert v.level == "weak" and v.ok is True
        assert "text_backed" in v.failed()

    def test_case_difference_is_tolerated(self):
        trace = [row("single_object", {"object_id": "chair_1", "label": "Chair"})]
        assert verify(submission("chair"), trace=trace).level == "supported"


# ---------------------------------------------------------------------------
# 结构
# ---------------------------------------------------------------------------


class TestReportShape:
    def test_levels_are_declared(self):
        assert LEVELS == ("supported", "weak", "unsupported", "abstained")

    def test_to_dict_is_json_serialisable(self, scene):
        import json

        v = verify(submission(2.5024, targets=("chair_1",)), trace=DISTANCE_TRACE, scene=scene)
        json.dumps(v.to_dict(), ensure_ascii=False)

    def test_numbers_are_capped_in_the_report(self):
        trace = [row("list_objects", list(range(200)))]
        v = verify(submission(-1.0), trace=trace)
        assert len(v.to_dict()["numbers"]) == 32
        assert v.to_dict()["n_numbers"] == 200

    def test_summary_line_names_the_failed_checks(self):
        v = verify(submission(998.65), trace=DISTANCE_TRACE)
        assert "numeric_backed" in v.summary_line()

    def test_hard_and_soft_failures_are_separable(self, scene):
        v = verify(submission(2.5024, targets=("nope_9",), unknown=("nope_9",)),
                   trace=DISTANCE_TRACE, scene=scene)
        assert v.failed(hard_only=True) == ()
        assert set(v.failed()) >= {"targets_in_scene", "targets_exist"}

    def test_a_hard_failure_dominates_a_soft_one(self, scene):
        v = verify(submission(998.65, targets=("nope_9",), unknown=("nope_9",)),
                   trace=DISTANCE_TRACE, scene=scene)
        assert v.level == "unsupported"

    def test_check_dataclass_round_trips(self):
        c = Check(name="x", ok=False, detail="d", hard=False)
        assert c.to_dict() == {"name": "x", "ok": False, "detail": "d", "hard": False}

    def test_verdict_ok_property(self):
        assert Verdict(level="supported").ok is True
        assert Verdict(level="weak").ok is True
        assert Verdict(level="abstained").ok is True
        assert Verdict(level="unsupported").ok is False


# ---------------------------------------------------------------------------
# 派生量：答案不是某个工具返回值，而是由它算出来的
# ---------------------------------------------------------------------------

#: 4 幅画的宽度。均值 0.445 不是其中任何一个 —— 正是「派生量」的形态。
MEAN_TRACE = [
    row("find_object",
        [{"object_id": "picture_%d" % i, "extent_m": {"w": w}}
         for i, w in enumerate([0.31, 0.52, 0.48, 0.47])],
        evidence={"label": "picture", "count": 4}),
]

#: 「场景里 9 个物体」—— 人口数由显式字段给出。
NINE_TRACE = [row("list_objects",
                  [{"object_id": "o_%d" % i, "label": "thing"} for i in range(9)],
                  evidence={"returned": 9, "total_objects": 9, "total_in_scene": 9})]


class TestDerivedAnswers:
    """★ 组合题的答案大多是**派生量**：均值落在两个数之间、计数是个新整数。

    原口径要求「答案逐字落在 trace 的数上」，于是把**答对的**也扔了 ——
    实测 10 道组合题里有 3 道（30%）被判 unsupported。加工具治不了这个。

    这一组用例同时钉住两件事：新口径**接住了**派生量，且它**没有**放宽到
    「差不多就行」（998.65 那条线不能因为改口径而丢）。
    """

    def test_a_mean_is_accepted_but_only_as_a_weak_tier(self):
        v = verify(submission(0.445), trace=MEAN_TRACE)
        assert v.matched_from.startswith("derived:")
        assert v.level == "weak"           # 对得上，但没逐字出现 ⟹ 不冒充 supported
        assert v.ok is True                # 它仍然**是一个答案**，不是编的

    def test_a_mean_outside_the_interval_is_still_unsupported(self):
        assert verify(submission(998.65), trace=MEAN_TRACE).level == "unsupported"

    def test_a_count_is_bounded_by_the_population_field(self):
        v = verify(submission(3), trace=NINE_TRACE)
        assert v.matched_from == "derived:count" and v.level == "weak"

    def test_a_count_larger_than_the_population_is_not_forgiven(self):
        trace = [row("list_objects", [], evidence={"total_objects": 9})]
        assert verify(submission(11), trace=trace).level == "unsupported"

    def test_count_tier_needs_an_explicit_population_field(self):
        """★ 不能用 `len(返回值列表)` 当上界 —— 那会让「id 里的 1」重新变成证据。"""
        assert verify(submission(1.0), trace=[row("list_objects", [], evidence={})]).level == "unsupported"

    def test_point_count_is_not_a_population(self):
        """`n_points=23841` 是「一个物体有几万个点」，与「场景里有几个物体」无关。"""
        trace = [row("get_3d_extent", {"w": 1.0, "h": 1.0, "l": 1.0},
                     evidence={"n_points": 23841})]
        assert verify(submission(500), trace=trace).level == "unsupported"

    def test_sums_are_deliberately_not_covered(self):
        """求和的上界是「最大值 × 个数」，宽到会放走差一个数量级的答案 —— 故意不认。

        这是**已知缺口**，写在 `_derived_tier` 的 docstring 里：要治它得让
        `submit` 支持声明式派生（把被聚合的输入集合一起报上来逐项核对）。
        """
        trace = [row("calculate_distance", 2.5024, evidence={"distance_m": 2.5024})]
        assert verify(submission(7.5072), trace=trace).level == "unsupported"   # 3 × 2.5024

    def test_exact_match_never_degrades_to_the_weak_tier(self):
        v = verify(submission(2.5024), trace=DISTANCE_TRACE)
        assert v.matched_from == "value" and v.level == "supported"


class TestCensusIsNotEvidence:
    """★ `label_counts` 是 `{"sofa": 1, "picture": 4}` —— 一组 1~4 的小整数。

    它让「4 个」恒为 supported、「3 个」恒为 unsupported：那一档的判断力来自
    数字的**大小**，不来自正确性。同一个量在 `returned` 里按**本次查询**给了一份，
    所以删掉普查副本不会拿掉任何合法出处，只删掉假阳性。
    """

    def test_label_counts_no_longer_makes_small_integers_true(self):
        trace = [row("list_objects", [],
                     evidence={"label_counts": {"sofa": 1, "picture": 4}})]
        assert verify(submission(4.0), trace=trace).level == "unsupported"

    def test_the_query_scoped_copy_still_works(self):
        trace = [row("list_objects", [],
                     evidence={"returned": 4, "total_objects": 9,
                               "label_counts": {"sofa": 1}})]
        v = verify(submission(4.0), trace=trace)
        assert v.ok and v.matched_from == "evidence"

    def test_census_subtree_is_skipped_entirely(self):
        pools = collect_pools([row("list_objects", [], evidence={"label_counts": {"a": 7}})])
        assert 7 not in pools.measures and 7 not in pools.populations


class TestIdentifiersAndParamsAreNotEvidence:
    """★ 与 `label_counts` **同形**的第二个假出处：`method = "geometry_v1"`。

    实测抓到的经过（不是设想）：

        `query_relation` 的 `evidence["method"]` 恒为 `RELATIONS_METHOD = "geometry_v1"`；
        旧 `_ID_RE` 只抹「下划线**紧跟**数字」的 id（`chair_1`），抹不掉 `_v1` 这种
        **版本号后缀** ⟹ `1` 被 `_NUM_RE` 抠出来，每调一次就往证据池塞一个
        `('method', 1.0)`。而 C8「桌子前面有几个物体」的正确答案恰好是 `1` ⟹
        它被判成 `supported`，出处却是**算法版本号**。

    这两个用例钉的是**不变量**而不是那次具体经过：
    「键名表明这个位置放的不是量」⟹ 里面的数字不许当证据。
    同类里最先被撞上的是 `method`，但名单里还有 `tol`（答案恰好等于自己传进去的阈值
    也不是证据）与 `object_id`。
    """

    def test_version_tag_digit_is_not_evidence(self):
        trace = [row("query_relation", False,
                     evidence={"method": "geometry_v1", "relation": "front_of"})]
        assert verify(submission(1.0), trace=trace).level == "unsupported"

    def test_tolerance_param_is_not_evidence(self):
        # tol=0.0 会让答案 0 恒真有出处 —— 「答案等于我传进去的阈值」不是证据。
        trace = [row("query_relation", False,
                     evidence={"method": "geometry_v1", "tol": 0.0},
                     args={"tol": 0.0})]
        assert verify(submission(0.0), trace=trace).level == "unsupported"

    def test_request_count_k_is_not_evidence(self):
        # find_nearest(k=1) 的 k 是「让返回几个」，不是「场景里有几个」。
        trace = [row("find_nearest", [{"object_id": "sofa_1"}], args={"k": 1})]
        assert verify(submission(1.0), trace=trace).level == "unsupported"

    def test_non_quantity_keys_are_skipped_entirely(self):
        pools = collect_pools([
            row("query_relation", False,
                evidence={"method": "geometry_v1", "tol": 0.0, "object_id": "chair_1"},
                args={"k": 3}),
        ])
        for bad in (1.0, 0.0, 3.0):
            assert bad not in pools.values and bad not in pools.evidence
            assert bad not in pools.measures and bad not in pools.populations

    def test_version_suffix_ids_are_stripped_from_strings(self):
        """`geometry_v1` 这种「下划线 + 字母 + 数字」也要被 `_ID_RE` 抹掉。

        旧正则 `\\b[A-Za-z_][A-Za-z_0-9]*_(\\d+)\\b` 认不出它 —— 这是**根因**，
        上面按键名跳过只是把后果挡住。两层都要钉：改坏了任一层，
        下一次换个键名（例如某个 `note` 字段）同样的数字又会漏进来。
        """
        from agents.verifier import _ID_RE

        for s in ("geometry_v1", "chair_1", "sofa_chair_1", "sam_v2"):
            assert _ID_RE.sub(" ", s).strip() == "", s
        # 反例：合法的量不能被误抹（`extent_m` / `a_z` 后面跟的是 `=`，不是数字）
        assert "1.263" in _ID_RE.sub(" ", "extent_m=1.263")
        assert "3.17" in _ID_RE.sub(" ", "a_z=3.17")


class TestProvenanceToolReproducesTheLeak:
    """★ `scripts/show_answer_provenance.py` 的「修复前重放」必须**还复现得出**那个泄漏。

    为什么这条用例值得存在：那个脚本里冻了一份**已知有缺陷**的旧 `_ID_RE`。
    它唯一的用途就是重放修复前的口径，让「修复到底修掉了什么」可以被核对。
    如果哪天有人顺手把那份旧正则也「修好」了，重放表就会变成空白 ——
    **而空白与「没有泄漏」长得一模一样**，最坏的是没人会发现。

    所以这里用**真实的 C8 trace 形状**（`query_relation` 的 evidence）钉住：
    旧口径下 `method` 会给出 `1.0`，新口径下一个数都不给。
    """

    #: C8 的真实 evidence 形状（字段与取值类型都照抄，数值取自实际 trace）。
    _REL_EVIDENCE = {
        "method": "geometry_v1", "scene_id": "points_probe", "relation": "front_of",
        "a": "sofa_1", "b": "table_1",
        "delta_z": 1.3031622767448425, "tol": 0.0,
        "a_z": 3.1745684146881104, "b_z": 1.8714061379432678,
    }

    def test_before_fix_replay_still_finds_the_version_tag(self):
        import importlib.util

        root = ROOT / "scripts" / "show_answer_provenance.py"
        spec = importlib.util.spec_from_file_location("_prov", root)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        trace = [row("query_relation", False, evidence=self._REL_EVIDENCE,
                     args={"relation": "front_of", "a": "sofa_1", "b": "table_1", "tol": 0.0})]
        before = mod._hits(1.0, mod._pairs_of_trace_before_fix(trace))
        assert ("method", 1.0) in before, "重放表失效了：旧口径下 method 应当给出 1.0"
        # 当前口径下，同一个 trace 一个都不给 —— 这才是修复的内容。
        fv, fe = mod._pairs_of_trace(trace)
        assert mod._hits(1.0, fv) == [] and mod._hits(1.0, fe) == []
        assert mod._hits(0.0, fe) == []          # tol=0.0 也必须被挡住
