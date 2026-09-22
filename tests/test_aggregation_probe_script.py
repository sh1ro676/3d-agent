#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聚合段探针（`scripts/run_aggregation_probe.py`）的回归测试。

**整脚本只跑一次**（module 级 fixture，约 10 秒），所有断言读它的 JSON。
理由：探针的价值在于各档**在同一个场景、同一套参数下**互相印证，
拆成小函数分别跑就失去了那个「同一份」的前提。

钉住的都是**反直觉**的实测结果 —— 顺直觉的那些不需要测试，
因为将来改坏了会有人立刻发现；反直觉的改坏了没人会发现：

* `bbox_fallback` 的纵深净代价**恰好为 0**（我原本的推论是「偏向远处」）；
  但**尺寸**代价在 pad=0 就已经是米级 —— 只看中位数会得出完全相反的结论。
* `robust_extent` 在 box 路径上**剔了但没剔对**（阈值被第二团自己抬大）。
* `no_valid_points` 的连带扰动是**内禀**的，不是注入方式的缺陷。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_aggregation_probe import main  # noqa: E402


@pytest.fixture(scope="module")
def probe(tmp_path_factory) -> dict:
    """跑一次整个探针，返回 `{exit_code, data}`。"""
    out = tmp_path_factory.mktemp("aggregation") / "aggregation_probe.json"
    code = main(["--quiet", "--json-out", str(out)])
    return {
        "exit_code": code,
        "data": json.loads(out.read_text(encoding="utf-8")),
    }


@pytest.fixture(scope="module")
def data(probe) -> dict:
    return probe["data"]


# ---------------------------------------------------------------------------
# 总体
# ---------------------------------------------------------------------------


class TestScriptHealth:
    def test_exit_code_is_zero(self, probe):
        """有任何 problem 时退出码非 0 —— 失败要响，不能只写在 JSON 里。"""
        assert probe["exit_code"] == 0

    def test_no_problems_were_recorded(self, data):
        assert data["problems"] == []

    def test_every_section_carries_its_scope_statement(self, data):
        """口径必须随数字一起落盘。

        没有 `scope` 的数字到了别人手里就会按他自己的口径读 ——
        而「净代价」与「相对盒心的误差」正好是两个容易被互换的东西。
        """
        for key in ("bbox_fallback", "no_valid_points", "intrinsics_guards"):
            assert data[key]["scope"].strip(), f"{key} 缺 scope"
        assert data["coverage_boundary"]["scope"].strip()
        assert data["extent_failure_mechanism"]["note"].strip()

    def test_config_declares_it_is_zero_api_and_names_the_path(self, data):
        cfg = data["config"]
        assert cfg["zero_api"] is True
        assert cfg["n_foreground"] == 9
        assert len(cfg["path"]) == 4


# ---------------------------------------------------------------------------
# ① 基线
# ---------------------------------------------------------------------------


class TestBaseline:
    def test_clean_build(self, data):
        b = data["baseline"]
        assert b["n_nodes"] == 9
        assert b["n_fallbacks"] == 0
        assert b["n_dropped"] == 0
        assert b["n_detections_raw"] == 9
        assert b["warnings"] == []

    def test_intrinsics_really_reached_the_perception_layer(self, data):
        """`[⑥ 6a/6b]` 保证这件事会响；这里保证**正常路径**也确实是 provided。

        两条一起才完整：只测守卫会响，不知道正常时是什么状态；
        只测正常状态，不知道那个状态是不是蒙对的。
        """
        b = data["baseline"]
        assert b["intrinsics_source"] == "provided"
        assert b["fov"]["plausible"] is True
        assert b["perception_kind"] == "SyntheticPerception"

    def test_gravity_axis_is_reliable_on_the_full_scene(self, data):
        """9 物体 + 背景时 `estimate_up_axis` 可靠 —— 与单物体单测里的
        `too_few_points` 形成对照：那一条是场景属性，不是全局行为。"""
        b = data["baseline"]
        assert b["up_axis_reliable"] is True
        assert b["up_axis"] == "-y"


# ---------------------------------------------------------------------------
# ② 掩码抹空
# ---------------------------------------------------------------------------


class TestBboxFallback:
    def test_injection_applied_to_every_object(self, data):
        fb = data["bbox_fallback"]
        assert fb["n_objects_injected"] == 9
        assert fb["n_applied"] == 9
        assert fb["missed"] == [], "有物体没走出降级 ⟹ 报告会显示「一切正常」"

    def test_each_row_actually_has_zero_mask_points_and_some_box_points(self, data):
        """降级的判据是「掩码取不到点、框内**有**点」—— 两头都要验。

        只验一头，一个「掩码本来就空」的物体也能混进来，而它走的其实是
        `no_valid_points`，两条路径的账就混在一起了。
        """
        for r in data["bbox_fallback"]["rows"]:
            assert r["n_mask_points"] == 0, r["object_id"]
            assert r["n_box_points"] > 0, r["object_id"]

    def test_the_delta_is_nonzero_so_the_injection_is_the_only_variable(self, data):
        """净代价若恒为 0，那说明注入没生效 —— 而不是「降级没代价」。"""
        for r in data["bbox_fallback"]["rows"]:
            assert r["delta_m"]["total_m"] > 0.0, r["object_id"]

    def test_depth_delta_is_exactly_zero_at_high_coverage(self, data):
        """★ 反直觉的实测结果，必须钉住。

        我原本的推论是「框内背景点的 z 更大 ⟹ 降级把物体系统性推向远处」。
        实测是 9/9 个物体的纵深中位数**恰好不变**（差为精确 0），
        原因是逐轴中位数在背景占比不到一半时取不到它们。
        这条结论的边界由 `[②b]` 量出 —— 一旦有人改了 `centroid_of` 的
        口径（例如换成均值），这里会立刻变红，而那是必须被知道的事。
        """
        rows = data["bbox_fallback"]["rows"]
        assert all(r["delta_centroid_3d"][2] == 0.0 for r in rows), [
            (r["object_id"], r["delta_centroid_3d"][2]) for r in rows
        ]
        assert data["bbox_fallback"]["depth_direction"]["n_farther"] == 0

    def test_extent_median_is_zero_while_its_max_is_meters(self, data):
        """★ 本档最重要的读数：**中位数掩盖了一个米级的尾部**。

        单看中位数会得出「bbox_fallback 只影响质心、不影响尺寸」——
        而事实正好相反：质心是毫米~百毫米级，尺寸是**米级**。
        这条断言要求两个数**同时**成立，否则那个静默形态就没被复现。
        """
        ex = data["bbox_fallback"]["extent_l1_m"]
        assert ex["median"] == 0.0
        assert ex["max"] > 5.0, ex["max"]
        assert 0 < ex["n_changed"] < data["bbox_fallback"]["n_objects_injected"]

    def test_box_coverage_is_reported_because_it_is_the_reading_axis(self, data):
        """覆盖率是解读这一档的**自变量**（背景占比过半才拉得动中位数）。"""
        bc = data["bbox_fallback"]["box_coverage"]
        assert 0.0 < bc["min"] <= bc["median"] <= bc["max"] <= 1.0
        assert "46.2%" in bc["note"], "必须点明真实链路的量级不同，否则会被直接外推"


# ---------------------------------------------------------------------------
# ②b 覆盖率扫描
# ---------------------------------------------------------------------------


class TestCoverageBoundary:
    def test_coverage_falls_monotonically_as_the_box_grows(self, data):
        covs = [r["coverage_median"] for r in data["coverage_boundary"]["rows"]]
        assert covs == sorted(covs, reverse=True)
        assert covs[0] > covs[-1]

    def test_depth_finally_moves_and_the_crossing_point_is_recorded(self, data):
        """「纵深不变」的边界必须能被指出来，否则那条结论会被当成普遍命题。"""
        cb = data["coverage_boundary"]
        assert cb["first_pad_with_depth_change"] is not None
        pad = cb["first_pad_with_depth_change"]
        by_pad = {r["box_pad_px"]: r for r in cb["rows"]}
        assert by_pad[pad]["n_depth_changed"] > 0
        # 在临界点**之前**，纵深必须一个都没动
        for r in cb["rows"]:
            if r["box_pad_px"] < pad:
                assert r["n_depth_changed"] == 0, r["box_pad_px"]

    def test_lateral_cost_saturates_while_depth_takes_over(self, data):
        """两轴的代价有接力顺序：横向先涨、然后饱和，纵深随后接管。

        这是本档唯一的结构性结论，所以钉住它 —— 如果哪天横向跟着纵深一起
        暴涨，说明有个实现细节变了，而那条「接力」的解释就不再成立。

        注意「临界点那一档」的形态：已经有 4 个物体的纵深变了，但**中位数**还是
        0 —— 少数派先动。所以 `n_depth_changed` 与 `median_depth_m` 是两个读数，
        不可能互相替代（这一条本身也值得钉住）。
        """
        cb = data["coverage_boundary"]
        rows, pd = cb["rows"], cb["first_pad_with_depth_change"]
        idx = next(i for i, r in enumerate(rows) if r["box_pad_px"] == pd)
        crit, last = rows[idx], rows[-1]

        assert crit["n_depth_changed"] > 0
        assert crit["median_depth_m"] == 0.0, "临界那一档中位数还不该动（少数派先动）"
        assert last["median_lateral_m"] < 2.0 * crit["median_lateral_m"], "横向应当饱和"
        # 0.9 而不是 1.0：实测中位数恰好是 1000.0 mm，浮点上差 4e-16 就会把
        # 一个「米级」的读数判成「不到一米」。判据追的是量级，不是那一位小数。
        assert last["median_depth_m"] > 0.9, "纵深应当涨到米级"
        assert last["median_depth_m"] > last["median_lateral_m"], "纵深应当已经接管"

    def test_the_known_bias_is_declared_in_the_scope(self, data):
        """pad ≥ 16 时框超出画面、覆盖率分母未裁剪 ⟹ 读数偏低。
        已知偏差必须写出来，而不是让读者自己发现。"""
        assert "偏低" in data["coverage_boundary"]["scope"]

    def test_depth_change_is_counted_with_its_sign(self, data):
        """★ 判据必须作用在**带符号**的位移上，不能作用在误差量上。

        实测 `cabinet_1` 在 pad=16 时纵深变化是 **−45.9 mm**（变近）。
        `gm.centroid_error(...)["depth_m"]` 是**非负**的误差量 ——
        在它上面写 `abs(v) > 0` 等于什么都没做（本档第一版就是这么写的，
        而且看起来比 `> 0` 更严谨），负向位移会被静默丢掉。

        这条测试**自证**自己的必要性：它要求整张表里确实存在负向位移。
        哪天场景换了、没有负向位移了，它应当变红并指出「这条判据没被验证」，
        而不是继续用一条无区别的判据假装通过。
        """
        rows = data["coverage_boundary"]["rows"]
        assert any(r["n_depth_decreased"] > 0 for r in rows), (
            "整张表里没有一个纵深变近的物体 ⟹ 符号判据在这份数据上没被验证"
        )
        for r in rows:
            assert r["n_depth_changed"] == r["n_depth_increased"] + r["n_depth_decreased"]
        # 有些档位（低 pad）三个计数都该是 0 —— 那正是「覆盖率还高」的定义，
        # 所以不能要求每一行都严格大于，只能要求**全表**存在符号分歧。
        assert sum(r["n_depth_decreased"] for r in rows) > 0


# ---------------------------------------------------------------------------
# ②c 尺寸失效机制
# ---------------------------------------------------------------------------


class TestExtentFailureMechanism:
    def test_rejection_happens_but_does_not_remove_the_second_cluster(self, data):
        """★ 「剔了 ≠ 剔对了」：阈值 `k_p90 × r_p90` 被第二团自己抬大，
        于是被判为离群的是更远的墙，第二团被当成了主体的一部分。

        所以这里的断言是「**确实剔了**（max > 0）**且** z 跨度照样暴涨」——
        两者同时成立才是那个机制；只报 `n_rejected = 0` 会把它说成
        「一个都没剔」，那是本文件第一版写过的错话。
        """
        em = data["extent_failure_mechanism"]
        assert em["box_n_rejected"]["max"] > 0, "完全没有剔除很难解释米级跨度"
        assert em["z_span_growth"]["max"] > 5.0

    def test_planar_objects_have_undefined_growth_not_zero(self, data):
        """可见面是平面（挂画/显示器）时 mask 侧 z 跨度为 0，涨幅无定义。

        记成 0 会读成「没变化」，而那正好相反 —— 这类物体任何 z 读数都是纯误差。
        """
        em = data["extent_failure_mechanism"]
        assert em["z_span_growth"]["n_undefined"] > 0
        for r in em["rows"]:
            if r["mask"]["z_span_m"] == 0.0:
                assert r["z_span_growth"] is None, r["object_id"]

    def test_the_mechanism_note_names_the_design_premise_it_breaks(self, data):
        note = data["extent_failure_mechanism"]["note"]
        assert "r_p90" in note and "第二团" in note
        assert "本轮不实现" in note, "修法方向要标成建议，不能读成已经改了"


# ---------------------------------------------------------------------------
# ③ 深度空洞
# ---------------------------------------------------------------------------


class TestNoValidPoints:
    def test_every_object_disappears(self, data):
        nv = data["no_valid_points"]
        assert nv["n_applied"] == 9
        assert nv["missed"] == []
        for r in nv["rows"]:
            assert r["n_nodes_now"] == r["n_nodes_base"] - 1, r["object_id"]

    def test_edge_loss_is_fully_explained_by_the_dropped_node(self, data):
        """边损失必须**全部**与目标相关。

        若出现「不涉及目标的边也丢了」，说明有别的东西变了（最可能是
        `up_axis`）—— 那时丢的边就不能记在「物体消失」这一条上。
        """
        nv = data["no_valid_points"]
        for r in nv["rows"]:
            assert r["n_edges_lost_unrelated"] == 0, r["object_id"]
            assert r["n_edges_lost"] == r["n_edges_lost_touching_target"]

    def test_gravity_axis_never_changed(self, data):
        assert data["no_valid_points"]["side_effects"]["n_up_axis_changed"] == 0

    def test_side_effects_are_present_and_declared_intrinsic(self, data):
        """★ 连带扰动是**内禀属性**，不是注入方式的缺陷 ——
        触发条件要求「框内所有像素都没有深度」，而框内必然有别人的像素。

        所以这里要求：既**确实有**连带扰动（否则说明注入太温柔、没测到真东西），
        又**在 note 里说明了它为什么不可避免**。
        """
        se = data["no_valid_points"]["side_effects"]
        assert se["n_rows_with_side_effects"] > 0
        assert se["max_other_delta_m"] > 0.0
        assert "内禀" in se["note"]
        assert se["n_above_relation_tol"] >= 0
        assert se["relation_tol_m"] > 0.0


# ---------------------------------------------------------------------------
# ④ 双路对照
# ---------------------------------------------------------------------------


class TestDualPath:
    def test_every_object_matches_bit_for_bit(self, data):
        """★ 尺子自检：builder 的 ④ 段与 `visible_geometry` 必须**逐位相等**。

        两条实现是分别写的，所以在同一份点云 + 同一份掩码上给出同样的数字，
        是「聚合段没有偷偷改数」的强证据。用 `==` 不用 `approx`：
        `resample_mask_to` 在网格相同时返回同一块内存，所以「几乎相等」
        本身就是缺陷信号。
        """
        dp = data["dual_path"]
        assert dp["n_checked"] == 9
        assert dp["n_mismatch"] == 0, dp["mismatches"][:2]
        assert "==" in dp["criterion"]


# ---------------------------------------------------------------------------
# ⑤ prompt 召回
# ---------------------------------------------------------------------------


class TestPromptRecall:
    def test_the_default_prompt_misses_six_of_nine_labels(self, data):
        """★ 这张表让「prompt 漏类别 = 图里少物体」变成一个**可见**的后果。

        默认 prompt 只写 5 类，夹具里有 9 类 ⟹ 实得 3 个节点。
        而模型不会报错、场景图格式合法、下游只会得到 `NOT_FOUND` ——
        正是本项目最关心的那种静默形态。
        """
        pr = data["prompt_recall"]
        default = pr["variants"]["default"]
        full = pr["variants"]["full"]
        assert "painting" in pr["labels_in_fixture"]
        assert default["n_nodes"] == 3
        assert full["n_nodes"] == 9
        assert len(default["labels_missing"]) == 6
        assert default["labels_missing"] == sorted(default["labels_missing"])

    def test_fewer_nodes_means_far_fewer_edges(self, data):
        """召回不全的代价会被关系层放大：3 节点 12 边 vs 9 节点 145 边。"""
        pr = data["prompt_recall"]["variants"]
        assert pr["default"]["n_edges"] < pr["full"]["n_edges"] / 10


# ---------------------------------------------------------------------------
# ⑥ 内参守卫
# ---------------------------------------------------------------------------


class TestIntrinsicsGuards:
    def test_both_guards_fire(self, data):
        """★ 「守卫从未生效」比「守卫失败」更难发现 —— 所以这里断言的是
        `ok is True`（**响**了），而不是「没有崩溃」。"""
        gu = data["intrinsics_guards"]
        assert gu["6a_wrong_known_intrinsics"]["ok"] is True
        assert gu["6a_wrong_known_intrinsics"]["raised"] == "SyntheticIntrinsicsError"
        assert gu["6b_require_camera_K_but_none_given"]["ok"] is True
        assert gu["6b_require_camera_K_but_none_given"]["raised"] == (
            "SyntheticIntrinsicsError"
        )

    def test_guard_6a_reports_both_the_given_and_the_truth(self, data):
        rec = data["intrinsics_guards"]["6a_wrong_known_intrinsics"]
        assert rec["given"] != rec["truth"]
        assert len(rec["given"]) == 4 and len(rec["truth"]) == 4

    def test_the_source_field_is_declared_degraded_in_the_bridge(self, data):
        """★ 6c 是一条**警告**，不是好消息。

        没传内参、lift 收到 `None`，而 `DepthField.intrinsics_source` 依然是
        `provided` ⟹ 该字段在合成桥里是**结构性事实**，不能用来判断
        「内参条件是否生效」。拿它论证「内参生效」是循环论证。
        """
        gu = data["intrinsics_guards"]
        c = gu["6c_no_known_intrinsics"]
        assert c["raised"] is None, "不传内参是允许的，不该抛"
        assert c["intrinsics_source"] == "provided"
        assert c["lift_received_camera_K"] is False
        assert "structural" in gu["scope"] or "结构性" in gu["scope"]

    def test_implausible_fov_but_provided_still_warns(self, data):
        """`provided` 来源也可能视场不可信（内参与分辨率不匹配）——
        builder 里那条独立分支必须有测试覆盖，它此前只活在代码里。"""
        rec = data["intrinsics_guards"]["6d_implausible_fov_but_provided"]
        assert rec["plausible"] is False
        assert rec["n_fov_warnings"] == 1
        assert "视场不可信" in (rec["warning"] or "")
        assert rec["intrinsics_source"] == "provided"
