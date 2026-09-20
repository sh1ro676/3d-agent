"""L5 `tools/scene_report.py` 的单元测试 —— 零 GPU、零模型权重、零联网。

这一层是 L1–L5 里**唯一**只对已有 `SceneGraph` 做派生计算的一层，所以它可以被完整单测；
反过来说，如果这组测试需要 GPU，就说明 L5 违反了自己的设计前提（§11.1 的「零额外模型成本」）。

测试刻意钉住三类东西：
  1. **报告契约** —— 字段形状与关系覆盖，因为报告会落盘被人引用；
  2. **诚实性** —— 不可信的米数必须自己声明（尺度未校正 / 内参是猜的 / 降级质心）；
  3. **反事实的正确性** —— 剪枝后必须**重算**而不是复用全场景摘要（这一条真的写错过）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_graph.schema import BBox3D, Node, SceneGraph  # noqa: E402
from tools import load_tools  # noqa: E402
from tools import scene_report  # noqa: E402
from tools.registry import ToolArgumentError, ToolContext  # noqa: E402
from tools.result import ErrorCode, Recovery  # noqa: E402
from tools.scene_report import (  # noqa: E402
    DEFAULT_MAX_PAIRS,
    counterfactual,
    describe_scene,
    diagnose_failure,
    summarize_scene,
)

LOADED = load_tools()


# ----------------------------------------------------------------------------
# 夹具
# ----------------------------------------------------------------------------


def mk_node(
    node_id: str,
    label: str,
    xyz: tuple[float, float, float],
    size: tuple[float, float, float] = (0.4, 0.4, 0.4),
    *,
    with_bbox: bool = True,
    score: float = 0.8,
    centroid_source: str = "mask",
) -> Node:
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
        bbox_2d=(100.0, 120.0, 220.0, 300.0),
        mask_ref=f"masks/{node_id}.png",
        centroid_3d=(x, y, z),
        extent_3d=(w, h, l),
        bbox_3d=bbox,
        n_points=4200,
        attributes={"color": "grey"},
        centroid_source=centroid_source,  # type: ignore[arg-type]
    )


@pytest.fixture()
def scene() -> SceneGraph:
    """5 物体小客厅。全部带 `bbox_3d`，所以每对都有 11 条关系（含 on/inside）。"""
    return SceneGraph(
        scene_id="living_room_01",
        image_id="img_000",
        camera_intrinsics=[[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        up_axis="-y",
        nodes=(
            mk_node("door_1", "door", (-1.20, 0.00, 1.40), (0.90, 2.00, 0.10)),
            mk_node("chair_1", "chair", (0.80, 0.10, 2.90), (0.50, 0.90, 0.50)),
            mk_node("chair_2", "chair", (1.60, 0.10, 2.20), (0.50, 0.90, 0.50)),
            mk_node("sofa_1", "sofa", (0.31, -0.42, 2.86), (1.92, 0.78, 0.91)),
            mk_node("table_1", "table", (-0.44, -0.61, 3.12), (0.86, 0.45, 0.86)),
        ),
        build_meta={
            "detector": "grounding-dino-tiny",
            "intrinsics_source": "provided",
            "up_axis_reliable": True,
            "up_axis_tilt_deg": 11.95,
        },
    )


@pytest.fixture()
def ctx(scene: SceneGraph) -> ToolContext:
    return ToolContext(scene=scene)


@pytest.fixture()
def chain_scene() -> SceneGraph:
    """三点一线：`book_1` 夹在 `lamp_1` 与 `rug_1` 中间。

    几何是刻意选的，好让反事实的断言**决定性**而非「跑通即可」：
        lamp→book 0.20 m、book→lamp 0.20 m、rug→book 1.80 m
    一旦移除 `book_1`，两端的最近邻都必须**改口**：
        lamp→rug 2.00 m、rug→lamp 2.00 m
    如果实现复用了全场景摘要，`lamp_1` 的最近邻仍会是被移走的 `book_1` —— 测试会红。
    """
    return SceneGraph(
        scene_id="chain",
        image_id="img_chain",
        nodes=(
            mk_node("lamp_1", "lamp", (0.0, 0.0, 1.00)),
            mk_node("book_1", "book", (0.0, 0.0, 1.20)),
            mk_node("rug_1", "rug", (0.0, 0.0, 3.00)),
        ),
        build_meta={"intrinsics_source": "provided", "up_axis_reliable": True},
    )


def _by_type(relations: list[dict], rel_type: str) -> list[dict]:
    return [e for e in relations if e["type"] == rel_type]


# ----------------------------------------------------------------------------
# describe_scene
# ----------------------------------------------------------------------------


class TestDescribeScene:
    def test_registered_and_has_no_ctx_in_public_signature(self):
        """公开签名里不该出现 ctx —— 那是注入，不是模型参数（registry 的约定）。"""
        from tools.registry import TOOL_REGISTRY, tool_parameters

        assert "describe_scene" in LOADED
        assert tool_parameters("describe_scene") == ("scene_id", "detail")
        assert TOOL_REGISTRY["describe_scene"] is not None

    def test_report_shape(self, ctx: ToolContext):
        res = describe_scene(ctx)
        assert res.ok
        report = res.value
        assert report["scene_id"] == "living_room_01"
        assert report["image_id"] == "img_000"
        assert len(report["objects"]) == 5

        obj = report["objects"][0]
        assert set(obj) >= {
            "id", "label", "centroid_m", "extent_m",
            "confidence", "centroid_source", "n_points", "attributes",
        }
        # extent 是米制的三元组，且必须是正数 —— 报告里的数字一旦不合理，
        # 整份报告就失去被引用的资格。
        assert len(obj["centroid_m"]) == 3
        assert all(v > 0 for v in obj["extent_m"].values())

    def test_zero_model_cost(self, ctx: ToolContext):
        """L5 的核心主张：不加载任何权重、不碰 GPU。

        `gpu_peak_mb` 恒为 None 是这条主张最直接的证据 —— 没有任何一处在采样显存。
        """
        res = describe_scene(ctx)
        assert res.meta is not None
        assert res.meta.gpu_peak_mb is None
        assert res.evidence["method"] == "geometry_v1"
        assert res.meta.latency_ms < 100.0

    def test_every_pair_gets_all_eleven_relations(self, ctx: ToolContext):
        """5 个物体 → 10 对 × 11 条 = 110 条。数量对不上就说明有对漏了或有对重了。"""
        res = describe_scene(ctx)
        rels = res.value["relations"]
        assert len(rels) == 10 * 11
        assert res.value["truncated"] is False

        # 每对都必须有且只有一条 distance，且带量纲。
        pairs = {(e["a"], e["b"]) for e in rels}
        assert len(pairs) == 10
        dist = _by_type(rels, "distance")
        assert len(dist) == 10
        assert all("value_m" in e and e["value_m"] >= 0.0 for e in dist)
        # 布尔关系用 `value`，不用 `value_m` —— 两种量纲不能混在一个键里。
        for e in rels:
            assert ("value_m" in e) == (e["type"] == "distance")
            assert ("value" in e) == (e["type"] != "distance")

    def test_on_and_inside_skipped_without_bbox(self, ctx: ToolContext):
        """缺 `bbox_3d` 时 on/inside 应当**缺席**而不是补 null。

        `pairwise` 的约定是「不因一项缺数据就丢掉整对关系」，所以其余 9 条照给。
        """
        nodes = tuple(
            mk_node(n.id, n.label, n.centroid_3d, n.extent_3d, with_bbox=(n.id != "sofa_1"))
            for n in ctx.scene.nodes  # type: ignore[union-attr]
        )
        ctx.scene = ctx.scene.model_copy(update={"nodes": nodes})  # type: ignore[union-attr]
        res = describe_scene(ctx)
        assert res.ok
        rels = res.value["relations"]
        # 涉及 sofa_1 的 4 对少 2 条：10*11 - 4*2 = 102
        assert len(rels) == 102
        assert not [
            e for e in rels if e["type"] in ("on", "inside") and "sofa_1" in (e["a"], e["b"])
        ]

    def test_metric_dedup_keeps_only_decisive_quantities(self, ctx: ToolContext):
        """回归测试：只删**可从报告别处无损还原**的质心分量，绝不删判别量。

        这里钉住的是两件曾经写错的事：
          ① 判别量必须留在该条自身里 —— `left_of` 的判据就是 `delta_x < -tol`，
             删掉它「为什么判定为左」就无法从这条复算，metric 也就失去存在理由；
          ② 字段集必须**确定** —— 早先按「值与该对 distance 条相同」来删，
             而 `round(x, 6) == x` 只在少数数上成立，于是同一个 `left_of` 关系
             在不同对上留下了不同的字段集（(sofa,picture) 留有 delta_x、(door,sofa) 没有）。
             报告要能逐字节 diff，字段集就不能随数值抖动。
        """
        rels = describe_scene(ctx).value["relations"]
        # ⚠ 对是**无序**的，且在报告里按节点原序输出（`i < j`，见 `_all_pairs`），
        # 所以这里必须按集合匹配，不能假定 a 就是 sofa_1。
        pair = [e for e in rels if {e["a"], e["b"]} == {"sofa_1", "door_1"}]
        assert pair
        dist = next(e for e in pair if e["type"] == "distance")
        left = next(e for e in pair if e["type"] == "left_of")

        # distance 是该对的参照条：保留完整基础量（含质心分量）。
        assert {"distance_m", "delta_x", "delta_y", "delta_z",
                "a_x", "a_y", "a_z", "b_x", "b_y", "b_z"}.issubset(dist["metric"])
        # 其余关系保留自己的判别量。
        assert "delta_x" in left["metric"], "判别量被误删了"
        assert "tol" in left["metric"]
        # 但质心分量必须去掉 —— 它们就是 objects[].centroid_m，无损可还原。
        assert "a_x" not in left["metric"]
        assert "b_x" not in left["metric"]

    def test_metric_field_set_is_identical_for_every_pair(self, ctx: ToolContext):
        """同一关系类型在**所有对**上必须留下同一组字段名。

        这是上一条的加强版：断言的是**确定性**本身，而不是某一个对的结果。
        只比较键集合，不比较值 —— 因为值本来就该随几何变化。
        """
        rels = describe_scene(ctx).value["relations"]
        key_sets: dict[str, set[frozenset]] = {}
        for e in rels:
            key_sets.setdefault(e["type"], set()).add(frozenset(e["metric"]))
        unstable = {t: s for t, s in key_sets.items() if len(s) > 1}
        assert not unstable, f"这些关系类型的字段集在不同对上不一致：{unstable}"

    def test_scale_not_calibrated_is_declared(self, ctx: ToolContext):
        """未校正尺度必须自己说出来 —— 否则 1.42 m 会被当成真值引用。"""
        res = describe_scene(ctx)
        assert res.value["quality"]["scale_calibrated"] is False
        assert any("尺度未校正" in c for c in res.value["caveats"])

    def test_calibrated_scale_adds_no_caveat(self, scene: SceneGraph):
        calibrated = scene.model_copy(
            update={"scale_factor": 1.234, "build_meta": {**scene.build_meta, "scale_calibrated": True}}
        )
        res = describe_scene(ToolContext(scene=calibrated))
        assert res.value["quality"]["scale_calibrated"] is True
        assert not any("尺度未校正" in c for c in res.value["caveats"])

    def test_predicted_intrinsics_is_flagged(self, scene: SceneGraph):
        predicted = scene.model_copy(
            update={"build_meta": {**scene.build_meta, "intrinsics_source": "predicted"}}
        )
        res = describe_scene(ToolContext(scene=predicted))
        assert res.value["quality"]["intrinsics_source"] == "predicted"
        assert any("内参来自模型预测" in c for c in res.value["caveats"])

    def test_bbox_fallback_centroid_is_flagged(self, scene: SceneGraph):
        nodes = tuple(
            mk_node(n.id, n.label, n.centroid_3d, n.extent_3d,
                    centroid_source="bbox_fallback" if n.id == "sofa_1" else "mask")
            for n in scene.nodes
        )
        res = describe_scene(ToolContext(scene=scene.model_copy(update={"nodes": nodes})))
        assert any("降级路径" in c for c in res.value["caveats"])

    def test_brief_is_o_n_and_keeps_nearest_neighbour(self, ctx: ToolContext):
        brief = describe_scene(ctx, detail="brief")
        full = describe_scene(ctx, detail="full")
        assert brief.value["relations"] == []
        assert len(brief.value["relation_summary"]["nearest_neighbour"]) == 5
        assert brief.value["relation_summary"]["n_pairs"] == 10
        assert len(json.dumps(brief.value, ensure_ascii=False)) < len(
            json.dumps(full.value, ensure_ascii=False)
        )

    def test_default_max_pairs_is_a_pair_based_cap(self, ctx: ToolContext, monkeypatch):
        """截断必须**声明**，且按「对」而不是按「条」砍 —— 半对会让一致率算错。"""
        assert DEFAULT_MAX_PAIRS == 200
        monkeypatch.setattr(scene_report, "DEFAULT_MAX_PAIRS", 3)
        res = describe_scene(ctx)
        assert res.value["truncated"] is True
        assert res.value["relation_summary"] == {"n_pairs_total": 10, "n_pairs_reported": 3}
        assert len(res.value["relations"]) == 3 * 11
        assert any("截断" in c for c in res.value["caveats"])

    def test_bad_detail_raises_argument_error_not_tool_failure(self, ctx: ToolContext):
        """值域写错是「程序写错了」，必须冒泡 —— 不能被翻成工具失败。

        否则失败诊断会把责任算到工具头上，「工具调用成功率」这个指标就是说谎的。
        """
        with pytest.raises(ToolArgumentError):
            describe_scene(ctx, detail="verbose")
        # 而且它不该在 trace 里留下一条失败记录。
        assert not [r for r in ctx.trace if not r["result"]["ok"]]

    def test_scene_id_mismatch_is_not_found(self, ctx: ToolContext):
        res = describe_scene(ctx, scene_id="other_scene")
        assert not res.ok
        assert res.error is not None and res.error.code is ErrorCode.NOT_FOUND

    def test_no_scene_returns_actionable_hint(self):
        res = describe_scene(ToolContext())
        assert not res.ok
        assert res.error is not None and res.error.code is ErrorCode.NOT_FOUND
        assert "build_scene_graph" in res.error.context["hint"]


# ----------------------------------------------------------------------------
# summarize_scene
# ----------------------------------------------------------------------------


class TestSummarizeScene:
    def test_template_is_byte_identical_across_calls(self, ctx: ToolContext):
        """确定性是默认行为的前提：报告要进指标比较，LLM 采样噪声不能混进来。"""
        a = summarize_scene(ctx)
        b = summarize_scene(ctx)
        assert a.ok and b.ok
        assert a.value == b.value
        assert a.evidence["mode"] == "template"
        assert a.evidence["deterministic"] is True

    def test_template_mentions_inventory_and_scale_caveat(self, ctx: ToolContext):
        text = summarize_scene(ctx).value
        assert "5 个物体" in text
        assert "chair × 2" in text
        assert "尺度未校正" in text

    def test_use_llm_without_summarizer_is_capability_disabled(self, ctx: ToolContext):
        res = summarize_scene(ctx, use_llm=True)
        assert not res.ok
        assert res.error is not None
        assert res.error.code is ErrorCode.CAPABILITY_DISABLED
        # 消融开关的恢复动作必须指向「用几何替代」或「弃答」，不是「重试」。
        assert Recovery.USE_GEOMETRY in res.error.recovery
        assert res.error.context["capability"] == "summarizer"
        assert "summarizer" in res.error.context["hint"]

    def test_injected_summarizer_is_used(self, ctx: ToolContext):
        seen: dict = {}

        def fake_summarizer(report: dict) -> str:
            seen["report"] = report
            return "自定义摘要"

        ctx.flags["summarizer"] = fake_summarizer
        res = summarize_scene(ctx, use_llm=True)
        assert res.ok
        assert res.value == "自定义摘要"
        assert res.evidence["mode"] == "llm"
        assert res.evidence["deterministic"] is False
        # 传给 summarizer 的必须是**完整报告**（含逐对关系），不是 brief 版。
        assert seen["report"]["objects"]
        assert seen["report"]["relations"]

    def test_summarizer_exception_becomes_recoverable_failure(self, ctx: ToolContext):
        def boom(_report: dict) -> str:
            raise RuntimeError("模型服务 502")

        ctx.flags["summarizer"] = boom
        res = summarize_scene(ctx, use_llm=True)
        assert not res.ok
        assert res.error is not None and res.error.code is ErrorCode.DEGENERATE
        assert "RuntimeError" in res.error.message
        # 摘要失败不该让整条流水线断掉 —— 必须提示还有确定性模板这条退路。
        assert "use_llm=False" in res.error.context["recovery_note"]


# ----------------------------------------------------------------------------
# diagnose_failure
# ----------------------------------------------------------------------------


class TestDiagnoseFailure:
    def test_clean_scene_yields_no_findings(self):
        """完全健康的场景不该被硬挑出一个 stage —— 那会让诊断本身变成噪声。"""
        clean = SceneGraph(
            scene_id="clean",
            image_id="img_clean",
            nodes=(mk_node("chair_1", "chair", (0.0, 0.0, 2.0)),),
            build_meta={
                "intrinsics_source": "provided",
                "up_axis_reliable": True,
                "scale_calibrated": True,
            },
        )
        res = diagnose_failure(ToolContext(scene=clean))
        assert res.ok
        assert res.value["findings"] == []
        assert res.value["stage"] is None

    def test_scale_outranks_detection_as_root_cause(self, scene: SceneGraph):
        """核心语义：内参错是**根因**，尺寸越界多半只是它的症状，必须排在前面。

        这条断言直接对应 scripts/inspect_scene.py 里那条注释
        （「先解决这里，再去看其他嫌疑人 —— 它们多半是同一件事的症状」）。
        """
        broken = scene.model_copy(
            update={
                "nodes": tuple(scene.nodes)
                + (mk_node("blob_1", "blob", (0.0, 0.0, 3.0), (6.0, 0.4, 0.4)),),
                "build_meta": {
                    **scene.build_meta,
                    "intrinsics_source": "predicted",
                },
            }
        )
        res = diagnose_failure(ToolContext(scene=broken))
        assert res.ok
        assert res.value["stage"] == "尺度"
        # findings 按严重度降序，且第一条必须是尺度那条。
        assert res.value["findings"][0]["stage"] == "尺度"
        overrun = [f for f in res.value["findings"] if "三维尺寸超过" in f["reason"]]
        assert overrun, "尺寸越界应当被报出来（作为症状）"
        assert "尺度" in overrun[0]["fix"], "症状的修复建议必须指向根因检查"

    def test_tool_failures_are_grouped_by_error_code(self, ctx: ToolContext):
        from tools.spatial import get_object

        get_object(ctx, object_id="nope_1")
        get_object(ctx, object_id="also_nope")
        res = diagnose_failure(ctx, question_id="q007")
        assert res.ok
        tool_findings = [f for f in res.value["findings"] if f["stage"] == "工具"]
        assert len(tool_findings) == 1, "同一错误码应当聚合成一条，而不是两条"
        assert "NOT_IN_SCENE" in tool_findings[0]["reason"]
        assert "2 次" in tool_findings[0]["reason"]
        assert res.value["question_id"] == "q007"

    def test_question_text_resolved_from_flag(self, ctx: ToolContext):
        ctx.flags["questions"] = {"q001": "沙发左边是什么？"}
        res = diagnose_failure(ctx, question_id="q001")
        assert res.value["question"] == "沙发左边是什么？"
        assert res.evidence["question_resolved"] is True
        # 题集缺席时如实报 False，而不是伪造一个题干。
        assert diagnose_failure(ctx, question_id="q999").evidence["question_resolved"] is False

    def test_program_error_is_its_own_stage_not_a_tool_failure(self, ctx: ToolContext):
        """`ToolArgumentError` 冒泡出来后由 Agent 层记为「程序」错误。

        它**不能**被算进「工具失败」—— 否则工具调用成功率这个指标会说谎。
        """
        with pytest.raises(ToolArgumentError):
            describe_scene(ctx, detail="nope")

        ctx.flags["program_error"] = "TypeError: describe_scene() got an unexpected keyword 'detial'"
        res = diagnose_failure(ctx)
        assert res.ok
        stages = {f["stage"] for f in res.value["findings"]}
        assert "程序" in stages
        assert "工具" not in stages, "程序错误不得被计入工具失败"

    def test_low_point_object_flagged_in_both_caveat_and_diagnosis(self, scene: SceneGraph):
        """点数不足的**伪物体**必须在两处都被说出来。

        只在 `diagnose_failure` 里报是不够的：`describe_scene` 才是那份会被引用的交付物，
        它会把 objects 直接读成「这个房间里有 4 幅画」。若其中 3 幅其实是几十个像素的
        低分检测框而报告一声不吭，报告就在说谎 —— 这比数字不准更严重。

        阈值 1000 来自实测的自然分界（伪物体 809–965 vs 真实物体 16137–30660，17 倍空隙）。
        """
        ghost = scene.model_copy(
            update={"nodes": tuple(scene.nodes) + (mk_node("ghost_1", "picture", (0.0, 0.0, 4.0)),)}
        )
        nodes = tuple(
            n.model_copy(update={"n_points": 420 if n.id == "ghost_1" else n.n_points})
            for n in ghost.nodes
        )
        ghost = ghost.model_copy(update={"nodes": nodes})
        ctx = ToolContext(scene=ghost)

        report = describe_scene(ctx).value
        assert report["quality"]["n_low_point_objects"] == 1
        assert any("伪物体" in c for c in report["caveats"])

        diag = diagnose_failure(ctx).value
        hit = [f for f in diag["findings"] if "掩码点数不足" in f["reason"]]
        assert hit and "ghost_1" in hit[0]["detail"]

    def test_healthy_point_count_is_not_flagged(self, ctx: ToolContext):
        """反向断言：真实场景那 6 个万级点数的物体不得被误报。"""
        report = describe_scene(ctx).value
        assert report["quality"]["n_low_point_objects"] == 0
        assert not any("伪物体" in c for c in report["caveats"])

    def test_up_axis_unreliable_is_reported(self, scene: SceneGraph):
        shaky = scene.model_copy(
            update={
                "build_meta": {
                    **scene.build_meta,
                    "up_axis_reliable": False,
                    "up_axis_reason": "tilt_too_large",
                }
            }
        )
        res = diagnose_failure(ToolContext(scene=shaky))
        assert any("重力方向不可靠" in f["reason"] for f in res.value["findings"])
        assert any("above" in f["detail"] for f in res.value["findings"])

    def test_empty_scene_is_a_detection_failure(self):
        empty = SceneGraph(scene_id="empty", image_id="img_e", nodes=())
        res = diagnose_failure(ToolContext(scene=empty))
        assert res.ok
        assert res.value["stage"] == "检测"
        assert any("没有任何物体" in f["reason"] for f in res.value["findings"])


# ----------------------------------------------------------------------------
# counterfactual
# ----------------------------------------------------------------------------


class TestCounterfactual:
    def test_removing_one_object_drops_exactly_its_pairs(self, ctx: ToolContext):
        before = describe_scene(ctx).value["relations"]
        res = counterfactual(ctx, remove=["sofa_1"])
        assert res.ok
        after = res.value["diff"]["n_relations_after"]

        # 5 物体 10 对 → 4 物体 6 对；110 条 → 66 条。
        assert res.value["diff"]["n_relations_before"] == len(before) == 110
        assert after == 6 * 11
        # 消失的每一条都必然涉及被移除的物体 —— 否则说明删多了。
        assert all("sofa_1" in (e["a"], e["b"]) for e in res.value["diff"]["removed_relations"])
        assert len(res.value["diff"]["removed_relations"]) == 110 - 66

    def test_nearest_neighbour_is_recomputed_on_pruned_scene(self, chain_scene: SceneGraph):
        """★ 回归测试：剪枝后必须**重算**摘要，不能复用全场景那一份。

        写错过一次：复用会让 `lamp_1` 的最近邻仍指向已被移走的 `book_1` ——
        一个看起来完全合理、实际已经不存在的答案。这类错最危险的地方是它不报错。
        """
        ctx = ToolContext(scene=chain_scene)
        before = counterfactual(ctx, remove=["rug_1"]).value["relation_summary"]["nearest_neighbour"]
        # book_1 夹在中间，所以移走它之后两端都必须改口。
        assert before["lamp_1"] == {"object_id": "book_1", "distance_m": 0.2}

        res = counterfactual(ctx, remove=["book_1"])
        assert res.ok
        nn = res.value["relation_summary"]["nearest_neighbour"]
        assert "book_1" not in nn, "被移除的物体不得出现在最近邻表里"
        assert nn["lamp_1"] == {"object_id": "rug_1", "distance_m": 2.0}
        assert nn["rug_1"] == {"object_id": "lamp_1", "distance_m": 2.0}

    def test_report_objects_match_kept_nodes(self, ctx: ToolContext):
        res = counterfactual(ctx, remove=["sofa_1", "door_1"])
        assert res.ok
        ids = {o["id"] for o in res.value["objects"]}
        assert "sofa_1" not in ids and "door_1" not in ids
        assert len(ids) == 3
        assert res.value["quality"]["counterfactual_removed"] == ["door_1", "sofa_1"]

    def test_no_models_rerun(self, ctx: ToolContext):
        """这是本工具的核心主张：纯图操作。用证据字段把它钉住。"""
        res = counterfactual(ctx, remove=["table_1"])
        assert res.evidence["models_rerun"] == 0
        assert res.evidence["method"] == "geometry_v1"

    def test_hypothetical_view_is_labelled(self, ctx: ToolContext):
        """反事实结果绝不能被误当成真实观测 —— 必须自带标注。"""
        res = counterfactual(ctx, remove=["table_1"])
        assert any("反事实" in c for c in res.value["caveats"])

    def test_unknown_id_is_hallucination_capture(self, ctx: ToolContext):
        res = counterfactual(ctx, remove=["ghost_1"])
        assert not res.ok
        assert res.error is not None
        assert res.error.code is ErrorCode.NOT_IN_SCENE
        assert "ghost_1" in res.error.context["object_id"]
        # 带上合法 id 列表，模型据此改写程序 —— 这正是幻觉捕获点的价值。
        assert "sofa_1" in res.error.context["known_ids"]

    def test_removing_everything_does_not_crash(self, ctx: ToolContext):
        res = counterfactual(ctx, remove=list(ctx.scene.ids()))  # type: ignore[union-attr]
        assert res.ok
        assert res.value["objects"] == []
        assert res.value["relations"] == []
        assert res.value["relation_summary"]["nearest_neighbour"] == {}
        assert res.value["diff"]["n_relations_after"] == 0
        assert res.value["diff"]["flipped_relations"] == []

    def test_remove_is_optional(self, ctx: ToolContext):
        res = counterfactual(ctx)
        assert res.ok
        assert res.value["diff"]["removed_objects"] == []
        assert res.value["diff"]["n_relations_after"] == res.value["diff"]["n_relations_before"]
