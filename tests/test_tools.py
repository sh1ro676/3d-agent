"""`tools/` 的单元测试 —— 不需要 GPU、不需要模型权重、不需要联网。

覆盖的是「外部层」的职责：存在性校验、错误码、恢复建议、证据组装、能力开关、trace。
数学本身在 `scene_graph/tests/test_relations.py` 里测。

这组测试同时是**错误码契约的回归测试**：§13.3(1) 把 6 个错误码写进了文档，
而文档是会腐烂的 —— 这里用断言把它钉在代码上。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_graph.schema import BBox3D, Edge, Node, SceneGraph  # noqa: E402
from tools import load_tools  # noqa: E402
from tools.geometry import (  # noqa: E402
    calculate_angle,
    calculate_distance,
    get_3d_extent,
    get_3d_position,
)
from tools.registry import (  # noqa: E402
    TOOL_REGISTRY,
    ToolArgumentError,
    ToolContext,
    tool_parameters,
    tool_names,
)
from tools.result import ErrorCode, Recovery, ToolResult  # noqa: E402
from tools.version import TOOLS_VERSION  # noqa: E402
from tools.spatial import (  # noqa: E402
    find_farthest,
    find_nearest,
    find_object,
    get_object,
    list_objects,
    query_relation,
    single_object,
)

LOADED = load_tools()


# ----------------------------------------------------------------------------
# 场景夹具
# ----------------------------------------------------------------------------


def mk_node(
    node_id: str,
    label: str,
    xyz: tuple[float, float, float],
    size: tuple[float, float, float],
    *,
    with_bbox: bool = True,
    score: float = 0.8,
    n_points: int = 4200,
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
        n_points=n_points,
    )


@pytest.fixture()
def scene() -> SceneGraph:
    """一个小客厅：1 门、2 椅、1 沙发、1 桌。

    几何是刻意选的：chair_1 到 door_1 比 chair_2 到 door_1 **更近**，
    于是 find_nearest 的排序结果可断言（不是「跑通即可」的空测试）。
    """
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
        edges=(),
        build_meta={"detector": "grounding-dino-tiny", "tools_version": "1.0.0"},
    )


@pytest.fixture()
def ctx(scene: SceneGraph) -> ToolContext:
    return ToolContext(scene=scene)


# ----------------------------------------------------------------------------
# 信封的不变量
# ----------------------------------------------------------------------------


class TestEnvelopeInvariants:
    def test_failure_without_error_is_unconstructible(self):
        """这是「堵死静默失败」的地方。

        早期基线的做法：命名空间里没有 `final_result` 就取 `""`，
        不报错、不算失败，直接当答案拿去评分。同样的对象在这里构造不出来。
        """
        with pytest.raises(ValueError, match="静默失败"):
            ToolResult(ok=False, value=None)

    def test_success_with_error_is_unconstructible(self):
        with pytest.raises(ValueError, match="不应携带 error"):
            ToolResult(
                ok=True,
                value=1,
                error=_err(ErrorCode.NOT_FOUND, "x"),
            )

    def test_every_error_code_has_recovery(self):
        """6 个码都必须有恢复动作 —— 否则模型拿到错误也不知道干什么。"""
        for code in ErrorCode:
            res = ToolResult.failure(code, "boom")
            assert res.error is not None
            assert len(res.error.recovery) >= 1
            assert all(isinstance(r, Recovery) for r in res.error.recovery)

    def test_enum_not_string_is_required_for_recovery_lookup(self):
        """回归测试：ErrorCode 是 str-mixin 枚举但 hash 基于成员名，
        传字符串会在 `error.recovery` 里 KeyError。这个坑真的踩过。"""
        res = ToolResult.failure(ErrorCode.DEGENERATE, "x")
        assert res.error is not None
        assert res.error.recovery[0] is Recovery.CHANGE_ANCHOR
        assert ErrorCode("DEGENERATE") is ErrorCode.DEGENERATE

    def test_to_dict_shape_and_json_roundtrip(self):
        res = ToolResult.success(value={"a": 1}, evidence={"formula": "x+y"})
        d = res.to_dict()
        assert set(d) == {"ok", "value", "evidence", "error", "meta"}
        json.dumps(d)   # 不抛异常即可

    def test_failure_to_dict_exposes_recovery_to_the_model(self):
        res = ToolResult.failure(ErrorCode.NOT_FOUND, "没有门", tool="t")
        d = res.to_dict()
        assert d["error"]["code"] == "NOT_FOUND"
        assert d["error"]["recovery"] == ["retry_query", "abstain"]

    def test_unwrap_raises_on_failure(self):
        from tools.result import ToolFailure

        with pytest.raises(ToolFailure):
            ToolResult.failure(ErrorCode.NOT_FOUND, "x").unwrap()
        assert ToolResult.success(7).unwrap() == 7

    def test_bool_reflects_ok(self):
        assert ToolResult.success(0)
        assert not ToolResult.failure(ErrorCode.NOT_FOUND, "x")


def _err(code: ErrorCode, message: str):
    from tools.result import ToolError

    return ToolError(code=code, message=message)


# ----------------------------------------------------------------------------
# 注册表
# ----------------------------------------------------------------------------


class TestRegistry:
    def test_expected_tools_are_registered(self):
        """把工具全集钉死在测试里 —— 工具集合本身就是「实验臂」的一部分。

        加了工具就会让这条红，于是「工具集变了」必须是一次有意识的决定：
        它同时意味着 `TOOLS_VERSION` 该升版本，否则两次实验的工具集不可比。
        """
        assert set(LOADED) == {
            # L1/L2 感知与场景读取（L1 的感知类工具在 agent 层按需导入，不在这里）
            "get_3d_position", "get_3d_extent", "calculate_distance", "calculate_angle",
            "list_objects", "get_object", "find_object", "single_object",
            "find_nearest", "find_farthest", "query_relation",
            # L5 场景级输出（2026-09-16 加入，TOOLS_VERSION → 1.1.0）
            "describe_scene", "summarize_scene", "diagnose_failure", "counterfactual",
            # L4 视觉语义（2026-09-18 加入，TOOLS_VERSION → 1.2.0）
            # ⚠ 它是**唯一带 `capability="vlm"` 的工具**，也是消融臂 E-1 的开关对象：
            #   `ctx.vlm=None` 时它在**动作空间里仍然存在**，只是运行时返回
            #   `CAPABILITY_DISABLED`。**不要**在关掉视觉时把它从清单里删掉 ——
            #   那样两次实验的提示词就一起变了，"少了一个能力"与"少了一个工具"
            #   会混在同一个差异里，归因不再成立（见 executors 的 `QA_TOOLSET`）。
            "get_attributes",
        }

    def test_public_signature_hides_ctx(self):
        """ctx 是我们的注入，不是模型的参数 —— 它不该出现在工具文档里。"""
        for name in tool_names():
            assert "ctx" not in tool_parameters(name), name

    def test_signature_is_introspectable_for_static_check(self):
        """AST 静态检查的输入就是这些签名（§13.3(2) 第 3 项）。"""
        assert set(tool_parameters("query_relation")) == {"scene_id", "relation", "a", "b", "tol"}
        assert set(tool_parameters("list_objects")) == {"scene_id", "label", "limit"}
        assert set(tool_parameters("calculate_angle")) == {"scene_id", "a", "b", "c"}


# ----------------------------------------------------------------------------
# 错误码契约
# ----------------------------------------------------------------------------


class TestErrorCodes:
    def test_not_in_scene_is_the_hallucination_trap(self, ctx: ToolContext):
        res = get_3d_position(ctx, object_id="chair_9")   # 编的 id
        assert not res.ok
        assert res.error.code is ErrorCode.NOT_IN_SCENE
        assert res.error.recovery == (Recovery.READ_SCENE,)
        # 关键：把合法 id 一起给回去，模型才能改对
        assert "chair_1" in res.error.context["known_ids"]
        assert len(res.error.context["known_ids"]) == 5

    def test_not_found_lists_available_labels(self, ctx: ToolContext):
        res = find_object(ctx, label="helicopter")
        assert res.error.code is ErrorCode.NOT_FOUND
        assert res.error.recovery[0] is Recovery.RETRY_QUERY
        assert "sofa" in res.error.context["available_labels"]

    def test_ambiguous_returns_candidates_not_a_coin_flip(self, ctx: ToolContext):
        res = single_object(ctx, label="chair")
        assert res.error.code is ErrorCode.AMBIGUOUS
        assert res.error.recovery[0] is Recovery.ADD_CONSTRAINT
        cands = res.error.context["candidates"]
        assert {c["object_id"] for c in cands} == {"chair_1", "chair_2"}

    def test_single_object_succeeds_when_unique(self, ctx: ToolContext):
        res = single_object(ctx, label="sofa")
        assert res.ok
        assert res.value["object_id"] == "sofa_1"

    def test_degenerate_when_bbox_missing(self):
        """缺 bbox_3d 的节点用 anchor='bbox_center' → DEGENERATE + 换 anchor 建议。"""
        bare = mk_node("lonely_1", "thing", (0.0, 0.0, 2.0), (1, 1, 1), with_bbox=False)
        c = ToolContext(scene=SceneGraph(scene_id="s", image_id="i", nodes=(bare,)))
        res = get_3d_position(c, object_id="lonely_1", anchor="bbox_center")
        assert res.error.code is ErrorCode.DEGENERATE
        assert res.error.recovery[0] is Recovery.CHANGE_ANCHOR
        assert "anchor" in res.error.context["hint"]

    def test_capability_disabled_is_a_switch_not_an_exception(self, scene: SceneGraph):
        """关掉 vlm 这个动作本身不产生异常 —— 它产生一个程序必须处理的结构化事件。"""
        c = ToolContext(scene=scene, vlm=None)
        assert not c.enabled("vlm")
        c2 = ToolContext(scene=scene, vlm=object())
        assert c2.enabled("vlm")


# ----------------------------------------------------------------------------
# 参数错误必须冒泡（不能被算成「工具失败」）
# ----------------------------------------------------------------------------


class TestArgumentErrorsBubble:
    def test_bad_anchor_raises_not_returns(self, ctx: ToolContext):
        """值域写错 = 程序写错，必须归到失败诊断的「程序」类。

        如果这里被吞成一个失败的 ToolResult，「工具调用成功率」这个指标会说谎 ——
        它会随着模型写错参数而下降，看起来像工具的问题。
        """
        with pytest.raises(ToolArgumentError, match="anchor"):
            get_3d_position(ctx, object_id="chair_1", anchor="middle")

    def test_unknown_relation_raises(self, ctx: ToolContext):
        with pytest.raises(ToolArgumentError, match="未知关系"):
            query_relation(ctx, relation="between", a="chair_1", b="door_1")

    def test_list_objects_rejects_negative_limit(self, ctx: ToolContext):
        with pytest.raises(ToolArgumentError):
            list_objects(ctx, limit=-1)


# ----------------------------------------------------------------------------
# 几何工具
# ----------------------------------------------------------------------------


class TestGeometryTools:
    def test_get_3d_position_returns_meters(self, ctx: ToolContext):
        res = get_3d_position(ctx, object_id="sofa_1")
        assert res.ok
        assert res.value == pytest.approx([0.31, -0.42, 2.86])
        assert res.evidence["anchor"] == "mask_median"
        assert res.evidence["n_points"] == 4200

    def test_anchor_switch_is_the_ablation_knob(self):
        """centroid 与 bbox_center 的差就是 §17 那组数字的来源。

        这里特意造一个「质心明显偏离框中心」的节点（模拟沙发那种框里混了大片背景），
        断言开关真的会改变结果 —— 否则这个消融维度是假的。
        """
        node = mk_node("sofa_1", "sofa", (0.31, -0.42, 2.86), (1.92, 0.78, 0.91))
        node = node.model_copy(update={"centroid_3d": (0.518, -0.42, 2.86)})  # 偏 208 mm
        c = ToolContext(scene=SceneGraph(scene_id="s", image_id="i", nodes=(node,)))

        centroid = get_3d_position(c, object_id="sofa_1", anchor="centroid").value
        box = get_3d_position(c, object_id="sofa_1", anchor="bbox_center").value
        shift = sum((centroid[i] - box[i]) ** 2 for i in range(3)) ** 0.5
        assert shift == pytest.approx(0.208, abs=0.01)

    def test_get_3d_extent_is_metric(self, ctx: ToolContext):
        res = get_3d_extent(ctx, object_id="sofa_1")
        assert res.value == pytest.approx({"w": 1.92, "h": 0.78, "l": 0.91})

    def test_calculate_distance_is_euclidean(self, ctx: ToolContext):
        res = calculate_distance(ctx, a="door_1", b="sofa_1")
        # door(-1.20,0,1.40) ↔ sofa(0.31,-0.42,2.86)
        expected = (1.51**2 + (-0.42) ** 2 + 1.46**2) ** 0.5
        assert res.value == pytest.approx(expected, rel=1e-9)
        assert res.evidence["formula"].startswith("||centroid_a")

    def test_calculate_angle_vertex_is_b(self, ctx: ToolContext):
        res = calculate_angle(ctx, a="door_1", b="chair_1", c="chair_2")
        assert res.ok
        assert 0.0 <= res.value <= 180.0
        assert res.evidence["vertex"] == "chair_1"
        json.dumps(res.to_dict())

    def test_calculate_angle_degenerate_when_points_coincide(self):
        a = mk_node("a", "x", (1.0, 0.0, 2.0), (0.1, 0.1, 0.1))
        b = mk_node("b", "x", (1.0, 0.0, 2.0), (0.1, 0.1, 0.1))   # 与 a 重合
        c_node = mk_node("c", "x", (0.0, 0.0, 2.0), (0.1, 0.1, 0.1))
        ct = ToolContext(scene=SceneGraph(scene_id="s", image_id="i", nodes=(a, b, c_node)))
        res = calculate_angle(ct, a="a", b="b", c="c")
        assert res.error.code is ErrorCode.DEGENERATE
        assert res.error.recovery[0] is Recovery.CHANGE_ANCHOR


# ----------------------------------------------------------------------------
# 场景查询与空间关系
# ----------------------------------------------------------------------------


class TestSpatialTools:
    def test_list_objects_is_the_only_source_of_legal_ids(self, ctx: ToolContext):
        res = list_objects(ctx)
        ids = {o["object_id"] for o in res.value}
        assert ids == {"door_1", "chair_1", "chair_2", "sofa_1", "table_1"}
        assert res.evidence["label_counts"] == {"door": 1, "chair": 2, "sofa": 1, "table": 1}

    def test_list_objects_limit_and_filter(self, ctx: ToolContext):
        assert len(list_objects(ctx, label="chair").value) == 2
        assert len(list_objects(ctx, limit=2).value) == 2
        assert list_objects(ctx, limit=2).evidence["total_in_scene"] == 5

    def test_brief_payload_is_compact(self, ctx: ToolContext):
        """观察文本必须压缩 —— 否则 20 个物体就能吃掉几千 token 的上下文。"""
        one = list_objects(ctx, limit=1).value[0]
        assert set(one) == {"object_id", "label", "score", "centroid_m", "extent_m"}
        assert "mask_ref" not in one and "bbox_2d" not in one

    def test_get_object_carries_attributes_mask_ref_and_bbox_2d(self, ctx: ToolContext):
        res = get_object(ctx, object_id="sofa_1")
        assert res.value["mask_ref"] == "masks/sofa_1.png"
        assert res.value["attributes"] == {}
        # `bbox_2d` 只在**单物体路径**上给：它是「图像左半边有几个物体」这类
        # 二维区域题唯一的判据，而 `list_objects` 刻意不带它
        # （一次返回 9~20 个物体，每个多 4 个像素数会把观察文本撑大近一倍，
        #  见 test_brief_payload_is_compact）。
        assert res.value["bbox_2d"] == [100.0, 120.0, 220.0, 300.0]

    def test_find_nearest_excludes_the_anchor_itself(self, ctx: ToolContext):
        res = find_nearest(ctx, anchor="door_1", label="chair")
        assert res.ok
        # chair_1 距 door 比 chair_2 近，见 fixture 注释
        assert res.value[0]["object_id"] == "chair_1"
        assert res.evidence["n_candidates"] == 2

    def test_find_nearest_cannot_return_the_anchor(self, ctx: ToolContext):
        res = find_nearest(ctx, anchor="chair_1", label="chair")
        assert res.value[0]["object_id"] == "chair_2"   # 不是 chair_1 自己
        assert all(o["object_id"] != "chair_1" for o in res.value)

    def test_find_nearest_when_only_the_anchor_matches(self, ctx: ToolContext):
        res = find_nearest(ctx, anchor="sofa_1", label="sofa")
        assert res.error.code is ErrorCode.NOT_FOUND
        assert "除 anchor 自身外" in res.error.message

    def test_find_farthest_is_the_reverse_order(self, ctx: ToolContext):
        near = find_nearest(ctx, anchor="door_1", label="chair", k=2)
        far = find_farthest(ctx, anchor="door_1", label="chair", k=2)
        assert [o["object_id"] for o in near.value] == list(
            reversed([o["object_id"] for o in far.value])
        )

    def test_ranked_evidence_makes_the_ranking_checkable(self, ctx: ToolContext):
        """evidence 里带全量排名 —— 答案可脱离程序复算，这就是「可审计」。"""
        res = find_nearest(ctx, anchor="door_1", label="chair")
        ranked = res.evidence["ranked"]
        assert len(ranked) == 2
        assert ranked[0]["distance_m"] <= ranked[1]["distance_m"]

    @pytest.mark.parametrize(
        "relation,expected",
        [
            ("left_of", True),      # chair_1.x=0.80 > door_1.x=-1.20 → 用的是反向断言
            ("right_of", False),
            ("behind", True),       # chair_1.z=2.90 > door_1.z=1.40
            ("front_of", False),
            ("near", False),        # 2.5 m > 1.0 m 默认阈值
        ],
    )
    def test_query_relation_dispatch(self, ctx: ToolContext, relation: str, expected: bool):
        res = query_relation(ctx, relation=relation, a="chair_1", b="door_1")
        assert res.ok
        if relation == "left_of":
            assert res.value is False         # chair_1 在 door 右边
        elif relation == "right_of":
            assert res.value is True
        else:
            assert res.value is expected

    def test_query_relation_returns_distance(self, ctx: ToolContext):
        res = query_relation(ctx, relation="distance", a="door_1", b="chair_1")
        assert isinstance(res.value, float)
        assert res.value == pytest.approx(2.503, abs=0.01)

    def test_query_relation_honours_scene_up_axis(self, ctx: ToolContext):
        """up_axis 从场景图透传 —— 换一个 up_axis，above/below 必须翻转。"""
        flipped = ctx.scene.model_copy(update={"up_axis": "+y"})
        c = ToolContext(scene=flipped)
        normal = query_relation(ctx, relation="above", a="sofa_1", b="table_1").value
        reversed_ = query_relation(c, relation="above", a="sofa_1", b="table_1").value
        assert normal is not reversed_

    def test_query_relation_carries_metric_evidence(self, ctx: ToolContext):
        res = query_relation(ctx, relation="left_of", a="door_1", b="chair_1")
        assert res.evidence["delta_x"] == pytest.approx(-2.0)
        assert res.evidence["a"] == "door_1" and res.evidence["b"] == "chair_1"
        assert res.evidence["method"] == "geometry_v1"


# ----------------------------------------------------------------------------
# 运行时：计时与 trace
# ----------------------------------------------------------------------------


class TestRuntime:
    def test_latency_and_tool_name_are_stamped(self, ctx: ToolContext):
        res = list_objects(ctx)
        assert res.meta is not None
        assert res.meta.tool == "list_objects"
        assert res.meta.latency_ms >= 0.0
        # 装饰器必须把库版本戳进 meta（而不是留 result.py 的默认值就算完）。
        assert res.meta.version == TOOLS_VERSION
        # 同时把**当前值**钉住：升版本时这条会红，于是它成为一次有意识的决定，
        # 而不是「反正断言跟着常量走、版本悄悄变了也没人知道」。
        # 1.2.0 = 加入 `get_attributes`（L4 视觉语义），工具数 11 → 12。
        # 1.3.0 = **工具数不变**，但工具文档（= 逐字节进 prompt 的那部分）与
        #         `get_object` 的 value 变了 ⟹ 照样是实验条件变了，照样升版本。
        assert TOOLS_VERSION == "1.3.0"
        assert len(res.meta.tool_call_id) == 13   # "c" + 12 hex

    def test_trace_records_every_call_including_failures(self, ctx: ToolContext):
        list_objects(ctx)
        get_3d_position(ctx, object_id="nope_1")
        assert len(ctx.trace) == 2
        assert ctx.trace[0]["tool"] == "list_objects"
        assert ctx.trace[1]["result"]["ok"] is False
        assert ctx.trace[1]["result"]["error"]["code"] == "NOT_IN_SCENE"

    def test_trace_args_are_bound_to_names(self, ctx: ToolContext):
        query_relation(ctx, relation="left_of", a="door_1", b="chair_1", tol=0.1)
        args = ctx.trace[-1]["args"]
        assert args["relation"] == "left_of"
        assert args["tol"] == 0.1
        assert "ctx" not in args

    def test_trace_is_jsonl_serializable(self, ctx: ToolContext):
        list_objects(ctx)
        json.dumps(list(ctx.trace_jsonl()))

    def test_trace_can_be_switched_off_for_cheap_runs(self, scene: SceneGraph):
        c = ToolContext(scene=scene, record_trace=False)
        list_objects(c)
        assert c.trace == []

    def test_tool_context_scene_is_immutable(self, ctx: ToolContext):
        """工具不能改场景 —— frozen 模型保证了「工具调用只读」这条不变量。"""
        with pytest.raises(Exception):
            ctx.scene.scene_id = "hacked"  # type: ignore[misc]

    def test_no_scene_gives_actionable_error(self):
        c = ToolContext(scene=None)
        res = list_objects(c)
        assert res.error.code is ErrorCode.NOT_FOUND
        assert "build_scene_graph" in res.error.context["hint"]

    def test_scene_id_mismatch_is_rejected(self, ctx: ToolContext):
        res = list_objects(ctx, scene_id="another_room")
        assert res.error.code is ErrorCode.NOT_FOUND
        assert "current_scene_id" in res.error.context


# ----------------------------------------------------------------------------
# 场景图本身
# ----------------------------------------------------------------------------


class TestSceneGraph:
    def test_label_lookup_is_case_insensitive(self, scene: SceneGraph):
        """GroundingDINO 的 prompt 必须小写，但模型写 'Chair' 很常见 —— 在这里吃掉。"""
        assert len(scene.by_label("Chair")) == 2
        assert len(scene.by_label(" CHAIR ")) == 2

    def test_label_counts_is_the_whole_scene_hint(self, scene: SceneGraph):
        """scene_hint 只有清单与计数、**没有坐标** —— 这是 §13.3(2) 的核心约束。"""
        hint = scene.label_counts()
        assert hint == {"door": 1, "chair": 2, "sofa": 1, "table": 1}
        assert all(isinstance(v, int) for v in hint.values())

    def test_node_raises_keyerror_instead_of_returning_none(self, scene: SceneGraph):
        """返回 None 会让调用方忘记处理；抛 KeyError 逼它显式翻译成 NOT_IN_SCENE。"""
        with pytest.raises(KeyError):
            scene.node("ghost_1")

    def test_edge_carries_metric_and_method(self):
        e = Edge(source="a", target="b", relation="left_of", value=True,
                 metric={"delta_x": -1.0}, method="geometry_v1")
        assert e.metric["delta_x"] == -1.0
        json.dumps(e.model_dump())

    def test_node_rejects_unknown_field(self):
        with pytest.raises(Exception):
            Node(id="x", label="y", centroid_3d=(0, 0, 1), typo_field=1)  # type: ignore[call-arg]

    def test_intrinsics_must_be_3x3(self):
        with pytest.raises(Exception, match="3×3"):
            SceneGraph(scene_id="s", image_id="i", camera_intrinsics=[[1.0, 2.0]])

    def test_numpy_vectors_are_accepted(self):
        """上游是 numpy（点云、PCA 结果），构造 Node 时不该要求手工转 list。"""
        np = pytest.importorskip("numpy")
        n = Node(id="a", label="b", centroid_3d=np.array([0.1, 0.2, 0.3]))
        assert n.centroid_3d == pytest.approx((0.1, 0.2, 0.3))
