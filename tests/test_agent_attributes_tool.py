"""`tools/attributes.py`（L4 视觉语义工具）的单测 —— 零联网、零 GPU。

这一层是「角色②」与「动作空间」之间的接缝，所以最值钱的断言是**消融开关那条**：

    `ctx.vlm = None` 时工具必须返回 `CAPABILITY_DISABLED`，
    而**不是**从动作空间里消失。

两者效果完全不同：把工具从名单里摘掉，模型根本不知道有这条路，
于是「关掉视觉」就变成了「换了一个动作空间」，消融不再干净（§13.3(7)）。

另外三条也很关键：
    · 低置信度 → `LOW_CONFIDENCE` 失败，但**值仍在 context 里**（程序有机会自己取舍）；
    · 不确定的结果**不许写进场景图缓存** —— 否则一次猜测会在后续所有题目里被当成事实；
    · 参数写错（未知属性 / 空闭集）→ `ToolArgumentError` **冒泡**，
      因为它要归因到「程序」而不是「工具」。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.executor import QA_TOOLSET, build_namespace  # noqa: E402
from llm.vlm import ATTRS, Attribute, DescribeCall, VLMError  # noqa: E402
from scene_graph.schema import Node, SceneGraph  # noqa: E402
from tools import load_tools  # noqa: E402
from tools.registry import TOOL_REGISTRY, ToolArgumentError, ToolContext  # noqa: E402
from tools.result import ErrorCode  # noqa: E402

load_tools()
get_attributes = TOOL_REGISTRY["get_attributes"]


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


def node_of(node_id: str, label: str, bbox=(10.0, 20.0, 60.0, 90.0)) -> Node:
    return Node(
        id=node_id, label=label, score=0.8,
        bbox_2d=bbox, mask_ref=f"masks/{node_id}.png",
        centroid_3d=(0.5, 0.1, 1.5), extent_3d=(0.5, 0.9, 0.5),
        bbox_3d=None, n_points=1000,
    )


@pytest.fixture
def scene() -> SceneGraph:
    return SceneGraph(
        scene_id="unit_scene", image_id="rgb.png", up_axis="-y",
        nodes=(node_of("chair_1", "chair"), node_of("sofa_1", "sofa")),
        build_meta={"image_hw": [480, 640]},
    )


class FakeVLM:
    """duck-typed 的角色② —— 只实现工具真正用到的那两个成员（`describe` / `last_call`）。"""

    def __init__(self, values=None, *, last_call=None, raises=None):
        self.values = values if values is not None else [
            Attribute(name="color", value="black", confidence=0.9, raw="black")]
        self.calls: list[dict] = []
        self._last = last_call or DescribeCall(
            model="fake-vlm", threshold=0.6, requested=("color",), returned=("color",),
            cropped=True, elapsed_s=0.42, usage={"prompt_tokens": 120, "completion_tokens": 20})
        self.raises = raises

    def describe(self, image, region, attrs, candidates=None):
        self.calls.append({"image": image, "region": region,
                           "attrs": tuple(attrs), "candidates": candidates})
        if self.raises is not None:
            raise self.raises
        return list(self.values)

    @property
    def last_call(self):
        return self._last


def ctx_with(scene, *, vlm=None, images=None, record_trace=True) -> ToolContext:
    ctx = ToolContext(scene=scene, record_trace=record_trace)
    ctx.vlm = vlm
    if images is not None:
        ctx.images.update(images)
    return ctx


# ---------------------------------------------------------------------------
# ★ 消融开关
# ---------------------------------------------------------------------------


class TestCapabilityGate:
    def test_vlm_none_returns_capability_disabled(self, scene):
        res = get_attributes(ctx_with(scene, vlm=None, images={"rgb.png": object()}),
                             object_id="chair_1")
        assert res.ok is False
        assert res.error.code is ErrorCode.CAPABILITY_DISABLED
        assert res.error.context["capability"] == "vlm"

    def test_disabled_recovery_points_at_geometry(self, scene):
        res = get_attributes(ctx_with(scene, vlm=None), object_id="chair_1")
        assert "use_geometry" in [r.value for r in res.error.recovery]

    def test_disabled_call_is_still_recorded_in_the_trace(self, scene):
        """消融造成的失败必须能被统计 —— 不记 trace 就等于它没发生过。"""
        ctx = ctx_with(scene, vlm=None)
        get_attributes(ctx, object_id="chair_1")
        assert len(ctx.trace) == 1 and ctx.trace[0]["tool"] == "get_attributes"
        assert ctx.trace[0]["result"]["error"]["code"] == "CAPABILITY_DISABLED"

    def test_the_tool_stays_in_the_action_space_when_disabled(self):
        """★ 关掉视觉**不等于**把工具从动作空间摘掉（见模块 docstring）。"""
        assert "get_attributes" in QA_TOOLSET
        ns = build_namespace(ToolContext(), lambda *a, **k: None)
        assert "get_attributes" in ns


# ---------------------------------------------------------------------------
# 取图
# ---------------------------------------------------------------------------


class TestImageLookup:
    def test_missing_image_is_not_found_with_a_hint(self, scene):
        res = get_attributes(ctx_with(scene, vlm=FakeVLM()), object_id="chair_1")
        assert res.ok is False and res.error.code is ErrorCode.NOT_FOUND
        assert "ctx.images" in res.error.context["hint"]

    def test_image_keyed_by_image_id_is_used(self, scene):
        v = FakeVLM()
        ctx = ctx_with(scene, vlm=v, images={"rgb.png": "IMG"})
        assert get_attributes(ctx, object_id="chair_1").ok is True
        assert v.calls[0]["image"] == "IMG"

    def test_single_image_fallback_when_the_key_does_not_match(self, scene):
        v = FakeVLM()
        ctx = ctx_with(scene, vlm=v, images={"something_else": "IMG"})
        assert get_attributes(ctx, object_id="chair_1").ok is True

    def test_multiple_images_without_a_matching_key_refuses_to_guess(self, scene):
        """多图时**不猜** —— 猜错会把 A 图的颜色安到 B 图的物体上，之后查不出来。"""
        ctx = ctx_with(scene, vlm=FakeVLM(), images={"a": "A", "b": "B"})
        res = get_attributes(ctx, object_id="chair_1")
        assert res.ok is False and res.error.code is ErrorCode.NOT_FOUND


# ---------------------------------------------------------------------------
# 正常路径
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_returns_the_attributes_and_the_evidence(self, scene):
        v = FakeVLM()
        res = get_attributes(ctx_with(scene, vlm=v, images={"rgb.png": "IMG"}),
                             object_id="chair_1", attrs=["color"])
        assert res.ok is True
        assert res.value == [{"name": "color", "value": "black", "confidence": 0.9,
                              "source": "vlm", "in_closed_set": True, "raw": "black"}]
        ev = res.evidence
        assert ev["object_id"] == "chair_1" and ev["source"] == "vlm"
        assert ev["model"] == "fake-vlm" and ev["threshold"] == 0.6
        assert ev["values"] == {"color": "black"}
        assert ev["region"]["bbox_2d"] == [10.0, 20.0, 60.0, 90.0]

    def test_region_carries_the_bbox_so_the_role_can_crop(self, scene):
        v = FakeVLM()
        get_attributes(ctx_with(scene, vlm=v, images={"rgb.png": "IMG"}), object_id="chair_1")
        assert v.calls[0]["region"].bbox == (10.0, 20.0, 60.0, 90.0)
        assert v.calls[0]["region"].label == "chair"

    def test_region_gets_the_image_size_from_build_meta(self, scene):
        v = FakeVLM()
        get_attributes(ctx_with(scene, vlm=v, images={"rgb.png": "IMG"}), object_id="chair_1")
        assert v.calls[0]["region"].image_size == (640, 480)

    def test_attrs_default_to_everything(self, scene):
        v = FakeVLM()
        get_attributes(ctx_with(scene, vlm=v, images={"rgb.png": "IMG"}), object_id="chair_1")
        assert v.calls[0]["attrs"] == ATTRS

    def test_a_bare_string_attr_is_accepted(self, scene):
        v = FakeVLM()
        get_attributes(ctx_with(scene, vlm=v, images={"rgb.png": "IMG"}),
                       object_id="chair_1", attrs="color")
        assert v.calls[0]["attrs"] == ("color",)

    def test_meta_carries_the_tool_name_and_version(self, scene):
        res = get_attributes(ctx_with(scene, vlm=FakeVLM(), images={"rgb.png": "IMG"}),
                             object_id="chair_1")
        assert res.meta.tool == "get_attributes" and res.meta.version

    def test_hallucinated_id_hits_the_hallucination_trap_first(self, scene):
        """id 不存在时的错误码必须是 `NOT_IN_SCENE`（幻觉捕获点），不是别的。"""
        res = get_attributes(ctx_with(scene, vlm=FakeVLM(), images={"rgb.png": "IMG"}),
                             object_id="chair_9")
        assert res.error.code is ErrorCode.NOT_IN_SCENE
        assert "chair_1" in res.error.context["known_ids"]


# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------


class TestCache:
    def test_confident_values_are_written_back_to_the_scene_graph(self, scene):
        get_attributes(ctx_with(scene, vlm=FakeVLM(), images={"rgb.png": "IMG"}),
                       object_id="chair_1", attrs=["color"])
        assert scene.node("chair_1").attributes["color"] == "black"

    def test_second_call_is_served_from_cache_without_calling_the_model(self, scene):
        v = FakeVLM()
        ctx = ctx_with(scene, vlm=v, images={"rgb.png": "IMG"})
        get_attributes(ctx, object_id="chair_1", attrs=["color"])
        res = get_attributes(ctx, object_id="chair_1", attrs=["color"])
        assert res.ok is True and res.meta.cached is True
        assert res.evidence["cached"] is True
        assert len(v.calls) == 1                       # ★ 只调了一次
        assert res.value[0]["source"] == "scene_graph_cache"

    def test_a_closed_set_request_bypasses_the_cache(self, scene):
        """给了闭集就**不**吃缓存：缓存里存的是没有约束时的答案。"""
        v = FakeVLM()
        ctx = ctx_with(scene, vlm=v, images={"rgb.png": "IMG"})
        get_attributes(ctx, object_id="chair_1", attrs=["color"])
        get_attributes(ctx, object_id="chair_1", attrs=["color"],
                       candidates={"color": ["black", "white"]})
        assert len(v.calls) == 2
        assert v.calls[1]["candidates"] == {"color": ["black", "white"]}

    def test_uncertain_values_are_never_cached(self, scene):
        """★ 低置信度的值**不许**进场景图 —— 否则一次猜测会在后面每道题里变成事实。"""
        low = [Attribute(name="color", value="charcoal", confidence=0.0,
                         in_closed_set=False, raw="charcoal")]
        v = FakeVLM(low)
        res = get_attributes(ctx_with(scene, vlm=v, images={"rgb.png": "IMG"}),
                             object_id="chair_1", attrs=["color"])
        assert res.ok is False
        assert scene.node("chair_1").attributes == {}


# ---------------------------------------------------------------------------
# 不确定必须上报
# ---------------------------------------------------------------------------


class TestUncertaintyIsReported:
    def _low_call(self, *, violations=(), low=("color",)) -> DescribeCall:
        return DescribeCall(model="fake-vlm", threshold=0.6,
                            requested=("color",), returned=("color",),
                            violations=violations, low_confidence=low)

    def test_low_confidence_becomes_a_low_confidence_failure(self, scene):
        v = FakeVLM(last_call=self._low_call())
        res = get_attributes(ctx_with(scene, vlm=v, images={"rgb.png": "IMG"}),
                             object_id="chair_1", attrs=["color"])
        assert res.ok is False and res.error.code is ErrorCode.LOW_CONFIDENCE
        assert "report_uncertain" in [r.value for r in res.error.recovery]

    def test_the_values_survive_inside_the_context(self, scene):
        """失败但**值仍在** —— 与 `AMBIGUOUS` 带回 `candidates` 同一套做法：
        让程序有机会自己判断，但绝不让「不确定」看起来像「确定」。"""
        v = FakeVLM(last_call=self._low_call())
        res = get_attributes(ctx_with(scene, vlm=v, images={"rgb.png": "IMG"}),
                             object_id="chair_1", attrs=["color"])
        assert res.error.context["attributes"][0]["value"] == "black"
        assert res.error.context["low_confidence"] == ["color"]

    def test_closed_set_violation_is_named_in_the_message(self, scene):
        v = FakeVLM(last_call=self._low_call(violations=("color",)))
        res = get_attributes(ctx_with(scene, vlm=v, images={"rgb.png": "IMG"}),
                             object_id="chair_1", attrs=["color"],
                             candidates={"color": ["black", "white"]})
        assert res.ok is False and "闭集" in res.error.message
        assert res.error.context["violations"] == ["color"]

    def test_evidence_is_still_attached_to_the_failure(self, scene):
        v = FakeVLM(last_call=self._low_call())
        res = get_attributes(ctx_with(scene, vlm=v, images={"rgb.png": "IMG"}),
                             object_id="chair_1", attrs=["color"])
        assert res.evidence["object_id"] == "chair_1"    # 失败也要能回溯


class TestBackendFailure:
    def test_vlm_error_becomes_a_disabled_result_with_a_reason(self, scene):
        """后端挂了**不该**把整题打断 —— 对程序来说要做的动作与「能力被关掉」完全一样。

        §13.3(1) 的六个错误码是冻结契约，所以不新加第七个；
        两种情况靠 `context.reason` 区分，统计上仍然分得开。
        """
        v = FakeVLM(raises=VLMError("连接超时"))
        res = get_attributes(ctx_with(scene, vlm=v, images={"rgb.png": "IMG"}),
                             object_id="chair_1", attrs=["color"])
        assert res.ok is False and res.error.code is ErrorCode.CAPABILITY_DISABLED
        assert res.error.context["reason"] == "backend_unavailable"
        assert "use_geometry" in [r.value for r in res.error.recovery]

    def test_backend_failure_is_distinguishable_from_the_ablation_switch(self, scene):
        """两种 `CAPABILITY_DISABLED` 在统计上必须分得开 —— 靠 reason 字段。"""
        ablated = get_attributes(ctx_with(scene, vlm=None), object_id="chair_1")
        failed = get_attributes(
            ctx_with(scene, vlm=FakeVLM(raises=VLMError("x")), images={"rgb.png": "IMG"}),
            object_id="chair_1", attrs=["color"])
        assert "reason" not in ablated.error.context
        assert failed.error.context["reason"] == "backend_unavailable"


# ---------------------------------------------------------------------------
# 参数校验：错误必须冒泡，归因到「程序」
# ---------------------------------------------------------------------------


class TestArgumentValidation:
    def _ctx(self, scene):
        return ctx_with(scene, vlm=FakeVLM(), images={"rgb.png": "IMG"})

    def test_unknown_attribute_raises_and_bubbles(self, scene):
        with pytest.raises(ToolArgumentError, match="不支持的属性"):
            get_attributes(self._ctx(scene), object_id="chair_1", attrs=["brand"])

    def test_a_spatial_attribute_is_refused_with_the_geometry_hint(self, scene):
        """`left_of` 这类请求必须被拒，并指向几何工具 —— 而不是让角色② 去猜。"""
        with pytest.raises(ToolArgumentError, match="get_3d_position"):
            get_attributes(self._ctx(scene), object_id="chair_1", attrs=["left_of"])

    def test_empty_candidates_list_is_refused(self, scene):
        with pytest.raises(ToolArgumentError, match="空闭集"):
            get_attributes(self._ctx(scene), object_id="chair_1",
                           attrs=["color"], candidates={"color": []})

    def test_candidates_for_an_attribute_that_was_not_requested_is_refused(self, scene):
        """静默忽略会让模型以为自己约束上了。"""
        with pytest.raises(ToolArgumentError, match="attrs 里没有要它"):
            get_attributes(self._ctx(scene), object_id="chair_1",
                           attrs=["color"], candidates={"material": ["wood"]})

    def test_candidates_with_an_unknown_attribute_is_refused(self, scene):
        with pytest.raises(ToolArgumentError, match="未知属性"):
            get_attributes(self._ctx(scene), object_id="chair_1",
                           attrs=["color"], candidates={"brand": ["x"]})

    def test_candidates_must_be_a_dict(self, scene):
        with pytest.raises(ToolArgumentError, match="必须是"):
            get_attributes(self._ctx(scene), object_id="chair_1",
                           attrs=["color"], candidates=["black"])

    def test_bad_attrs_type_is_refused(self, scene):
        with pytest.raises(ToolArgumentError, match="字符串或字符串列表"):
            get_attributes(self._ctx(scene), object_id="chair_1", attrs=42)

    def test_empty_string_attrs_is_refused(self, scene):
        with pytest.raises(ToolArgumentError, match="解析后为空"):
            get_attributes(self._ctx(scene), object_id="chair_1", attrs=["", "  "])

    def test_a_candidates_string_value_is_accepted_as_a_one_element_set(self, scene):
        v = FakeVLM()
        ctx = ctx_with(scene, vlm=v, images={"rgb.png": "IMG"})
        get_attributes(ctx, object_id="chair_1", attrs=["color"], candidates={"color": "black"})
        assert v.calls[0]["candidates"] == {"color": ["black"]}


class TestRegistryHygiene:
    def test_tool_is_registered_exactly_once(self):
        assert "get_attributes" in TOOL_REGISTRY

    def test_public_signature_hides_ctx(self):
        params = list(TOOL_REGISTRY["get_attributes"].__signature__.parameters)
        assert params == ["scene_id", "object_id", "attrs", "candidates"]

    def test_docstring_first_paragraph_goes_into_the_prompt(self):
        from llm.schema import docs_text

        text = docs_text(tools=["get_attributes"])
        assert "get_attributes(" in text
        # prompt 里只放摘要，不放那段很长的设计理由
        assert "缓存" not in text and "消融" not in text
