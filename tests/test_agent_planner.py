"""`agents/planner.py`（臂 G：planning-then-synthesis）的单测 —— 零联网、零 GPU。

臂 G 是一条**可开关的增量干预**：多一次 LLM 调用，换"程序里少犯结构性错误"。
它值不值只有跑出来才知道，所以这里不替它辩护，只钉住三件事：

    1. **口径**：它改的是生成前的上下文，**不改控制流** ——
       计划不产生任何工具调用。报告里必须叫 planning-then-synthesis，
       不能写成「逐步规划」那种形态。
    2. **提示词只差那一段**：`planner="off"` 与 `"on"` 的 synthesize 提示词
       逐字节只差计划块 —— 否则两次实验的差异就归因不清了。
    3. **计划里出现的数字要被记下来**：计划会原样进下一次合成的提示词，
       而合成端***无从分辨***一个数是模型猜的还是工具算的。这是臂 G 特有的污染路径。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.loop import AgentLoop  # noqa: E402
from agents.planner import PLANNER_MODES, Plan, plan, render_plan_block  # noqa: E402
from agents.prompts.system import build_plan_system_prompt, build_plan_user_prompt  # noqa: E402
from agents.prompts.system import PLAN_BLOCK_TEMPLATE, build_user_prompt, render_template  # noqa: E402
from llm.adapter import LLMError, LLMReply, UsageLedger  # noqa: E402
from scene_graph.schema import Node, SceneGraph  # noqa: E402
from tools import load_tools  # noqa: E402
from tools.registry import ToolContext  # noqa: E402

load_tools()

HINT = {"objects": {"chair": 2, "door": 1}, "n_objects": 3,
        "image_size": [640, 480], "scene_id": "s", "image_id": "rgb.png"}

GOOD_PLAN = json.dumps({
    "tool_categories": ["list_objects", "calculate_distance"],
    "steps": ["用 list_objects 拿到物体的 object_id", "用 calculate_distance 求两者距离"],
}, ensure_ascii=False)

GOOD_PROG = "```python\nres = list_objects()\nsubmit(len(res.value), evidence=['len(list_objects())'])\n```"


class FakeClient:
    def __init__(self, *script):
        self.script = list(script)
        self.calls: list[dict] = []
        self.usage = UsageLedger()

    def chat(self, messages, *, purpose="chat"):
        self.calls.append({"messages": [dict(m) for m in messages], "purpose": purpose})
        if not self.script:
            raise AssertionError("client 被多调了一次")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            self.usage.add_failure()
            raise item
        reply = LLMReply(text=item, purpose=purpose, model_returned="fake-model",
                         finish_reason="stop",
                         usage={"prompt_tokens": 400, "completion_tokens": 80},
                         elapsed_s=0.01, attempts=1)
        self.usage.add(reply, 0.0005)
        return reply


@pytest.fixture
def scene() -> SceneGraph:
    return SceneGraph(
        scene_id="unit_scene", image_id="rgb.png", up_axis="-y",
        nodes=(Node(id="chair_1", label="chair", centroid_3d=(0.8, 0.1, 2.9)),
               Node(id="door_1", label="door", centroid_3d=(-1.2, 0.0, 1.4))),
        build_meta={"image_hw": [480, 640]},
    )


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


class TestPlanParsing:
    def test_a_fenced_json_plan_is_parsed(self):
        p = plan(FakeClient("```json\n" + GOOD_PLAN + "\n```"),
                 question="q", scene_hint=HINT)
        assert p.ok is True
        assert p.tool_categories == ("list_objects", "calculate_distance")
        assert len(p.steps) == 2
        assert p.text and "list_objects" in p.text

    def test_a_bare_json_plan_is_parsed(self):
        assert plan(FakeClient(GOOD_PLAN), question="q", scene_hint=HINT).ok is True

    def test_a_json_list_is_accepted_as_steps(self):
        p = plan(FakeClient(json.dumps(["第一步", "第二步"])), question="q", scene_hint=HINT)
        assert p.ok is True and p.steps == ("第一步", "第二步")

    def test_non_json_becomes_a_recorded_failure(self):
        p = plan(FakeClient("我先想想……"), question="q", scene_hint=HINT)
        assert p.ok is False and "不是 JSON" in p.error

    def test_empty_plan_is_a_failure(self):
        p = plan(FakeClient('{"steps": [], "tool_categories": []}'),
                 question="q", scene_hint=HINT)
        assert p.ok is False and "既没有" in p.error

    def test_llm_error_is_recorded_not_raised(self):
        """臂 G 是可选增强 —— 它不该把整题拖挂（那种失败会被误读成"模型不会做题"）。"""
        p = plan(FakeClient(LLMError("HTTP 500")), question="q", scene_hint=HINT)
        assert p.ok is False and "HTTP 500" in p.error

    def test_purpose_is_distinguishable_in_the_usage_ledger(self):
        """臂 G 的成本必须能单独归因 —— `by_purpose` 里有 `plan` 这一项。"""
        c = FakeClient(GOOD_PLAN)
        plan(c, question="q", scene_hint=HINT)
        assert "plan" in c.usage.snapshot()["by_purpose"]


# ---------------------------------------------------------------------------
# ★ 数字污染与类别核对
# ---------------------------------------------------------------------------


class TestPlanIsAudited:
    def test_numbers_in_the_plan_are_recorded(self):
        """★ 计划里出现数字必须留痕 —— 它会进下一次合成，而合成端分不清它的来源。"""
        payload = json.dumps({"steps": ["沙发大约 1.5 米宽，再算长宽比"]}, ensure_ascii=False)
        p = plan(FakeClient(payload), question="q", scene_hint=HINT)
        assert p.ok is True
        assert "1.5" in p.numbers
        assert p.to_dict()["numbers_in_plan"]

    def test_a_clean_plan_has_no_numbers(self):
        """★ 干净计划不许被误报 —— 否则指标饱和，"真夹带数字"就再也分不出来。

        `GOOD_PLAN` 有两步，渲染成块后会带上我们加的序号 `1.` / `2.`；
        这两位数**不是**模型写的，不能进 `numbers`。
        """
        p = plan(FakeClient(GOOD_PLAN), question="q", scene_hint=HINT)
        assert p.numbers == ()
        assert "1. " in p.text                      # 序号确实渲染出来了 ≠ 被记成数字

    def test_categories_outside_the_action_space_are_flagged(self):
        payload = json.dumps({"tool_categories": ["list_objects", "get_weather"],
                              "steps": ["x"]}, ensure_ascii=False)
        p = plan(FakeClient(payload), question="q", scene_hint=HINT)
        assert p.unknown_categories == ("get_weather",)

    def test_coarse_category_words_are_not_counted_as_mistakes(self):
        """「几何」「场景读取」这类粗粒度说法对组织思路有用，不算记错工具集。"""
        payload = json.dumps({"tool_categories": ["几何", "视觉", "geometry", "aggregate"],
                              "steps": ["x"]}, ensure_ascii=False)
        p = plan(FakeClient(payload), question="q", scene_hint=HINT)
        assert p.unknown_categories == ()

    def test_the_plan_is_customisable_per_arm(self):
        """核对用的类别清单就是本次的动作空间 —— 换臂时不该还用旧名单。"""
        payload = json.dumps({"tool_categories": ["get_attributes"], "steps": ["x"]})
        p = plan(FakeClient(payload), question="q", scene_hint=HINT,
                 tools=("list_objects", "get_attributes"))
        assert p.unknown_categories == ()


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------


class TestRendering:
    def test_block_lists_categories_then_numbered_steps(self):
        text = render_plan_block(Plan(ok=True, tool_categories=("a", "b"),
                                      steps=("第一步", "第二步")))
        assert text.index("需要的工具类别") < text.index("步骤")
        assert "  1. 第一步" in text and "  2. 第二步" in text

    def test_block_keeps_the_models_wording_verbatim(self):
        """我们**不**替模型改写它的计划 —— 否则观测到的就不是"它想了多少"。"""
        text = render_plan_block(Plan(ok=True, steps=("用 list_objects 拿到 id",)))
        assert "用 list_objects 拿到 id" in text

    def test_to_dict_is_json_serialisable(self):
        json.dumps(Plan(ok=True, steps=("a",), tool_categories=("b",)).to_dict(),
                   ensure_ascii=False)


# ---------------------------------------------------------------------------
# ★ 提示词：只差那一段
# ---------------------------------------------------------------------------


class TestPromptDelta:
    def test_without_a_plan_there_is_no_plan_block(self):
        text = build_user_prompt("q", HINT, answer_type="float")
        assert "你上一轮列出的计划" not in text
        assert "{{" not in text                       # 占位符必须被收干净

    def test_empty_and_none_plans_are_equivalent(self):
        a = build_user_prompt("q", HINT, plan_text=None)
        b = build_user_prompt("q", HINT, plan_text="   ")
        assert a == b

    def test_with_a_plan_only_the_block_is_added(self):
        """★ 两次实验的差异必须**只**是这一段，否则归因不成立。"""
        without = build_user_prompt("q", HINT, answer_type="float")
        with_plan = build_user_prompt("q", HINT, answer_type="float", plan_text="A → B")
        assert "你上一轮列出的计划" in with_plan
        assert "A → B" in with_plan
        # 把计划块**当成一整段连续切片**摘掉，剩下必须与不带计划时**逐字**相同。
        # ⚠ 这里刻意不用"按行过滤"那种启发式：它得靠关键词猜哪些行属于计划块，
        # 一旦模板多几个含"计划"字样的句子，筛选规则自己就会漏删几行，
        # 于是"其余部分没变"这个断言会因为**筛选规则不严谨**而假绿 ——
        # 而这条断言存在的全部意义就是防这一类静默差异。
        block = render_template(PLAN_BLOCK_TEMPLATE, {"PLAN": "A → B"})
        assert with_plan.count(block) == 1
        i = with_plan.index(block)
        assert with_plan[:i] + with_plan[i + len(block):] == without

    def test_the_plan_block_warns_that_numbers_must_not_be_trusted(self):
        text = build_user_prompt("q", HINT, plan_text="大约 1.5 米")
        assert "只能" in text and "工具返回值" in text

    def test_plan_system_prompt_carries_the_tool_docs(self):
        text = build_plan_system_prompt(None, ("list_objects", "calculate_distance"))
        assert "list_objects(" in text and "calculate_distance(" in text

    def test_plan_system_prompt_forbids_numbers_and_code(self):
        text = build_plan_system_prompt(None, ("list_objects",))
        assert "不要写任何数字" in text and "不要写代码" in text

    def test_plan_system_prompt_shows_a_valid_single_brace_json_example(self):
        """JSON 示例必须用**单**花括号渲染给模型 —— 双花括号会原样送出去。"""
        text = build_plan_system_prompt(None, ("list_objects",))
        assert '{"tool_categories"' in text
        assert "{{" not in text

    def test_plan_user_prompt_has_no_coordinates(self):
        text = build_plan_user_prompt("哪把椅子离门最近？", HINT)
        assert "centroid" not in text and "0.8" not in text
        assert '"chair": 2' in text

    def test_plan_and_synthesis_share_the_same_scene_hint(self):
        """两个角色的输入必须同源 —— 「计划有没有引入额外信息」的答案是确定的：没有。"""
        plan_text = build_plan_user_prompt("q", HINT)
        synth_text = build_user_prompt("q", HINT)
        assert json.dumps(HINT, ensure_ascii=False, sort_keys=True) in plan_text
        assert json.dumps(HINT, ensure_ascii=False, sort_keys=True) in synth_text


# ---------------------------------------------------------------------------
# 循环接线
# ---------------------------------------------------------------------------


class TestLoopWiring:
    def test_planner_off_costs_exactly_one_call(self, scene):
        c = FakeClient(GOOD_PROG)
        loop = AgentLoop(c, ctx=ToolContext(scene=scene), planner="off")
        run = loop.run("问")
        assert len(c.calls) == 1
        assert [s["stage"] for s in run.stages][0] != "plan"
        assert run.plan["skipped"] is True and run.plan["reason"] == "planner=off"

    def test_planner_on_adds_one_call_and_a_plan_stage(self, scene):
        c = FakeClient(GOOD_PLAN, GOOD_PROG)
        loop = AgentLoop(c, ctx=ToolContext(scene=scene), planner="on")
        run = loop.run("问")
        assert len(c.calls) == 2
        assert c.calls[0]["purpose"] == "plan"
        assert c.calls[1]["purpose"] == "synthesize"
        assert [s["stage"] for s in run.stages][0] == "plan"
        assert run.stages[0]["ok"] is True
        assert run.plan["ok"] is True and run.plan["n_steps"] == 2

    def test_the_plan_reaches_the_synthesis_prompt(self, scene):
        c = FakeClient(GOOD_PLAN, GOOD_PROG)
        AgentLoop(c, ctx=ToolContext(scene=scene), planner="on").run("问")
        user = c.calls[1]["messages"][1]["content"]
        assert "你上一轮列出的计划" in user and "list_objects" in user

    def test_planner_off_synthesis_prompt_has_no_plan_section(self, scene):
        c = FakeClient(GOOD_PROG)
        AgentLoop(c, ctx=ToolContext(scene=scene), planner="off").run("问")
        assert "你上一轮列出的计划" not in c.calls[0]["messages"][1]["content"]

    def test_a_failed_plan_falls_back_to_plain_synthesis(self, scene):
        """★ 计划失败不该把整题拖挂：退回无计划合成，并把失败记进 stages。"""
        c = FakeClient("这不是 JSON", GOOD_PROG)
        loop = AgentLoop(c, ctx=ToolContext(scene=scene), planner="on")
        run = loop.run("问")
        assert run.status == "ok"
        assert run.stages[0]["stage"] == "plan" and run.stages[0]["ok"] is False
        assert run.plan["ok"] is False
        assert "你上一轮列出的计划" not in c.calls[1]["messages"][1]["content"]

    def test_plan_cost_is_attributable(self, scene):
        c = FakeClient(GOOD_PLAN, GOOD_PROG)
        run = AgentLoop(c, ctx=ToolContext(scene=scene), planner="on").run("问")
        assert set(run.usage["by_purpose"]) == {"plan", "synthesize"}

    def test_a_bogus_planner_value_is_refused(self, scene):
        with pytest.raises(ValueError, match="planner"):
            AgentLoop(FakeClient(), ctx=ToolContext(scene=scene), planner="maybe")

    def test_modes_are_declared(self):
        assert PLANNER_MODES == ("off", "on")

    def test_switches_record_the_planner_state(self, scene):
        loop = AgentLoop(FakeClient(), ctx=ToolContext(scene=scene), planner="on")
        assert loop.switches()["planner"] == "on"
        assert loop.switches()["n_tools"] == loop.switches()["n_tools"]      # 可读即可
        assert "get_attributes" in loop.switches()["toolset"]
