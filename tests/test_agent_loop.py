"""`agents/loop.py` 的端到端单测 —— 用**假 client**，零联网、零 GPU、零成本。

覆盖的是循环层的全部结局与两条重试路径：

    synthesize → static_check → execute → render
                      ↑______ 定向重生成（≤2 次）______|
                      ↑______ 执行失败同样回灌 _________|

结局：ok / abstained / static_failed / exec_failed / llm_error 五种都要有一条用例，
否则「新加了一种状态但没人统计它」这类错误会静默发生。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.loop import AgentLoop, STATUSES  # noqa: E402
from agents.memory import WorkingMemory  # noqa: E402
from llm.adapter import LLMError, LLMReply, UsageLedger  # noqa: E402
from scene_graph.schema import BBox3D, Node, SceneGraph  # noqa: E402
from tools import load_tools  # noqa: E402
from tools.registry import ToolContext  # noqa: E402

load_tools()


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


def mk(node_id, label, xyz, size=(0.5, 0.9, 0.5), score=0.8) -> Node:
    x, y, z = xyz
    w, h, l = size
    return Node(
        id=node_id, label=label, score=score,
        bbox_2d=(10.0, 10.0, 50.0, 60.0), mask_ref=f"masks/{node_id}.png",
        centroid_3d=(x, y, z), extent_3d=(w, h, l),
        bbox_3d=BBox3D(min=(x - w / 2, y - h / 2, z - l / 2), max=(x + w / 2, y + h / 2, z + l / 2)),
        n_points=1000,
    )


@pytest.fixture
def scene() -> SceneGraph:
    return SceneGraph(
        scene_id="unit_scene", image_id="rgb.png", up_axis="-y",
        nodes=(
            mk("door_1", "door", (-1.2, 0.0, 1.4), (0.9, 2.0, 0.1), 0.91),
            mk("chair_1", "chair", (0.8, 0.1, 2.9)),
            mk("chair_2", "chair", (1.6, 0.1, 2.2)),
        ),
        build_meta={"image_size": [640, 480]},
    )


class FakeClient:
    """按顺序吐出预设回复（或抛预设异常）。记账与真实 client 同结构。

    ⚠ `chat` 必须接受 `deadline` 并**记下来**，不能只写 `**kwargs` 吞掉：
    预算有没有真的传到 LLM 调用这一层，是本项目要能**断言**的一件事
    （只写 `**_ignored` 的话，「预算传丢了」和「预算传对了」在测试里长得一样）。
    """

    def __init__(self, *script):
        self.script = list(script)
        self.calls: list[dict] = []
        self.usage = UsageLedger()

    def chat(self, messages, *, purpose="chat", deadline=None):
        self.calls.append({"messages": [dict(m) for m in messages], "purpose": purpose,
                           "deadline": deadline})
        if not self.script:
            raise AssertionError("client 被多调了一次")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            # 与真实 client 一致：失败必须计数，否则报告里的失败率会凭空偏低。
            self.usage.add_failure()
            raise item
        reply = LLMReply(
            text=item, purpose=purpose, model_returned="fake-model",
            finish_reason="stop", usage={"prompt_tokens": 500, "completion_tokens": 120},
            elapsed_s=0.01, attempts=1,
        )
        self.usage.add(reply, 0.001)
        return reply


def prog_submit(value="1.0", evidence="'x'") -> str:
    """最小合规程序：直接 submit 一个字面值。"""
    return "```python\nsubmit(%s, evidence=[%s])\n```" % (value, evidence)


GOOD_PROG = (
    "```python\n"
    "res = find_nearest(anchor='door_1', label='chair', k=1)\n"
    "if not res.ok:\n"
    "    submit('unknown', evidence=['find_nearest failed'])\n"
    "submit(res.value[0]['distance_m'], target_ids=[res.value[0]['object_id']],\n"
    "       evidence=['find_nearest(door_1, chair) = %s' % res.value[0]['distance_m']])\n"
    "```"
)

BAD_STATIC = "```python\nres = list_objects('chair')\nfinal_result = 1\n```"
BAD_RUNTIME = "```python\nx = ctx.scene\nsubmit(1, evidence=['x'])\n```"


# ---------------------------------------------------------------------------
# 主路径
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_one_llm_call_and_a_complete_run(self, scene):
        c = FakeClient(GOOD_PROG)
        loop = AgentLoop(c, ctx=ToolContext(scene=scene))
        run = loop.run("哪把椅子离门最近？", answer_type="float")
        assert run.status == "ok" and len(c.calls) == 1
        assert run.answer == pytest.approx(2.502, abs=0.01)     # sqrt(2^2+0.1^2+1.5^2)
        assert run.attempts == 1 and run.tool_calls == 1
        assert run.evidence and "find_nearest" in run.evidence[0]
        assert run.render.startswith("2.")
        assert run.tools_version

    def test_stage_sequence(self, scene):
        loop = AgentLoop(FakeClient(GOOD_PROG), ctx=ToolContext(scene=scene))
        run = loop.run("问")
        # ⚠ 默认（`planner="off"`）时序列里**没有** `plan` 阶段 —— 臂 G 关掉时必须
        #   连那一轮 LLM 调用都不发生，否则"关掉规划"只是"不把它读进提示词"，
        #   而两次实验的**花费**会悄悄不一样。
        #   `verify` 是零成本的证据校验，永远在最后（见 agents/verifier.py）。
        assert [s["stage"] for s in run.stages] == [
            "synthesize", "static_check", "execute", "render", "verify"]
        assert all(s["ok"] for s in run.stages)

    def test_system_message_carries_tool_docs_and_user_carries_scene_hint(self, scene):
        c = FakeClient(GOOD_PROG)
        AgentLoop(c, ctx=ToolContext(scene=scene)).run("问", answer_type="float")
        system, user = c.calls[0]["messages"][0], c.calls[0]["messages"][1]
        assert system["role"] == "system" and "list_objects(" in system["content"]
        assert "describe_scene" not in system["content"]        # L5 不在动作空间
        assert user["role"] == "user" and '"chair": 2' in user["content"]
        # 场景清单里**没有**坐标
        assert "centroid" not in user["content"] and "0.8" not in user["content"]
        assert c.calls[0]["purpose"] == "synthesize"

    def test_usage_delta_covers_only_this_run(self, scene):
        c = FakeClient(GOOD_PROG, GOOD_PROG)
        loop = AgentLoop(c, ctx=ToolContext(scene=scene))
        loop.run("第一题")
        run2 = loop.run("第二题")
        assert run2.usage["calls"] == 1
        assert c.usage.calls == 2

    def test_abstain_is_a_distinct_status(self, scene):
        c = FakeClient('```python\nsubmit("unknown", evidence=["场景里没有 door"])\n```')
        run = AgentLoop(c, ctx=ToolContext(scene=scene)).run("门在哪？")
        assert run.status == "abstained" and run.abstained is True
        assert run.answer == "unknown"

    def test_all_statuses_are_declared(self):
        assert set(STATUSES) == {"ok", "abstained", "static_failed", "exec_failed",
                                 "llm_error", "deadline"}

    def test_deadline_is_not_folded_into_llm_error(self):
        """`deadline` 必须与 `llm_error` **并存**，不能合并。

        两者处置方向相反：`llm_error` 要查链路（网络/端点/模型），`deadline` 要查
        预算够不够、以及「为什么单次调用变慢了」。合并成一条之后，报告里的失败归因
        会指向错误的方向，而两个状态各自的计数也就失去了意义。
        """
        assert "deadline" in STATUSES and "llm_error" in STATUSES
        assert len(set(STATUSES)) == len(STATUSES), "状态枚举里不能有重复值"


# ---------------------------------------------------------------------------
# 重试：定向，而不是"再来一次"
# ---------------------------------------------------------------------------


class TestRetry:
    def test_static_failure_is_followed_by_a_fix_and_succeeds(self, scene):
        c = FakeClient(BAD_STATIC, GOOD_PROG)
        run = AgentLoop(c, ctx=ToolContext(scene=scene)).run("问")
        assert run.status == "ok" and len(c.calls) == 2
        assert run.attempts == 2
        retry_msgs = c.calls[1]["messages"]
        assert retry_msgs[-1]["role"] == "user"
        assert "位置参数" in retry_msgs[-1]["content"]      # 定向：带上了具体原因
        assert "res = list_objects('chair')" in retry_msgs[-2]["content"]   # 带上了上一次的程序
        assert [s["stage"] for s in run.stages][:3] == ["synthesize", "static_check", "synthesize"]

    def test_runtime_failure_is_followed_by_a_fix_and_succeeds(self, scene):
        """`ctx` 在命名空间里不存在 → 运行期 NameError → 重试时改了写法。"""
        c = FakeClient(BAD_RUNTIME, GOOD_PROG)
        run = AgentLoop(c, ctx=ToolContext(scene=scene)).run("问")
        assert run.status == "ok"
        feedback = c.calls[1]["messages"][-1]["content"]
        assert "NameError" in feedback and "ctx" in feedback

    def test_tool_error_code_reaches_the_feedback(self, scene):
        """幻觉 id → NOT_IN_SCENE → 反馈里要带上错误码与合法 id 清单。"""
        bad = ("```python\n"
               "r = get_3d_position(object_id='chair_9')\n"
               "submit(r.value, evidence=['x'])\n"
               "```")
        c = FakeClient(bad, GOOD_PROG)
        run = AgentLoop(c, ctx=ToolContext(scene=scene)).run("问")
        assert run.status == "ok"
        feedback = c.calls[1]["messages"][-1]["content"]
        assert "NOT_IN_SCENE" in feedback
        assert "chair_1" in feedback          # 合法 id 清单进了反馈（幻觉捕获点的价值）

    def test_retries_are_capped_and_status_is_classified(self, scene):
        c = FakeClient(BAD_STATIC, BAD_STATIC, BAD_STATIC)
        run = AgentLoop(c, ctx=ToolContext(scene=scene), max_synthesis_retries=2).run("问")
        assert run.status == "static_failed" and len(c.calls) == 3
        assert run.failure["stage"] == "static_check"

    def test_zero_retries_means_one_call(self, scene):
        c = FakeClient(BAD_STATIC)
        run = AgentLoop(c, ctx=ToolContext(scene=scene), max_synthesis_retries=0).run("问")
        assert len(c.calls) == 1 and run.status == "static_failed"

    def test_exec_failure_status_carries_the_stage(self, scene):
        bad = "```python\nx = 1 / 0\nsubmit(x, evidence=['x'])\n```"
        c = FakeClient(bad, bad, bad)
        run = AgentLoop(c, ctx=ToolContext(scene=scene)).run("问")
        assert run.status == "exec_failed"
        assert run.failure["stage"] == "runtime"
        assert run.failure["lineno"] == 1        # `x = 1 / 0` 是程序的第 1 行

    def test_missing_submit_is_caught_statically_at_zero_cost(self, scene):
        """没有 `submit` 的程序**根本不会被执行** —— 一次工具调用都不发生。

        这就是静态检查在「花钱之前」拦住的价值：GPU 一次没动。
        """
        src = "```python\nx = 1\n```"
        c = FakeClient(src, src, src)
        run = AgentLoop(c, ctx=ToolContext(scene=scene)).run("问")
        assert run.status == "static_failed" and run.tool_calls == 0
        assert "submit" in c.calls[1]["messages"][-1]["content"]

    def test_no_submit_at_runtime(self, scene):
        """`submit` 写在走不到的分支里 → 静态检查过得去，运行期才不会交答案。

        这条对应最贵的那类失败：命名空间里没有 `final_result` 就给空串，
        然后**静默算错**。这里必须明确判 `no_submit`。
        """
        src = "```python\nif False:\n    submit(1, evidence=['x'])\n```"
        c = FakeClient(src, src, src)
        run = AgentLoop(c, ctx=ToolContext(scene=scene)).run("问")
        assert run.status == "exec_failed"
        assert run.failure["stage"] == "no_submit"


# ---------------------------------------------------------------------------
# 失败与边界
# ---------------------------------------------------------------------------


class TestFailures:
    def test_llm_error_is_recorded_not_raised(self, scene):
        c = FakeClient(LLMError("HTTP 400: bad", status=400))
        run = AgentLoop(c, ctx=ToolContext(scene=scene)).run("问")
        assert run.status == "llm_error"
        assert run.failure["stage"] == "llm_error" and run.failure["status"] == 400
        assert c.usage.failed_calls == 1

    def test_no_client_raises_before_any_work(self, scene):
        with pytest.raises(ValueError, match="没有 client"):
            AgentLoop(ctx=ToolContext(scene=scene)).run("问")

    def test_no_scene_raises(self):
        with pytest.raises(ValueError, match="没有场景"):
            AgentLoop(FakeClient()).run("问")

    def test_scene_can_be_passed_per_run(self, scene):
        loop = AgentLoop(FakeClient(GOOD_PROG))
        assert loop.run("问", scene=scene).status == "ok"

    def test_tool_loop_is_refused_not_silently_downgraded(self):
        """E′ 未实现时必须**当场报错**：静默跑主路径会让那组消融失去意义。"""
        with pytest.raises(NotImplementedError, match="tool_loop"):
            AgentLoop(FakeClient(), action_space="tool_loop")


class TestRender:
    def test_template_render_keeps_the_number_verbatim(self, scene):
        c = FakeClient(prog_submit("2.103456", "'calculated'"))
        run = AgentLoop(c, ctx=ToolContext(scene=scene)).run("问", answer_type="float")
        assert run.render == "2.10346"          # %.6g —— 不四舍五入到 2 位
        assert run.render_report.get("mode") == "template"

    def test_llm_polish_that_invents_a_number_falls_back(self, scene):
        c = FakeClient(prog_submit("2.1", "'x'"),
                       "大约是 2.1 米，也就是 6.9 英尺吧")     # 6.9 是凭空出现的
        run = AgentLoop(c, ctx=ToolContext(scene=scene),
                        render="llm").run("有多远？", answer_type="float")
        assert run.render == "2.1"
        assert run.render_report["used_fallback"] is True
        assert "6.9" in run.render_report["introduced_numbers"]

    def test_llm_polish_without_new_numbers_is_used(self, scene):
        c = FakeClient(prog_submit("2.1", "'x'"), "该物体的答案为 2.1 米。")
        run = AgentLoop(c, ctx=ToolContext(scene=scene),
                        render="llm").run("有多远？", answer_type="float")
        assert run.render == "该物体的答案为 2.1 米。"
        assert run.render_report["used_fallback"] is False


class TestMemory:
    def test_run_records_memory_for_replay(self, scene):
        c = FakeClient(BAD_STATIC, GOOD_PROG)
        run = AgentLoop(c, ctx=ToolContext(scene=scene)).run("问")
        assert run.memory["n_attempts"] == 1
        assert run.memory["attempts"][0]["stage"] == "static_check"

    def test_injected_memory_is_used_and_filled(self, scene):
        mem = WorkingMemory()
        run = AgentLoop(FakeClient(GOOD_PROG), ctx=ToolContext(scene=scene)).run("问", memory=mem)
        assert mem.question == "问" and mem.scene_hint["objects"]["chair"] == 2
        assert "chair_1" in mem.confirmed_ids          # 从成功 trace 里学到的
        assert run.memory["confirmed_ids"]

    def test_to_dict_is_json_serialisable(self, scene):
        import json

        run = AgentLoop(FakeClient(GOOD_PROG), ctx=ToolContext(scene=scene)).run("问")
        json.dumps(run.to_dict(), ensure_ascii=False)
