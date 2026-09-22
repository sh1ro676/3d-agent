"""溯源类缺陷的守卫：**「这条记录究竟是谁产生的」必须可回答。**

三个缺陷装在同一个文件里，因为它们共享同一类失败模式 ——
**记录自洽、格式合法、不报错，但回答的是另一个问题**：

  B1  `_resolve_ctx` 静默忽略显式传入的场景 ⟹ 「问 B、算 A」，
      而落盘的 `scene_id` 取自旧场景，整条记录看起来无懈可击。
  B3  非重试阶段的 `attempt` 填的是**阶段下标** ⟹ 「render 是第 7 次尝试」
      这句话格式合法且单调递增，所以没人怀疑它。
  B4  末轮静态检查失败时，`program` 是那段从未执行的程序，`trace` 来自上一轮
      ⟹ 拿 `program` 解释 `trace` 得到的是**错归因**，而两个字段各自看都正常。

共同点：它们不会抛异常，只会让后续的分析**安静地算出错的结论**。
所以这里的断言都要求「错配这件事本身在数据里可见」，
而不是只要求「不再发生」—— 后者无法防止将来以别的形式再出现。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.loop import AgentLoop, _NON_ATTEMPT_STAGES, scene_conflict  # noqa: E402
from llm.adapter import LLMError, LLMReply, UsageLedger  # noqa: E402
from scene_graph.schema import BBox3D, Node, SceneGraph  # noqa: E402
from tools import load_tools  # noqa: E402
from tools.registry import ToolContext  # noqa: E402

load_tools()

#: 这些阶段**是**重试轮次，`attempt` 必须是 1..N 的整数。
_ATTEMPT_STAGES = ("synthesize", "static_check", "execute")


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


def mk(node_id, label, xyz, size=(0.5, 0.9, 0.5)) -> Node:
    x, y, z = xyz
    w, h, l = size
    return Node(
        id=node_id, label=label, score=0.8,
        bbox_2d=(10.0, 10.0, 50.0, 60.0), mask_ref=f"masks/{node_id}.png",
        centroid_3d=(x, y, z), extent_3d=(w, h, l),
        bbox_3d=BBox3D(min=(x - w / 2, y - h / 2, z - l / 2),
                       max=(x + w / 2, y + h / 2, z + l / 2)),
        n_points=1000,
    )


def _scene(scene_id: str) -> SceneGraph:
    return SceneGraph(
        scene_id=scene_id, image_id="rgb.png", up_axis="-y",
        nodes=(mk("door_1", "door", (-1.2, 0.0, 1.4), (0.9, 2.0, 0.1)),
               mk("chair_1", "chair", (0.8, 0.1, 2.9))),
        build_meta={"image_size": [640, 480]},
    )


@pytest.fixture
def scene() -> SceneGraph:
    return _scene("unit_scene")


class FakeClient:
    def __init__(self, *script):
        self.script = list(script)
        self.calls: list[dict] = []
        self.usage = UsageLedger()

    def chat(self, messages, *, purpose="chat", deadline=None):
        self.calls.append({"purpose": purpose, "deadline": deadline})
        if not self.script:
            raise AssertionError("client 被多调了一次")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            self.usage.add_failure()
            raise item
        reply = LLMReply(text=item, purpose=purpose, model_returned="fake-model",
                         finish_reason="stop",
                         usage={"prompt_tokens": 300, "completion_tokens": 60},
                         elapsed_s=0.01, attempts=1)
        self.usage.add(reply, 0.001)
        return reply


GOOD_PROG = "```python\nsubmit(1.0, evidence=['literal'])\n```"
#: 静态检查过不了（位置参数写错）—— 于是这一轮的 source **从未被执行**。
BAD_STATIC = "```python\nres = list_objects('chair')\nfinal_result = 1\n```"
#: 调用一个真实工具、产生一条真实 trace，然后自己炸掉。
#:
#: ⚠ `submit(...)` 那行**必须存在，哪怕不可达**：静态检查查的是 AST 里「有没有调用
#: `submit`」（`synthesizer.py` 的 `has_submit`），不是运行期能不能走到。
#: 少了它这一轮会被静态检查拦下、根本不会执行 —— 那样就造不出本文件要测的
#: 「trace 来自这一轮，program 来自另一轮」这个组合。
CALLS_THEN_FAILS = (
    "```python\n"
    "res = find_nearest(anchor='door_1', label='chair', k=1)\n"
    "raise RuntimeError('boom after a real call')\n"
    "submit(res.value[0]['distance_m'], evidence=['unreachable on purpose'])\n"
    "```"
)


# ---------------------------------------------------------------------------
# B1：显式传入的场景不能被静默忽略
# ---------------------------------------------------------------------------


class TestSceneConflictJudge:
    """`scene_conflict` 是纯函数 —— 判据可以零成本单独验。"""

    def test_the_same_object_is_not_a_conflict(self):
        s = _scene("a")
        assert scene_conflict(s, s) is None

    def test_a_missing_side_is_not_a_conflict(self):
        s = _scene("a")
        assert scene_conflict(None, s) is None
        assert scene_conflict(s, None) is None

    def test_the_same_scene_id_is_not_a_conflict(self):
        """同一个 id 的两份对象语义上是同一个场景（权威文件只有一份）。

        这条是**刻意的宽容**：缓存 session 这条路靠它才走得通 ——
        每一轮都从磁盘重载 scene 的调用方不该因为对象不同就被拒。
        """
        assert scene_conflict(_scene("a"), _scene("a")) is None

    def test_different_scene_ids_are_a_conflict(self):
        why = scene_conflict(_scene("a"), _scene("b"))
        assert why is not None
        assert "'a'" in why and "'b'" in why, "报错必须点名两个 scene_id"

    def test_unknown_ids_default_to_conflict(self):
        """取不到 id 时**默认判为冲突**，不放行。

        放行等于把「判断不了」静默记成「没问题」，而这里正是最需要响的地方。
        """

        class Anonymous:
            pass

        why = scene_conflict(Anonymous(), Anonymous())
        assert why is not None

    def test_one_sided_id_is_a_conflict(self):
        class Anonymous:
            pass

        assert scene_conflict(_scene("a"), Anonymous()) is not None
        assert scene_conflict(Anonymous(), _scene("a")) is not None


class TestResolveCtx:
    def test_conflicting_scene_raises_instead_of_answering_from_the_old_one(self):
        """★ 核心用例：拿 A 的 ctx 去问 B，必须报错而不是给出 A 的答案。

        修复前这里**不报错**：`run()` 会用 A 的场景算完，返回一个
        `scene_id="a"` 的正常记录 —— 调用方以为问的是 B。
        """
        a, b = _scene("scene_a"), _scene("scene_b")
        c = FakeClient(GOOD_PROG)
        loop = AgentLoop(c, ctx=ToolContext(scene=a))

        with pytest.raises(ValueError) as ei:
            loop.run("问 B", scene=b)

        msg = str(ei.value)
        assert "scene_a" in msg and "scene_b" in msg
        assert c.calls == [], "报错要发生在**调用模型之前**，不能先算完再抱怨"
        assert loop.ctx.scene is a, "ctx 里的场景不能被悄悄换掉"

    def test_same_scene_id_is_still_accepted(self):
        """不回归：同一场景重复传 ⟹ 照常工作（这也是复用 ctx 的快路径）。"""
        c = FakeClient(GOOD_PROG)
        loop = AgentLoop(c, ctx=ToolContext(scene=_scene("same")))
        run = loop.run("问", scene=_scene("same"))
        assert run.status == "ok" and run.scene_id == "same"

    def test_the_same_object_is_still_accepted(self):
        s = _scene("same")
        c = FakeClient(GOOD_PROG)
        run = AgentLoop(c, ctx=ToolContext(scene=s)).run("问", scene=s)
        assert run.status == "ok" and run.scene_id == "same"

    def test_an_empty_ctx_still_takes_the_scene(self, scene):
        """不回归：ctx 没有场景时，`scene=` 仍然要能把它填上。"""
        from tools.registry import ToolContext as TC

        loop = AgentLoop.__new__(AgentLoop)          # 绕过 __init__，只为测这一个方法
        loop.ctx = TC(scene=None)
        loop.record_trace = True
        resolved = loop._resolve_ctx(scene)
        assert resolved.scene is scene

    def test_two_questions_on_one_scene_still_share_the_ctx(self, scene):
        """不回归：缓存 session 的用法（同一个 loop 连续问同一场景）必须照常。"""
        c = FakeClient(GOOD_PROG, GOOD_PROG)
        loop = AgentLoop(c, ctx=ToolContext(scene=scene))
        assert loop.run("第一题").status == "ok"
        assert loop.run("第二题").status == "ok"
        assert len(c.calls) == 2


# ---------------------------------------------------------------------------
# B3：非重试阶段不能编一个 attempt 出来
# ---------------------------------------------------------------------------


class TestStageAttemptIsNeverFabricated:
    def test_non_retry_stages_have_no_attempt_number(self, scene):
        """`plan` / `render` / `verify` 的 `attempt` 必须是 `None`。

        修复前它们填的是 `len(stages)`（阶段下标）—— 一个看起来像轮次的数。
        """
        c = FakeClient('{"steps": ["先 list_objects"]}', GOOD_PROG)
        run = AgentLoop(c, ctx=ToolContext(scene=scene), planner="on").run("问")
        rows = {s["stage"]: s for s in run.stages}
        for name in _NON_ATTEMPT_STAGES:
            assert name in rows, "这个用例需要 plan/render/verify 三个阶段都在"
            assert rows[name]["attempt"] is None, (
                "%s 不是重试轮次，attempt 不能是编出来的数（收到 %r）"
                % (name, rows[name]["attempt"]))

    def test_retry_stages_carry_the_real_round_number(self, scene):
        c = FakeClient(GOOD_PROG)
        run = AgentLoop(c, ctx=ToolContext(scene=scene)).run("问")
        rounds = [s["attempt"] for s in run.stages if s["stage"] in _ATTEMPT_STAGES]
        assert rounds == [1, 1, 1]

    def test_round_numbers_follow_the_retry_sequence(self, scene):
        """两轮：synthesize 与 execute 的轮次必须是 [1, 2]，不是 [0, 1]。"""
        c = FakeClient(BAD_STATIC, GOOD_PROG)
        run = AgentLoop(c, ctx=ToolContext(scene=scene)).run("问")
        assert [s["attempt"] for s in run.stages if s["stage"] == "synthesize"] == [1, 2]
        assert [s["attempt"] for s in run.stages if s["stage"] == "execute"] == [2]

    def test_render_and_verify_say_which_round_they_apply_to(self, scene):
        """`applies_to_attempt` 才是这两个阶段真正有用的信息。"""
        c = FakeClient(BAD_STATIC, GOOD_PROG)
        run = AgentLoop(c, ctx=ToolContext(scene=scene)).run("问")
        rows = {s["stage"]: s for s in run.stages}
        assert rows["render"]["applies_to_attempt"] == 2
        assert rows["verify"]["applies_to_attempt"] == 2

    def test_attempt_is_none_if_nothing_ever_executed(self, scene):
        """一轮都没执行过 ⟹ `applies_to_attempt` 是 None（而不是 0 或 1）。"""
        c = FakeClient(BAD_STATIC)
        run = AgentLoop(c, ctx=ToolContext(scene=scene),
                        max_synthesis_retries=0).run("问")
        rows = {s["stage"]: s for s in run.stages}
        assert rows["render"]["applies_to_attempt"] is None
        assert rows["verify"]["applies_to_attempt"] is None

    def test_the_full_invariant_holds(self, scene):
        """一条不变量，覆盖全部阶段行 —— 新增阶段时它会立刻红。

        `attempt` 是 `None` ⟺ 该阶段不是重试轮次；否则它必须是正整数。
        """
        c = FakeClient('{"steps": ["x"]}', BAD_STATIC, CALLS_THEN_FAILS, GOOD_PROG)
        run = AgentLoop(c, ctx=ToolContext(scene=scene), planner="on").run("问")
        for row in run.stages:
            a = row["attempt"]
            if row["stage"] in _NON_ATTEMPT_STAGES:
                assert a is None, "%s 的 attempt 应为 None，收到 %r" % (row["stage"], a)
            else:
                assert isinstance(a, int) and a >= 1, (
                    "%s 的 attempt 应为正整数，收到 %r" % (row["stage"], a))


# ---------------------------------------------------------------------------
# B4：program 与 trace 必须能对上，对不上也要看得见
# ---------------------------------------------------------------------------


class TestProgramAndTraceProvenance:
    def test_happy_path_is_same_source(self, scene):
        c = FakeClient(GOOD_PROG)
        run = AgentLoop(c, ctx=ToolContext(scene=scene)).run("问")
        assert run.program_matches_trace is True
        assert run.executed_attempt == 1
        assert run.executed_program == run.program

    def test_last_round_static_failure_makes_the_mismatch_visible(self, scene):
        """★ 核心用例：第 1 轮跑过（产生 trace）、**末轮**静态检查没过。

        修复前：`program` 是末轮那段**从未执行**的代码，`trace` 是第 1 轮的，
        两者并排落盘，看不出不是一套。现在错配必须是**显式可见**的。

        脚本给满 3 轮（默认 `max_synthesis_retries=2` ⟹ 共 3 次生成）——
        这个场景本身就要求「末轮才是静态失败」，少给一份脚本会变成
        「client 被多调了一次」，那是脚本没铺满，不是被测行为的问题。
        """
        c = FakeClient(CALLS_THEN_FAILS, BAD_STATIC, BAD_STATIC)
        run = AgentLoop(c, ctx=ToolContext(scene=scene)).run("问")

        assert run.status == "static_failed"
        assert run.trace, "第 1 轮真的调用过工具，trace 不该是空的"

        # 归属必须清楚
        assert run.executed_attempt == 1
        assert run.program_matches_trace is False
        assert run.executed_program != run.program

        # 而且要用 trace 的内容**证伪**「program 解释了 trace」这件事
        tools_called = {row.get("tool") for row in run.trace}
        assert "find_nearest" in tools_called
        assert "find_nearest" not in run.program, \
            "第 2 轮那段程序不含 find_nearest —— 它不可能是这段 trace 的来源"
        assert "find_nearest" in run.executed_program, \
            "executed_program 才是与 trace 同源的那一段"

    def test_no_trace_is_not_a_mismatch(self, scene):
        """一次都没执行过 ⟹ 没有 trace 可以对错，不能把它报成「错配」。

        这条防的是反向错误：把「未测量」当成「测到不一致」。
        """
        c = FakeClient(BAD_STATIC)
        run = AgentLoop(c, ctx=ToolContext(scene=scene),
                        max_synthesis_retries=0).run("问")
        assert run.trace == ()
        assert run.executed_program == ""
        assert run.executed_attempt is None
        assert run.program_matches_trace is True

    def test_mismatch_flag_is_in_the_serialised_record(self, scene):
        """派生量必须落盘 —— 读产物的人不必自己重算这条判据。"""
        c = FakeClient(CALLS_THEN_FAILS, BAD_STATIC, BAD_STATIC)
        run = AgentLoop(c, ctx=ToolContext(scene=scene)).run("问")
        data = run.to_dict()
        assert data["program_matches_trace"] is False
        assert data["executed_attempt"] == 1
        assert data["executed_program"] == run.executed_program

    def test_executed_program_is_recorded_at_execution_time(self, scene):
        """记录点在**执行的那一刻**，不是循环结束后按状态猜。

        用一个「执行失败 → 重试成功」的序列来验：两次执行各产生一段 trace 时，
        `executed_program` 必须跟着**最后一次执行**走，而不是跟着最后一次生成走。
        """
        first = CALLS_THEN_FAILS
        second = ("```python\nres = list_objects()\n"
                  "submit(len(res.value), evidence=['len(list_objects())'])\n```")
        c = FakeClient(first, second)
        run = AgentLoop(c, ctx=ToolContext(scene=scene)).run("问")

        assert run.status == "ok"
        assert run.executed_attempt == 2
        assert run.program_matches_trace is True
        assert run.executed_program == run.program
        assert "list_objects" in run.executed_program
        assert "find_nearest" not in run.executed_program

    def test_a_stale_trace_is_not_reused_when_the_last_round_fails_at_runtime(self, scene):
        """补一个容易漏的组合：末轮**执行了**但失败 ⟹ 仍然是同源的。"""
        c = FakeClient("```python\nx = 1 / 0\nsubmit(x, evidence=['x'])\n```")
        run = AgentLoop(c, ctx=ToolContext(scene=scene),
                        max_synthesis_retries=0).run("问")
        assert run.status == "exec_failed"
        assert run.executed_attempt == 1
        assert run.program_matches_trace is True
