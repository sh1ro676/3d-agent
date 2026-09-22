"""整题**总预算**的测试 —— 零联网、零 GPU、零 API 费用。

要钉住的是三件事，一件比一件容易做错：

**① 不设预算时，行为必须与加这个参数之前逐字相同。**
   这是「新增一个可选参数」的最低要求，也是最容易被漏掉的一条：
   只测「设了预算会生效」，不测「不设预算时没变」，就无法排除
   「预算悄悄改变了默认路径」。所以 `deadline=None` 必须一路传成
   `chat(deadline=None)`，且 HTTP 超时**等于**配置值本身。

**② 预算必须在每次 HTTP 尝试前生效，而不是只在轮次边界。**
   按代码里的常数算（`SPATIAL_TIMEOUT=180`、`SPATIAL_MAX_RETRIES=2`）：

       单次 chat 最坏 = 3 × 180 + 退避 6 = 546 s
       一次问答最多 4 次 chat（3 次 synthesize ＋ 臂 G 1 次 plan）
       ⟹ 只在轮次边界检查时，「预算」的实际上界是 `预算 + 546 s`

   那不是预算，是装饰。所以这里要**直接断言 transport 收到的 timeout 被压小了** ——
   不测这一条，就无法区分「预算真的生效」与「预算只是个记在产物里的字段」。

**③ 「超预算停手」与「模型调用失败」是两种结局。**
   `deadline` vs `llm_error`，处置方向相反（查预算 / 查链路）。
   合并计数会让报告里的失败归因指向错误的方向。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agents.loop as loop_module  # noqa: E402
from agents.loop import AgentLoop, STATUSES  # noqa: E402
from llm.adapter import (  # noqa: E402
    DEADLINE_MESSAGE,
    LLMClient,
    LLMError,
    LLMReply,
    LLMSettings,
    UsageLedger,
    call_timeout_s,
    remaining_s,
    retry_sleep_s,
)
from llm.adapter import _RetryableError  # noqa: E402
from scene_graph.schema import Node, SceneGraph  # noqa: E402
from tools import load_tools  # noqa: E402
from tools.registry import ToolContext  # noqa: E402

load_tools()


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """退避必须被抽掉 —— 否则一个用例会真的睡好几秒。"""
    import llm.adapter as adapter

    monkeypatch.setattr(adapter.time, "sleep", lambda *_: None)


class FrozenClock:
    """可控单调钟，**只装在 `agents.loop` 这一个模块上**。

    为什么需要它：要测「轮次边界的检查点真的会拦住下一次调用」，就必须让
    「已超预算」这件事**确定发生**。用 `total_budget_s=1e-6` 那种办法依赖真实时钟，
    在慢机器上是绿的、在快机器上可能翻红 —— 一个偶尔失效的守卫比没有守卫更难发现。
    这里让 `monotonic()` 第一次返回 0（用于算 deadline），之后返回一个大数，
    于是「已过点」是逻辑必然，而不是和计时器赛跑。

    `perf_counter` 保持真实实现：`elapsed_s` 是要报给人看的，不该被伪造。
    """

    def __init__(self) -> None:
        self.monotonic_calls = 0

    def monotonic(self) -> float:
        self.monotonic_calls += 1
        return 0.0 if self.monotonic_calls == 1 else 1.0e6

    @staticmethod
    def perf_counter() -> float:
        return time.perf_counter()


@pytest.fixture
def clock(monkeypatch) -> FrozenClock:
    c = FrozenClock()
    monkeypatch.setattr(loop_module, "time", c)
    return c


def reply_body(text="ok"):
    return {
        "model": "fake-model",
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


class FakeTransport:
    """按脚本返回；每次调用都把收到的 `timeout` 记下来 —— 这是本文件的主角。"""

    def __init__(self, *script):
        self.script = list(script)
        self.calls: list[dict] = []

    def __call__(self, url, payload, headers, timeout):
        self.calls.append({"url": url, "timeout": timeout})
        if not self.script:
            raise AssertionError("transport 被多调了一次")
        nxt = self.script.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt


def settings(**kw) -> LLMSettings:
    base = dict(base_url="https://example.invalid/v1", api_key="sk-test-1234",
                model="fake-model", temperature=0.2, max_tokens=1024,
                max_retries=2, timeout=180.0)
    base.update(kw)
    return LLMSettings(**base)


@pytest.fixture
def scene() -> SceneGraph:
    return SceneGraph(
        scene_id="budget_scene", image_id="rgb.png", up_axis="-y",
        nodes=(Node(id="chair_1", label="chair", centroid_3d=(0.8, 0.1, 2.9)),),
        build_meta={"image_size": [640, 480]},
    )


GOOD_PROG = "```python\nsubmit(1.0, evidence=['literal'])\n```"


class LoopFakeClient:
    """循环层用的假 client。**记录 `deadline`**，而不是用 `**kwargs` 吞掉 ——
    否则「预算传丢了」与「预算传对了」在测试里长得一样。"""

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
                         usage={"prompt_tokens": 500, "completion_tokens": 120},
                         elapsed_s=0.01, attempts=1)
        self.usage.add(reply, 0.001)
        return reply


# ---------------------------------------------------------------------------
# 纯函数：四个判据都可以零成本单独验
# ---------------------------------------------------------------------------


class TestPureHelpers:
    def test_remaining_is_none_without_a_deadline(self):
        assert remaining_s(None) is None

    def test_remaining_counts_down(self):
        left = remaining_s(time.monotonic() + 10.0)
        assert 9.0 < left <= 10.0

    def test_call_timeout_keeps_the_configured_value_without_a_deadline(self):
        """① 的守卫：没有预算 ⟹ 超时**逐字**等于配置值（180.0，不是 179.9）。"""
        assert call_timeout_s(180.0, None) == 180.0

    def test_call_timeout_is_capped_by_the_remaining_budget(self):
        """② 的守卫：剩余 5 s ⟹ 这一次 HTTP 只能给 5 s，不是 180 s。"""
        assert call_timeout_s(180.0, 5.0) == 5.0

    def test_call_timeout_has_a_usable_floor(self):
        """剩余只剩 1 ms 时不能把 1 ms 传下去。

        `urlopen(timeout=0)` 的语义是"非阻塞"（立刻抛），不是"不超时"。
        所以下界是 0.1 s；它只在"马上就到点"时生效，代价可以忽略。
        """
        assert call_timeout_s(180.0, 0.001) == pytest.approx(0.1)

    def test_retry_sleep_matches_the_original_formula_without_a_deadline(self):
        """① 的守卫，另一半：退避公式必须还是 `min(2**attempt, 16)`。

        这条是把「超时 180 s × 3 + 退避 6 s = 546 s」那个算式钉在测试里 ——
        改了退避就等于改了最坏路径，而 546 / 2184 这两个数是报告里要被引用的。
        """
        assert retry_sleep_s(1, None) == 2.0
        assert retry_sleep_s(2, None) == 4.0
        assert retry_sleep_s(3, None) == 8.0
        assert retry_sleep_s(5, None) == 16.0        # 封顶 16
        assert retry_sleep_s(9, None) == 16.0
        assert (3 * 180.0 + retry_sleep_s(1, None) + retry_sleep_s(2, None)) == 546.0

    def test_retry_sleep_is_capped_to_the_deadline(self):
        """睡过预算再抛「超预算」也是超预算，但账要能算清。"""
        assert retry_sleep_s(2, time.monotonic() + 1.0) <= 1.0
        assert retry_sleep_s(2, time.monotonic() - 1.0) == 0.0


# ---------------------------------------------------------------------------
# LLMClient.chat(deadline=...)
# ---------------------------------------------------------------------------


class TestChatDeadline:
    def test_no_deadline_is_byte_for_byte_the_old_behaviour(self, tmp_path):
        """① 最重要的一条：不传 deadline ⟹ 超时就是配置值，且能正常返回。"""
        t = FakeTransport(reply_body("hi"))
        c = LLMClient(settings(timeout=180.0), transport=t,
                      log_path=str(tmp_path / "c.jsonl"))
        reply = c.chat([{"role": "user", "content": "x"}])
        assert reply.text == "hi"
        assert t.calls[0]["timeout"] == 180.0

    def test_expired_deadline_fails_before_sending_anything(self, tmp_path):
        """③ 的守卫（下）：已经过点 ⟹ **一个请求都不发**，且失败要计数。"""
        t = FakeTransport(reply_body("never"))
        c = LLMClient(settings(), transport=t, log_path=str(tmp_path / "c.jsonl"))
        with pytest.raises(LLMError) as ei:
            c.chat([{"role": "user", "content": "x"}],
                   deadline=time.monotonic() - 1.0)
        assert t.calls == [], "超预算时不该发出任何请求"
        assert ei.value.over_budget is True
        assert DEADLINE_MESSAGE in str(ei.value)
        assert c.usage.failed_calls == 1, "「一个请求都没发出去」也必须计入失败"

    def test_deadline_shrinks_the_http_timeout(self, tmp_path):
        """② 的核心断言：预算 5 s ⟹ transport 收到的 timeout 是 5 左右，不是 180。"""
        t = FakeTransport(reply_body("hi"))
        c = LLMClient(settings(timeout=180.0), transport=t,
                      log_path=str(tmp_path / "c.jsonl"))
        c.chat([{"role": "user", "content": "x"}], deadline=time.monotonic() + 5.0)
        assert 4.0 < t.calls[0]["timeout"] <= 5.0

    def test_deadline_stops_the_retry_loop(self, tmp_path, monkeypatch):
        """② 的另一半：重试链上也要受约束 —— 第一次失败后若已过点，不再发第二次。

        用打桩的 `remaining_s` 造出确定的时间线（而不是真等）：
        先给 5 s 让第一次尝试发出去，第二次尝试时已过点。
        """
        import llm.adapter as adapter

        timeline = [5.0, -1.0]

        def fake_remaining(deadline):
            return timeline.pop(0) if timeline else -1.0

        monkeypatch.setattr(adapter, "remaining_s", fake_remaining)

        t = FakeTransport(_RetryableError("HTTP 503"), reply_body("never"))
        c = LLMClient(settings(), transport=t, log_path=str(tmp_path / "c.jsonl"))
        with pytest.raises(LLMError) as ei:
            c.chat([{"role": "user", "content": "x"}], deadline=time.monotonic() + 5.0)
        assert len(t.calls) == 1, "超预算后不该再发第二次请求"
        assert ei.value.over_budget is True

    def test_plain_llm_error_is_not_flagged_as_over_budget(self, tmp_path):
        """③ 的守卫（上）：普通失败**不能**被标成 over_budget，否则两种结局会混。"""
        t = FakeTransport(_RetryableError("boom"), _RetryableError("boom"),
                          _RetryableError("boom"))
        c = LLMClient(settings(), transport=t, log_path=str(tmp_path / "c.jsonl"))
        with pytest.raises(LLMError) as ei:
            c.chat([{"role": "user", "content": "x"}])
        assert ei.value.over_budget is False
        assert c.usage.failed_calls == 1

    def test_llmerror_default_flag_is_false(self):
        """默认必须是 False：老调用点一个都没改，不能因为新字段改变既有语义。"""
        assert LLMError("x").over_budget is False


# ---------------------------------------------------------------------------
# 循环层：预算的接线
# ---------------------------------------------------------------------------


class TestLoopBudgetWiring:
    def test_no_budget_passes_none_all_the_way_down(self, scene):
        """① 的守卫（循环层）：不设预算 ⟹ 每次 chat 收到的都是 `None`。"""
        c = LoopFakeClient(GOOD_PROG)
        run = AgentLoop(c, ctx=ToolContext(scene=scene)).run("问")
        assert run.status == "ok"
        assert [call["deadline"] for call in c.calls] == [None]

    def test_budget_reaches_every_llm_call(self, scene):
        """② 的守卫（循环层）：设了预算 ⟹ 每一次 chat 拿到同一个**绝对** deadline。"""
        c = LoopFakeClient(GOOD_PROG)
        before = time.monotonic()
        run = AgentLoop(c, ctx=ToolContext(scene=scene), total_budget_s=30.0).run("问")
        assert run.status == "ok"
        deadline = c.calls[0]["deadline"]
        assert deadline is not None
        assert before + 30.0 <= deadline <= before + 31.0
        assert all(call["deadline"] == deadline for call in c.calls), \
            "每一次调用必须共用同一个 deadline，不能各自重新计时"

    def test_plan_costs_against_the_same_budget(self, scene):
        """臂 G 的计划调用也吃同一份预算（不另给额度）。"""
        c = LoopFakeClient('{"steps": ["先 list_objects"]}', GOOD_PROG)
        run = AgentLoop(c, ctx=ToolContext(scene=scene), planner="on",
                        total_budget_s=30.0).run("问")
        assert run.status == "ok"
        assert [call["purpose"] for call in c.calls] == ["plan", "synthesize"]
        assert c.calls[0]["deadline"] == c.calls[1]["deadline"] is not None

    def test_budget_is_part_of_the_switch_fingerprint(self, scene):
        """预算进 `switches`：`None` 与 300 s 是**两次不同的实验**。"""
        c = LoopFakeClient(GOOD_PROG)
        loop = AgentLoop(c, ctx=ToolContext(scene=scene), total_budget_s=300.0)
        loop.run("问")
        assert loop.switches()["total_budget_s"] == 300.0

        c2 = LoopFakeClient(GOOD_PROG)
        loop2 = AgentLoop(c2, ctx=ToolContext(scene=scene))
        loop2.run("问")
        assert loop2.switches()["total_budget_s"] is None

    def test_zero_budget_is_rejected_at_construction(self, scene):
        """`0` 不是「不设上界」，是「一开始就超预算」。

        允许它会得到一个「每次都以 deadline 收尾」的循环，而原因看起来像模型故障。
        """
        c = LoopFakeClient(GOOD_PROG)
        with pytest.raises(ValueError, match="total_budget_s"):
            AgentLoop(c, ctx=ToolContext(scene=scene), total_budget_s=0)
        with pytest.raises(ValueError, match="total_budget_s"):
            AgentLoop(c, ctx=ToolContext(scene=scene), total_budget_s=-1.0)


# ---------------------------------------------------------------------------
# 循环层：超预算是一种**独立**的结局
# ---------------------------------------------------------------------------


class TestDeadlineIsADistinctOutcome:
    def test_expired_budget_stops_before_the_first_call(self, scene, clock):
        """③ 的守卫：预算已过点 ⟹ 状态是 `deadline`，且**一次都没调**模型。"""
        c = LoopFakeClient(GOOD_PROG)
        run = AgentLoop(c, ctx=ToolContext(scene=scene), total_budget_s=30.0).run("问")
        assert c.calls == [], "已经超预算了就不该再发请求"
        assert run.status == "deadline"
        assert run.failure["stage"] == "deadline"
        assert run.failure["budget_s"] == pytest.approx(30.0)
        assert "主动停手" in run.failure["message"]
        assert "轮次边界" in run.failure["where"]

    def test_the_synthesize_stage_row_records_the_stop(self, scene, clock):
        """阶段表里也要留下这一笔 —— 否则「阶段停在 plan 之后」看不出为什么。"""
        c = LoopFakeClient(GOOD_PROG)
        run = AgentLoop(c, ctx=ToolContext(scene=scene), total_budget_s=30.0).run("问")
        synth_rows = [s for s in run.stages if s["stage"] == "synthesize"]
        assert len(synth_rows) == 1 and synth_rows[0]["ok"] is False
        assert "预算" in synth_rows[0]["detail"]

    def test_over_budget_llm_error_becomes_deadline(self, scene):
        """`chat` 从内部报超预算 ⟹ 也必须翻成 `deadline`，不能落到 `llm_error`。"""
        c = LoopFakeClient(LLMError("超预算了", over_budget=True))
        run = AgentLoop(c, ctx=ToolContext(scene=scene), total_budget_s=30.0).run("问")
        assert run.status == "deadline"
        assert run.status != "llm_error"
        assert "调用内部" in run.failure["where"]

    def test_ordinary_llm_error_stays_llm_error(self, scene):
        """反向守卫：普通调用失败**不能**被误记成超预算。"""
        c = LoopFakeClient(LLMError("HTTP 500", status=500))
        run = AgentLoop(c, ctx=ToolContext(scene=scene), total_budget_s=30.0).run("问")
        assert run.status == "llm_error"
        assert run.failure["status"] == 500

    def test_both_outcomes_are_declared_and_distinct(self):
        assert "deadline" in STATUSES and "llm_error" in STATUSES
        assert len(set(STATUSES)) == len(STATUSES)

    def test_deadline_is_counted_as_not_ok(self, scene, clock):
        """`run.ok` 必须为 False —— 超预算不是一种成功。"""
        c = LoopFakeClient(GOOD_PROG)
        run = AgentLoop(c, ctx=ToolContext(scene=scene), total_budget_s=30.0).run("问")
        assert run.ok is False
        assert run.answer is None
