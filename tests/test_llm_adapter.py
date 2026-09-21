"""`llm/adapter.py` 的单元测试 —— **零联网**（注入假 transport）、零 GPU。

这组测试的重点不是"能不能调通"，而是**四种失败必须是响的**：

    ① 配置写错 → 抛，而不是回退到默认值（`EXTRA_BODY` 写坏的代价最大，见 §7.4）
    ② 缺 key   → 立刻抛，不重试（把配置错误当网络抖动重试 = 把 0.1 s 拖成几分钟）
    ③ 4xx      → 不重试；429/5xx → 重试
    ④ 密钥     → 进得了环境变量，出不了 `describe()` / 摘要

第 ④ 条是本项目的一条红线：任何把 key 写进产物的路径都是严重缺陷。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.adapter import (  # noqa: E402
    DEFAULT_PRICE_TABLE,
    ENV_ALIASES,
    LLMClient,
    LLMError,
    LLMSettings,
    UsageLedger,
    cache_tokens,
    mask_secret,
    usage_delta,
)
from llm.adapter import _FatalError, _RetryableError  # noqa: E402


# ---------------------------------------------------------------------------
# 假 transport
# ---------------------------------------------------------------------------


def reply_body(text="ok", *, model="deepseek-flash", finish="stop",
               prompt_tokens=100, completion_tokens=20, usage_extra=None,
               reasoning=None):
    """一个成功响应体。`usage_extra` 用来模拟不同服务商的用量字段。

    ⚠ **刻意不默认塞入缓存字段**：默认的假服务商就是「什么都不报」那种，
    这样「没报」这条路径才是被默认覆盖的路径（它比「报了 0」更容易被漏掉）。
    """
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
             "total_tokens": prompt_tokens + completion_tokens}
    if usage_extra:
        usage.update(usage_extra)
    message = {"content": text}
    if reasoning:
        message["reasoning_content"] = reasoning
    return {
        "model": model,
        "choices": [{"message": message, "finish_reason": finish}],
        "usage": usage,
    }


class FakeTransport:
    """按脚本返回。`script` 的每一项是 dict（成功）或 Exception（抛出）。"""

    def __init__(self, *script):
        self.script = list(script)
        self.calls: list[dict] = []

    def __call__(self, url, payload, headers, timeout):
        self.calls.append({"url": url, "payload": dict(payload), "headers": dict(headers),
                           "timeout": timeout})
        if not self.script:
            raise AssertionError("transport 被多调了一次")
        nxt = self.script.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """重试退避必须被抽掉 —— 否则一个用例会真的睡 2 秒。"""
    import llm.adapter as adapter

    monkeypatch.setattr(adapter.time, "sleep", lambda *_: None)


def settings(**kw):
    base = dict(base_url="https://example.invalid/v1", api_key="sk-test-1234",
                model="deepseek-flash", temperature=0.2, max_tokens=1024,
                max_retries=2, timeout=5.0)
    base.update(kw)
    return LLMSettings(**base)


# ---------------------------------------------------------------------------
# 设置：写错必须响
# ---------------------------------------------------------------------------


class TestSettingsFromEnv:
    def test_defaults_when_nothing_set(self):
        s = LLMSettings.from_env("text", environ={})
        assert s.base_url == "https://api.deepseek.com"
        assert s.model == "deepseek-flash"
        assert s.api_key == ""
        assert s.ready() is False

    def test_every_entry_has_exactly_one_key_name(self):
        """**结构上禁止隐式回退**：每个设置项只许有一个键名。

        双轨（规范名 + 历史别名）的风险是同一个设置两个来源、优先级靠约定而不是
        靠事实 —— 那是配置层面最典型的混淆变量。曾经存在过这样一条双轨，已随
        它服务的那条实验路径一起删除。这里从结构上钉死：任何人往 ENV_ALIASES 里
        加第二个键名，这条断言立刻变红。
        """
        multi = {k: v for k, v in ENV_ALIASES.items() if len(v) != 1}
        assert not multi, "这些设置项有多个键名（等于隐式回退）：%s" % multi
        assert ENV_ALIASES, "ENV_ALIASES 不该为空"

    def test_empty_value_counts_as_unset(self):
        """空串视同未设（与 `llm_env` 同口径）—— 不把空值当成一份有效配置。"""
        env = {"SPATIAL_API_KEY": "", "SPATIAL_MODEL": "deepseek-flash"}
        s = LLMSettings.from_env("text", environ=env)
        assert s.api_key == "" and s.ready() is False
        assert s.model == "deepseek-flash"

    # 反面：静默回退成默认值会让报告里出现一份"不是你写的"配置
    def test_bad_temperature_raises_instead_of_falling_back(self):
        with pytest.raises(ValueError, match="不是浮点数"):
            LLMSettings.from_env("text", environ={"SPATIAL_TEMPERATURE": "0.2x"})

    def test_bad_int_raises(self):
        with pytest.raises(ValueError, match="不是整数"):
            LLMSettings.from_env("text", environ={"SPATIAL_MAX_TOKENS": "8k"})

    def test_broken_extra_body_raises_loudly(self):
        """最贵的一种配置错误：写坏了 → thinking 没关 → temperature 不生效。"""
        with pytest.raises(ValueError) as ei:
            LLMSettings.from_env("text", environ={"SPATIAL_EXTRA_BODY": '{"thinking": '})
        assert "EXTRA_BODY" in str(ei.value) or "SPATIAL_EXTRA_BODY" in str(ei.value)

    def test_extra_body_passes_through_verbatim(self):
        raw = '{"thinking": {"type": "disabled"}}'
        s = LLMSettings.from_env("text", environ={"SPATIAL_EXTRA_BODY": raw})
        assert s.extra_body == {"thinking": {"type": "disabled"}}

    def test_vision_falls_back_to_text_endpoint(self):
        env = {"SPATIAL_BASE_URL": "https://text.invalid", "SPATIAL_MODEL": "m",
               "SPATIAL_API_KEY": "sk-a"}
        v = LLMSettings.from_env("vision", environ=env)
        assert v.base_url == "https://text.invalid" and v.model == "m"

    def test_vision_uses_its_own_endpoint_when_configured(self):
        env = {"SPATIAL_VISION_BASE_URL": "https://v.invalid", "SPATIAL_VISION_MODEL": "vm",
               "SPATIAL_VISION_API_KEY": "sk-v", "SPATIAL_API_KEY": "sk-t"}
        v = LLMSettings.from_env("vision", environ=env)
        assert v.base_url == "https://v.invalid" and v.model == "vm" and v.label == "vision"


class TestSettingsSurface:
    def test_endpoint_no_double_slash(self):
        assert settings(base_url="https://x.invalid/v1/").endpoint == "https://x.invalid/v1/chat/completions"

    def test_describe_never_leaks_key(self):
        d = settings(api_key="sk-super-secret-9999").describe()
        blob = json.dumps(d, ensure_ascii=False)
        assert "sk-super-secret-9999" not in blob
        assert d["api_key"] == "***9999"

    def test_cost_uses_price_table(self):
        s = settings()
        assert s.price() == (3.0, 9.0)
        assert s.cost_cny({"prompt_tokens": 1_000_000, "completion_tokens": 0}) == pytest.approx(3.0)

    def test_unknown_model_has_no_price(self):
        """没收录的模型宁可不算钱，也不猜 —— 否则报告里会出现来源不明的价格。"""
        s = settings(model="some-unknown-model")
        assert s.price() is None
        assert s.cost_cny({"prompt_tokens": 1000, "completion_tokens": 10}) is None

    def test_mask_secret_short_value(self):
        assert mask_secret("ab") == "***"
        assert mask_secret("") == "(empty)"
        assert mask_secret(None) == "(empty)"

    def test_default_price_table_has_deepseek_flash(self):
        assert DEFAULT_PRICE_TABLE["deepseek-flash"] == (3.0, 9.0)


# ---------------------------------------------------------------------------
# 调用：重试策略与失败分级
# ---------------------------------------------------------------------------


class TestChat:
    def test_happy_path_and_payload_shape(self, tmp_path):
        t = FakeTransport(reply_body("hello"))
        c = LLMClient(settings(extra_body={"thinking": {"type": "disabled"}}),
                      transport=t, log_path=str(tmp_path / "calls.jsonl"))
        reply = c.chat([{"role": "user", "content": "hi"}], purpose="unit")
        assert reply.text == "hello" and reply.attempts == 1
        assert reply.truncated is False
        p = t.calls[0]["payload"]
        assert p["model"] == "deepseek-flash" and p["max_tokens"] == 1024
        assert p["thinking"] == {"type": "disabled"}          # extra_body 原样透传
        assert t.calls[0]["headers"]["Authorization"] == "Bearer sk-test-1234"

    def test_missing_key_fails_without_calling_transport(self, tmp_path):
        t = FakeTransport()
        c = LLMClient(settings(api_key=""), transport=t, log_path=str(tmp_path / "c.jsonl"))
        with pytest.raises(LLMError, match="缺少 API key"):
            c.chat([{"role": "user", "content": "hi"}])
        assert t.calls == []                                  # ← 一个请求都没发
        assert c.usage.failed_calls == 1

    def test_429_is_retried_then_succeeds(self, tmp_path):
        t = FakeTransport(_RetryableError("HTTP 429: slow down"), reply_body("ok"))
        c = LLMClient(settings(), transport=t, log_path=str(tmp_path / "c.jsonl"))
        assert c.chat([{"role": "user", "content": "hi"}]).attempts == 2
        assert c.usage.retried_calls == 1

    def test_400_is_not_retried(self, tmp_path):
        t = FakeTransport(_FatalError("HTTP 400: bad request"), reply_body("never"))
        c = LLMClient(settings(), transport=t, log_path=str(tmp_path / "c.jsonl"))
        with pytest.raises(LLMError, match="不可重试"):
            c.chat([{"role": "user", "content": "hi"}])
        assert len(t.calls) == 1                              # 只发了一次
        assert c.usage.failed_calls == 1

    def test_exhausted_retries_raise_with_status_none(self, tmp_path):
        t = FakeTransport(_RetryableError("boom"), _RetryableError("boom"), _RetryableError("boom"))
        c = LLMClient(settings(max_retries=2), transport=t, log_path=str(tmp_path / "c.jsonl"))
        with pytest.raises(LLMError, match="重试 2 次仍失败"):
            c.chat([{"role": "user", "content": "hi"}])
        assert len(t.calls) == 3                              # 1 + 2 次重试

    def test_truncation_is_flagged(self, tmp_path):
        t = FakeTransport(reply_body("half a prog", finish="length"))
        c = LLMClient(settings(), transport=t, log_path=str(tmp_path / "c.jsonl"))
        reply = c.chat([{"role": "user", "content": "hi"}])
        assert reply.truncated is True
        assert c.usage.truncated == 1

    def test_malformed_response_raises_and_logs(self, tmp_path):
        t = FakeTransport({"nope": 1})
        log = tmp_path / "c.jsonl"
        c = LLMClient(settings(), transport=t, log_path=str(log))
        with pytest.raises(LLMError, match="响应结构异常"):
            c.chat([{"role": "user", "content": "hi"}])
        assert c.usage.failed_calls == 1

    def test_call_log_has_tokens_and_cost(self, tmp_path):
        log = tmp_path / "c.jsonl"
        c = LLMClient(settings(), transport=FakeTransport(reply_body()), log_path=str(log))
        c.chat([{"role": "user", "content": "hi"}], purpose="synthesize")
        rec = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
        assert rec["purpose"] == "synthesize" and rec["ok"] is True
        assert rec["prompt_tokens"] == 100 and rec["arm"] == "agent"
        assert rec["est_cost_cny"] == pytest.approx((100 * 3.0 + 20 * 9.0) / 1e6)

    def test_call_log_does_not_store_prompt_by_default(self, tmp_path):
        log = tmp_path / "c.jsonl"
        c = LLMClient(settings(), transport=FakeTransport(reply_body()), log_path=str(log))
        c.chat([{"role": "user", "content": "绝密题目"}])
        assert "绝密题目" not in log.read_text(encoding="utf-8")
        assert json.loads(log.read_text(encoding="utf-8").splitlines()[0])["msg_digest"]["text_chars"] == 4


# ---------------------------------------------------------------------------
# 记账
# ---------------------------------------------------------------------------


class TestUsage:
    def test_delta_of_single_call(self, tmp_path):
        c = LLMClient(settings(), transport=FakeTransport(reply_body()), log_path=str(tmp_path / "c.jsonl"))
        before = c.usage.snapshot()
        c.chat([{"role": "user", "content": "hi"}], purpose="synthesize")
        d = usage_delta(before, c.usage.snapshot())
        assert d["calls"] == 1 and d["prompt_tokens"] == 100
        assert d["cost_cny"] == pytest.approx((100 * 3.0 + 20 * 9.0) / 1e6)

    def test_by_purpose_is_split(self, tmp_path):
        c = LLMClient(settings(), transport=FakeTransport(reply_body(), reply_body()),
                      log_path=str(tmp_path / "c.jsonl"))
        c.chat([{"role": "user", "content": "a"}], purpose="synthesize")
        c.chat([{"role": "user", "content": "b"}], purpose="polish")
        bp = c.usage.snapshot()["by_purpose"]
        assert set(bp) == {"synthesize", "polish"} and bp["synthesize"]["calls"] == 1

    def test_unpriced_calls_are_counted_not_hidden(self, tmp_path):
        c = LLMClient(settings(model="mystery-model"),
                      transport=FakeTransport(reply_body()), log_path=str(tmp_path / "c.jsonl"))
        c.chat([{"role": "user", "content": "hi"}])
        snap = c.usage.snapshot()
        assert snap["unpriced_calls"] == 1 and snap["cost_cny"] == 0.0

    def test_delta_handles_missing_keys(self):
        d = usage_delta({}, {"calls": 2, "cost_cny": 1.5})
        assert d["calls"] == 2 and d["prompt_tokens"] == 0 and d["by_purpose"] == {}

    def test_ledger_snapshot_is_a_copy(self):
        led = UsageLedger()
        led.calls = 3
        snap = led.snapshot()
        led.calls = 9
        assert snap["calls"] == 3


# ---------------------------------------------------------------------------
# 缓存计量：**「报了 0」和「没报」必须分得开**
#
# 背景：`synthesizer` 把逐字节稳定的工具文档放进 system，注释声称「能吃服务端
# 前缀缓存」。但账本原先只取 prompt_tokens / completion_tokens ⟹ **这个承诺
# 无法被证实**，于是「多轮方案到底多花多少钱」这个问题也答不了。
#
# 这组测试守的就是那条区分线：把「没报」误当成「报了 0」，会让一个**没测量的
# 问题**看起来像一个**已测量的负面结论**。
# ---------------------------------------------------------------------------


class TestCacheTokensParsing:
    """纯函数，逐字段名验证。**不归一成一种叫法** —— 记下来源才能发现改名。"""

    def test_openai_style_prompt_tokens_details(self):
        v, src = cache_tokens({"prompt_tokens": 900,
                               "prompt_tokens_details": {"cached_tokens": 640}})
        assert v == 640 and src == "prompt_tokens_details.cached_tokens"

    def test_deepseek_native_prompt_cache_hit_tokens(self):
        v, src = cache_tokens({"prompt_tokens": 900, "prompt_cache_hit_tokens": 512})
        assert v == 512 and src == "prompt_cache_hit_tokens"

    def test_anthropic_style_cache_read_input_tokens(self):
        v, src = cache_tokens({"cache_read_input_tokens": 256})
        assert v == 256 and src == "cache_read_input_tokens"

    def test_provider_silent_returns_none_not_zero(self):
        """★ 这条是本组的核心：**没报 ≠ 没命中**。"""
        v, src = cache_tokens({"prompt_tokens": 900, "completion_tokens": 20})
        assert v is None and src is None

    def test_explicit_zero_is_reported_as_zero(self):
        """服务商**明确报 0** 与「压根没报」结果不同：前者是「测过了，没命中」。"""
        v, src = cache_tokens({"prompt_tokens_details": {"cached_tokens": 0}})
        assert v == 0 and src == "prompt_tokens_details.cached_tokens"

    def test_nested_field_of_wrong_type_does_not_crash(self):
        v, src = cache_tokens({"prompt_tokens_details": "not-a-mapping"})
        assert v is None and src is None

    def test_garbage_value_falls_through_to_absent(self):
        v, src = cache_tokens({"prompt_cache_hit_tokens": "abc", "prompt_tokens": 900})
        assert v is None and src is None

    def test_empty_usage_is_absent(self):
        assert cache_tokens({}) == (None, None)


class TestCacheAccounting:
    def test_reported_hit_is_accumulated(self, tmp_path):
        body = reply_body(prompt_tokens=1000,
                          usage_extra={"prompt_tokens_details": {"cached_tokens": 768}})
        c = LLMClient(settings(), transport=FakeTransport(body),
                      log_path=str(tmp_path / "c.jsonl"))
        c.chat([{"role": "user", "content": "hi"}])
        snap = c.usage.snapshot()
        assert snap["cached_tokens"] == 768
        assert snap["cache_reported_calls"] == 1

    def test_silent_provider_leaves_hit_rate_unknown(self, tmp_path):
        """没报 ⟹ 命中率是 `None`。**绝不能是 0.0**，否则等于谎报「测过且没命中」。"""
        c = LLMClient(settings(), transport=FakeTransport(reply_body()),
                      log_path=str(tmp_path / "c.jsonl"))
        c.chat([{"role": "user", "content": "hi"}])
        assert c.usage.cache_reported_calls == 0
        assert c.usage.cached_tokens == 0
        assert c.usage.cache_hit_rate() is None

    def test_hit_rate_uses_cached_over_prompt(self, tmp_path):
        body = reply_body(prompt_tokens=1000,
                          usage_extra={"prompt_tokens_details": {"cached_tokens": 750}})
        c = LLMClient(settings(), transport=FakeTransport(body),
                      log_path=str(tmp_path / "c.jsonl"))
        c.chat([{"role": "user", "content": "hi"}])
        assert c.usage.cache_hit_rate() == pytest.approx(0.75)

    def test_cache_is_split_by_purpose(self, tmp_path):
        body = reply_body(prompt_tokens=800,
                          usage_extra={"prompt_cache_hit_tokens": 600})
        c = LLMClient(settings(), transport=FakeTransport(body),
                      log_path=str(tmp_path / "c.jsonl"))
        c.chat([{"role": "user", "content": "hi"}], purpose="synthesize")
        bp = c.usage.snapshot()["by_purpose"]
        assert bp["synthesize"]["cached_tokens"] == 600

    def test_delta_carries_cache_fields(self, tmp_path):
        """单题缓存率靠 `usage_delta` 算 ⟹ 增量里必须有这两个键。"""
        body = reply_body(usage_extra={"prompt_tokens_details": {"cached_tokens": 64}})
        c = LLMClient(settings(), transport=FakeTransport(body),
                      log_path=str(tmp_path / "c.jsonl"))
        before = c.usage.snapshot()
        c.chat([{"role": "user", "content": "hi"}])
        d = usage_delta(before, c.usage.snapshot())
        assert d["cached_tokens"] == 64 and d["cache_reported_calls"] == 1


class TestCallLogDiagnostics:
    """调用日志是**唯一能回答「服务商到底报了什么」**的地方。"""

    def test_log_records_usage_keys_and_cache_source(self, tmp_path):
        log = tmp_path / "c.jsonl"
        body = reply_body(usage_extra={"prompt_tokens_details": {"cached_tokens": 32},
                                       "prompt_cache_miss_tokens": 68})
        c = LLMClient(settings(), transport=FakeTransport(body), log_path=str(log))
        c.chat([{"role": "user", "content": "hi"}])
        rec = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
        assert rec["cached_tokens"] == 32
        assert rec["cache_source"] == "prompt_tokens_details.cached_tokens"
        assert rec["usage_keys"] == ["completion_tokens", "prompt_cache_miss_tokens",
                                     "prompt_tokens", "prompt_tokens_details",
                                     "total_tokens"]

    def test_log_cache_source_none_when_provider_silent(self, tmp_path):
        """`cache_source` 从有值变 `None` ＝ 服务商改了字段名。
        这是「缓存悄悄失效」的唯一告警信号，必须留在日志里。"""
        log = tmp_path / "c.jsonl"
        c = LLMClient(settings(), transport=FakeTransport(reply_body()), log_path=str(log))
        c.chat([{"role": "user", "content": "hi"}])
        rec = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
        assert rec["cached_tokens"] is None and rec["cache_source"] is None
        assert rec["usage_keys"] == ["completion_tokens", "prompt_tokens", "total_tokens"]


class TestFailedCallWallClock:
    """失败的时间**也是成本**。原先 `add_failure()` 不带参数 ⟹
    真实记录里出现过「花了 24.27 s，usage.calls=0」，总预算据此算会偏小。"""

    def test_add_failure_accumulates_elapsed(self):
        led = UsageLedger()
        led.add_failure(2.5)
        led.add_failure(1.25)
        assert led.failed_calls == 2
        assert led.failed_latency_s == pytest.approx(3.75)

    def test_add_failure_clamps_negative_and_defaults_to_zero(self):
        led = UsageLedger()
        led.add_failure()
        led.add_failure(-1.0)
        assert led.failed_latency_s == 0.0

    def test_failed_chat_leaves_positive_wall_clock(self, tmp_path, monkeypatch):
        """失败路径真的把耗时交上去了（假时钟，不依赖真实耗时）。"""
        import llm.adapter as adapter

        class _Clock:                       # 每次读表前进 0.5 s，永不见底
            def __init__(self):
                self.t = 0.0

            def __call__(self):
                self.t += 0.5
                return self.t

        monkeypatch.setattr(adapter.time, "perf_counter", _Clock())
        t = FakeTransport(_RetryableError("boom"))
        c = LLMClient(settings(max_retries=0), transport=t, log_path=str(tmp_path / "c.jsonl"))
        before = c.usage.snapshot()
        with pytest.raises(LLMError, match="重试 0 次仍失败"):
            c.chat([{"role": "user", "content": "hi"}])
        d = usage_delta(before, c.usage.snapshot())
        assert d["calls"] == 0 and d["failed_calls"] == 1
        assert d["failed_latency_s"] > 0        # 曾经这一项恒为 0

    def test_missing_key_failure_also_records_wall_clock(self, tmp_path, monkeypatch):
        """预检失败（一个请求都没发）同样记账 —— 它是「配置错」而不是「免费」。"""
        import llm.adapter as adapter

        class _Clock:
            def __init__(self):
                self.t = 0.0

            def __call__(self):
                self.t += 0.25
                return self.t

        monkeypatch.setattr(adapter.time, "perf_counter", _Clock())
        c = LLMClient(settings(api_key=""), transport=FakeTransport(),
                      log_path=str(tmp_path / "c.jsonl"))
        with pytest.raises(LLMError, match="缺少 API key"):
            c.chat([{"role": "user", "content": "hi"}])
        assert c.usage.failed_latency_s > 0
