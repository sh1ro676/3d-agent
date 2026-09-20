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
    LLMClient,
    LLMError,
    LLMSettings,
    UsageLedger,
    mask_secret,
    usage_delta,
)
from llm.adapter import _FatalError, _RetryableError  # noqa: E402


# ---------------------------------------------------------------------------
# 假 transport
# ---------------------------------------------------------------------------


def reply_body(text="ok", *, model="deepseek-flash", finish="stop",
               prompt_tokens=100, completion_tokens=20):
    return {
        "model": model,
        "choices": [{"message": {"content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
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

    def test_canonical_key_wins_over_legacy(self):
        env = {"SPATIAL_MODEL": "new-model", "VADAR_MODEL": "old-model"}
        assert LLMSettings.from_env("text", environ=env).model == "new-model"

    def test_falls_back_to_legacy_key(self):
        """自研臂没写 SPATIAL_* 时，必须沿用基线臂那一套（一份配置文件驱动两臂）。"""
        env = {"VADAR_MODEL": "deepseek-flash", "VADAR_API_KEY": "sk-a"}
        s = LLMSettings.from_env("text", environ=env)
        assert s.model == "deepseek-flash" and s.api_key == "sk-a"

    def test_empty_value_counts_as_unset(self):
        env = {"SPATIAL_API_KEY": "", "VADAR_API_KEY": "sk-legacy"}
        assert LLMSettings.from_env("text", environ=env).api_key == "sk-legacy"

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
