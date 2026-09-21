#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""llm/adapter.py —— OpenAI 兼容后端客户端（自研 Agent 侧的唯一对外出口）。

本文件是**自研 Agent 侧的唯一对外出口**：一个形状正常的 OpenAI 兼容客户端。
它不需要迁就任何第三方代码的调用约定，因为 `agents/` 是我们自己写的。
所有入口都读同一套环境变量名，所以**「换了模型」这件事只改一处**
（见下面 ENV_ALIASES）。

四个刻意的设计决定
==================

1. **键名统一为 `SPATIAL_*`，不做隐式回退。**
   只用一套键名：`configs/llm_backend.env` 里写的是哪套配置，跑出来的就是哪套。
   曾经存在过「规范名 + 历史别名」的双轨（为的是同时兼容另一条实验路径），
   已随那条路径一起删除 —— 双轨的风险是**同一个设置有两个来源**，
   而先后顺序一旦不明确，它就是配置层面最典型的混淆变量。

2. **配置写错就抛，不静默用默认值。**
   `SPATIAL_TEMPERATURE=0.2x` 这种笔误如果被"容错"成默认值 0.2，
   实验会带着**看起来正常但并非你写的那份**配置跑完。
   最贵的例子是 `EXTRA_BODY`：写坏了 → thinking 没关 → temperature 不生效 →
   「低温可复现」这句话就不成立了（§7.4 已实测），而报告里不会有任何痕迹。
   所以 `extra_body` / `price_table` 解析失败一律 `ValueError`。

3. **缺 key 立刻抛 `LLMError`，不重试。**
   本项目的早期运行器实测过这条：把配置错误当成网络抖动去重试，
   只会把一个 0.1 秒的失败拖成几分钟，然后给出同样失败的结论。

4. **调用日志按臂分开写。**
   默认 `logs/agent_llm_calls.jsonl`（可用 `SPATIAL_CALL_LOG` 指到别处）。
   不同臂混进同一个 JSONL，token/成本就再也拆不开 —— 而那正是消融表要的数。

用法
====
    from llm.adapter import LLMClient, load_backend_env
    load_backend_env()                      # 读 configs/llm_backend.env（可选，但入口都该调）
    client = LLMClient()                    # 从环境变量组装
    reply = client.chat([{"role": "user", "content": "ping"}], purpose="selftest")
    print(reply.text, client.usage.snapshot())

单测时不联网：`LLMClient(transport=fake)` —— `transport(url, payload, headers, timeout) -> dict`。
"""

from __future__ import annotations

import copy
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "ENV_ALIASES",
    "DEFAULT_PRICE_TABLE",
    "DEFAULT_CALL_LOG",
    "DEFAULT_TEXT_MODEL",
    "DEFAULT_BASE_URL",
    "LLMError",
    "LLMSettings",
    "LLMReply",
    "UsageLedger",
    "LLMClient",
    "load_backend_env",
    "usage_delta",
    "mask_secret",
]

#: 每个设置项对应的环境变量名。一律单名，**不做隐式回退** —— 见模块 docstring 第 1 条。
ENV_ALIASES: Mapping[str, tuple[str, ...]] = {
    "base_url": ("SPATIAL_BASE_URL",),
    "api_key": ("SPATIAL_API_KEY",),
    "model": ("SPATIAL_MODEL",),
    "temperature": ("SPATIAL_TEMPERATURE",),
    "max_tokens": ("SPATIAL_MAX_TOKENS",),
    "max_retries": ("SPATIAL_MAX_RETRIES",),
    "timeout": ("SPATIAL_TIMEOUT",),
    "extra_body": ("SPATIAL_EXTRA_BODY",),
    "price_table": ("SPATIAL_PRICE_TABLE",),
    "call_log": ("SPATIAL_CALL_LOG",),
    "log_prompts": ("SPATIAL_LOG_PROMPTS",),
    "vision_base_url": ("SPATIAL_VISION_BASE_URL",),
    "vision_api_key": ("SPATIAL_VISION_API_KEY",),
    "vision_model": ("SPATIAL_VISION_MODEL",),
    "vision_max_tokens": ("SPATIAL_VISION_MAX_TOKENS",),
    "env_file": ("SPATIAL_ENV_FILE",),
}

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TEXT_MODEL = "deepseek-flash"
DEFAULT_CALL_LOG = os.path.join("logs", "agent_llm_calls.jsonl")

#: 内置价目表（人民币 / 百万 token，[输入, 输出]）。
#: 只收录**官方公开价**；deepseek 取**高峰价**（保守估计，且便于统一口径）。
#: 未收录的模型：调用照跑、token 照记，但**不猜价格**（`cost_cny=None`），
#: 由 `UsageLedger.unpriced_calls` 显式计数 —— 「算不出来的成本」必须看得见。
DEFAULT_PRICE_TABLE: Mapping[str, Sequence[float]] = {
    "deepseek-flash": (3.0, 9.0),
    "deepseek-v4-flash": (3.0, 9.0),
    "deepseek-v4-pro": (9.0, 27.0),
}


# ============================================================================
# 1. 错误与密钥呈现
# ============================================================================


class LLMError(RuntimeError):
    """一次调用彻底失败（重试耗尽 / 配置错误 / 不可重试的 4xx）。

    调用方（`agents/loop.py`）应当把它转成一条**被记录在案的失败**，
    而不是让它冒到顶层把整轮实验打断 —— 但绝不能吞掉不记。
    """

    def __init__(self, message: str, *, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class _RetryableError(Exception):
    """内部：网络层异常 / 429 / 5xx —— 值得重试。"""


class _FatalError(Exception):
    """内部：4xx（除 429）—— 再试多少次都一样。"""


def mask_secret(value: Any) -> str:
    """密钥的对外呈现。与 `llm_env.mask()` 同口径：只露后 4 位。"""
    if not value:
        return "(empty)"
    v = str(value)
    return "***" + v[-4:] if len(v) > 4 else "***"


# ============================================================================
# 2. 设置
# ============================================================================


def _lookup(names: Sequence[str], default: str | None, environ: Mapping[str, str]) -> str | None:
    for n in names:
        v = environ.get(n)
        if v not in (None, ""):
            return v
    return default


def _parse_float(names: Sequence[str], raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            "环境变量 %s 的值 %r 不是浮点数。**故意不回退到默认值 %r** —— "
            "静默回退会让实验带着一份不是你写的配置跑完。" % (names[0], raw, default)
        ) from None


def _parse_int(names: Sequence[str], raw: str | None, default: int) -> int:
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise ValueError(
            "环境变量 %s 的值 %r 不是整数。故意不回退（同上）。" % (names[0], raw)
        ) from None


def _parse_json(names: Sequence[str], raw: str | None, default: Any) -> Any:
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "环境变量 %s 不是合法 JSON（%s）。\n"
            "  ⚠ 这个键**绝不能**静默忽略：最贵的例子是 SPATIAL_EXTRA_BODY 写坏 → "
            "thinking 没关 → temperature 不生效 → 「低温可复现」不成立，而报告里没有痕迹。\n"
            "  收到的值：%r" % (names[0], exc, raw)
        ) from None


@dataclass(frozen=True)
class LLMSettings:
    """一个端点（文本或视觉）的完整配置。frozen —— 跑起来之后不许改。"""

    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    model: str = DEFAULT_TEXT_MODEL
    temperature: float = 0.2
    max_tokens: int = 4096
    max_retries: int = 2
    timeout: float = 180.0
    extra_body: Mapping[str, Any] | None = None
    price_table: Mapping[str, Sequence[float]] = field(default_factory=lambda: dict(DEFAULT_PRICE_TABLE))
    label: str = "text"

    # -- 组装 ----------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        label: str = "text",
        *,
        environ: Mapping[str, str] | None = None,
        vision_fallback: "LLMSettings | None" = None,
    ) -> "LLMSettings":
        """`label="vision"` 时读 `*_VISION_*`；全部为空则回落到 `vision_fallback`
        （默认就是文本端点 —— 与桥接层同口径：deepseek-flash 自带 Vision，
        不需要第二套端点）。"""
        env = os.environ if environ is None else environ
        vprefix = "vision_" if label == "vision" else ""

        def names(key: str) -> tuple[str, ...]:
            """视觉端点只对 `base_url` / `api_key` / `model` / `max_tokens` 有独立键名
            （与桥接层同口径）；温度、重试、超时、extra_body、价目表**两个端点共用**。
            所以这里有 `.get(...) or 回退` 而不是硬索引 —— 硬索引会让
            `SPATIAL_VISION_MODEL` 一填就 KeyError('vision_temperature')。
            """
            if vprefix:
                return ENV_ALIASES.get(vprefix + key) or ENV_ALIASES[key]
            return ENV_ALIASES[key]

        if label == "vision":
            raw_model = _lookup(names("model"), None, env)
            raw_base = _lookup(names("base_url"), None, env)
            if raw_model is None and raw_base is None:
                if vision_fallback is not None:
                    return vision_fallback
                # 没配视觉端点 → 复用文本端点的全部参数（含 key）
                return cls.from_env("text", environ=env)

        max_tokens = _parse_int(
            names("max_tokens"),
            _lookup(names("max_tokens"), None, env),
            4096,
        )
        return cls(
            base_url=_lookup(names("base_url"), DEFAULT_BASE_URL, env) or DEFAULT_BASE_URL,
            api_key=_lookup(names("api_key"), "", env) or "",
            model=_lookup(names("model"), DEFAULT_TEXT_MODEL, env) or DEFAULT_TEXT_MODEL,
            temperature=_parse_float(names("temperature"), _lookup(names("temperature"), None, env), 0.2),
            max_tokens=max_tokens,
            max_retries=_parse_int(names("max_retries"), _lookup(names("max_retries"), None, env), 2),
            timeout=_parse_float(names("timeout"), _lookup(names("timeout"), None, env), 180.0),
            extra_body=_parse_json(names("extra_body"), _lookup(names("extra_body"), None, env), None),
            price_table=_parse_json(
                names("price_table"),
                _lookup(names("price_table"), None, env),
                dict(DEFAULT_PRICE_TABLE),
            ),
            label=label,
        )

    # -- 便利 ----------------------------------------------------------------

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def ready(self) -> bool:
        """能不能发起调用。缺 key 是最常见的一种 —— 调用方据此快速失败。"""
        return bool(self.api_key and self.base_url and self.model)

    def price(self) -> tuple[float, float] | None:
        """`(输入价, 输出价)` / 百万 token。没收录的模型返回 None（不猜）。"""
        for key, val in (self.price_table or {}).items():
            if key and key in self.model and isinstance(val, (list, tuple)) and len(val) == 2:
                return float(val[0]), float(val[1])
        return None

    def cost_cny(self, usage: Mapping[str, Any]) -> float | None:
        p = self.price()
        if p is None:
            return None
        pin = usage.get("prompt_tokens") or 0
        pout = usage.get("completion_tokens") or 0
        return (pin * p[0] + pout * p[1]) / 1e6

    def describe(self) -> dict[str, Any]:
        """进实验产物的形态。**密钥只以掩码出现** —— 这条不能破。"""
        return {
            "label": self.label,
            "base_url": self.base_url,
            "model": self.model,
            "api_key": mask_secret(self.api_key),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries,
            "timeout_s": self.timeout,
            "extra_body": dict(self.extra_body) if self.extra_body else None,
            "price_cny_per_mtok": self.price(),
            "ready": self.ready(),
        }


# ============================================================================
# 3. 计量
# ============================================================================


@dataclass
class UsageLedger:
    """累计用量。跑批时把「一个题集花掉多少」直接读出来，不必事后解析日志。"""

    calls: int = 0
    failed_calls: int = 0
    retried_calls: int = 0
    truncated: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_cny: float = 0.0
    unpriced_calls: int = 0
    latency_s: float = 0.0
    by_purpose: dict[str, dict[str, float]] = field(default_factory=dict)

    def add(self, reply: "LLMReply", cost: float | None) -> None:
        self.calls += 1
        if reply.attempts > 1:
            self.retried_calls += 1
        if reply.truncated:
            self.truncated += 1
        pt = int(reply.usage.get("prompt_tokens") or 0)
        ct = int(reply.usage.get("completion_tokens") or 0)
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.latency_s += reply.elapsed_s
        if cost is None:
            self.unpriced_calls += 1
        else:
            self.cost_cny += cost
        slot = self.by_purpose.setdefault(
            reply.purpose, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                            "cost_cny": 0.0, "latency_s": 0.0}
        )
        slot["calls"] += 1
        slot["prompt_tokens"] += pt
        slot["completion_tokens"] += ct
        slot["cost_cny"] += cost or 0.0
        slot["latency_s"] += reply.elapsed_s

    def add_failure(self) -> None:
        self.failed_calls += 1

    def snapshot(self) -> dict[str, Any]:
        """深拷贝 —— 单题成本靠 `usage_delta(before, after)` 算。"""
        return copy.deepcopy(
            {
                "calls": self.calls,
                "failed_calls": self.failed_calls,
                "retried_calls": self.retried_calls,
                "truncated": self.truncated,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cost_cny": round(self.cost_cny, 6),
                "unpriced_calls": self.unpriced_calls,
                "latency_s": round(self.latency_s, 3),
                "by_purpose": self.by_purpose,
            }
        )


def usage_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """两个 snapshot 之差 —— 「这一题花了多少」唯一正确的算法。

    为什么不做成 `UsageLedger.diff`：快照是**纯数据**，差值也该是纯函数，
    这样它可以在没有 client 的地方被单测（含负数、缺键、by_purpose 变空）。
    """
    out: dict[str, Any] = {}
    for k in ("calls", "failed_calls", "retried_calls", "truncated",
              "prompt_tokens", "completion_tokens", "unpriced_calls",
              "cost_cny", "latency_s"):
        out[k] = round(float(after.get(k, 0) or 0) - float(before.get(k, 0) or 0), 6)
    bp: dict[str, dict[str, float]] = {}
    for name, slot in (after.get("by_purpose") or {}).items():
        prev = (before.get("by_purpose") or {}).get(name) or {}
        d = {k: round(float(slot.get(k, 0) or 0) - float(prev.get(k, 0) or 0), 6) for k in slot}
        if any(v for v in d.values()):
            bp[name] = d
    out["by_purpose"] = bp
    return out


# ============================================================================
# 4. HTTP 传输（可注入，所以单测零联网）
# ============================================================================


def _http_transport(url: str, payload: Mapping[str, Any], headers: Mapping[str, str],
                    timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = ""
        try:
            raw = exc.read().decode("utf-8", errors="replace")[:600]
        except Exception:                     # noqa: BLE001  读错误体失败不该掩盖 HTTP 状态
            pass
        if exc.code == 429 or exc.code >= 500:
            raise _RetryableError("HTTP %s: %s" % (exc.code, raw)) from None
        raise _FatalError("HTTP %s: %s" % (exc.code, raw)) from None
    except Exception as exc:                  # noqa: BLE001  网络层统统可重试
        raise _RetryableError("%s: %s" % (type(exc).__name__, exc)) from None


Transport = Callable[[str, Mapping[str, Any], Mapping[str, str], float], dict[str, Any]]


# ============================================================================
# 5. 调用
# ============================================================================


@dataclass(frozen=True)
class LLMReply:
    text: str
    purpose: str
    model_returned: str | None
    finish_reason: str | None
    usage: Mapping[str, Any]
    elapsed_s: float
    attempts: int
    has_reasoning: bool = False

    @property
    def truncated(self) -> bool:
        """`finish_reason == "length"` —— **必须被当成失败处理**。

        `max_tokens` 不传是个经典坑：程序合成输出被静默截断，
        生成一个语法不完整的程序，然后表现为「模型不会写程序」。
        """
        return self.finish_reason == "length"

    def to_dict(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "model_returned": self.model_returned,
            "finish_reason": self.finish_reason,
            "truncated": self.truncated,
            "has_reasoning": self.has_reasoning,
            "elapsed_s": round(self.elapsed_s, 3),
            "attempts": self.attempts,
            "usage": dict(self.usage),
            "text_chars": len(self.text),
        }


class LLMClient:
    """一个端点的客户端。**调用日志与用量都在这里，不在调用方** ——
    这样「哪一步花了多少钱」不需要调用方配合记账，不会漏记。"""

    def __init__(
        self,
        settings: LLMSettings | None = None,
        *,
        transport: Transport | None = None,
        log_path: str | None = None,
        log_prompts: bool | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        env = os.environ if environ is None else environ
        self.settings = settings or LLMSettings.from_env("text", environ=env)
        self._transport: Transport = transport or _http_transport
        self.log_path = (
            log_path
            or _lookup(ENV_ALIASES["call_log"], DEFAULT_CALL_LOG, env)
            or DEFAULT_CALL_LOG
        )
        if log_prompts is None:
            log_prompts = (_lookup(ENV_ALIASES["log_prompts"], "0", env) == "1")
        self.log_prompts = bool(log_prompts)
        self.usage = UsageLedger()

    # -- 预检 ----------------------------------------------------------------

    def check_ready(self) -> None:
        """缺 key / 缺端点就**立刻**抛 —— 不重试。

        本项目的早期运行器实测过：把配置错误当网络抖动重试，只会把一个 0.1 秒的
        失败拖成几分钟，结论还是同一个失败。
        """
        s = self.settings
        if not s.api_key:
            raise LLMError(
                "缺少 API key。请把 key 填进 configs/llm_backend.env 的 "
                "SPATIAL_API_KEY= 后重跑；"
                "当前端点 %s，模型 %s。" % (s.base_url, s.model)
            )
        if not s.base_url:
            raise LLMError("缺少 base_url（SPATIAL_BASE_URL）")
        if not s.model:
            raise LLMError("缺少 model（SPATIAL_MODEL）")

    # -- 主调用 --------------------------------------------------------------

    def chat(self, messages: Sequence[Mapping[str, Any]], *, purpose: str = "chat") -> LLMReply:
        """一次带重试的对话补全。失败抛 `LLMError`（已计入 `failed_calls`）。"""
        t_pre = time.perf_counter()
        try:
            self.check_ready()
        except LLMError as exc:
            # 预检失败也要记账 + 落日志。
            # 「一个请求都没发出去」和「发了但失败」是两种成本结构完全不同的失败，
            # 但都必须是**失败**：不计数的话，报告里的失败率会凭空偏低。
            self.usage.add_failure()
            self._log_failure(purpose, 1, str(exc), t_pre, retryable=False)
            raise
        s = self.settings
        payload: dict[str, Any] = {
            "model": s.model,
            "messages": list(messages),
            "temperature": s.temperature,
            "max_tokens": s.max_tokens,
        }
        if s.extra_body:
            # 原样透传，不猜参数名（§7.4：`thinking` 这类字段只能由服务商定义）
            payload.update(dict(s.extra_body))

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + s.api_key,
        }

        last_error = ""
        last_status: int | None = None
        for attempt in range(1, s.max_retries + 2):
            t0 = time.perf_counter()
            try:
                body = self._transport(s.endpoint, payload, headers, s.timeout)
            except _FatalError as exc:
                last_error, last_status = str(exc), _status_of(exc)
                self.usage.add_failure()
                self._log_failure(purpose, attempt, last_error, t0, retryable=False)
                raise LLMError("不可重试的调用失败：%s" % last_error,
                               status=last_status, body=last_error) from None
            except _RetryableError as exc:
                last_error, last_status = str(exc), None
                self._log_failure(purpose, attempt, last_error, t0, retryable=True)
                if attempt <= s.max_retries:
                    time.sleep(min(2 ** attempt, 16))
                continue
            except Exception as exc:          # noqa: BLE001  自定义 transport 的锅也要能看见
                last_error, last_status = "%s: %s" % (type(exc).__name__, exc), None
                self._log_failure(purpose, attempt, last_error, t0, retryable=True)
                if attempt <= s.max_retries:
                    time.sleep(min(2 ** attempt, 16))
                continue

            elapsed = time.perf_counter() - t0
            try:
                choice = body["choices"][0]
                message = choice.get("message") or {}
                text = message.get("content") or ""
            except (KeyError, IndexError, TypeError, AttributeError) as exc:
                last_error = "响应结构异常：%s（body 前 200 字：%r）" % (exc, str(body)[:200])
                self.usage.add_failure()
                self._log_failure(purpose, attempt, last_error, t0, retryable=False)
                raise LLMError(last_error) from None

            reply = LLMReply(
                text=text,
                purpose=purpose,
                model_returned=body.get("model"),
                finish_reason=choice.get("finish_reason"),
                usage=body.get("usage") or {},
                elapsed_s=elapsed,
                attempts=attempt,
                has_reasoning=bool(message.get("reasoning_content")),
            )
            cost = s.cost_cny(reply.usage)
            self.usage.add(reply, cost)
            self._log_success(reply, cost, messages)
            return reply

        self.usage.add_failure()
        raise LLMError(
            "重试 %d 次仍失败（status=%s）：%s" % (s.max_retries, last_status, last_error[:400]),
            status=last_status, body=last_error,
        )

    # -- 日志 ----------------------------------------------------------------

    def _write(self, rec: Mapping[str, Any]) -> None:
        try:
            path = os.path.abspath(self.log_path)
            d = os.path.dirname(path)
            if d and not os.path.isdir(d):
                os.makedirs(d, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:                     # noqa: BLE001
            # 记账失败不该让一次成功的推理变成失败；但也不能完全没有痕迹，
            # 所以退化到 stderr 一行。**静默吞掉会导致「成本凭空少了一半」**。
            import sys
            print("[llm] !! 调用日志写入失败：%s" % self.log_path, file=sys.stderr)

    def _base_rec(self, purpose: str, attempt: int, elapsed: float) -> dict[str, Any]:
        return {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "arm": "agent",
            "role": self.settings.label,
            "model": self.settings.model,
            "purpose": purpose,
            "attempts": attempt,
            "elapsed_s": round(elapsed, 3),
        }

    def _log_success(self, reply: LLMReply, cost: float | None,
                     messages: Sequence[Mapping[str, Any]]) -> None:
        rec = self._base_rec(reply.purpose, reply.attempts, reply.elapsed_s)
        rec.update({
            "ok": True,
            "finish_reason": reply.finish_reason,
            "truncated": reply.truncated,
            "model_returned": reply.model_returned,
            "has_reasoning": reply.has_reasoning,
            "prompt_tokens": reply.usage.get("prompt_tokens"),
            "completion_tokens": reply.usage.get("completion_tokens"),
            "est_cost_cny": None if cost is None else round(cost, 6),
            "msg_digest": _digest(messages),
        })
        if reply.truncated:
            rec["warning"] = ("finish_reason=length：输出被截断，程序可能残缺。"
                              "调大 SPATIAL_MAX_TOKENS。")
        if self.log_prompts:
            rec["messages"] = list(messages)
        self._write(rec)

    def _log_failure(self, purpose: str, attempt: int, error: str,
                     t0: float, *, retryable: bool) -> None:
        rec = self._base_rec(purpose, attempt, time.perf_counter() - t0)
        rec.update({"ok": False, "retryable": retryable, "error": error[:600]})
        self._write(rec)


def _status_of(exc: Exception) -> int | None:
    import re
    m = re.match(r"HTTP (\d{3})", str(exc))
    return int(m.group(1)) if m else None


def _digest(messages: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """消息历史的轻量指纹 —— 记形状不记内容（prompt 可能很长，且含题目）。"""
    images = chars = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if not isinstance(part, Mapping):
                    continue
                if part.get("type") == "image_url":
                    images += 1
                elif part.get("type") == "text":
                    chars += len(part.get("text") or "")
    return {"n": len(messages), "images": images, "text_chars": chars}


# ============================================================================
# 6. 配置文件的加载
# ============================================================================


def load_backend_env(path: str | None = None, *, environ: dict[str, str] | None = None) -> dict[str, Any]:
    """把 `configs/llm_backend.env` 读进环境变量，返回**脱敏**报告。

    刻意复用 `llm_env.load_env_file` 而不是自己写一份解析器：
    那是本项目里唯一一处「键值文件 → 环境变量」的实现，已经带着 60+ 条单测
    （重复键报错、空值视同未设、密钥不进摘要）。写第二份 = 造第二个会漂移的真相。
    """
    import llm_env

    if path is None:
        env = os.environ if environ is None else environ
        configured = _lookup(ENV_ALIASES["env_file"], None, env)
        path = configured or llm_env.default_path()
    return llm_env.load_env_file(path, environ=environ)
