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
    "CACHE_FIELDS",
    "LLMError",
    "LLMSettings",
    "LLMReply",
    "UsageLedger",
    "LLMClient",
    "load_backend_env",
    "cache_tokens",
    "usage_delta",
    "mask_secret",
    # 预算相关的纯函数与常量（`agents/loop.py` 的 total_budget_s 靠它们落地）
    "DEADLINE_MESSAGE",
    "remaining_s",
    "call_timeout_s",
    "retry_sleep_s",
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


#: 超出调用方给的预算时抛出的消息（`chat(..., deadline=...)`）。
#:
#: 定义成常量而不是就地写字符串，是为了让**测试能钉住它**，并且让日志里出现这句话时
#: 一眼能认出这是「我们主动停手」，不是「模型/网络失败」—— 两者的处置完全不同：
#: 前者要调预算或查为什么单次调用变慢，后者要查链路。
DEADLINE_MESSAGE = "调用前已超出调用方给的总预算，主动放弃这次请求（不是网络或模型失败）"


class LLMError(RuntimeError):
    """一次调用彻底失败（重试耗尽 / 配置错误 / 不可重试的 4xx / **超预算主动停手**）。

    调用方（`agents/loop.py`）应当把它转成一条**被记录在案的失败**，
    而不是让它冒到顶层把整轮实验打断 —— 但绝不能吞掉不记。
    """

    def __init__(self, message: str, *, status: int | None = None, body: str = "",
                 over_budget: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.body = body
        #: ★ True ⟺ 这是**我们**主动停手（预算用完），不是模型/网络失败。
        #:
        #: 为什么要一个显式布尔而不是让调用方去比消息字符串：`agents/loop.py` 要把
        #: 这两件事记成**两种不同的结局**（`llm_error` vs `deadline`）。靠字符串匹配
        #: 来分类，会在有人改一句文案时静默失效 —— 而失效的方向是「主动停手被记成
        #: 模型故障」，正好让报告里的失败归因失真。
        self.over_budget = bool(over_budget)


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


#: 单次 HTTP 尝试的最小超时。`urlopen(timeout=0)` 的语义是"非阻塞"（立刻抛），
#: 不是"不超时"，所以预算只剩几毫秒时不能把 0 传下去 —— 那会把一次本来能成的
#: 快速请求变成必然失败。0.1 s 是本机到国内端点一次正常往返的量级下界。
_MIN_CALL_TIMEOUT_S = 0.1


def remaining_s(deadline: float | None) -> float | None:
    """距离 `deadline` 还剩多少秒（None = 不设上界）。**纯函数，可单测。**

    ⚠ 这里用 `time.monotonic()` 而不是 `time.time()`：预算衡量的是**时长**，
    而墙钟会被系统对时/NTP 调整往回拨。用墙钟算剩余量，一次对时就能让预算
    凭空多出或消失几秒 —— 一个偶尔失效的限时器比没有限时器更难发现。
    """
    if deadline is None:
        return None
    return deadline - time.monotonic()


def call_timeout_s(configured: float, left: float | None) -> float:
    """这一次 HTTP 尝试该用多长超时 = `min(配置值, 剩余预算)`，并留一个可用下界。

    ⚠ 为什么必须**取小**而不是直接用配置值：只用配置值的话，预算只剩 3 s 时
    仍然会发一个最长 180 s 的请求，于是「总预算」的实际上界变成
    「预算 + 180 s × 重试次数」。那不是预算，是装饰。
    """
    if left is None:
        return configured
    return max(_MIN_CALL_TIMEOUT_S, min(configured, left))


def retry_sleep_s(attempt: int, deadline: float | None) -> float:
    """重试退避时长，**封顶到 deadline**。与 `chat()` 里的 `min(2**attempt, 16)` 同口径。

    封顶的理由：睡过预算再抛「超预算」确实还是超预算，但那样「预算 300 s、
    实际用了 306 s」这种账会说不清 —— 而这些秒数在报告里是要被引用的。
    """
    base = float(min(2 ** attempt, 16))
    left = remaining_s(deadline)
    if left is None:
        return base
    return max(0.0, min(base, left))


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
        """按「输入 token × 输入价 + 输出 token × 输出价」估算人民币。

        ⚠ **刻意不套用缓存折扣**：价格表只有两个数，缓存价必须由服务商定义，
        猜一个系数等于把「估计」伪装成「测量」。当前口径**系统性偏高**
        （命中缓存的输入照全价算），对预算而言这是安全方向；
        真实成本以服务商账单为准。
        """
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


#: 「命中前缀缓存的输入 token」在不同服务商那里叫不同名字。
#: **不猜、不归一、全试一遍并把来源记下来** —— 猜错字段的后果不是报错，
#: 而是**永远读到 0**，然后被当成「缓存没生效」，据此做错架构决策。
#: 顺序即优先级，按「OpenAI 兼容面最广」排在最前。
CACHE_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("prompt_tokens_details.cached_tokens", ("prompt_tokens_details", "cached_tokens")),
    ("prompt_cache_hit_tokens", ("prompt_cache_hit_tokens",)),
    ("cache_read_input_tokens", ("cache_read_input_tokens",)),
)


def cache_tokens(usage: Mapping[str, Any]) -> tuple[int | None, str | None]:
    """取「命中前缀缓存的输入 token」，返回 `(值, 字段名)`。

    **没找到时返回 `(None, None)`，不返回 0** —— 这个区分是刻意的：
    「服务商报了 0」和「服务商根本没报这个字段」是两件不同的事。
    合并成 0 的后果是「缓存到底生不生效」这个问题**永远无法回答**，
    而它恰好是判断多轮方案真实成本的关键（本项目已踩过一次同类坑：
    把「没实现」当成「实现了但数字是 0」）。
    """
    for label, path in CACHE_FIELDS:
        cur: Any = usage
        for part in path:
            cur = cur.get(part) if isinstance(cur, Mapping) else None
            if cur is None:
                break
        if cur is None:
            continue
        try:
            return int(cur), label
        except (TypeError, ValueError):
            continue
    return None, None


@dataclass
class UsageLedger:
    """累计用量。跑批时把「一个题集花掉多少」直接读出来，不必事后解析日志。"""

    calls: int = 0
    failed_calls: int = 0
    retried_calls: int = 0
    truncated: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: 命中前缀缓存的输入 token 合计。**只有服务商报了才累加**（见下一字段）。
    cached_tokens: int = 0
    #: 回报过缓存字段的调用次数。判断缓存率时**必须**以它为分母：
    #: `cache_reported_calls == 0` 表示服务商没报，不是「缓存没命中」。
    cache_reported_calls: int = 0
    cost_cny: float = 0.0
    unpriced_calls: int = 0
    latency_s: float = 0.0
    #: 失败调用耗掉的时间。**失败也要算进墙钟** —— 否则「花了 24 秒却记 calls=0」
    #: 这种记录会让总耗时被系统性算少，而它正是全量跑批的预算依据。
    failed_latency_s: float = 0.0
    by_purpose: dict[str, dict[str, float]] = field(default_factory=dict)

    def add(self, reply: "LLMReply", cost: float | None) -> None:
        self.calls += 1
        if reply.attempts > 1:
            self.retried_calls += 1
        if reply.truncated:
            self.truncated += 1
        pt = int(reply.usage.get("prompt_tokens") or 0)
        ct = int(reply.usage.get("completion_tokens") or 0)
        hit, _src = cache_tokens(reply.usage)
        self.prompt_tokens += pt
        self.completion_tokens += ct
        if hit is not None:
            self.cached_tokens += hit
            self.cache_reported_calls += 1
        self.latency_s += reply.elapsed_s
        if cost is None:
            self.unpriced_calls += 1
        else:
            self.cost_cny += cost
        slot = self.by_purpose.setdefault(
            reply.purpose, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                            "cached_tokens": 0, "cost_cny": 0.0, "latency_s": 0.0}
        )
        slot["calls"] += 1
        slot["prompt_tokens"] += pt
        slot["completion_tokens"] += ct
        if hit is not None:
            slot["cached_tokens"] += hit
        slot["cost_cny"] += cost or 0.0
        slot["latency_s"] += reply.elapsed_s

    def add_failure(self, elapsed_s: float = 0.0) -> None:
        self.failed_calls += 1
        self.failed_latency_s += max(0.0, float(elapsed_s or 0.0))

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
                "cached_tokens": self.cached_tokens,
                "cache_reported_calls": self.cache_reported_calls,
                "cost_cny": round(self.cost_cny, 6),
                "unpriced_calls": self.unpriced_calls,
                "latency_s": round(self.latency_s, 3),
                "failed_latency_s": round(self.failed_latency_s, 3),
                "by_purpose": self.by_purpose,
            }
        )

    def cache_hit_rate(self) -> float | None:
        """缓存命中率 = 命中 token / **账本里全部调用的 prompt token 合计**。

        **没报过就返回 None，不返回 0.0。** 返回 0 会让调用方以为
        「测过了，没命中」，而真相是「压根没测到」——两者该做的决策完全相反。

        口径说明：同一个端点要么全报、要么全不报，那种情况下这个比值是精确的；
        若只有部分调用回报（异常配置），分母偏大 ⟹ **结果偏小（保守）**。
        """
        if not self.cache_reported_calls:
            return None
        return self.cached_tokens / max(1, self.prompt_tokens)


def usage_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """两个 snapshot 之差 —— 「这一题花了多少」唯一正确的算法。

    为什么不做成 `UsageLedger.diff`：快照是**纯数据**，差值也该是纯函数，
    这样它可以在没有 client 的地方被单测（含负数、缺键、by_purpose 变空）。
    """
    out: dict[str, Any] = {}
    for k in ("calls", "failed_calls", "retried_calls", "truncated",
              "prompt_tokens", "completion_tokens",
              "cached_tokens", "cache_reported_calls",
              "unpriced_calls", "cost_cny", "latency_s", "failed_latency_s"):
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

    def chat(self, messages: Sequence[Mapping[str, Any]], *, purpose: str = "chat",
             deadline: float | None = None) -> LLMReply:
        """一次带重试的对话补全。失败抛 `LLMError`（已计入 `failed_calls`）。

        `deadline` 是 `time.monotonic()` 坐标系下的**绝对时刻**（None = 不设上界）。
        它的检查点是**每一次 HTTP 尝试之前**，不是调用方那一层 —— 理由见循环体内
        那段注释，一句话：否则预算的实际上界会变成「预算 + 546 s」，等于没定。
        """
        t_pre = time.perf_counter()
        try:
            self.check_ready()
        except LLMError as exc:
            # 预检失败也要记账 + 落日志。
            # 「一个请求都没发出去」和「发了但失败」是两种成本结构完全不同的失败，
            # 但都必须是**失败**：不计数的话，报告里的失败率会凭空偏低。
            self.usage.add_failure(time.perf_counter() - t_pre)
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
            # ★ 预算的**唯一有效检查点**就在这里。
            #   为什么不能只放在调用方（`AgentLoop.run` 的轮次边界）：调用方只看得到
            #   「两次 synthesize 之间」，而一次 `chat()` 内部合法地含
            #   `max_retries + 1` 次 HTTP 尝试，每次都允许用满 `s.timeout`。
            #   按代码里的常数算（SPATIAL_TIMEOUT=180、SPATIAL_MAX_RETRIES=2）：
            #       单次 chat 最坏 = 3 × 180 + (2 + 4) = 546 s
            #       一次问答最多 3 次 synthesize（+ 臂 G 1 次 plan）= 4 次 chat
            #       ⟹ deadline 若只查轮次边界，上界 = 预算 + 546 s，而不是预算。
            #   只有这里同时知道「还剩多少」和「怎么把它变成 urlopen(timeout=)」。
            left = remaining_s(deadline)
            if left is not None and left <= 0:
                # 「一个请求都没发出去」也要**计一次失败**：与前面预检失败同一条口径，
                # 不计数会让报告里的失败率凭空偏低。落日志便于事后归因。
                self.usage.add_failure(time.perf_counter() - t_pre)
                self._log_failure(purpose, attempt, DEADLINE_MESSAGE, t_pre, retryable=False)
                raise LLMError(DEADLINE_MESSAGE, status=None, body=DEADLINE_MESSAGE,
                               over_budget=True)

            attempt_timeout = call_timeout_s(s.timeout, left)
            t0 = time.perf_counter()
            try:
                body = self._transport(s.endpoint, payload, headers, attempt_timeout)
            except _FatalError as exc:
                last_error, last_status = str(exc), _status_of(exc)
                self.usage.add_failure(time.perf_counter() - t0)
                self._log_failure(purpose, attempt, last_error, t0, retryable=False)
                raise LLMError("不可重试的调用失败：%s" % last_error,
                               status=last_status, body=last_error) from None
            except _RetryableError as exc:
                last_error, last_status = str(exc), None
                self._log_failure(purpose, attempt, last_error, t0, retryable=True)
                if attempt <= s.max_retries:
                    time.sleep(retry_sleep_s(attempt, deadline))
                continue
            except Exception as exc:          # noqa: BLE001  自定义 transport 的锅也要能看见
                last_error, last_status = "%s: %s" % (type(exc).__name__, exc), None
                self._log_failure(purpose, attempt, last_error, t0, retryable=True)
                if attempt <= s.max_retries:
                    time.sleep(retry_sleep_s(attempt, deadline))
                continue

            elapsed = time.perf_counter() - t0
            try:
                choice = body["choices"][0]
                message = choice.get("message") or {}
                text = message.get("content") or ""
            except (KeyError, IndexError, TypeError, AttributeError) as exc:
                last_error = "响应结构异常：%s（body 前 200 字：%r）" % (exc, str(body)[:200])
                self.usage.add_failure(time.perf_counter() - t0)
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

        self.usage.add_failure(time.perf_counter() - t_pre)
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
        hit, hit_src = cache_tokens(reply.usage)
        rec.update({
            "ok": True,
            "finish_reason": reply.finish_reason,
            "truncated": reply.truncated,
            "model_returned": reply.model_returned,
            "has_reasoning": reply.has_reasoning,
            "prompt_tokens": reply.usage.get("prompt_tokens"),
            "completion_tokens": reply.usage.get("completion_tokens"),
            #: 命中前缀缓存的输入 token；`None` = **服务商没报这个字段**
            #: （不是「报了 0」）。两者含义相反，日志里必须能分开。
            "cached_tokens": hit,
            #: 命中的值是从哪个字段名读到的。**服务商换了字段名时，
            #: 这一列会从有值变成 None** —— 那才是「缓存悄悄失效」的告警信号。
            "cache_source": hit_src,
            #: 服务商回传的 usage 顶层键。不解析、不过滤、原样留档：
            #: 「它到底报不报缓存」这个问题，靠这一列一次真跑就能回答，
            #: 不必再去猜字段名或翻文档。
            "usage_keys": sorted(str(k) for k in reply.usage.keys()),
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
