#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agents/planner.py —— **planning-then-synthesis**（消融臂 G）。

⚠ 命名口径，先说清楚，因为它决定了报告怎么写
============================================

**这不是「逐步规划」那一套。** 在「动作空间 = Python 程序」这个范式下，
「先说计划再写程序」的含义变了：

    早期基线的 planner      每一步都决定下一个动作，逐步推进，中间结果进上下文
    本文件的 planning       生成程序**之前**，先让模型列出「这题需要哪几类工具、
                            按什么顺序、每一步读什么量」，然后**仍然由一次程序合成
                            把整段推理写成一段代码**

差别在于**计划有没有参与控制流**：本文件产出的计划只是一段前置文本，
它进了下一次合成的提示词，但不产生任何工具调用、不改变执行图。
所以报告里必须叫 **planning-then-synthesis**，不能沿用「逐步规划」的叙事 ——
否则读者会以为臂 G 改的是循环层，而它改的只是**生成前的上下文**。

于是臂 G 与 E′（`action_space="tool_loop"`）是**两个不同的维度**：

    臂 G（本文件）    单轮程序合成 + 前置计划      改的是"想不想清楚"
    E′（tool_loop）   多轮逐步决策                 改的是"能不能回头"

为什么它值得做成开关（`planner="on"/"off"`）
==========================================
因为它是一个**纯增量成本**的干预：多一次 LLM 调用，换"程序里少犯结构性错误"。
它到底值不值，只有跑出来才知道 —— 而这正是消融该回答的问题。
所以本文件只负责把这条臂实现得干净、可复现，**不负责替它辩护**。

三个刻意的设计决定
================

1. **计划里不许出现数字。** 提示词明确禁止，而且产出会被**扫描并记录**
   （`Plan.numbers`）—— 不阻断（那等于替模型改稿），但要可查：
   如果某个计划里出现了"约 1.5 米"，那个数字会随计划一起进下一次合成的提示词，
   而程序合成**无从分辨**它是模型猜的还是工具算的。这是臂 G 特有的污染路径，
   必须留痕。
   ⚠ 扫描范围**只限模型写的那几行**，不含渲染时我们加的步骤序号 ——
   否则每个多步计划都"含数字"，指标饱和、真污染反而看不出来。

2. **工具类别要对着真实动作空间核对。** 计划里列出的类别若不在 `toolset` 里，
   记为 `unknown_categories`。它是个**免费的质量指标**：模型对工具集的记忆有多准。

3. **计划失败不致命。** `plan()` 出任何问题（网络失败 / 不是 JSON）都返回
   `ok=False` 的 `Plan`，调用方**退回无计划合成**并把它记进 `stages`。
   理由：臂 G 是一次"可选的增强"，让它把整题拖挂等于用一个实验开关制造失败，
   而那种失败在报告里会被误读成"模型不会做题"。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from agents.executor import QA_TOOLSET
from llm.adapter import LLMError

__all__ = ["PLANNER_MODES", "Plan", "plan", "render_plan_block"]

#: 开关取值。`"off"` 是主路径。
PLANNER_MODES: tuple[str, ...] = ("off", "on")

_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class Plan:
    """一次前置规划的产物。`ok=False` 时 `error` 说明原因，计划文本为空。"""

    ok: bool = False
    steps: tuple[str, ...] = ()
    tool_categories: tuple[str, ...] = ()
    text: str = ""
    #: **模型自己写的那几行**里出现的数字（**不含**我们渲染时加的步骤序号）——
    #: 臂 G 的污染观测点，见模块 docstring 第 1 条。
    numbers: tuple[str, ...] = ()
    #: 计划里列了、但不在本次动作空间里的工具/类别。
    unknown_categories: tuple[str, ...] = ()
    reply: Any | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "steps": list(self.steps),
            "n_steps": len(self.steps),
            "tool_categories": list(self.tool_categories),
            "unknown_categories": list(self.unknown_categories),
            "numbers_in_plan": list(self.numbers),
            "text": self.text,
            "reply": self.reply.to_dict() if self.reply is not None else None,
            "error": self.error,
        }


def _extract_json(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("计划回复为空")
    for candidate in [b for b in _FENCE_RE.findall(raw)] + [raw]:
        try:
            return json.loads(candidate.strip())
        except (ValueError, TypeError):
            pass
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except (ValueError, TypeError):
            pass
    raise ValueError("计划回复不是 JSON：%r" % raw[:200])


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            text = str(v).strip()
            if text:
                out.append(text)
        return out
    return [str(value).strip()]


def plan(
    client: Any,
    *,
    question: str,
    scene_hint: Mapping[str, Any],
    tool_docs_text: str | None = None,
    tools: Sequence[str] | None = None,
    answer_type: str | None = None,
    purpose: str = "plan",
    deadline: float | None = None,
) -> Plan:
    """调一次 LLM 产出一份**前置计划**。**不做重试**（与 synthesizer 同一分工：
    重试策略属于循环层）。

    任何失败都翻成 `ok=False`，不抛 `LLMError` —— 见模块 docstring 第 3 条。

    ⚠ `deadline`（`time.monotonic()` 绝对时刻 / None）：臂 G 这一次调用**计入同一份
    总预算**，不另给额度。所以总预算不够时，臂 G 可能把预算耗尽、让随后的 synthesis
    直接以「超预算」收尾 —— 这是有意的：预算的口径是「整题的 LLM 时间」，
    按调用方切额度会让「预算 300 s」这句话变成「每个环节各 300 s」。
    """
    from agents.prompts import system as prompt_system

    wanted = tuple(QA_TOOLSET if tools is None else tools)
    messages = [
        {"role": "system", "content": prompt_system.build_plan_system_prompt(
            tool_docs_text, wanted)},
        {"role": "user", "content": prompt_system.build_plan_user_prompt(
            question, scene_hint, answer_type=answer_type)},
    ]

    try:
        reply = client.chat(messages, purpose=purpose, deadline=deadline)
    except LLMError as exc:
        return Plan(ok=False, error=str(exc)[:400])

    try:
        payload = _extract_json(reply.text)
    except ValueError as exc:
        return Plan(ok=False, reply=reply, error=str(exc)[:400])

    if isinstance(payload, Mapping):
        steps = _as_list(payload.get("steps"))
        cats = _as_list(payload.get("tool_categories") or payload.get("tools"))
    elif isinstance(payload, list):
        steps, cats = _as_list(payload), []
    else:
        return Plan(ok=False, reply=reply, error="计划 JSON 结构无法识别：%r" % (payload,))

    if not steps and not cats:
        return Plan(ok=False, reply=reply, error="计划里既没有 steps 也没有 tool_categories")

    text = render_plan_block(Plan(steps=tuple(steps), tool_categories=tuple(cats)))
    # ⚠ 扫描对象是**模型自己写的那几行**，不是渲染后的整块。
    # 因为 `render_plan_block` 会给我们给 steps 加序号（"1." "2."），若对整块扫描，
    # **任何 ≥2 步的计划**都会被判成"含数字" —— 指标恒定饱和之后，真正夹带了
    # "约 1.5 米"的那个计划就再也分不出来，而只有后者才会污染下一次合成。
    authored = "\n".join(list(cats) + list(steps))
    numbers = tuple(dict.fromkeys(_NUM_RE.findall(authored)))
    known = set(wanted)
    unknown = tuple(c for c in cats if c not in known and c not in _CATEGORY_ALIASES)

    return Plan(
        ok=True,
        steps=tuple(steps),
        tool_categories=tuple(cats),
        text=text,
        numbers=numbers,
        unknown_categories=unknown,
        reply=reply,
    )


#: 计划里允许出现、但**不是工具名**的粗粒度类别词。
#: 「几何」「场景读取」这类说法对模型组织思路有用，不该被算成"记错了工具集"。
_CATEGORY_ALIASES: frozenset[str] = frozenset({
    "geometry", "geometric", "scene", "scene_read", "perception", "vision",
    "visual", "semantic", "filter", "aggregate", "arithmetic", "arithmetic_ops",
    "几何", "场景", "场景读取", "感知", "视觉", "语义", "筛选", "聚合", "算术",
})


def render_plan_block(p: Plan) -> str:
    """把计划渲染成一段文本 —— **唯一**注入下一次合成的地方。

    工具类别与步骤都保留原始措辞（不翻译、不归纳）：一旦我们替模型改写了它的计划，
    臂 G 观测到的就不再是"模型自己想清楚了多少"，而是"我们帮它想了多少"。
    """
    lines: list[str] = []
    if p.tool_categories:
        lines.append("需要的工具类别：" + "、".join(p.tool_categories))
    if p.steps:
        lines.append("步骤：")
        for i, s in enumerate(p.steps, 1):
            lines.append("  %d. %s" % (i, s))
    return "\n".join(lines)
