#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agents/memory.py —— 工作记忆（**不跨问题**）。

VADAR 没有记忆：每次调用都是全新对话，前面查到过的物体 id、失败原因全部丢掉，
于是同一个幻觉 id 会在重试里被反复编出来。

这里只记两样东西，两样都有明确用途：

1. **失败原因（压缩后）** —— 重试轮的唯一输入。
   刻意**不记完整 traceback**：那是 VADAR 的做法（`agents.py` 把 traceback 原样回灌），
   它把几百 token 花在调用栈上，而模型真正需要的是
   「第 12 行 `get_3d_position()` 拿到了 NOT_IN_SCENE，合法 id 是 [...]」。
   压缩点就是本文件存在的理由（§18 Phase 7 风险②：观察必须压缩）。

2. **已确认的 object_id** —— 从成功的 `list_objects` / `find_object` 的 trace 里抽出来。
   下一次重试时作为「这些 id 是真的」的正面证据。
   它同时是幻觉率的**分母**：模型编了几个不在这个集合里的 id，直接数得出来。

为什么明确写「不跨问题」
--------------------
跨题记忆会让第 5 题的表现取决于第 1–4 题 —— 那是一条**无法控制的混淆变量**，
而且它破坏「每题独立可复现」。要做长期记忆就单独开一条实验臂，
不要偷偷加进主路径。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["Attempt", "WorkingMemory", "MAX_FEEDBACK_ATTEMPTS", "MAX_MESSAGE_CHARS"]

#: 反馈里最多回放几次尝试。留 2 次：再往前的那几次错误通常已经被前一次覆盖，
#: 全部塞进去只会让提示词变长、让模型去修一个已经修好的问题。
MAX_FEEDBACK_ATTEMPTS = 2

#: 单条错误消息的字符上限。工具报错里带了 `available_labels` 这类长列表，
#: 截断是必要的；但**只截尾巴**，因为错误种类与行号都在开头。
MAX_MESSAGE_CHARS = 400


@dataclass(frozen=True)
class Attempt:
    """一次失败的尝试。字段与 `executor.ExecOutcome` / `StaticCheckReport` 对齐，
    这样「静态检查失败」与「执行失败」能用同一个结构被记录与回放。"""

    index: int
    stage: str
    message: str
    code: str | None = None
    lineno: int | None = None
    hint: str | None = None
    recovery: tuple[str, ...] = ()
    source_excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "stage": self.stage,
            "code": self.code,
            "lineno": self.lineno,
            "message": self.message,
            "hint": self.hint,
            "recovery": list(self.recovery),
            "source_excerpt": self.source_excerpt,
        }

    def line(self) -> str:
        bits = ["第 %d 次尝试失败于 `%s`" % (self.index, self.stage)]
        if self.lineno:
            bits.append("（第 %d 行）" % self.lineno)
        text = "%s：%s" % ("".join(bits), self.message)
        if self.code:
            text = "%s\n  错误码 %s" % (text, self.code)
        if self.recovery:
            text = "%s，建议动作 %s" % (text, " / ".join(self.recovery))
        if self.hint:
            text = "%s\n  提示：%s" % (text, self.hint)
        return text


@dataclass
class WorkingMemory:
    """一轮问答的工作记忆。`loop.run()` 每题新建一个。"""

    question: str = ""
    scene_hint: Mapping[str, Any] = field(default_factory=dict)
    attempts: list[Attempt] = field(default_factory=list)
    #: 已确认存在的 object_id（来自成功工具调用的返回值）。
    confirmed_ids: set[str] = field(default_factory=set)
    #: 用户/上游明确给出的事实（例如题目里写明的已知尺寸）。
    facts: dict[str, Any] = field(default_factory=dict)

    # -- 记录 ----------------------------------------------------------------

    def record_attempt(
        self,
        stage: str,
        message: str,
        *,
        code: str | None = None,
        lineno: int | None = None,
        hint: str | None = None,
        recovery: Sequence[str] = (),
        source: str = "",
    ) -> Attempt:
        attempt = Attempt(
            index=len(self.attempts) + 1,
            stage=stage,
            code=code,
            lineno=lineno,
            message=_clip(message, MAX_MESSAGE_CHARS),
            hint=hint,
            recovery=tuple(recovery),
            source_excerpt=_excerpt(source),
        )
        self.attempts.append(attempt)
        return attempt

    def learn_from_trace(self, trace: Iterable[Mapping[str, Any]]) -> list[str]:
        """从 trace 里抽出「确实存在」的 id。返回本次新学到的。

        只认成功返回的读取类工具。`NOT_IN_SCENE` 的 `known_ids` **也学**：
        那条错误里带的正是服务端认可的全部 id，是最权威的一份清单。
        """
        learned: list[str] = []
        for row in trace:
            result = row.get("result") or {}
            tool = row.get("tool")
            if result.get("ok"):
                if tool == "list_objects" and isinstance(result.get("value"), list):
                    learned.extend(str(item.get("object_id")) for item in result["value"]
                                   if isinstance(item, Mapping) and item.get("object_id"))
                elif tool in ("find_object", "single_object", "find_nearest", "find_farthest"):
                    value = result.get("value")
                    rows = value if isinstance(value, list) else [value]
                    learned.extend(str(item.get("object_id")) for item in rows
                                   if isinstance(item, Mapping) and item.get("object_id"))
            else:
                ctx = ((result.get("error") or {}).get("context") or {})
                known = ctx.get("known_ids")
                if isinstance(known, list):
                    learned.extend(str(k) for k in known)
        new = [i for i in dict.fromkeys(learned) if i and i not in self.confirmed_ids]
        self.confirmed_ids.update(new)
        return new

    # -- 回放 ----------------------------------------------------------------

    def feedback(self, *, max_attempts: int = MAX_FEEDBACK_ATTEMPTS) -> str | None:
        """压成一段给模型看的失败说明。没有失败过就返回 None。"""
        if not self.attempts:
            return None
        recent = self.attempts[-max_attempts:]
        lines = [a.line() for a in recent]
        if len(self.attempts) > len(recent):
            lines.insert(0, "（此前还有 %d 次更早的失败，已省略）"
                         % (len(self.attempts) - len(recent)))
        if self.confirmed_ids:
            ids = sorted(self.confirmed_ids)
            shown = ", ".join(ids[:24])
            if len(ids) > 24:
                shown += " …（共 %d 个）" % len(ids)
            lines.append("场景里**确实存在**的 object_id（只能从这些里选）：%s" % shown)
        return "\n".join(lines)

    def reset(self) -> None:
        self.attempts.clear()
        self.confirmed_ids.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "scene_hint": dict(self.scene_hint),
            "n_attempts": len(self.attempts),
            "attempts": [a.to_dict() for a in self.attempts],
            "confirmed_ids": sorted(self.confirmed_ids),
            "facts": dict(self.facts),
        }


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 12].rstrip() + " …（已截断）"


def _excerpt(source: str, limit: int = 240) -> str:
    """失败现场的代码片段 —— 只留前几行。整段程序已经在 run 记录里，不必重复。"""
    lines = [ln for ln in (source or "").splitlines() if ln.strip()]
    text = "\n".join(lines[:6])
    if len(text) > limit:
        text = text[:limit].rstrip() + " …"
    return text
