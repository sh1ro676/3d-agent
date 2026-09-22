#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""llm/render.py —— 答案的自然语言渲染。**模板为主，LLM 仅可选润色。**

为什么这一层值得单独存在（§13.3(5) 最后一段）
============================================
这是整条链路里**最后一个可能篡改数字的环节**。

前面所有环节都在保护数字：工具返回带 `evidence`、`submit()` 锁定 `answer`、
`answer_type` 前置约束类型。如果最后交给 LLM 「自由地把答案说成人话」，
那前面全部白做 —— 模型可以把 2.103 说成「大约 2 米」，
而报告里那个数字就再也对不上 `result.json`。

所以：
    · `render_answer()` 只做**拼装**，数值槽位由 `submit` 的参数锁定；
    · 需要自然解释时才开一次 LLM 润色，且 `polish_guard()` **强制校验**
      「润色后不许出现渲染前没有的数字」—— 只允许删减与改述，不允许新增或改写数值。

`polish_guard` 是**集合包含**而不是相等：润色文本允许省略某个数字
（例如只说「2.10 米」而不提原始精度），但不允许引入新数字。
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "NUMBER_RE",
    "format_scalar",
    "numbers_in",
    "render_answer",
    "polish_guard",
    "POLISH_SYSTEM_PROMPT",
    "polish_prompt",
    "DEFAULT_TEMPLATES",
]

#: 数字识别。刻意**不含**千分位与科学计数法：答案是几何量，
#: 出现 `1e-3` 或 `1,234` 说明上游哪里已经不对了，不该被这条正则掩盖。
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

#: 模板表。`{answer}` / `{unit}` / `{n_targets}` 是槽位。
#: 加模板必须同时加测试 —— 模板是**唯一**会把数字写进最终文本的地方。
DEFAULT_TEMPLATES: Mapping[str, str] = {
    "plain": "{answer}{unit}",
    "scene": "{answer}{unit}",
    "objects": "{answer}",
}


def format_scalar(value: Any) -> str:
    """把 `submit` 的 answer 变成文本。

    浮点用 `%.6g`：既不会把 2.103 变成 "2.1030000000000004"，
    也不会像 `%.2f` 那样**悄悄丢掉精度**（那是另一种数字篡改）。
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != value:                       # NaN
            return "nan"
        return "%.6g" % value
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_scalar(v) for v in value) + "]"
    return str(value)


def numbers_in(text: str) -> tuple[str, ...]:
    return tuple(NUMBER_RE.findall(text or ""))


def render_answer(
    answer: Any,
    *,
    unit: str = "",
    template: str = "plain",
    target_labels: Sequence[str] = (),
    extra: str | None = None,
) -> str:
    """模板渲染。**纯函数** —— 没有 LLM、没有随机性，同输入同输出。

    `target_labels` 是「被指认的物体」的人话名字（来自 `submit.target_ids`）。
    它只做展示，不参与任何数值 —— 高亮目标由 `target_ids` 驱动 Viewer，
    不让 LLM 再猜一次该高亮谁（§13.3(5)）。
    """
    tpl = DEFAULT_TEMPLATES.get(template)
    if tpl is None:
        raise KeyError("未知渲染模板 %r；可用：%s" % (template, sorted(DEFAULT_TEMPLATES)))
    text = tpl.format(
        answer=format_scalar(answer),
        unit=unit,
        n_targets=len(target_labels),
    )
    if target_labels:
        text = "%s（指认：%s）" % (text, "、".join(target_labels))
    if extra:
        text = "%s %s" % (text, extra)
    return text.strip()


def polish_guard(rendered: str, polished: str) -> tuple[bool, list[str]]:
    """润色是否越界。返回 `(ok, 越界的数字)`。

    判据：**润色后的数字集合必须是渲染前的子集**。
    只多不少 = 模型在编数字；改了某个数字 = 模型在篡改答案。
    两种都拒绝 —— 拒绝后调用方应当**回退到模板文本**，而不是"再润色一次"
    （重试一个会篡改数值的环节，只是在赌下一次它不篡改）。
    """
    allowed = set(numbers_in(rendered))
    introduced = [n for n in numbers_in(polished) if n not in allowed]
    return (not introduced), introduced


POLISH_SYSTEM_PROMPT = (
    "你的唯一任务是把给定的结论句改写得更自然。"
    "**绝对禁止**出现原文中没有的任何数字、单位数值或新的事实；"
    "可以删减、可以换词，不可以新增。只输出改写后的那一句话，不要解释。"
)


def polish_prompt(rendered: str, *, question: str) -> str:
    return (
        "问题：%s\n"
        "已锁定的结论：%s\n"
        "请改写这一句（不得新增任何数字）：" % (question, rendered)
    )


def polish(client: Any, rendered: str, *, question: str,
           purpose: str = "polish", deadline: float | None = None
           ) -> tuple[str, dict[str, Any]]:
    """可选润色。**失败或越界一律回退到模板文本**，并如实记录原因。

    返回 `(text, report)`。`report` 里 `used_fallback` 为 True 时，
    `text` 就是 `rendered` —— 于是「渲染层是否引入数字漂移」这个消融维度
    （§13.3(7) `render="llm"`）有了一份可直接统计的记录。

    `deadline` 与 synthesize / plan 同一口径：它**也**吃整题的那份总预算。
    润色是可选项，超预算时它失败并回退到模板文本 —— 这正是想要的行为
    （答案已经算出来了，不该为了把句子写好看而把预算耗光）。
    """
    from llm.adapter import LLMError

    report: dict[str, Any] = {"mode": "llm", "used_fallback": False, "reason": None,
                              "rendered": rendered, "introduced_numbers": []}
    try:
        reply = client.chat(
            [
                {"role": "system", "content": POLISH_SYSTEM_PROMPT},
                {"role": "user", "content": polish_prompt(rendered, question=question)},
            ],
            purpose=purpose,
            deadline=deadline,
        )
    except LLMError as exc:
        report.update({"used_fallback": True, "reason": "llm_error: %s" % exc})
        return rendered, report

    text = (reply.text or "").strip()
    if not text:
        report.update({"used_fallback": True, "reason": "empty_reply"})
        return rendered, report

    ok, introduced = polish_guard(rendered, text)
    report["polished"] = text
    report["introduced_numbers"] = introduced
    if not ok:
        report.update({"used_fallback": True,
                       "reason": "introduced_numbers: %s" % ",".join(introduced)})
        return rendered, report
    report["reply"] = reply.to_dict()
    return text, report
