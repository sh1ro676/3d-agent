#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agents/loop.py —— 主循环。**自己写的循环，不用 LangGraph**（§13.2）。

一次问答的全部步骤（主路径）：

    问题 ──► synthesize（1 次 LLM）──► static_check（零成本）
                                              │ 不过 → 带失败原因重新生成（≤2 次）
                                              ▼
                                          execute（沙箱 + 工具库 + trace）
                                              │ 失败 → 同样定向重新生成
                                              ▼
                                       submit(answer, evidence)
                                              ▼
                                        render（模板，数值槽位锁定）
                                              ▼
                                         AgentRun

三个刻意的设计决定
================

**① 重试是「定向」的，不是「再来一次」。**
失败原因经过 `WorkingMemory` 压缩后回灌：行号 + 错误码 + 恢复动作。
早期基线回灌的是原始 traceback，模型得自己读出「错在哪、该怎么救」——
这里的 `recovery` 是工具层写死的枚举，模型直接照做。

**② 重试上限 2 次（共 3 次生成）。**
超过之后每次重试的收益在实测里掉得很快（同一类错误）而成本线性上涨。
更重要的原因：**上限必须是显式的常数**，否则「3 次」与「7 次」的两次实验
在报告里会长得一模一样。

**③ 状态是枚举，不是「成功/失败」两态。**
`ok / abstained / static_failed / exec_failed / llm_error` 五种。
把「弃答」与「答错」分开统计是本项目的一条硬要求（§13.3(5) 的 `Recovery.ABSTAIN`）：
模型如实说「图里没有门」和模型编了一个米数，对系统的评价完全不同。

`action_space` 与消融臂
=====================
`action_space="program"` 是主路径；`"tool_loop"`（E′，多轮逐步决策）**尚未实现**，
构造时就明确报错而不是静默走主路径 —— 详见 `__init__` 里的说明。

`planner="on"` 是**臂 G**（planning-then-synthesis）：生成前先让模型列一次思路，
那段文本插进下一次合成的 user 消息里。它**不改控制流** —— 计划不产生任何工具调用，
所以报告里必须叫 planning-then-synthesis，不能写成「逐步规划」（见 `agents/planner.py`）。

谁来校验答案
===========
执行成功**不等于**答案可信。所以每次跑完都过一遍 `agents/verifier.py`：
它检查答案有没有落在工具返回过的数上。结论（`supported/weak/unsupported/abstained`）
进 `AgentRun.verdict`，**与答案一起落盘** —— 于是「答对了但没证据」和
「答错了但证据齐全」在记录里是两件不同的事，能分开统计。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from agents.executor import QA_TOOLSET, execute_program
from agents.memory import WorkingMemory
from agents.planner import PLANNER_MODES, Plan
from agents.planner import plan as run_plan
from agents.prompts.system import scene_hint_for
from agents.synthesizer import SynthesisResult, synthesize
from agents.verifier import verify
from llm.adapter import LLMError, usage_delta

__all__ = ["AgentRun", "AgentLoop", "STATUSES", "SUPPORTED_ACTION_SPACES"]

#: 结局枚举。加值时必须同时在 `evaluation/metrics.py` 的口径里交代它算不算对
#: —— 否则新状态会静默地既不算对也不算错。
STATUSES = ("ok", "abstained", "static_failed", "exec_failed", "llm_error")

#: 主路径允许的动作空间。E′ 是另一条臂，故意不给"近似实现"（见 `__init__`）。
SUPPORTED_ACTION_SPACES = ("program",)


@dataclass(frozen=True)
class AgentRun:
    """一次完整问答的产物。`to_dict()` 直接落盘，字段名与基线臂的记录对齐。"""

    question: str
    scene_id: str
    status: str
    answer: Any = None
    answer_type: str | None = None
    target_ids: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    abstained: bool = False
    program: str = ""
    program_fenced: bool = True
    render: str = ""
    render_report: Mapping[str, Any] = field(default_factory=dict)
    stages: tuple[Mapping[str, Any], ...] = ()
    attempts: int = 0
    usage: Mapping[str, Any] = field(default_factory=dict)
    failure: Mapping[str, Any] | None = None
    tool_calls: int = 0
    trace: tuple[Mapping[str, Any], ...] = ()
    elapsed_s: float = 0.0
    tools_version: str = ""
    memory: Mapping[str, Any] = field(default_factory=dict)
    #: ★ 答案的证据校验结论（`agents/verifier.py`）。与答案同时落盘 ——
    #: 「答对了但没证据」与「答错了但证据齐全」必须能分开统计。
    verdict: Mapping[str, Any] = field(default_factory=dict)
    #: 臂 G 的前置计划（`planner="off"` 时是 `{"ok": False, "skipped": True}`）。
    plan: Mapping[str, Any] = field(default_factory=dict)
    #: 本次用的开关组合 —— 消融表按它分组，不从别处反推。
    switches: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "abstained")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["target_ids"] = list(self.target_ids)
        data["evidence"] = list(self.evidence)
        data["stages"] = list(self.stages)
        data["trace"] = list(self.trace)
        return data


class AgentLoop:
    """主循环。

    `client` 可以延后注入（`AgentLoop(ctx=...)` 之后 `loop.client = ...`），
    这是为了让「不花钱的干跑」（`--program-file` / `--dry-run`）能复用同一条装配路径。
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        ctx: Any | None = None,
        tool_docs_text: str | None = None,
        max_synthesis_retries: int = 2,
        exec_timeout_s: float = 60.0,
        action_space: str = "program",
        render: str = "template",
        planner: str = "off",
        record_trace: bool = True,
        toolset: Sequence[str] | None = None,
    ) -> None:
        if action_space not in SUPPORTED_ACTION_SPACES:
            raise NotImplementedError(
                "action_space=%r 尚未实现（当前只支持 %s）。\n"
                "  E′（多轮 tool-calling）是**消融臂**，不是主路径的「近似实现」：\n"
                "  它与主路径共用同一套工具库与执行器，只换循环层 —— 所以它必须是被\n"
                "  单独实现出来的一条臂。用一个「循环几次」的伪实现冒充它，会让\n"
                "  「程序合成 vs 逐步决策」这组消融失去意义（§13.1）。"
                % (action_space, SUPPORTED_ACTION_SPACES)
            )
        if planner not in PLANNER_MODES:
            raise ValueError(
                "planner=%r 未知（可用：%s）。臂 G 的两种取值必须是显式的 —— "
                "「没写就是 off」会让两次实验的开关组合在记录里长得一样。"
                % (planner, PLANNER_MODES)
            )
        self.client = client
        self.ctx = ctx
        self.tool_docs_text = tool_docs_text
        self.max_synthesis_retries = int(max_synthesis_retries)
        self.exec_timeout_s = float(exec_timeout_s)
        self.action_space = action_space
        self.render = render
        #: 臂 G 开关。`"on"` 时每次题目会**多一次 LLM 调用**（成本可归因，见 usage.by_purpose）。
        self.planner = planner
        self.record_trace = bool(record_trace)
        #: 动作空间。**同一份名单喂给三处**（提示词 / 静态检查 / 命名空间）——
        #: 见 `executor.QA_TOOLSET` 的说明与 `tests/test_agent_toolset.py`。
        self.toolset = tuple(QA_TOOLSET if toolset is None else toolset)
        if ctx is not None:
            ctx.record_trace = self.record_trace

    # -- 组装 ----------------------------------------------------------------

    def _resolve_ctx(self, scene: Any | None) -> Any:
        from tools.registry import ToolContext

        ctx = self.ctx
        if ctx is None:
            ctx = ToolContext(scene=scene, record_trace=self.record_trace)
            self.ctx = ctx
        elif scene is not None and ctx.scene is None:
            ctx.scene = scene
        if ctx.scene is None:
            raise ValueError(
                "没有场景。run() 需要 `scene=` 或一个已带 scene 的 ToolContext —— "
                "场景图必须在循环之外构建（builder 要 GPU，循环层不该碰它）。"
            )
        return ctx

    # -- 主循环 --------------------------------------------------------------

    def run(
        self,
        question: str,
        *,
        scene: Any | None = None,
        answer_type: str | None = None,
        memory: WorkingMemory | None = None,
    ) -> AgentRun:
        if self.client is None:
            raise ValueError("AgentLoop 没有 client —— 没法生成程序（干跑请直接调 execute_program）")

        t0 = time.perf_counter()
        ctx = self._resolve_ctx(scene)
        scene_obj = ctx.scene
        mem = memory if memory is not None else WorkingMemory()
        mem.question = question
        mem.scene_hint = scene_hint_for(scene_obj)

        before = self.client.usage.snapshot()
        stages: list[dict[str, Any]] = []
        outcome = None
        synthesis: SynthesisResult | None = None
        status = "exec_failed"
        failure: dict[str, Any] | None = None

        # -- 臂 G：前置计划（生成程序之前，整题只做一次） ----------------------
        plan_obj = Plan(ok=False, error="planner=off")
        if self.planner == "on":
            try:
                plan_obj = run_plan(
                    self.client,
                    question=question,
                    scene_hint=mem.scene_hint,
                    tool_docs_text=self.tool_docs_text,
                    tools=self.toolset,
                    answer_type=answer_type,
                )
            except Exception as exc:          # noqa: BLE001  计划是可选项，不该拖挂整题
                plan_obj = Plan(ok=False, error="%s: %s" % (type(exc).__name__, exc))
            stages.append({
                "stage": "plan", "attempt": 1, "ok": plan_obj.ok,
                "detail": plan_obj.to_dict() if plan_obj.ok else plan_obj.error,
            })
        plan_text = plan_obj.text if plan_obj.ok else None

        for attempt in range(1, self.max_synthesis_retries + 2):
            try:
                synthesis = synthesize(
                    self.client,
                    question=question,
                    scene_hint=mem.scene_hint,
                    tool_docs_text=self.tool_docs_text,
                    answer_type=answer_type,
                    feedback=mem.feedback(),
                    previous_source=synthesis.source if synthesis else None,
                    tools=self.toolset,
                    plan_text=plan_text,
                )
            except LLMError as exc:
                stages.append({"stage": "synthesize", "attempt": attempt, "ok": False,
                               "detail": str(exc)[:400]})
                status = "llm_error"
                failure = {"stage": "llm_error", "message": str(exc)[:400],
                           "status": getattr(exc, "status", None)}
                break

            stages.append({
                "stage": "synthesize", "attempt": attempt, "ok": True,
                "chars": len(synthesis.source), "fenced": synthesis.fenced,
                "reply": synthesis.reply.to_dict() if synthesis.reply else None,
            })

            if not synthesis.check.ok:
                stages.append({"stage": "static_check", "attempt": attempt, "ok": False,
                               "check": synthesis.check.to_dict()})
                mem.record_attempt("static_check", synthesis.check.feedback(),
                                   source=synthesis.source)
                status = "static_failed"
                failure = {"stage": "static_check",
                           "message": synthesis.check.feedback()[:800]}
                continue

            stages.append({"stage": "static_check", "attempt": attempt, "ok": True,
                           "n_tool_calls": len(synthesis.check.tool_calls),
                           "tool_calls": list(synthesis.check.tool_calls)})

            outcome = execute_program(
                synthesis.source, ctx,
                answer_type=answer_type,
                timeout_s=self.exec_timeout_s,
                toolset=self.toolset,
            )
            mem.learn_from_trace(outcome.trace)
            stages.append({
                "stage": "execute", "attempt": attempt, "ok": outcome.ok,
                "detail": outcome.stage, "message": outcome.message[:400],
                "lineno": outcome.lineno, "tool_calls": outcome.tool_calls,
                "duration_ms": round(outcome.duration_ms, 3),
                "trace_errors": outcome.to_dict()["trace_errors"],
            })

            if outcome.ok:
                status = "abstained" if (outcome.submission and outcome.submission.abstained) else "ok"
                failure = None
                break

            code, recovery, hint = _last_tool_error(outcome.trace)
            mem.record_attempt(
                "execute:" + outcome.stage,
                outcome.message,
                code=code, recovery=recovery, hint=hint,
                lineno=outcome.lineno, source=synthesis.source,
            )
            status = "exec_failed"
            failure = {"stage": outcome.stage, "message": outcome.message[:800],
                       "lineno": outcome.lineno, "tool_error_code": code}

        # -- 渲染 ------------------------------------------------------------
        render_text = ""
        render_report: dict[str, Any] = {"mode": self.render, "used_fallback": False}
        submission = outcome.submission if (outcome is not None and outcome.ok) else None
        if submission is not None:
            from llm.render import render_answer

            render_text = render_answer(
                submission.answer,
                target_labels=submission.target_ids,
                template="plain",
            )
            if self.render == "llm":
                from llm.render import polish

                render_text, render_report = polish(
                    self.client, render_text, question=question)
        stages.append({"stage": "render", "attempt": len(stages), "ok": bool(render_text),
                       "detail": render_report})

        # -- 证据校验（零成本，每次都做） ---------------------------------------
        # 执行成功 != 答案可信。这一步检查「答案有没有落在工具返回过的数上」，
        # 结论与答案一起落盘 —— 于是「答对了但没证据」和「答错了但证据齐全」分得开。
        verdict = verify(
            submission,
            trace=(outcome.trace if outcome is not None else ()),
            scene=scene_obj,
            answer_type=answer_type,
        )
        stages.append({"stage": "verify", "attempt": len(stages), "ok": verdict.ok,
                       "detail": verdict.to_dict()})

        after = self.client.usage.snapshot()
        return AgentRun(
            question=question,
            scene_id=getattr(scene_obj, "scene_id", ""),
            status=status,
            answer=submission.answer if submission else None,
            answer_type=answer_type,
            target_ids=submission.target_ids if submission else (),
            evidence=submission.evidence if submission else (),
            abstained=bool(submission and submission.abstained),
            program=(synthesis.source if synthesis else ""),
            program_fenced=bool(synthesis.fenced) if synthesis else True,
            render=render_text,
            render_report=render_report,
            stages=tuple(stages),
            attempts=sum(1 for s in stages if s["stage"] == "synthesize"),
            usage=usage_delta(before, after),
            failure=failure,
            tool_calls=outcome.tool_calls if outcome is not None else 0,
            trace=outcome.trace if outcome is not None else (),
            elapsed_s=round(time.perf_counter() - t0, 3),
            tools_version=_tools_version(),
            memory=mem.to_dict(),
            verdict=verdict.to_dict(),
            plan=plan_obj.to_dict() if self.planner == "on"
            else {"ok": False, "skipped": True, "reason": "planner=off"},
            switches=self.switches(),
        )

    # -- 开关组合 -------------------------------------------------------------

    def switches(self) -> dict[str, Any]:
        """本次运行的开关组合 —— 消融表按它分组。

        单独一个方法而不是散落在 `to_dict()` 里，是因为它同时也是**环境指纹**的一部分：
        「这两次结果能不能放进同一张表」取决于它是否逐字段相等，而不取决于我们的记忆。
        """
        vlm = getattr(self.ctx, "vlm", None) if self.ctx is not None else None
        return {
            "action_space": self.action_space,
            "planner": self.planner,
            "render": self.render,
            "toolset": list(self.toolset),
            "n_tools": len(self.toolset),
            "max_synthesis_retries": self.max_synthesis_retries,
            "exec_timeout_s": self.exec_timeout_s,
            "vlm": bool(vlm is not None),
            "vlm_model": (getattr(getattr(vlm, "settings", None), "model", None)
                          if vlm is not None else None),
        }


def _last_tool_error(trace: Sequence[Mapping[str, Any]]) -> tuple[str | None, tuple[str, ...], str | None]:
    """trace 里最后一次工具失败的 (错误码, 恢复动作, 提示)。

    「最后一处」而不是「第一处」：程序可能先探一个不存在的 id（得到 NOT_IN_SCENE）
    然后自己纠正过来了，真正的死因是最后那一处。
    """
    for row in reversed(list(trace)):
        result = row.get("result") or {}
        if result.get("ok"):
            continue
        err = result.get("error") or {}
        hint = (err.get("context") or {}).get("hint")
        return err.get("code"), tuple(err.get("recovery") or ()), hint
    return None, (), None


def _tools_version() -> str:
    try:
        from tools.version import TOOLS_VERSION

        return TOOLS_VERSION
    except Exception:                      # noqa: BLE001  版本号拿不到不该让一次问答失败
        return ""
