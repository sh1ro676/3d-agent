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
`ok / abstained / static_failed / exec_failed / llm_error / deadline` 六种。
把「弃答」与「答错」分开统计是本项目的一条硬要求（§13.3(5) 的 `Recovery.ABSTAIN`）：
模型如实说「图里没有门」和模型编了一个米数，对系统的评价完全不同。
同理，「模型调用失败」（`llm_error`）与「我们主动停手，因为超出了预算」（`deadline`）
也是两件事：前者要查链路，后者要查预算或「为什么单次调用变慢了」。
把它们合并成一条，报告里的失败归因就会指向错误的方向。

**④ 总预算（`total_budget_s`）是显式的，且**必须**在 `chat()` 内部落地。**
`None`（默认）= 不设上界，行为与加这个参数之前**逐字相同**。
设成数值时，`run()` 拿到一个绝对 deadline，并把它穿透
`plan` / `synthesize` / `polish` → `LLMClient.chat`。

为什么检查点不在循环层、而在 `chat()` 里：循环层只看得到「两次 synthesize 之间」，
而一次 `chat()` 内部合法地含 `max_retries + 1` 次 HTTP 尝试、每次都允许用满
`s.timeout`。按代码里的常数算（`SPATIAL_TIMEOUT=180`、`SPATIAL_MAX_RETRIES=2`）：

    单次 chat 最坏 = 3 × 180 + (2 + 4) = 546 s
    一次问答最多 4 次 chat（3 次 synthesize ＋ 臂 G 1 次 plan）
    ⟹ 只查轮次边界时，「预算」的实际上界 = 预算 + 546 s，那是装饰不是预算。

ⓘ 这个数不是估的：`546` 与 `2184` 是照上面两个常数的取值算出来的，
`SPATIAL_MAX_RETRIES=2` 在 `configs/llm_backend.env:62` 里显式写着。
实测（2026-09-22，`logs/agent_runs/` 里 54 条真实 run 的 `elapsed_s`）：
中位数 1.8 s、p90 3.9 s、最大 24.3 s —— 最大那条还是 `llm_error`。
⟹ 正常题离超时很远，`180 s` 的单次超时本身就已经是实测最大值的 7 倍多。
预算的意义在这条长尾上，不在平均值上。

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
#:
#: ⚠ `"deadline"` 与 `"llm_error"` **不能合并**，两者的处置方向相反。
#: `run_agent.py` 的退出码判据 `status in ("ok", "abstained")`（第 331 行）
#: 让新状态自动落在「不算对」那一侧，这是对的；但计数器（`tally("status")`）
#: 会把它单列一行，于是「这次是超预算停的」在汇总里看得见。
STATUSES = ("ok", "abstained", "static_failed", "exec_failed", "llm_error", "deadline")

#: 主路径允许的动作空间。E′ 是另一条臂，故意不给"近似实现"（见 `__init__`）。
SUPPORTED_ACTION_SPACES = ("program",)

#: 这些阶段**不是**重试轮次，所以 `attempt` 必须是 `None` 而不是一个编出来的数。
#: 见 `AgentRun.stages` 的字段说明。
_NON_ATTEMPT_STAGES = ("plan", "render", "verify")


def scene_conflict(existing: Any, incoming: Any) -> str | None:
    """显式传入的场景与上下文里已有的场景**是否冲突**。返回说明（None = 不冲突）。

    **纯函数**，不碰 GPU、不构造 `ToolContext` —— 于是这条判据可以零成本单测。

    为什么需要它：`_resolve_ctx` 原先的语义是
    「ctx 已有 scene ⟹ **静默忽略** 调用方传来的 scene」。后果不是报错，是
    **答非所问**：调用方以为在问 B，实际用的是 A，而返回的 `scene_id` 字段
    还是从 `ctx.scene` 取的 —— 整条记录自洽得看不出问题。
    它会在「缓存 session 以提速」这条路上必踩：缓存了 A 的 session 之后，
    再问 B 就是拿 A 的场景回答 B 的问题。

    判据按**优先级**取，不猜：
    ① 同一个对象（`is`）⟹ 不冲突（最常见的「复用同一份 ctx」）。
    ② 两边都有非空 `scene_id` ⟹ 比 id。**不同 id 一定冲突** —— 这是本函数
       要抓的那一类。
    ③ 取不到 id 时退化为 `is` 比较（②已覆盖 `is`，所以这里只会是「无法判断」）
       ⟹ 如实返回**冲突**，而不是默认放行。默认放行等于把「判断不了」
       静默记成「没问题」，而这里正是最需要响的地方。
    """
    if existing is None or incoming is None:
        return None
    if existing is incoming:
        return None
    a = getattr(existing, "scene_id", None)
    b = getattr(incoming, "scene_id", None)
    if a and b and a == b:
        # 同一个 scene_id 的两份对象：内容以 `dataset/scenes/<id>/scene.json` 为准，
        # 语义上是同一个场景 ⟹ 不算冲突（复用 ctx 里那份即可，包括它的 vlm/缓存）。
        return None
    return (
        "显式传入的场景与 ToolContext 里已有的场景不是同一个："
        "已有 scene_id=%r，传入 scene_id=%r。\n"
        "  这条以前是**静默忽略**传入值的 —— 于是「问 B、算 A」不会报错，"
        "而记录里的 scene_id 还取自旧场景，自洽得看不出来。\n"
        "  两种改法：① 同一个场景就别再传 `scene=`（走 ctx 复用，也是快路径）；"
        "  ② 换场景就换一个 `ToolContext`（或在同一个 loop 上只服务一个场景）。"
        % (a, b)
    )


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
    #: ★ **最后一次 synthesis 写出来的**程序（与基线臂的字段名对齐）。
    #:
    #: ⚠ 它**不一定被执行过**：末轮 `static_check` 没过时，它是一段从未运行的程序，
    #: 而 `trace` 来自更早的一轮。哪段程序产生了 `trace` 要看 `executed_program`。
    #: 这两个字段都存在，是为了让「program↔trace 同源」这件事**可验证**而不是靠约定 ——
    #: 见 `executed_program` 与 `to_dict()["program_matches_trace"]`。
    program: str = ""
    program_fenced: bool = True
    render: str = ""
    render_report: Mapping[str, Any] = field(default_factory=dict)
    #: 每个阶段的记录。**不是**所有阶段都是重试轮次：
    #:   `synthesize` / `static_check` / `execute` —— 是，`attempt` = 第几轮（1..N）
    #:   `plan` / `render` / `verify`            —— 不是，`attempt` **恒为 None**，
    #:                                             它们依据第几轮的产物记在 `applies_to_attempt`
    #:
    #: ⚠ 这条以前是错的：那三个非重试阶段填的是 `len(stages)`，也就是**阶段下标**。
    #: 它是个**看起来像 attempt 的数**，格式合法、单调、不报错，所以谁也没发现
    #: 「render 是第 7 次尝试」这句话毫无意义。现在 `attempt=None` 让"这不是轮次"
    #: 在数据里就是显式的，`applies_to_attempt` 才承载真正有用的信息。
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
    #: ★ **真正产生了 `trace` 的那段程序**。没有执行过任何一轮时是空串。
    #:
    #: 为什么必须单独存一份而不能从 `program` 推：末轮 `static_failed` 时
    #: `program` 是那段没跑成的程序，而 `trace` 属于上一轮 —— 用 `program` 去解释
    #: `trace` 就是拿一段从未执行过的代码去解释一堆它没产生过的工具调用。
    #: `scripts/probe_combination_run.py` 与 `mine_glue_patterns.py` 都在按
    #: `program` 重算归因，所以这个错配会让它们**静默**算出错的结论。
    executed_program: str = ""
    #: `executed_program` 是第几轮 synthesis 的产物（`None` = 没执行过）。
    executed_attempt: int | None = None

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "abstained")

    @property
    def program_matches_trace(self) -> bool:
        """`program` 与 `trace` 是否同源。

        没有 trace（没执行过）时恒为 True —— 「没有 trace 可以对错」不是一种错配。
        """
        if not self.trace:
            return True
        return self.program == self.executed_program

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["target_ids"] = list(self.target_ids)
        data["evidence"] = list(self.evidence)
        data["stages"] = list(self.stages)
        data["trace"] = list(self.trace)
        # 派生量显式落盘：读产物的人不必自己重算这条判据，也不会漏掉它。
        data["program_matches_trace"] = self.program_matches_trace
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
        total_budget_s: float | None = None,
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
        #: ★ 整题的总预算（秒）。`None` = 不设上界（与加这个参数之前逐字相同）。
        #:
        #: ⚠ 校验放在构造时而不是 `run()` 里：`0` 或负数会让**第一次**调用就直接
        #: 超预算，得到一个「永远 deadline」的循环 —— 那看起来像模型坏了。
        #: 想表达"不设上界"请用 `None`，不要用 `0`。
        if total_budget_s is not None:
            total_budget_s = float(total_budget_s)
            if total_budget_s <= 0:
                raise ValueError(
                    "total_budget_s 必须为正数或 None（收到 %r）。\n"
                    "  `0` 不是「不设上界」，是「一开始就超预算」—— 那会让每一次"
                    "问答都以 deadline 收尾，而原因看起来像模型故障。\n"
                    "  不设上界请显式传 None。" % (total_budget_s,)
                )
        self.total_budget_s = total_budget_s
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
        elif scene is not None:
            # ★ 显式传了 scene 而 ctx 里已经有别的 scene ⟹ **报错，不要静默忽略**。
            #   原先这里什么都不做，于是「问 B、算 A」不报错、不留痕，
            #   而落盘的 scene_id 取自 ctx.scene，整条记录自洽得看不出问题。
            #   判据与理由见 `scene_conflict`（纯函数，已单测）。
            why = scene_conflict(ctx.scene, scene)
            if why:
                raise ValueError(why)
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
        # ★ 预算的**绝对** deadline（`time.monotonic()` 坐标系）。`None` = 不设上界。
        #   传绝对时刻而不是「每次调用减一减」：后者会让串行链路上的每一段各自
        #   重新计时，加起来能超出预算好几倍。
        deadline = (None if self.total_budget_s is None
                    else time.monotonic() + self.total_budget_s)
        # ★ 产生 `trace` 的那段程序 / 它是第几轮的产物。见 `AgentRun.executed_program`。
        #   在**执行的那一刻**记下来，而不是循环结束后按最后状态猜 —— 猜不出来，
        #   这正是 B4 那个缺陷的成因。
        executed_program = ""
        executed_attempt: int | None = None

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
                    deadline=deadline,
                )
            except Exception as exc:          # noqa: BLE001  计划是可选项，不该拖挂整题
                plan_obj = Plan(ok=False, error="%s: %s" % (type(exc).__name__, exc))
            stages.append({
                # 计划整题只做一次，**不是**一次重试轮次 ⟹ `attempt` 必须是 None。
                # 原先填的是字面量 1，它看起来像「第 1 轮」，实际是编的。
                "stage": "plan", "attempt": None, "applies_to_attempt": None,
                "ok": plan_obj.ok,
                "detail": plan_obj.to_dict() if plan_obj.ok else plan_obj.error,
            })
        plan_text = plan_obj.text if plan_obj.ok else None

        for attempt in range(1, self.max_synthesis_retries + 2):
            # ★ 轮次边界的这一次检查只是**提前止损**（少发一个注定超预算的请求），
            #   真正的上界由 `chat(deadline=)` 内部保证 —— 那段算术见模块 docstring ④。
            if deadline is not None and time.monotonic() >= deadline:
                status = "deadline"
                failure = _deadline_failure(
                    budget_s=self.total_budget_s, attempt=attempt,
                    elapsed_s=time.perf_counter() - t0,
                    where="轮次边界（还没开始第 %d 轮 synthesis）" % attempt,
                )
                stages.append({"stage": "synthesize", "attempt": attempt, "ok": False,
                               "detail": failure["message"][:400]})
                break
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
                    deadline=deadline,
                )
            except LLMError as exc:
                stages.append({"stage": "synthesize", "attempt": attempt, "ok": False,
                               "detail": str(exc)[:400]})
                # ★ 「我们主动停手」与「模型调用失败」必须分成两种结局。
                #   靠 `exc.over_budget` 判，不靠比消息字符串 —— 文案一改就静默失效，
                #   而失效的方向是「超预算被记成模型故障」，正好让失败归因指向错的方向。
                if getattr(exc, "over_budget", False):
                    status = "deadline"
                    failure = _deadline_failure(
                        budget_s=self.total_budget_s, attempt=attempt,
                        elapsed_s=time.perf_counter() - t0,
                        where="第 %d 轮 synthesis 的调用内部（HTTP 尝试之前）" % attempt,
                    )
                else:
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
            # ★ 在执行的那一刻记下「谁产生了这段 trace」。
            #   循环结束后再回头看是**猜**不出来的：末轮可能只有 static_check 失败，
            #   那时 `synthesis.source` 已经是那段没跑成的程序了。
            executed_program = synthesis.source
            executed_attempt = attempt
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

                # 润色也吃同一份总预算；超预算时它回退到模板文本（答案已经算出来了，
                # 不该为了把句子写好看而把预算耗光）。
                render_text, render_report = polish(
                    self.client, render_text, question=question, deadline=deadline)
        # ⚠ `attempt` 恒为 None：渲染**不是**一次重试轮次。原先填 `len(stages)`，
        #   那是阶段下标 —— 一个看起来像轮次的数。真正有用的是 `applies_to_attempt`：
        #   这一段渲染的是第几轮产出的答案。
        stages.append({"stage": "render", "attempt": None,
                       "applies_to_attempt": executed_attempt,
                       "ok": bool(render_text), "detail": render_report})

        # -- 证据校验（零成本，每次都做） ---------------------------------------
        # 执行成功 != 答案可信。这一步检查「答案有没有落在工具返回过的数上」，
        # 结论与答案一起落盘 —— 于是「答对了但没证据」和「答错了但证据齐全」分得开。
        verdict = verify(
            submission,
            trace=(outcome.trace if outcome is not None else ()),
            scene=scene_obj,
            answer_type=answer_type,
        )
        stages.append({"stage": "verify", "attempt": None,
                       "applies_to_attempt": executed_attempt,
                       "ok": verdict.ok, "detail": verdict.to_dict()})

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
            # ★ 见 `AgentRun.executed_program`：`trace` 的同源程序。
            executed_program=executed_program,
            executed_attempt=executed_attempt,
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
            # ★ 预算进开关组合：`None` 与具体的秒数是**两次不同的实验**，
            #   消融表按 switches 分组，所以它必须在这里，不能被漏掉。
            "total_budget_s": self.total_budget_s,
            "vlm": bool(vlm is not None),
            "vlm_model": (getattr(getattr(vlm, "settings", None), "model", None)
                          if vlm is not None else None),
        }


def _deadline_failure(*, budget_s: float | None, attempt: int,
                      elapsed_s: float, where: str) -> dict[str, Any]:
    """超预算停手时的 `failure` 记录。**纯函数，零 API 可单测。**

    ⚠ 它必须带上 `budget_s` / `elapsed_s` / `where`，而不只是一句「超时了」：
    否则事后只有「这次没答出来」，而**答不出是哪种答不出**（预算太小 vs 链路变慢）
    无从判断。`where` 还区分了是「轮次边界提前止损」还是「调用内部放弃」——
    前者说明前一阶段已经吃掉了大部分预算，后者说明单次调用本身慢。
    """
    return {
        "stage": "deadline",
        "message": ("整题超出总预算：budget_s=%s，已用 %.1fs，停在第 %d 轮。"
                    "检查点：%s。\n"
                    "  ⚠ 这不等于模型答错 —— 是**我们**主动停手。它与 llm_error"
                    "分开计数，处置方向也不同（调预算 / 查为什么变慢）。"
                    % (budget_s, elapsed_s, attempt, where)),
        "budget_s": budget_s,
        "elapsed_s": round(elapsed_s, 3),
        "attempt": attempt,
        "where": where,
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
