#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agents/verifier.py —— 答案的**事后证据校验**。零模型成本、零 GPU。

它回答一个问题：**这个答案有没有落在工具返回过的数上？**

为什么必须单独有这一步
====================
`submit(answer, evidence=[...])` 只保证了「程序给了一条证据字符串」，
它**没有**保证「那条证据里真的出现过这个数」。实测里最容易发生的是这一种：

    d = calculate_distance(a=sofa_1, b=table_1)
    ...
    submit(2.103, evidence=["用了 calculate_distance"])     # ← 数从哪来的？

这不是假想：基线臂跑出来的 998.65 / 1048 这一类答案，程序里的每一步都"成功"了，
证据字段也填了，但最终交出的数**在整条 trace 里找不到出处**。所以：

    PRTS（协议遵守率）管的是「写没写证据」；
    本文件管的是「证据对不对得上」。两者是两条不同的指标，不能合并。

四级结论，而不是「对/错」
======================
    supported    硬检查全过，软检查也过 —— 答案落在 trace 上，指认的 id 也在场景里
    weak         硬检查全过，但有软检查没过（target_ids 里有个不存在的 id，
                 或**答案只是个派生量**：它对得上，但没有逐字落在某个工具返回值上）
    unsupported  有硬检查没过 —— **这个答案不能当答案用**
    abstained    程序如实弃答（`submit("unknown", ...)`）
                  —— 单独一级，因为「如实说答不了」与「编了个数」不是同一件事（§13.3(5)）

硬检查（不过 = unsupported）
    ① `has_evidence`        —— evidence 非空（executor 已强制，这里复核并留痕）
    ② `tool_was_used`       —— trace 里至少有一次**成功**的工具调用
    ③ `answer_finite`       —— 数值答案必须是有限数
    ④ `numeric_backed`      —— 数值答案必须能在 trace 的返回值/证据里找到来源。
                              分两档，`Verdict.matched_from` 如实记录用了哪一档：
                                · 强档 `value` / `evidence` —— 答案**逐字**落在某个数上
                                · 弱档 `derived:within` / `derived:count` —— 答案是**派生量**
                                  （均值、计数），改用可验证的不变量判定，见 `_derived_tier`
软检查（不过 = weak）
    ⑤ `targets_in_scene`    —— target_ids 全部落在场景内
    ⑥ `text_backed`         —— 字符串答案能在 trace 里找到出处（颜色/类别类题目的对应物）
    ⑦ `numeric_backed_exact`—— 数值答案**只**靠弱档通过时不过。
                              ⟹ 「派生量」落在 `weak` 这一级，不会混进 `supported`：
                              「supported 率」这个指标因此不会随口径悄悄变宽，
                              想统计「有多少答案靠弱档撑着」时读 `matched_from` 即可。

⚠ **本文件不判"答案对不对"**：它没有真值，也不该有 —— 真值属于 `evaluation/`，
而 `agents/` 不许依赖 `evaluation/`（依赖方向见 §13.3(6)：测量器械不能长在被测方身上）。
它只判「这个答案有没有被系统自己的证据支持」。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["LEVELS", "Check", "Verdict", "Pools", "verify", "collect_pools", "collect_numbers",
           "NON_QUANTITY_KEYS", "CENSUS_KEYS"]

#: 结论取值。顺序即由强到弱。加值时必须在 `evaluation/` 的口径里交代它算不算对。
LEVELS = ("supported", "weak", "unsupported", "abstained")

#: 数值一致性容差。相对容差为主 —— 因为 evidence 里的数常被 `round(x, 4)` 过。
#: 绝对下限 1e-4 是为了兜住「答案本身就是 0 附近」的情况（相对容差在 0 处无意义）。
REL_TOL = 1e-3
ABS_TOL = 1e-4

#: 从 evidence 字符串里抠数字。允许负号与科学计数法。
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

#: 不该被当成"数字证据"的字符串片段 —— 它们是 **id / 版本号**，不是量。
#: 不排除的话 `sofa_1` / `table_2` 会给 trace 贡献出 1、2 这样的假数字，
#: 于是一个巧合的答案会被判成 supported。
#:
#: ⚠ 旧版是 `\b[A-Za-z_][A-Za-z_0-9]*_(\d+)\b`，只认「下划线**紧跟**数字」那一种形状。
#: 于是 `geometry_v1` **漏网** —— 下划线后面先有一个 `v`。实测后果：
#: `query_relation` 的 `evidence["method"]` 恒为 `"geometry_v1"`，每调一次就往证据池里
#: 塞一个 `('method', 1.0)`，而 C8「桌子前面有几个物体」的正确答案恰好是 `1` ⟹
#: 它被判成 **supported（逐字有出处）**，出处却是**算法版本号**。
#: 现在把 `_` 后面的字母也一起吞掉，`geometry_v1` 整段被抹。
#:
#: ⚠ 已知边界（写下来而不是假装没有）：`sam2.1-hiera-base-plus` 这种**用点号分段**的
#: 版本串仍会被抠出 `2.1` —— 它在正则层面无解，只能靠下面的 `_NON_QUANTITY_KEYS`
#: 按**键名**整棵子树跳过。两个机制各管一半，缺一不可。
_ID_RE = re.compile(r"\b[A-Za-z_][A-Za-z_0-9]*_[A-Za-z_0-9]*\d[A-Za-z_0-9]*")


@dataclass(frozen=True)
class Check:
    """一条检查。`hard=False` 的失败只降级到 weak，不判 unsupported。"""

    name: str
    ok: bool
    detail: str
    hard: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail, "hard": self.hard}


@dataclass(frozen=True)
class Verdict:
    level: str
    checks: tuple[Check, ...] = ()
    #: 从 trace 里收集到的、可以作为答案出处的数值（截断保存，够诊断即可）。
    numbers: tuple[float, ...] = ()
    #: 命中的那个数来自哪 —— `"value"` / `"evidence"` / `"value+evidence"`。
    matched_from: str = ""

    @property
    def ok(self) -> bool:
        return self.level in ("supported", "weak", "abstained")

    @property
    def supported(self) -> bool:
        return self.level == "supported"

    def failed(self, *, hard_only: bool = False) -> tuple[str, ...]:
        return tuple(c.name for c in self.checks if not c.ok and (c.hard or not hard_only))

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "ok": self.ok,
            "matched_from": self.matched_from,
            "checks": [c.to_dict() for c in self.checks],
            "failed": list(self.failed()),
            "failed_hard": list(self.failed(hard_only=True)),
            "numbers": [round(n, 6) for n in self.numbers[:32]],
            "n_numbers": len(self.numbers),
        }

    def summary_line(self) -> str:
        bad = self.failed()
        if not bad:
            return "%s（%d 项检查全过）" % (self.level, len(self.checks))
        return "%s（未通过：%s）" % (self.level, ", ".join(bad))


# ============================================================================
# 从 trace 里收集可用的数
# ============================================================================


#: 人口数（「场景里有几个」）类证据键 —— **不进「区间」池，但可以当计数档的上界**。
_POPULATION_KEYS = frozenset({
    "n_objects", "total_objects", "total_in_scene", "returned", "count", "n_candidates",
})

#: 其它计数键 —— 既不是量值、也不是人口数。`n_points` 是「一个物体的掩码里有几万个点」，
#: 与「场景里有几个物体」毫无关系；拿它当计数档上界会让那一档宽到 23841。
_OTHER_COUNT_KEYS = frozenset({"n_points"})

#: 普查类键 —— **整棵子树都不进数字池**。
#: `list_objects.evidence["label_counts"] == {"sofa": 1, "picture": 4, ...}` 是一组 1~4 的
#: 小整数，与题目毫不相干，却让「4 个」恒为 supported、「3 个」恒为 unsupported ——
#: 那种判断力来自数字的**大小**，不来自正确性。
#: 同一个量在 `returned` 里已经**按本次查询**给了一份（`list_objects(label='chair')`
#: 的 `returned` 就是椅子数），所以删掉普查副本不会拿掉任何合法出处，只删掉假阳性。
_CENSUS_KEYS = frozenset({"label_counts"})

#: **非量值键** —— 整棵子树不进数字池。三种味道，同一条性质：
#: **它们的值不是对场景的度量。**
#:
#:   ① 身份（`object_id="chair_1"`、`label="sofa"`、`a`/`b`/`anchor`、`relation="front_of"`）
#:      —— 里面有数字也纯属 id 的一部分。`_ID_RE` 已能抹掉 `chair_1` 那种形状，
#:      但 `box2`（无下划线）抹不掉；按键名跳过是第二层，不靠正则的形状枚举。
#:   ② 版本 / 来源（`method="geometry_v1"`、`tool`、`model`）—— 实测抓到的那一个，
#:      见 `_ID_RE` 的注释：它不是漏网的**例外**，而是这一类里最先被撞上的一个。
#:   ③ 请求参数（`tol=0.05`、`k=1`、`min_contain`）—— 这些是**程序让工具怎么做**，
#:      不是**系统量到了什么**。危险程度不比 ① 低：`tol=0.0` 会让答案 `0` 恒判 supported。
#:      「答案恰好等于我传进去的阈值」不是证据。
#:
#: 为什么不干脆把 `evidence` 里的字符串全都不抠数字：那样会把 `submit(evidence=[...])`
#: 里人类写的量（`"mean width = 0.4445 m"`）一起废掉 —— 那是**合法**的出处。
#: 所以判据只能是「按**键名**判断这个位置放的是不是量」，与 `_CENSUS_KEYS` 同一手法。
#:
#: ⚠ 代价（明写）：`tol` 从此不作为「逐字出处」。C8 的完美答案因此从 `supported`
#: 落回 `derived:count` → `weak`。**这才是对的**：计数本来就是派生量，
#: 之前那个 supported 靠的是版本号里的 1。见 `tests/test_agent_verifier.py` 的回归用例。
_IDENT_KEYS = frozenset({
    "object_id", "label", "filter_label", "relation", "scene_id",
    "a", "b", "anchor", "target_ids", "unknown_targets", "ids", "name",
})

_VERSION_KEYS = frozenset({"method", "source", "model", "version", "tool", "toolset"})

#: 请求参数 —— 程序传进去的阈值 / 条数，不是量到的量。
_PARAM_KEYS = frozenset({"tol", "k", "top_k", "limit", "near_m", "min_contain", "up"})

_NON_QUANTITY_KEYS = _IDENT_KEYS | _VERSION_KEYS | _PARAM_KEYS

_MEASURE_EXCLUDED = _POPULATION_KEYS | _OTHER_COUNT_KEYS

#: 公开别名 —— 探针要拿它们做「抹掉非量键，结论会不会翻转」的自洽性对照。
NON_QUANTITY_KEYS = _NON_QUANTITY_KEYS
CENSUS_KEYS = _CENSUS_KEYS

#: **可以被诊断工具安全删掉**的键 —— 比 `_NON_QUANTITY_KEYS` 窄。
#:
#: 为什么必须有这个更窄的集合（这是踩过的坑，不是洁癖）：
#: 自洽性对照要「把无关字段删掉再验一次」。但如果连 `object_id` / `label` 一起删，
#: **字符串答案的 `text_backed` 就没东西可找了** —— 实测后果是 C1/C2/C3/C7
#: （答案 `picture_1` 这类 id）从 `supported` 掉成 `weak`，于是对照把它们报成
#: 「巧合命中」，四例全是假阳性。**删掉了答案本身的出处，不叫发现巧合。**
#: 所以可删集合只含那些**不可能是答案文本**的键：普查 / 版本 / 请求参数。
PRUNABLE_KEYS = _CENSUS_KEYS | _VERSION_KEYS | _PARAM_KEYS


@dataclass(frozen=True)
class Pools:
    """trace 能提供的几池数 —— **证据强度依次递减**。

    `values` / `evidence` 供「逐字对上」用（强档）；`measures` / `populations`
    供「派生量」的不变量判定用（弱档，见 `_derived_tier`）。
    """

    values: tuple[float, ...] = ()
    evidence: tuple[float, ...] = ()
    measures: tuple[float, ...] = ()
    populations: tuple[float, ...] = ()


def _pairs(obj: Any, out: list[tuple[str, float]], *, skip_ids: bool, key: str = "") -> None:
    """把一棵 JSON 子树摊成 `(键名, 数值)` —— **键名必须留着**。

    留键名是为了让「这个数是什么」可判定：`n_points` 与 `distance_m` 都是浮点，
    但一个是「几万个点」、一个是「几米远」，混进同一个区间，判据就失去意义了。
    列表里的元素**继承外层的键**（`extent_m: {w, h, l}` 三项都算 `extent_m`）。
    """
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        v = float(obj)
        if math.isfinite(v):
            out.append((key, v))
        return
    if isinstance(obj, str):
        if skip_ids:
            obj = _ID_RE.sub(" ", obj)          # 先把 id 抹掉，再抠数字
        for m in _NUM_RE.finditer(obj):
            try:
                v = float(m.group(0))
            except ValueError:
                continue
            if math.isfinite(v):
                out.append((key, v))
        return
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            name = str(k)
            # 普查 / 非量值：**整棵子树跳过**（理由分别见 _CENSUS_KEYS、_NON_QUANTITY_KEYS）
            if name in _CENSUS_KEYS or name in _NON_QUANTITY_KEYS:
                continue
            _pairs(v, out, skip_ids=skip_ids, key=name)
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            _pairs(v, out, skip_ids=skip_ids, key=key)


def collect_pools(trace: Sequence[Mapping[str, Any]]) -> Pools:
    """把 trace 拆成四池数。`collect_numbers` 是它的兼容薄壳。"""
    from_val: list[tuple[str, float]] = []
    from_ev: list[tuple[str, float]] = []
    for row in trace:
        res = row.get("result") or {}
        if res.get("ok"):
            _pairs(res.get("value"), from_val, skip_ids=True)
        _pairs(res.get("evidence"), from_ev, skip_ids=True)
        _pairs(row.get("args"), from_ev, skip_ids=True)
    both = from_val + from_ev
    return Pools(
        values=tuple(v for _, v in from_val),
        evidence=tuple(v for _, v in from_ev),
        measures=tuple(v for k, v in both if k not in _MEASURE_EXCLUDED),
        populations=tuple(v for k, v in both if k in _POPULATION_KEYS),
    )


def collect_numbers(trace: Sequence[Mapping[str, Any]]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """兼容入口：返回 `(工具返回值里的数, 证据/参数里的数)`。

    分成两份是为了**证据强度不同**：
      · 工具返回值里的数（`result.value`）是"系统算出来的量"，最强；
      · 证据与参数里的数（`evidence` / `args`）通常是同一个量的副本，弱一档。
    答案落在第一份里 = 有出处；只落在第二份里 = 至少对得上（仍算通过，但记下来）。
    """
    pools = collect_pools(trace)
    return pools.values, pools.evidence


def _match(answer: float, pool: Iterable[float]) -> float | None:
    """在池子里找与 `answer` 一致（容差内）的那个数。返回命中的值或 None。"""
    for v in pool:
        if abs(v - answer) <= max(ABS_TOL, REL_TOL * abs(answer)):
            return v
    return None


def _derived_tier(answer: float, pools: Pools) -> tuple[str, str] | None:
    """弱档：答案是个**派生量**（不是某个工具返回值）时，改用可验证的不变量判定。

    为什么必须有这一档
    ----------------
    组合题的正确答案往往是我们**第一次**算出来的数：均值落在两个数之间、
    计数是个新整数。原口径要求「逐字落在 trace 的数上」，于是把**答对的**也扔了 ——
    实测 10 题里 3 题（30%）。加工具治不了这个，它是契约的问题。

    两档不变量（都只用 trace 里**真实出现过的数**，所以放不过凭空编的量）：

      · `derived:within` —— 任何聚合量（`mean` / `median` / 分位数 / 加权平均）
        必然落在被聚合集合的 `[min, max]` 内。C5「四幅画的平均宽度」正是这一档：
        `0.4445` 不是任何工具返回值，却一定在四个宽度之间。
      · `derived:count` —— 计数是**子集的大小**，`0 ≤ 答案 ≤ 人口数`。
        人口数取自 trace 里**显式的人口字段**（`total_objects` / `returned` / `count`），
        **不是**取 `len(某个列表)` —— 后者会让「id 里那个 1」重新变成证据
        （由 `test_number_1_does_not_match_a_sofa_1_id` 钉着）。

    ⚠ **不认 `sum`**：求和的上界是「最大值 × 个数」，宽到足以把
    「25.024 vs 2.5024」这种差一个数量级的答案放进来 —— 那正是
    `test_wrong_magnitude_is_not_forgiven` 钉住的那条线。
    求和类题目目前只能如实判 unsupported；真要治，得让 `submit` 支持
    **声明式派生**（把被聚合的输入集合一起报上来，逐项核对），见模块 docstring 末节。

    ⚠ **区间档的强度有上界**：池子里混着全场景的量值（各物体的高、宽、距离），
    所以区间会比「被聚合的那个集合」宽。它是**有界性检查**，不是出处检查 ——
    因此它落在 `weak` 而不是 `supported`，且 `matched_from` 会写明用了哪一档。
    """
    uniq = sorted(set(pools.measures))
    if len(uniq) >= 2 and uniq[0] <= answer <= uniq[-1]:
        return ("derived:within",
                "答案 %r 落在 trace 量值区间 [%r, %r] 内（%d 个不同量值）—— 聚合类派生量按此判据通过"
                % (answer, uniq[0], uniq[-1], len(uniq)))
    if pools.populations and float(answer).is_integer() and 0 <= answer <= max(pools.populations):
        return ("derived:count",
                "答案 %r 是某个子集的大小：0 ≤ 答案 ≤ 人口数 %r（取自 trace 的 total/returned 字段）"
                % (answer, max(pools.populations)))
    return None


# ============================================================================
# 校验
# ============================================================================


def verify(
    submission: Any | None,
    *,
    trace: Sequence[Mapping[str, Any]] = (),
    scene: Any | None = None,
    answer_type: str | None = None,
) -> Verdict:
    """对一次 `submit()` 的结果做证据校验。**不抛异常**（除了我们自己的 bug）。

    `submission=None` 表示没有答案可校验（执行失败 / 没调 submit）——
    那本身就是一个 unsupported 结论，而不是"跳过校验"。
    """
    checks: list[Check] = []

    if submission is None:
        return Verdict(
            level="unsupported",
            checks=(Check("has_submission", False, "没有 submit() 结果 —— 本题没有答案可校验"),),
        )

    abstained = bool(getattr(submission, "abstained", False))
    evidence = tuple(getattr(submission, "evidence", ()) or ())
    answer = getattr(submission, "answer", None)
    targets = tuple(getattr(submission, "target_ids", ()) or ())
    unknown_targets = tuple(getattr(submission, "unknown_targets", ()) or ())

    # ① 证据非空 --------------------------------------------------------------
    checks.append(Check(
        "has_evidence", bool(evidence),
        "evidence 有 %d 条" % len(evidence) if evidence
        else "evidence 为空 —— 答案不可回溯（§13.3(5) 要求至少 1 条）",
    ))

    # ② 用过工具 --------------------------------------------------------------
    ok_rows = [r for r in trace if (r.get("result") or {}).get("ok")]
    failed_rows = [r for r in trace if not (r.get("result") or {}).get("ok")]
    checks.append(Check(
        "tool_was_used", bool(ok_rows),
        "成功调用 %d 次工具（另有 %d 次失败）" % (len(ok_rows), len(failed_rows)) if ok_rows
        else "整条 trace 里没有一次成功的工具调用 —— 答案不可能来自几何量",
    ))

    # ③④ 数值答案 --------------------------------------------------------------
    is_numeric = (
        not abstained
        and isinstance(answer, (int, float))
        and not isinstance(answer, bool)
    )
    pools = collect_pools(trace)
    pool = tuple(pools.values) + tuple(pools.evidence)
    matched_from = ""

    if is_numeric:
        v = float(answer)
        finite = math.isfinite(v)
        checks.append(Check("answer_finite", finite,
                            "答案是有限数 %r" % v if finite else "答案是 nan/inf"))

        if not finite:
            checks.append(Check("numeric_backed", False, "答案不是有限数，无需也无法核对出处"))
        else:
            hit_v = _match(v, pools.values)
            if hit_v is not None:
                matched_from = "value"
                checks.append(Check("numeric_backed", True,
                                    "答案 %r 命中工具返回值 %r" % (v, hit_v)))
            else:
                hit_e = _match(v, pools.evidence)
                if hit_e is not None:
                    matched_from = "evidence"
                    checks.append(Check("numeric_backed", True,
                                        "答案 %r 命中证据/参数里的数 %r（弱一档：非工具返回值）"
                                        % (v, hit_e)))
                else:
                    tier = _derived_tier(v, pools)
                    if tier is not None:
                        matched_from = tier[0]
                        checks.append(Check("numeric_backed", True, tier[1]))
                    else:
                        checks.append(Check(
                            "numeric_backed", False,
                            "答案 %r 在 trace 的 %d 个数里找不到出处，也不落在任何"
                            "可验证的派生量不变量里 —— 这个数是凭空出现的" % (v, len(pool)),
                        ))

            # ★ 软检查：只有「逐字对上」才算强档。
            #   派生量是**对得上、但没有逐字出现在任何工具返回值里**，所以它落在
            #   `weak` 而不是 `supported` —— 这样「supported 率」不会随着口径变宽
            #   而悄悄上升；「有多少答案靠弱档撑着」读 `matched_from` 单独统计即可。
            checks.append(Check(
                "numeric_backed_exact",
                matched_from in ("value", "evidence"),
                ("答案逐字落在 trace 的数上（%s）" % matched_from
                 if matched_from in ("value", "evidence")
                 else "答案只靠 %s 这一弱档通过：对得上，但没有逐字出现在任何工具返回值里"
                      % (matched_from or "无")),
                hard=False,
            ))
    elif abstained:
        checks.append(Check("answer_finite", True, "弃答，无数值可校验"))
        matched_from = "abstain"
    else:
        # 字符串 / 其他答案类型：用「答案是否出现在 trace 的任一字符串里」作为对应物，
        # 但它是**软**检查 —— 类别名会被归一化（`Chair` vs `chair`），
        # 严格的字符串比对容易产生假阴性，不该因此把答案判成 unsupported。
        needle = str(answer or "").strip().lower()
        blob = json.dumps(trace, ensure_ascii=False, default=str).lower()
        checks.append(Check(
            "text_backed", bool(needle) and needle in blob,
            "答案 %r 出现在 trace 里" % answer if (needle and needle in blob)
            else "答案 %r 未在 trace 里逐字出现（可能是归一化差异，需人工看一眼）" % answer,
            hard=False,
        ))

    # ⑤ 指认的 id --------------------------------------------------------------
    if unknown_targets:
        detail = "target_ids 里这些 id 不在场景中：%s（幻觉，但不致命 —— 答案本身仍有效）" % (
            ", ".join(unknown_targets),)
    elif targets:
        detail = "%d 个 target_id 全部落在场景内" % len(targets)
    else:
        detail = "没有指认任何物体（不是错，但 Viewer 无法高亮，也无法核对指认是否正确）"
    checks.append(Check("targets_in_scene", not unknown_targets, detail, hard=False))

    # ⑥ 场景一致性（有场景时才有意义） -----------------------------------------
    if scene is not None and targets:
        missing = [t for t in targets if not scene.has(t)]
        checks.append(Check("targets_exist", not missing,
                            "全部存在于场景图" if not missing else "场景图里没有：%s" % (missing,),
                            hard=False))

    if abstained:
        level = "abstained"
    elif any(not c.ok and c.hard for c in checks):
        level = "unsupported"
    elif any(not c.ok for c in checks):
        level = "weak"
    else:
        level = "supported"

    return Verdict(
        level=level,
        checks=tuple(checks),
        numbers=pool,
        matched_from=matched_from,
    )
