#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/show_answer_provenance.py —— **「这个答案的出处是哪来的」**。

为什么需要它
============
`verifier` 只告诉你一个结论（`supported` / `weak` / `unsupported`）和一句
`matched_from`（`value` / `evidence` / `derived:*`）。**它不告诉你命中的是哪个字段。**

而「命中的是哪个字段」恰恰是这一类缺陷唯一的线索：

    `query_relation` 的 `evidence["method"]` 是 `"geometry_v1"`，`_ID_RE` 抹不掉
    `_v1` 这种版本号后缀 ⟹ `1` 被抠出来，被记成 `('method', 1.0)`。
    于是 C8「桌子前面有几个物体」的答案 `1` 判成 **supported（逐字有出处）** ——
    出处是**算法版本号**。

这个缺陷不是被某个检查抓到的，是**顺着「这个逐字出处是哪来的」查出来的**。
所以这条线索得有个常驻工具，而不是靠临时写脚本。

用法
====
    python scripts/show_answer_provenance.py                 # 全部题
    python scripts/show_answer_provenance.py --only C8,C8b
    python scripts/show_answer_provenance.py --key method,tol  # 只看这些键名

产物：`reports/answer_provenance.md`（本机 PowerShell 的 stdout 不回传，必须落盘）。

判读
====
- 命中 `value` 桶 = 有出处（系统算出来的量）；
- 命中 `evidence` 桶 = 对得上（可能是量，也可能是**无关常量**——看键名）；
- 什么都没命中 = 靠派生量档位（`weak`）。
⚠ 键名属于 `verifier.NON_QUANTITY_KEYS` 却在**改之前**命中过答案，就是一条已修的巧合。
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools as _tools  # noqa: E402

_tools.load_tools()

from agents.verifier import (  # noqa: E402
    ABS_TOL, NON_QUANTITY_KEYS, REL_TOL, _ID_RE, _NUM_RE, _pairs, collect_pools, verify,
)
from agents.executor import Submission  # noqa: E402
from scene_graph.store import load_scene, scene_dir  # noqa: E402
from tools.registry import ToolContext  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from probe_combination import SPECS, run_spec  # noqa: E402

DEFAULT_OUT = ROOT / "reports" / "answer_provenance.md"


def _pairs_of_trace(trace: list[dict[str, Any]]) -> tuple[list, list]:
    """把 trace 摊成 `(value 桶, evidence 桶)` 的 `(键, 值)` 列表。**走 verifier 的口径。**"""
    from_val: list[tuple[str, float]] = []
    from_ev: list[tuple[str, float]] = []
    for row in trace:
        res = row.get("result") or {}
        if res.get("ok"):
            _pairs(res.get("value"), from_val, skip_ids=True)
        _pairs(res.get("evidence"), from_ev, skip_ids=True)
        _pairs(row.get("args"), from_ev, skip_ids=True)
    return from_val, from_ev


#: **冻结的历史正则** —— 2026-09-18 及之前用的那一版。
#:
#: 为什么要在这个脚本里留一份**已知有缺陷**的正则：不这样的话，「修复到底修掉了什么」
#: 就永远只是一句转述 —— 因为当前的 `_ID_RE` 已经把 `geometry_v1` 抹干净了，
#: 用**现在**的口径重放，是什么都看不出来的（我第一次就是这么写的，结果 C8 那一行空白）。
#: 要能核对，就必须能重放**当时**的口径。
#:
#: ⚠ **改它 = 篡改历史。** 它的唯一用途是当 `_pairs_of_trace_before_fix` 的输入；
#: `tests/` 里有一条用例保证它确实**复现得出**那个泄漏。
#: 旧版只认「下划线**紧跟**数字」：`chair_1` 抹得掉，`geometry_v1` 抹不掉。
_OLD_ID_RE = re.compile(r"\b[A-Za-z_][A-Za-z_0-9]*_(\d+)\b")


def _pairs_of_trace_before_fix(trace: list[dict[str, Any]]) -> list[tuple[str, float]]:
    """重放**修复前**的口径：旧 `_ID_RE` + **不按任何键名排除**。

    于是答案会命中什么，就是当时会发生什么。命中的键若属于 `NON_QUANTITY_KEYS`，
    那是一条**在真实 trace 上验证过的假出处**，不是构造出来的例子。

    ⚠ 它只重放**已经知道名字**的键。下一个同类缺陷若用了新键名，这张表照样空白 ——
    别把「表里没漏」当成「没有漏」。
    """
    out: list[tuple[str, float]] = []

    def walk(obj: Any, key: str) -> None:
        if isinstance(obj, bool) or obj is None:
            return
        if isinstance(obj, (int, float)):
            if math.isfinite(float(obj)):
                out.append((key, float(obj)))
            return
        if isinstance(obj, str):
            for m in _NUM_RE.finditer(_OLD_ID_RE.sub(" ", obj)):
                try:
                    v = float(m.group(0))
                except ValueError:
                    continue
                if math.isfinite(v):
                    out.append((key, v))
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, str(k))
            return
        if isinstance(obj, (list, tuple, set, frozenset)):
            for v in obj:
                walk(v, key)

    for row in trace:
        res = row.get("result") or {}
        if res.get("ok"):
            walk(res.get("value"), "")
        walk(res.get("evidence"), "")
        walk(row.get("args"), "")
    return out


def _hits(answer: Any, bucket: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """答案命中哪些 `(键, 值)`。**去重后按首次出现顺序返回。**

    去重是必须的：同一个键会被命中很多次（`query_relation` 调 8 次就有 8 个
    `('method', 1.0)`）。打印 8 遍同样的东西只会让人以为「有 8 个不同的问题」，
    而真正要看清的是「**是哪一个键**」。
    """
    if not isinstance(answer, (int, float)) or isinstance(answer, bool):
        return []
    a = float(answer)
    out: list[tuple[str, float]] = []
    for k, v in bucket:
        if abs(v - a) <= max(ABS_TOL, REL_TOL * abs(a)) and (k, v) not in out:
            out.append((k, v))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="打印每题答案在 trace 里的出处（零成本）")
    ap.add_argument("--scene", default="points_probe")
    ap.add_argument("--only", default="", help="逗号分隔的题号，默认全部")
    ap.add_argument("--key", default="", help="只看这些键名（逗号分隔）")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    scene_path = Path(args.scene)
    if not scene_path.exists():
        scene_path = scene_dir(args.scene)
    scene = load_scene(scene_path)

    only = {q.strip() for q in args.only.split(",") if q.strip()}
    keys = {k.strip() for k in args.key.split(",") if k.strip()}

    L: list[str] = []
    L.append("# 答案出处（provenance）—— `%s`" % args.scene)
    L.append("")
    L.append("每题：跑一遍它的**完美工具路线**，再看答案命中了 trace 里哪些 `(键, 值)`。")
    L.append("「命中 `evidence` 桶且键名不是量」= 巧合命中 —— 这正是 `method` 那一类的形状。")
    L.append("")

    flagged: list[str] = []
    for spec in SPECS:
        if only and spec.qid not in only:
            continue
        if spec.solve is None:
            continue
        r = run_spec(spec, scene)
        ctx = ToolContext(scene=scene, record_trace=True)
        try:
            spec.solve(ctx)
        except Exception as exc:                 # noqa: BLE001  探针不该因一题挂掉
            L.append("## %s —— 路线走不通：%s" % (spec.qid, exc))
            L.append("")
            continue

        sub = Submission(answer=r.answer, answer_type=spec.answer_type,
                         target_ids=tuple(r.targets), evidence=tuple(r.evidence))
        v = verify(sub, trace=ctx.trace, scene=scene, answer_type=spec.answer_type)
        fv, fe = _pairs_of_trace(ctx.trace)
        hv, he = _hits(r.answer, fv), _hits(r.answer, fe)

        L.append("## %s「%s」" % (spec.qid, spec.category))
        L.append("")
        L.append("- 答案 `%r` ｜ 结论 `%s` ｜ `matched_from` = `%s`"
                 % (r.answer, v.level, v.matched_from or "（非数值答案：走 `text_backed` 文本查找）"))
        if keys:
            hv = [p for p in hv if p[0] in keys]
            he = [p for p in he if p[0] in keys]
        L.append("- 命中 `value` 桶：%s" % ((", ".join("`%s=%r`" % p for p in hv)) or "—"))
        L.append("- 命中 `evidence` 桶：%s" % ((", ".join("`%s=%r`" % p for p in he)) or "—"))
        # 「修复前的世界」：用**旧正则 + 不按键排除**重放一次，看答案当时会命中什么。
        before = _pairs_of_trace_before_fix(ctx.trace)
        leaks = [p for p in _hits(r.answer, before) if p[0] in NON_QUANTITY_KEYS]
        if leaks:
            L.append("- ⚠ **修复前会命中**：%s ⟹ 这就是被挡掉的假出处"
                     % ", ".join("`%s=%r`" % p for p in leaks))
            flagged.append(spec.qid)
        pools = collect_pools(ctx.trace)
        L.append("- 池子规模：value %d ｜ evidence %d ｜ measures %d ｜ populations %r"
                 % (len(pools.values), len(pools.evidence), len(pools.measures), pools.populations))
        L.append("")

    L.append("---")
    L.append("")
    if flagged:
        L.append("⚠ **修复前会命中答案的题**：%s" % ", ".join(flagged))
        L.append("")
        L.append("这些就是**已经在真实 trace 上验证过**的假出处 —— 不是构造出来的例子。")
        L.append("两个机制一起把它们挡住：① `_ID_RE` 现在会连 `_v1` 这种版本号后缀一起抹掉；")
        L.append("② `verifier.NON_QUANTITY_KEYS` 按**键名**把这些子树整棵挡在数字池外。")
        L.append("于是它们在上表「命中」一栏是 `—`，答案落到 `derived:*` → `weak`。")
    else:
        L.append("✅ 本批题目里，没有任何整数答案会命中非量值键（id / 版本号 / 请求参数）。")
    L.append("")
    L.append("⚠ **这张表查不到什么**：它只重放**已经写进 `NON_QUANTITY_KEYS` 的键名**。")
    L.append("一个用了新键名的同类泄漏，在这张表上同样是 `—`。")
    L.append("要往下走一步，得换一种问法：不是「这个键是不是量」，")
    L.append("而是「答案有没有**独立于 trace** 的出处」——那需要契约层面的改动，本轮不做。")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("[provenance] %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
