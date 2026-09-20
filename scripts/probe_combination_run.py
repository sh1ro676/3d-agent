#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/probe_combination_run.py —— 组合题诊断的**后半段**：模型到底会不会组合。

前半段（`probe_combination.py`，零成本）已经回答了「现有工具返回值够不够」。
这里回答剩下的那一半：「够用 ≠ 会用」。

为什么不能只看答案对不对
========================
一道组合题答错，至少有三种互不相同的死法，而它们的**修法完全不同**：

    A. 程序里**根本没有组合算子**（没出现 `max(` / `sorted(` / `sum(` …）
       —— 模型压根没想到要组合。修法：提示词 / 动作空间导引。
    B. 程序里有组合算子，但**用错了**（`max` 没给 `key=`，或对错字段取极值）
       —— 是「会用但用得不对」。修法：返回值形状 / 字段命名。
    C. 程序组合正确、答案也对，但被 `verifier` 判 `unsupported`
       —— 属于证据契约问题（因④），**与模型无关**。

只看对错率，这三种会混成一个数字。所以本脚本对每一次运行都做**算子归因**：
直接在生成的程序源码上找组合算子，再和 GT（复用 `probe_combination.SPECS`）比。

真值只算一次
============
GT 直接 `import` 前半段的 `SPECS` —— 两个脚本共用同一份定义。
如果这里再写一份「我认为的正确答案」，两份真值迟早会漂移，
而漂移之后**看起来还都对**，这是最难查的一类问题。

成本
====
每题 1 次 LLM 调用 + 至多 `--max-retries` 次定向重试。8 题约 ¥0.5。
脚本在结尾打印累计成本，并且**不提供"跑全量"的开关** ——
501 题的账要单独算（见记忆里的空闲时段结论）。
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.probe_combination import (  # noqa: E402
    SPECS,
    Spec,
    one_call_solutions,
    run_spec,
)
from scripts.run_agent import build_session  # noqa: E402
from agents.executor import Submission  # noqa: E402
from agents.loop import AgentLoop  # noqa: E402
from agents.verifier import verify  # noqa: E402
from llm.adapter import LLMClient, LLMSettings  # noqa: E402
from scene_graph.store import load_scene, scene_dir  # noqa: E402

DEFAULT_OUT_DIR = ROOT / "reports"

#: 组合算子。**这是本脚本的核心观测量** —— 比答案对错更早地暴露失败类型。
#: 分两类：`KEYED` 说明模型知道「按某个字段取极值」，`BARE` 说明它只是随便取了个极值。
_KEYED_OPS = ("max(", "min(", "sorted(", "statistics.", "mean(", "median(", "Counter(",
              "filter(", "reduce(")
_BARE_OPS = ("sum(", "len(")
#: `key=` 出现 = 模型在用字段排序/取极值（而不是只对数字列表取极值）。
_KEY_ARG_RE = re.compile(r"\bkey\s*=")
#: 出现过 `for … in` —— **注意它同时匹配普通 `for x in y:` 语句**。
#:
#: ⚠ 这个名字是 2026-09-20 改的。原来它叫 `_COMPREHENSION_RE`、注释写「列表推导 /
#: 生成器」，于是产出的字段 `has_comprehension` 被当成「模型用了推导式」，
#: 而实际上这 10 道题里真正的推导式只有 4 道 —— **9 道题里出现的都是显式循环**。
#: 我据此在汇报与记忆里写过「9/10 用了推导式/带 key 算子」，那句话是错的。
#: 真实写法是「9/10 写了 `for` 循环扫描」，见 `reports/glue_mining.md`。
_ITERATION_RE = re.compile(r"\bfor\b[^\n]*\b in \b")

#: 答案比较时**顺序无意义**的题号：「哪两个物体最远」问的是一个集合。
#: 反例是 C6（按从左到右排序）—— 那里顺序就是答案本身，不能宽容。
_ORDER_INSENSITIVE = frozenset({"C12"})

#: 数值答案的比较容差。**刻意与 `agents/verifier.py` 用同一组常数**：
#: 两处各定一个容差，就会出现「脚本认为答对了、校验器认为没对上」的分叉，
#: 而分叉的表现是报告里出现一道自相矛盾的题。
_REL_TOL, _ABS_TOL = 1e-3, 1e-4


def same_answer(qid: str, gt: Any, answer: Any) -> bool:
    """GT 与模型答案是否一致。**不是字面相等** —— 三种答案形态各有一套口径。

    多 id 答案（`picture_4,picture_2,...`）尤其要注意：模型完全可能写成
    `"picture_4, picture_2"`（逗号后带空格）或换一个分隔符。
    用 `==` 比会把**答对的题**记成「组合错了」，于是「因① 模型不会组合」这个结论
    会被自己的测量口径伪造出来 —— 这是本脚本要防的最主要的一种自欺。
    """
    if isinstance(gt, bool) or isinstance(answer, bool):
        return bool(gt) == bool(answer)
    if isinstance(gt, (int, float)) and isinstance(answer, (int, float)):
        g = float(gt)
        return abs(float(answer) - g) <= max(_ABS_TOL, _REL_TOL * abs(g))
    if isinstance(gt, str) and isinstance(answer, str):
        gt_toks = [t for t in re.split(r"[,\s]+", gt.strip()) if t]
        an_toks = [t for t in re.split(r"[,\s]+", answer.strip()) if t]
        if len(gt_toks) > 1:
            if len(gt_toks) != len(an_toks):
                return False
            if qid in _ORDER_INSENSITIVE:
                return sorted(t.lower() for t in gt_toks) == sorted(t.lower() for t in an_toks)
            return [t.lower() for t in gt_toks] == [t.lower() for t in an_toks]
        return gt.strip().lower() == answer.strip().lower()
    return str(gt) == str(answer)


def has_real_comprehension(source: str) -> bool:
    """**真**推导式（`[... for ...]` / `(... for ...)` / `{... for ...}`）存在与否。

    这里破例用 AST：`has_iteration` 那个词法判据把普通 `for` 语句也算进去了，
    而「推导式」与「显式循环」是**两种不同的写法**，能不能合并成一句表达式
    恰恰是「模型会不会组合」这个观测点的关键 ⟹ 必须分清楚。
    词法匹配做不到这一点（`for` 后面跟不跟括号，正则很难可靠地区分）。

    解析失败时返回 False —— 但**不静默**：调用方 `ops_of` 会把 `parse_ok`
    一并写进产物，看到它就能知道这一栏是不是可信。
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(isinstance(n, (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp))
               for n in ast.walk(tree))


def ops_of(source: str) -> dict[str, Any]:
    """在程序源码上找组合算子。**以词法匹配为主，只对推导式破例用 AST。**

    不用 AST 的理由（对大部分字段成立）：这里要的是「模型有没有想到组合」这个
    **意图**层面的判断，`max(...)` 写在注释里也算它想到了。AST 会把注释丢掉。
    真正的合法性检查另有一处（`synthesizer.static_check`），不要在这里重复。
    """
    keyed = [op for op in _KEYED_OPS if op in source]
    bare = [op for op in _BARE_OPS if op in source]
    try:
        ast.parse(source)
        parse_ok = True
    except SyntaxError:
        parse_ok = False
    return {
        "keyed_ops": keyed,
        "bare_ops": bare,
        "has_key_arg": bool(_KEY_ARG_RE.search(source)),
        # 「出现过 for … in」—— 含普通 for 语句。**不是**「用了推导式」。
        "has_iteration": bool(_ITERATION_RE.search(source)),
        # 「真的用了推导式」—— AST 判的。
        "has_comprehension": has_real_comprehension(source),
        "parse_ok": parse_ok,
        "n_lines": len(source.splitlines()),
    }


def classify(*, qid: str, status: str, verdict: str, ops: dict[str, Any], gt: Any,
             answer: Any, abstained: bool) -> tuple[str, str]:
    """把一次运行归到 `OK / OK* / C / A / B` 之一，并给出一句判据。

    ⚠ **判序是「先看答案，再看算子」，不能反过来。**
    最初写成「先看有没有组合算子」是错的，实测里立刻现形：
    C2「哪幅画离镜子最近」用 `find_nearest` 一次调用就答对了，
    源码里当然没有 `max(` —— 于是被标成「压根没组合」。
    可它根本不是失败：**动作空间里已经有这个组合了**。
    判序反了会让「工具已经覆盖的组合」被统计成「模型不会组合」，
    而这个数字会直接误导「要不要加工具」的决定。

    所以现在的口径是：
      · 答案 = GT ⟹ `OK` / `OK*`（weak）/ `C`（被契约拒收）—— 三种都是**成功**。
        是否用了组合算子另记在 `used_combination` 字段里，不参与分类。
      · 答案 ≠ GT ⟹ `A`（源码里没有组合算子）/ `B`（有算子但算错）—— 失败归因。
    """
    if status in ("static_failed", "exec_failed", "llm_error"):
        if not (ops["keyed_ops"] or ops["bare_ops"]):
            return "B", "程序没过，且源码里没有组合算子 —— 失败点不在组合上"
        return "B", "程序没过，有组合算子 —— 组合写法的落地出了问题（看 failure.stage）"
    if abstained:
        return "A", "弃答 —— 现有返回值其实够用（见前半段的 10/13），属于没找到路"
    if same_answer(qid, gt, answer):
        # 先记住答案对不对，再看算子 —— 顺序不能反（见 docstring）。
        if verdict == "supported":
            return "OK", "答案与 GT 一致，证据被支持"
        if verdict == "weak":
            return "OK*", "答案与 GT 一致，但只判 weak（软检查没过：多值答案不逐字出现）"
        return "C", "答案与 GT 一致，但被契约判 %s —— 契约问题，与模型无关" % verdict
    # ⚠ 这里必须用 `used_combination`，**不能**只看 `has_comprehension`：
    #   后者现在只认真推导式，而手写 `for` + 累加器同样是组合。用它判会出现
    #   「写了循环、答案也错了，却被归成『压根没组合』（A）」，把失败类型判反。
    if used_combination(ops):
        return "B", "有组合算子但答案 ≠ GT —— 组合落了地但算错了"
    return "A", "答案 ≠ GT，且源码里没有组合算子"


def used_combination(ops: dict[str, Any]) -> bool:
    """模型有没有真的做「组合」。判据 = 带 key 的算子 / **任一形式的迭代** / `key=`。

    判据刻意放宽到包含普通 `for` 循环：手写 `for` + 累加变量**也是组合**
    （而且实测就是这 10 题里的主要形态，9/10）。把它排除在外，这个指标就只剩
    「有没有恰好写出 `max(`」这种措辞层面的东西。

    ⚠ 与它**不同**的一个观测点：`has_comprehension`（真推导式）。
    两者的差值 = 「用显式循环而不是一句推导式」的题数 —— 那个差值才有诊断价值，
    因为「能不能把循环压成表达式」才是组合能力的体现，而「写没写 for」不是。
    `len(` / `sum(` 不算：单独一个 `len()` 是在数返回值，与组合无关。
    """
    return bool(ops["keyed_ops"] or ops["has_iteration"]
                or ops["has_comprehension"] or ops["has_key_arg"])



def _gt_of(scene: Any, specs: list[Spec]) -> dict[str, dict[str, Any]]:
    """GT 表。

    `perfect_tool_calls` 是「GT 那条路线」的调用次数。
    `one_call` 是「只用一次 `list_objects` 能不能算出 GT」—— 这一列才是冗余的分母：
    GT 路线本身不一定最省（例如 C12 的 GT 路线要 36 次 `calculate_distance`，
    而一次 `list_objects` 拿到的质心就足以在纯 Python 里算完）。
    """
    oc = one_call_solutions(scene)
    gt: dict[str, dict[str, Any]] = {}
    for s in specs:
        r = run_spec(s, scene)
        oc_row = oc.get(s.qid) or {}
        gt[s.qid] = {
            "answer": r.answer, "error": r.error,
            "verdict_on_perfect": r.verdict_level,
            "perfect_tool_calls": r.n_tool_calls,
            "perfect_tools": r.tools_used,
            "one_call_answer": oc_row.get("answer"),
            "one_call_matches_gt": (oc_row.get("answer") is not None
                                    and same_answer(s.qid, r.answer, oc_row.get("answer"))),
        }
    return gt


def _record(spec: Spec, gt_row: dict[str, Any], d: dict[str, Any],
            *, elapsed_s: float | None = None, cost_cny: float | None = None) -> dict[str, Any]:
    """把一次运行的原始产物压成一条诊断记录。

    **真跑与离线复算共用这一处** —— 两处各写一份，就会出现「同一份 JSON 在两次
    分析里得到不同结论」这种最难排查的不一致。`--analyze` 走的也是这个函数。
    """
    ops = ops_of(d.get("program") or "")
    verdict_level = str((d.get("verdict") or {}).get("level") or "")
    cls, why = classify(qid=spec.qid, status=str(d.get("status") or ""),
                        verdict=verdict_level, ops=ops, gt=gt_row["answer"],
                        answer=d.get("answer"), abstained=bool(d.get("abstained")))
    perfect = int(gt_row.get("perfect_tool_calls") or 0)
    # 冗余的分母取「一次调用够不够」而不是 GT 路线的次数（见 _gt_of 的说明）。
    cheapest = 1 if gt_row.get("one_call_matches_gt") else perfect
    actual = int(d.get("tool_calls") or 0)
    return {
        "qid": spec.qid, "category": spec.category, "question": spec.question,
        "answer_type": spec.answer_type, "needs": spec.needs,
        "gt": gt_row["answer"], "gt_error": gt_row["error"],
        "gt_verdict_on_perfect": gt_row["verdict_on_perfect"],
        "perfect_tool_calls": perfect, "perfect_tools": gt_row.get("perfect_tools", []),
        "one_call_answer": gt_row.get("one_call_answer"),
        "one_call_matches_gt": bool(gt_row.get("one_call_matches_gt")),
        "min_tool_calls": cheapest,
        "answer": d.get("answer"), "abstained": bool(d.get("abstained")),
        "status": d.get("status"), "verdict": d.get("verdict") or {},
        "attempts": d.get("attempts"), "tool_calls": actual,
        # 冗余倍数：模型调了几次 / 最少几次。>1 = 返回值里已经有这个字段却没被用上。
        "call_ratio": (round(actual / cheapest, 2) if cheapest else None),
        "elapsed_s": elapsed_s if elapsed_s is not None else d.get("elapsed_s"),
        "cost_cny": cost_cny if cost_cny is not None else d.get("cost_cny"),
        "failure": d.get("failure"), "ops": ops,
        "used_combination": used_combination(ops),
        "program": d.get("program") or "",
        "trace_tools": d.get("trace_tools") or [row.get("tool") for row in (d.get("trace") or [])],
        # ★ 把**原始 trace 与 submit 参数**一起落盘。
        #   没有它们，`--analyze` 只能在旧 verdict 上重排报告 —— 口径改了也复算不出来，
        #   那就违背了本脚本「测量与分析分开」这条纪律（见 analyze 的 docstring）。
        #   有了它们，改 `agents/verifier.py` 的口径不必重跑模型。
        "evidence": list(d.get("evidence") or ()),
        "targets": list(d.get("target_ids") or ()),
        "trace": list(d.get("trace") or ()),
        "class": cls, "why": why,
    }


def _specs_of(scene: Any, args: argparse.Namespace) -> list[Spec]:
    wanted = [q.strip() for q in args.only.split(",") if q.strip()] or None
    return [s for s in SPECS
            if s.needs == "current_values" and s.solve is not None
            and (wanted is None or s.qid in wanted)]


def _load_scene_of(args: argparse.Namespace) -> tuple[Any, Path]:
    p = Path(args.scene)
    if not p.exists():
        p = scene_dir(args.scene)
    return load_scene(p), p


def _write(payload: dict[str, Any], args: argparse.Namespace, tag: str = "") -> Path:
    """落盘。**两种命名语义，别混**：

      · 真跑：`combination_run_<新时间戳>.json` —— 每一次都是一份**新的测量**。
      · 复算：`analyze()` 会先把 `args.out` 设成**源文件名 + `_reanalyzed`**，
        于是同一份测量只有一个复算产物，重算即覆盖。

    不加区分的话，同一个目录会同时存在「多份测量」与「同一份测量的多个版本」，
    而两者从文件名上看不出区别 —— 那时唯一的判据只剩 mtime，那太脆了。
    """

    out = Path(args.out) if args.out else (
        DEFAULT_OUT_DIR / ("combination_run_%s%s.json"
                           % (time.strftime("%Y%m%d_%H%M%S"), tag)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(_human(payload), encoding="utf-8")
    print("[combination-run] %s" % out)
    print("[combination-run] %s" % out.with_suffix(".md"))
    print("[combination-run] 累计成本 ¥%.4f" % payload["summary"]["cost_cny"])
    return out


def _reverify(row: dict[str, Any], spec: Spec, scene: Any) -> dict[str, Any]:
    """用**当前**的口径重算这一题：verdict + 组合算子 + 归因分类。

    这正是 `--analyze` 存在的理由：口径一定会被改（例如 2026-09-19 加了派生量档位、
    2026-09-20 把 `has_comprehension` 拆成「真推导式 / 出现过 for」），
    而每改一次口径就重跑一遍模型，会把「口径的影响」与「模型自身的波动」混在一起。

    ★ 所以**归因与分类也在复算范围里**：`ops` / `used_combination` / `class` / `why`
    全部从落盘的 `program` 重算。只重算 verdict 而把分类沿用旧值，会出现
    「verifier 是新口径、classify 是旧口径」的混合产物 —— 那种 JSON 最危险，
    因为它的每个字段单看都正常。

    ⚠ 2026-09-19 之前的产物没有落 `trace`，那时只能沿用落盘时的 verdict；
    这种情况如实标 `reverified=False`，免得把「旧口径的结论」当成「复算过的结论」。
    """
    trace = row.get("trace") or []
    program = str(row.get("program") or "")
    # 归因与分类**不依赖 trace**，所以即便没有 trace 也照样重算。
    ops = ops_of(program)
    cls, why = classify(qid=str(row.get("qid") or ""), status=str(row.get("status") or ""),
                        verdict=str((row.get("verdict") or {}).get("level") or ""),
                        ops=ops, gt=row.get("gt"), answer=row.get("answer"),
                        abstained=bool(row.get("abstained")))
    base = {**row, "ops": ops, "used_combination": used_combination(ops),
            "class": cls, "why": why}
    if not trace:
        return {**base, "reverified": False}
    targets = tuple(row.get("targets") or ())
    sub = Submission(
        answer=row.get("answer"), answer_type=spec.answer_type,
        target_ids=targets, evidence=tuple(row.get("evidence") or ()),
        # `unknown_targets` 是提交时的快照，没有落盘；用场景重算一遍，
        # 语义与 `execute_program` 里那次一致（都是「指认了但场景里没有」）。
        unknown_targets=tuple(t for t in targets
                              if scene is not None and not scene.has(t)),
        abstained=bool(row.get("abstained")),
    )
    verdict = verify(sub, trace=trace, scene=scene, answer_type=spec.answer_type)
    # 分类里用到的 verdict 也跟着更新（`class C` = 答对却被契约拒收，靠它判）
    cls2, why2 = classify(qid=str(row.get("qid") or ""), status=str(row.get("status") or ""),
                          verdict=str(verdict.level), ops=ops, gt=row.get("gt"),
                          answer=row.get("answer"), abstained=bool(row.get("abstained")))
    return {**base, "class": cls2, "why": why2,
            "verdict": verdict.to_dict(), "reverified": True}


def analyze(args: argparse.Namespace) -> int:
    """**零成本复算**：拿一份已保存的 JSON 重新做验证、归因与报告，不调模型。

    为什么必须有这条路：诊断口径（`verifier` / `classify` / `same_answer` / 指标定义）
    一定会被改。如果每改一次口径就要重跑一遍模型，那么「先跑便宜的一半、
    再决定跑不跑贵的一半」这条纪律就废了 —— 而且两次跑之间模型可能不同，
    口径改动的影响就与模型波动混在一起。把**测量**与**分析**分开，
    口径改动的影响就只落在分析上。
    """
    src_path = Path(args.analyze)
    src = json.loads(src_path.read_text(encoding="utf-8"))

    # ★ 复算的产物**按源文件名落盘**，不再另盖一个新时间戳。
    #
    # 为什么：`--analyze` 是零成本的，改一次口径就会重算一次。若每次都生成新名字，
    # 同一份 run 会在目录里堆出 N 份内容几乎相同的 `_reanalyzed`，而「哪份是当前的」
    # 只能靠文件 mtime 判断 —— 这正是**看起来正确、实际过期**的数字的来源。
    # 复算的是**同一份测量**，产物就该是**同一个位置**：重算即覆盖，天然幂等。
    # 真跑（不带 `--analyze`）仍然用时间戳命名 —— 那是**新的测量**，不该覆盖旧的。
    if args.out is None:
        args.out = str(src_path.with_name(src_path.stem + "_reanalyzed.json"))

    scene, scene_path = _load_scene_of(args)
    all_specs = {s.qid: s for s in SPECS}
    specs = [all_specs[r["qid"]] for r in src["runs"]]
    gt = _gt_of(scene, specs)
    runs = [_record(all_specs[r["qid"]], gt[r["qid"]], _reverify(r, all_specs[r["qid"]], scene))
            for r in src["runs"]]
    payload = {**src, "scene_path": str(scene_path), "summary": _summarize(runs),
               "reanalyzed_from": str(args.analyze),
               "reverified": all(r.get("reverified") for r in runs), "runs": runs}
    _write(payload, args, tag="_reanalyzed")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="组合题诊断（真跑 / 零成本复算）")
    ap.add_argument("--scene", default="points_probe")
    ap.add_argument("--only", default="", help="逗号分隔的题号，默认跑全部 needs=current_values 的题")
    ap.add_argument("--analyze", default=None,
                    help="★ 零成本：对一份已保存的 JSON 重新做归因与报告，不调模型")
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--env-file", default=None)
    ap.add_argument("--no-env-file", action="store_true")
    args = ap.parse_args(argv)

    if args.analyze:
        return analyze(args)

    scene, scene_path = _load_scene_of(args)
    specs = _specs_of(scene, args)
    if not specs:
        print("没有可跑的题")
        return 2

    gt = _gt_of(scene, specs)

    # ---- 装配（与 CLI / 演示后端共用 build_session）------------------------------
    session = build_session(scene_ref=args.scene, env_file=args.env_file,
                            no_env_file=args.no_env_file)
    if not session.ok:
        print("装配失败：%s" % session.error)
        return 2

    settings = LLMSettings.from_env("text")
    client = LLMClient(settings)
    client.check_ready()

    loop = AgentLoop(client, ctx=session.ctx, max_synthesis_retries=args.max_retries,
                     exec_timeout_s=args.timeout, render="template", planner="off",
                     toolset=session.toolset)

    runs: list[dict[str, Any]] = []
    for s in specs:
        t0 = time.perf_counter()
        run = loop.run(s.question, answer_type=s.answer_type)
        d = run.to_dict()
        rec = _record(s, gt[s.qid], d, elapsed_s=round(time.perf_counter() - t0, 3),
                      cost_cny=float((d.get("usage") or {}).get("cost_cny") or 0.0))
        runs.append(rec)
        print("[%s] %s  class=%s  answer=%r  gt=%r  calls=%s/%s"
              % (s.qid, run.status, rec["class"], run.answer, gt[s.qid]["answer"],
                 rec["tool_calls"], rec["perfect_tool_calls"]))

    payload = {
        "scene": args.scene, "scene_path": str(scene_path),
        "toolset": list(session.toolset), "switches": loop.switches(),
        "backend": settings.describe(),
        "gt_source": "scripts.probe_combination.SPECS（与前半段同一份实现）",
        "summary": _summarize(runs), "runs": runs,
    }
    _write(payload, args)
    return 0


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def tally(key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in runs:
            v = str(r.get(key))
            out[v] = out.get(v, 0) + 1
        return dict(sorted(out.items()))

    correct = [r for r in runs if same_answer(r["qid"], r["gt"], r["answer"]) and not r["abstained"]]
    rejected = [r for r in correct if r["class"] == "C"]
    ratios = [r["call_ratio"] for r in runs if r["call_ratio"]]
    return {
        "n": len(runs),
        "class": tally("class"),
        "status": tally("status"),
        "answer_matches_gt": len(correct),
        # ★ 最值钱的一列：**答案正确、却被自己的证据契约判 unsupported** 的题数。
        #   它和准确率无关 —— 它衡量的是「系统的验收标准会不会把对的答案扔掉」。
        "correct_but_rejected_by_contract": len(rejected),
        "used_combination": sum(1 for r in runs if r["used_combination"]),
        "keyed_ops_used": sum(1 for r in runs if r["ops"]["keyed_ops"]),
        "has_key_arg": sum(1 for r in runs if r["ops"]["has_key_arg"]),
        # ★ 这两个必须**一起**报，差值才是有信息量的那个数：
        #   `has_iteration` = 出现过 `for … in`（含普通 for 语句）
        #   `has_comprehension` = 真推导式（AST 判）
        #   差值 = 「用显式循环写、没压成一句表达式」的题数。
        "has_iteration": sum(1 for r in runs if r["ops"].get("has_iteration")),
        "has_comprehension": sum(1 for r in runs if r["ops"]["has_comprehension"]),
        "parse_ok": sum(1 for r in runs if r["ops"].get("parse_ok", True)),
        "tool_calls_total": sum(int(r["tool_calls"] or 0) for r in runs),
        "tool_calls_perfect_total": sum(int(r["perfect_tool_calls"] or 0) for r in runs),
        "tool_calls_min_total": sum(int(r["min_tool_calls"] or 0) for r in runs),
        # ★ 「一次 list_objects 就够」的题数 —— 直接量出「返回值里已经有却没被用上」的规模。
        "n_one_call_sufficient": sum(1 for r in runs if r["one_call_matches_gt"]),
        "call_ratio_median": (sorted(ratios)[len(ratios) // 2] if ratios else None),
        "cost_cny": round(sum(float(r.get("cost_cny") or 0.0) for r in runs), 6),
    }



def _human(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    L: list[str] = []
    L.append("# 组合题真跑诊断 —— %s" % payload["scene"])
    L.append("")
    L.append("真值来自 `scripts/probe_combination.py` 的同一份 SPECS（工具算出来的，不是脚本算的）。")
    L.append("")
    L.append("## 归因汇总")
    L.append("")
    L.append("- 题数 %d ｜ 与 GT 一致 **%d** ｜ 成本 ¥%.4f" % (s["n"], s["answer_matches_gt"], s["cost_cny"]))
    L.append("- 结局分布：`%s`" % json.dumps(s["class"], ensure_ascii=False))
    L.append("- 状态分布：`%s`" % json.dumps(s["status"], ensure_ascii=False))
    L.append("- 用了组合（**含显式 `for` 循环** / 带 key 算子 / `key=`）：**%d/%d**"
             % (s["used_combination"], s["n"]))
    L.append("  - 其中出现过 `for … in`：%d 题 ｜ **真推导式**：%d 题 ｜ 出现 `key=`：%d 题"
             % (s.get("has_iteration", 0), s["has_comprehension"], s["has_key_arg"]))
    L.append("  - ⚠ 差值是**有信息量**的那个数：它 = 「用显式循环写、没压成一句表达式」的题数。")
    L.append("    `has_iteration` 用的正是 `\\bfor\\b…in\\b`，它**同时匹配普通 `for x in y:` 语句**；")
    L.append("    早先这个字段叫 `has_comprehension`，于是「%d/%d 里出现过 for」被读成了"
             % (s.get("has_iteration", 0), s["n"]))
    L.append("    「%d/%d 用了推导式」—— 那句话是错的（2026-09-20 修正，见 `reports/glue_mining.md`）。"
             % (s.get("has_iteration", 0), s["n"]))
    L.append("- ★ **答对却被证据契约判 `unsupported`：%d 题**（与准确率无关）"
             % s["correct_but_rejected_by_contract"])
    L.append("- ★ 其中 **%d 题的全部所需几何量，一次 `list_objects` 就够**（上限：%d 次调用）"
             % (s["n_one_call_sufficient"], s["tool_calls_min_total"]))
    L.append("- 工具调用：实际 **%d** 次 ｜ GT 路线 %d 次 ｜ 最少 %d 次 ｜ 冗余倍数中位数 **%s**"
             % (s["tool_calls_total"], s["tool_calls_perfect_total"],
                s["tool_calls_min_total"], s["call_ratio_median"]))
    L.append("")
    L.append("⚠ 口径：「中位数」= **排序后取第 n/2 位**（见 `_summarize`）—— 偶数题数时取")
    L.append("偏大那一个，不是两数均值。同一份定义横跨各次运行所以可比，别与教科书中位数混用。")
    L.append("⚠ 分母（「最少几次」）取决于**当前**的 GT 与「一次 `list_objects` 够不够」的判定；")
    L.append("GT 口径一改，这一列就会跟着变 —— **模型没变而这一列变了**是可能的，")
    L.append("引用前先看 `reanalyzed_from` 与 `reverified` 两个字段。")
    L.append("")
    L.append("`class` 的含义：`OK/OK*` = 答对（证据 supported / weak）；`C` = 答对但证据被拒收；")
    L.append("`A` = 答错且没用组合；`B` = 答错但用了组合。**A 与 B 只描述失败**，")
    L.append("答对的题不归到 A/B —— 「工具已经覆盖这个组合」不该被统计成「模型不会组合」。")
    L.append("")
    L.append("`冗余倍数` = 实际调用次数 / 最少调用次数。>1 说明**模型没利用返回值里已有的字段**，")
    L.append("而不是「工具不够用」—— 这一列把「功能单薄」与「没看清返回值」分开了。")
    L.append("")
    L.append("`组合` 一列 = 出现过 `for … in`（**含手写循环**）/ 带 key 算子 / `key=` 之一。")
    L.append("它是**宽判据**：手写 `for` + 累加变量也算组合。窄的那个数是首页的「真推导式」。")
    L.append("")
    L.append("## 逐题")
    L.append("")
    L.append("| 题号 | class | 状态 | 答案 | GT | 证据 | 组合 | 调用(实际/最少) | 倍数 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in payload["runs"]:
        L.append("| %s | **%s** | %s | `%s` | `%s` | %s | %s | %s/%s | %s |"
                 % (r["qid"], r["class"], r["status"], r["answer"], r["gt"],
                    (r["verdict"] or {}).get("level", "—"),
                    "是" if r["used_combination"] else "**否**",
                    r["tool_calls"], r["min_tool_calls"],
                    r["call_ratio"] if r["call_ratio"] is not None else "—"))
    L.append("")
    for r in payload["runs"]:
        L.append("### %s —— %s" % (r["qid"], r["question"]))
        L.append("")
        L.append("- class **%s**：%s" % (r["class"], r["why"]))
        L.append("- 答案 `%s` ｜ GT `%s` ｜ 证据 `%s` ｜ 工具调用序列 `%s`"
                 % (r["answer"], r["gt"], (r["verdict"] or {}).get("level", "—"),
                    " → ".join(str(t) for t in r["trace_tools"]) or "—"))
        L.append("- 组合 `%s` ｜ 调用 %s 次（最少 %s 次，GT 路线 %s 次，倍数 %s）｜ 生成 %s 次 ｜ 耗时 %s s"
                 % ("是" if r["used_combination"] else "**否**", r["tool_calls"],
                    r["min_tool_calls"], r["perfect_tool_calls"], r["call_ratio"],
                    r["attempts"], r["elapsed_s"]))
        if r["failure"]:
            L.append("- 失败：`%s`" % json.dumps(r["failure"], ensure_ascii=False))
        L.append("")
        L.append("```python")
        L.append(r["program"].rstrip())
        L.append("```")
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
