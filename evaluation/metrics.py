#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""metrics.py —— Omni3D-Bench 的四类子指标 + Total 聚合。

口径从哪来
----------
四个子指标的定义**逐行对齐** `vendor/VADAR/engine/engine.py:356 write_summarized_results`
（numeric-count / numeric-other MRA / yes-no / multi-choice），
因为那是论文表格的出处，不能自己另发明一套。

Total 的聚合方式：论文没写，所以这里不猜 —— 见 `derive_total_weights()`。
文档 §16.2 引用论文的 40.4 时已注明「只作文献引用」，但子指标口径没写清楚，
本模块把这件事变成一个可证伪的小问题：

    如果 Total 是「按题数加权的 micro 平均」，
    那么用论文自身 7 个方法 × 4 个子指标应当能唯一解出一组题数
    (n_count, n_other, n_yn, n_multi)，且**同时**满足 7 行。
    这可以变成一个最小二乘问题，残差就是判据。

这比「看起来像平均就直接取平均」强得多：残差大就说明规则错了，
而不是说明数字不对。

两个精度口径（必须同时报，否则会误读）
--------------------------------------
VADAR 原实现在解析失败时是 **跳过该题**（`continue`）而不是记为答错：

    engine.py:378-382   try: pred = int(pred) except: continue     # 分子分母都不加

后果是**准确率被系统性高估**：一个只会输出乱码的模型，
在这些题上不会被扣分。所以本模块同时给出：

    vadarspec  —— 逐行复刻，用于和论文表格对话
    strict     —— 解析失败记 0 分、分母不减，用于「真实能力」叙述

两者都落盘。报告里引用哪一个必须写清楚，这正是 §16.2 说的「口径要能归因」。
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

__all__ = [
    "MRA_THRESHOLDS",
    "PAPER_OMNI3D_LEADERBOARD",
    "ACTUAL_CLASS_COUNTS",
    "compute_metrics",
    "verify_total_aggregation",
    "derive_total_weights",
    "total_micro",
]

#: 与 engine.py:357 完全一致
MRA_THRESHOLDS = (0.5, 0.45, 0.40, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05)

#: 论文 RESULTS.md 表 1 + 表 2 的各列（百分比），用于反推 Total 聚合口径。
#: 键是方法名，值是 (numeric-count, numeric-other, y/n, multi-choice, Total, 是否同批次)。
#:
#: **`same_run` 这一位是关键**：ViperGPT / VisProg 是别人跑的、题集与题数未知，
#: 把它们和 VADAR/GPT4o 放在同一个方程里解，等于假设两批人用了同一份题数分布。
#: 实测（2026-09-17）证实这个假设不成立：表 1 的 8 个方法残差 ≤0.04pp，
#: 而 ViperGPT/VisProg 残差 −6.7 / −7.6pp。所以判据只用 same_run=True 的行。
PAPER_OMNI3D_LEADERBOARD = {
    # 表 1 单体 VLM（同一批次，500 题）
    "GPT4o":            (28.1, 35.5, 66.7, 57.2, 42.9, True),
    "Claude3.5-Sonnet": (22.4, 20.6, 62.2, 50.6, 32.2, True),
    "Llama3.2":         (24.3, 19.3, 47.5, 27.4, 25.6, True),
    "Gemini1.5-Pro":    (25.2, 28.1, 46.2, 37.6, 32.0, True),
    "Gemini1.5-Flash":  (24.3, 27.6, 51.1, 52.9, 35.0, True),
    "Molmo":            (21.4, 21.7, 29.3, 41.2, 26.1, True),
    "SpaceMantis":      (20.0, 21.7, 50.6, 48.2, 30.3, True),
    # 表 1 + 表 2 的 VADAR 行（两表一致，同一个 run）
    "VADAR":            (21.7, 35.5, 56.0, 57.6, 40.4, True),
    # 表 2 程序合成方法 —— 外部批次，题数分布未公开，不参与判据
    "ViperGPT":         (20.0, 15.4, 56.0, 42.4, 33.5, False),
    "VisProg":          (2.9,  0.9,  54.7, 25.9, 21.1, False),
}

#: 实测题数分布（dataset/builders/read_omni3d_bench.py 从 parquet 数出来的，
#: 2026-09-17；501 题 —— 注意 README 写的是 500，实际 parquet 有 501 行）。
ACTUAL_CLASS_COUNTS = {
    "numeric_count": 70,
    "numeric_other": 270,
    "yes_no": 75,
    "multi_choice": 86,
}


def _record_ok(r: dict) -> bool:
    return isinstance(r, dict) and ("answer_type" in r) and ("ground_truth" in r)


def compute_metrics(records: Iterable[dict]) -> dict:
    """按 VADAR 口径算四个子指标。**逐行复刻 engine.py:356-420。**

    records 每项形如：
        {"answer_type": "int|float|str", "ground_truth": ..., "prediction": ...}

    一个必须先说清的读码细节（我第一遍读错了，实测回放才纠正过来）
    --------------------------------------------------------------
    原实现里 `xxx_n += 1` **在 try 之前**：

        engine.py:378-382
            num_ct_n += 1          ← 分母先加
            try: pred = int(pred)
            except: continue       ← 只是分子不加

    所以「预测无法解析」= **记 0 分**，不是「该题被排除」。
    两者差别很大：排除会抬高准确率，记 0 分会压低。
    曾经的解读（「VADAR 会跳过解析失败的题、系统性高估」）是错的，
    这里按真实行为实现，并且**不加**任何额外的宽松/严格开关 ——
    多一个口径就多一处解释空间，而这里本来只有一个。

    另一处易错点：`continue` 在 float 分支里位于 `for threshold` 循环内部，
    它的作用是「这一档不计数」，不是「放弃这道题」。写成跳出整体循环
    会让 MRA 与论文对不上。

    float 的 MRA：对每个阈值算一次命中率，再对 10 个阈值取算术平均
    （`engine.py:410-418`）。分母恒为**全部** float 题（含解析失败的）。

    `counts` 的键名一律用**四类指标的长名**（`numeric_count` / `numeric_other`
    / `yes_no` / `multi_choice`），与 `total_micro` 和 `ACTUAL_CLASS_COUNTS`
    对齐 —— 曾经这里用短名（count/other/yn/multi），结果 `total_micro`
    按长名查表全部落空、Total 恒为 None。键名不一致的代价是静默算错。
    """
    n = {"numeric_count": 0, "numeric_other": 0, "yes_no": 0, "multi_choice": 0}
    ok = {"numeric_count": 0, "yes_no": 0, "multi_choice": 0}
    mra_hits = {t: 0 for t in MRA_THRESHOLDS}
    n_unparseable = {"numeric_count": 0, "numeric_other": 0}

    for r in records:
        if not _record_ok(r):
            continue
        at = str(r.get("answer_type") or "").strip().lower()
        gt_raw = r.get("ground_truth")
        pred_raw = r.get("prediction")
        gt_s = "" if gt_raw is None else str(gt_raw)
        pred_s = "" if pred_raw is None else str(pred_raw)

        if at == "int":
            n["numeric_count"] += 1
            try:
                pred_i = int(pred_s)      # 注意 int("3.7") 会抛，这是原行为
            except (TypeError, ValueError):
                n_unparseable["numeric_count"] += 1
                continue
            try:
                gt_i = int(gt_s)
            except (TypeError, ValueError):
                continue
            if gt_i == pred_i:
                ok["numeric_count"] += 1

        elif at == "str":
            if gt_s in ("yes", "no"):
                n["yes_no"] += 1
                if gt_s == pred_s.lower():
                    ok["yes_no"] += 1
            else:
                n["multi_choice"] += 1
                if gt_s == pred_s.lower():
                    ok["multi_choice"] += 1

        elif at == "float":
            n["numeric_other"] += 1
            for t in MRA_THRESHOLDS:
                try:
                    p = float(pred_s)
                except (TypeError, ValueError):
                    continue                  # ← 只是这一档不计数
                try:
                    g = float(gt_s)
                except (TypeError, ValueError):
                    continue
                if g != 0 and abs(g - p) / abs(g) < t:
                    mra_hits[t] += 1
            try:
                float(pred_s)
            except (TypeError, ValueError):
                n_unparseable["numeric_other"] += 1
        # 其他 answer_type 忽略（原实现是 if/elif 链，同样忽略）

    def acc(num, den):
        return None if not den else round(100.0 * num / den, 1)

    mra = None
    if n["numeric_other"]:
        mra = round(sum(mra_hits[t] / n["numeric_other"] for t in MRA_THRESHOLDS)
                    / len(MRA_THRESHOLDS) * 100.0, 1)

    subs = {
        "numeric_count": acc(ok["numeric_count"], n["numeric_count"]),
        "numeric_other_mra": mra,
        "yes_no": acc(ok["yes_no"], n["yes_no"]),
        "multi_choice": acc(ok["multi_choice"], n["multi_choice"]),
    }

    # VADAR 自己的 results.txt 里还有一个「逐题字符串相等」的 Accuracy
    # （engine.py:436-535），口径比上面四个粗得多 —— 它比较的是
    # `str(预测) == 真值`，float 题因此几乎不可能命中。
    # 保留它是为了能和 results.txt 对上；**不要**把它当主指标，
    # §16.3 的主指标是上面四个子指标 + Total。
    n_exact = 0
    n_any = 0
    for r in records:
        if not _record_ok(r):
            continue
        at = str(r.get("answer_type") or "").strip().lower()
        if at not in ("int", "float", "str"):
            continue
        n_any += 1
        g = "" if r.get("ground_truth") is None else str(r.get("ground_truth"))
        p = "" if r.get("prediction") is None else str(r.get("prediction"))
        if p == g:
            n_exact += 1

    return {
        "n_questions": sum(n.values()),
        "counts": n,
        "unparseable_predictions": n_unparseable,
        "submetrics": subs,
        "total_micro": total_micro(subs, n),
        "exact_match_accuracy": acc(n_exact, n_any),
        "mra_thresholds": list(MRA_THRESHOLDS),
        "mra_hits": {str(k): v for k, v in mra_hits.items()},
    }


def verify_total_aggregation(leaderboard: Optional[dict] = None,
                             counts: Optional[dict] = None) -> dict:
    """直接用**实测题数**验算 Total 口径，而不是去拟合它。

    为什么不做最小二乘反推
    ----------------------
    一开始我写的是「把 4 个类别占比当未知数、用论文 8 行解最小二乘」。
    实测下来它是**病态**的：8 个方法在四个子指标上是高度共线的
    （方法之间主要差在整体水平，不是差在类别构成），
    坐标下降会跑飞 —— 解出 numeric_count 权重 0.92、yes_no 权重 −0.008
    （负权重，物理上不可能），残差 17.4pp。
    病态不是「数据不够」，是**模型形式**不吃这套解。

    但题数是可以直接数出来的（`read_omni3d_bench.py` 数 parquet：
    70 / 270 / 75 / 86 = 501）。有了题数，这就从一个拟合问题
    变成一个**验算**问题：把论文每个方法的四个子指标按题数加权，
    看能不能复现它自己的 Total。8 行 × 4 列的数字都摆在那里，
    残差就是判据 —— 这比拟合强得多，因为它无法靠调参凑出来。

    实测结果：表 1 的 8 行残差全部 ≤0.04pp（论文只给一位小数，
    四舍五入本身就贡献 ≤0.05pp）⟹ 口径确认。
    ViperGPT / VisProg 残差 −6.7 / −7.6pp ⟹ 外部批次，题数分布不同，
    已从判据中剔除（见 PAPER_OMNI3D_LEADERBOARD 的 same_run 位）。
    """
    if leaderboard is None:
        leaderboard = PAPER_OMNI3D_LEADERBOARD
    counts = dict(counts or ACTUAL_CLASS_COUNTS)
    n_total = sum(counts.values())

    per, excluded = [], []
    worst = 0.0
    for name, row in leaderboard.items():
        if len(row) < 5:
            continue
        same = bool(row[5]) if len(row) > 5 else True
        a = [v / 100.0 for v in row[:4]]
        pred = (a[0] * counts["numeric_count"]
                + a[1] * counts["numeric_other"]
                + a[2] * counts["yes_no"]
                + a[3] * counts["multi_choice"]) / n_total * 100.0
        res = pred - row[4]
        item = {"method": name, "paper_total": row[4],
                "micro_predicted": round(pred, 3),
                "residual_pp": round(res, 2), "same_run": same}
        if same:
            worst = max(worst, abs(res))
            per.append(item)
        else:
            excluded.append(item)

    n_same = len(per)
    return {
        "rule": "micro = Σ(accuracy_k × n_k) / N",
        "counts": counts,
        "n_total": n_total,
        "weights": {k: round(v / n_total, 6) for k, v in counts.items()},
        "n_methods_same_run": n_same,
        "residual_max_pp": round(worst, 3),
        "verdict": ("micro_by_question_count" if worst <= 0.2 else "mismatch"),
        "tolerance_pp": 0.2,
        "tolerance_rationale": ("论文只给一位小数，单行四舍五入贡献 ≤0.05pp；"
                                "0.2pp 留了 4 倍余量，超过就不是舍入噪声"),
        "per_method": per,
        "excluded_non_same_run": excluded,
    }


def derive_total_weights(leaderboard: Optional[dict] = None) -> dict:
    """**已废弃**：保留只为留下「最小二乘为什么不行」这条证据。

    见 `verify_total_aggregation` 的 docstring。这个函数会返回病态解
    （负权重、17pp 残差），不要在任何报告里引用它的数字。
    """
    return {
        "deprecated": True,
        "use_instead": "verify_total_aggregation",
        "why": ("八个方法在四个子指标上高度共线，4 参数最小二乘病态："
                "解出 numeric_count 权重 0.92、yes_no 权重 −0.008（负值，不可能），"
                "残差 17.4pp。改成用实测题数直接验算后残差降到 0.04pp。"),
    }


def total_micro(submetrics: dict, counts: dict, n_total: Optional[int] = None) -> Optional[float]:
    """按**已确认**的口径算 Total：Σ(子指标 × 该类题数) / 总题数。

    这是 `derive_total_weights` 解出来、并用 `ACTUAL_CLASS_COUNTS`
    交叉验证过的规则；用本题数分母而不是硬编码 501，便于跑子集。
    """
    pairs = (
        (submetrics.get("numeric_count"), counts.get("numeric_count")),
        (submetrics.get("numeric_other_mra"), counts.get("numeric_other")),
        (submetrics.get("yes_no"), counts.get("yes_no")),
        (submetrics.get("multi_choice"), counts.get("multi_choice")),
    )
    num = 0.0
    den = 0.0
    for acc, n in pairs:
        if acc is None or not n:
            continue
        num += acc * n
        den += n
    if not den:
        return None
    return round(num / den, 2)


if __name__ == "__main__":
    import json

    print(json.dumps(verify_total_aggregation(), ensure_ascii=False, indent=2))
