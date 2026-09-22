#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""内参敏感度探针的比对（原 `_tmp_compare_probe.py`，已固化 —— 可重放是它存在的理由）。

它回答一个问题
--------------
**端到端 QA 对内参错误有多敏感？**
做法：同一张图、同一批掩码、同一批题目，只在两档内参下各跑一遍，再逐题对照。

口径（**读数字前必读**）
------------------------
真值来源 = **参照档几何**（默认 `probe_instr_gt`，内参外部给定 fx=518.86）。
⚠ 它**不是标定真值**，只是「外部给定的内参」这一档算出来的几何 ——
Omni3D-Bench 上没有 GT 相机，所以本仓库不存在绝对真值。
因此本脚本量的是「预测档答案相对参照档答案的**偏离**」，是**相对量**，不是绝对精度。
把它读成"准确率"就错了。

唯一变量这件事**不靠假设，靠验**
--------------------------------
两档的提示词必须逐字节对齐，差异才可归因到几何。这一条曾经不成立：
`scene_hint_for` 把 `build_meta.intrinsics_source`（`predicted` / `provided`）
**透传进了提示词**，于是两档同时改了「几何」和「暗示」两个自变量
（已修，见 `agents/prompts/system.py` 的 ⚠⚠ 段与 `tests/test_agent_static_check.py`
的两条守卫）。所以本脚本跑之前**自己验一遍**（第 0 段），
验不过就退出 —— 宁可不出表，不要出一张不可归因的表。

判定两条轨，都报，因为它们回答不同问题
--------------------------------------
    · 连续轨：相对误差 = |答案 - 真值| / 真值  —— 不受容差主观性影响
    · 二元轨：若干容差档下的「对/错」  —— 这才是 QA 正确率的口径，
      但**容差会决定结论**（实测：沙发↔桌子两档差 13%，容差 5% 判错、20% 判对）

⚠ 所以第 4 段会**故意**报一张「同一批答案，只换容差」的表。
那不是凑数：本项目已经吃过一次「尺子错了 ≠ 数字错了」的亏 ——
第一版判定只认中文，而模型答的是 `left` / `back` / `behind`，把 3 道全对的题判成错，
还顺手把锅推给模型（"参照档也错"）。同一份数据、只换尺子，结论就变。

真值不写进题集文件
------------------
`dataset/probes/intrinsics_sensitivity_20q.json` 里**只有题目与分组**，没有答案。
真值由图上的几何实时算（`build_truth`），免得"尺子写死"之后被复制传播。
代价是**顺序即契约**：题集与 `build_truth` 必须逐题对齐。这里不靠注释提醒，
靠 `_assert_order_contract()` 硬校验（题目文本 + 分组都要对上），错位就退出。

用法
----
    python scripts/compare_intrinsics_probe.py
    python scripts/compare_intrinsics_probe.py --json-out reports/intrinsics_probe.json

退出码：0 正常；2 输入/对齐问题；3 提示词未对齐（除非 `--allow-hint-drift`）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENES = PROJECT_ROOT / "dataset" / "scenes"
RUNS = PROJECT_ROOT / "logs" / "agent_runs"

DEFAULT_QUESTIONS = PROJECT_ROOT / "dataset" / "probes" / "intrinsics_sensitivity_20q.json"
DEFAULT_PRED_SCENE = "probe_instr_pred"      # 内参 = 模型自预测（fx≈163.7）
DEFAULT_GT_SCENE = "probe_instr_gt"          # 内参 = 外部给定（fx≈518.9）—— 参照档，非真值
DEFAULT_PRED_RUN = RUNS / "probe_intrinsics_pred.json"
DEFAULT_GT_RUN = RUNS / "probe_intrinsics_gt.json"

#: 容差档。`0.05` 是本项目其它指标惯用的那档，列在最前只是为了好读。
TOLERANCES = (0.05, 0.005, 0.02, 0.10, 0.20)

#: 真值构造依赖这几个 id。缺任何一个都说明场景换了、题目语义已经不成立 ⟹ 直接退出。
REQUIRED_IDS = ("table_1", "chair_1", "mirror_1", "sofa_1")

CATEGORIES = ("abs", "rel", "cnt")
CATEGORY_NOTES = {
    "abs": "绝对米制 · 预期敏感",
    "rel": "相对序 · 预期免疫",
    "cnt": "计数类别 · 对照",
}

#: 别名表：把中英文答案与场景标签都归一到**规范标签**。
#: ⚠ 这张表是判定的一部分，改它等于改尺子 —— 与数字一起改、一起报。
LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "table": ("table", "桌子", "茶几", "桌"),
    "chair": ("chair", "椅子", "椅"),
    "sofa": ("sofa", "沙发"),
    "mirror": ("mirror", "镜子", "镜"),
    "picture": ("picture", "painting", "画"),
}

SIDE_ALIASES: dict[str, tuple[str, ...]] = {
    "左": ("left", "左"),
    "右": ("right", "右"),
    "前": ("front", "ahead", "前"),
    "后": ("back", "behind", "rear", "后"),
}


# ---------------------------------------------------------------------------
# 几何
# ---------------------------------------------------------------------------


def load_geom(scene_id: str) -> dict[str, dict[str, Any]]:
    path = SCENES / scene_id / "scene.json"
    if not path.is_file():
        raise FileNotFoundError("找不到场景：%s" % path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {n["id"]: n for n in raw["scene"]["nodes"]}


def dist(a: dict, b: dict) -> float:
    return math.sqrt(sum((a["centroid_3d"][i] - b["centroid_3d"][i]) ** 2 for i in range(3)))


def cam_dist(n: dict) -> float:
    """到相机的距离（相机系原点）。"""
    return math.sqrt(sum(v * v for v in n["centroid_3d"]))


# ---------------------------------------------------------------------------
# 真值构造 —— **顺序即契约**
# ---------------------------------------------------------------------------


def build_truth(g: dict[str, dict]) -> list[dict[str, Any]]:
    """逐题给出真值与判定方式。**题面文本必须与题集文件逐字相同**（见 `_assert_order_contract`）。

    `kind` 决定用哪把尺子：
        num   —— 数值，按相对误差判（真值与容差都进表）
        obj   —— 答"是哪一个物体"，按标签别名比对
        side  —— 答左/右/前/后，按方向别名比对
        set   —— 答一组类别
        yesno —— 是/否

    ⚠ 这个函数**与场景绑定**（它编码了"每道题问的到底是哪两个物体"）。
    换场景不是换参数，是重写这个函数 —— 写成配置只会让错误的绑定更难发现。
    """
    missing = [i for i in REQUIRED_IDS if i not in g]
    if missing:
        raise KeyError(
            "场景缺少真值构造所需的 id：%s；现有 id = %s。"
            "换场景必须重写 build_truth（它编码了每道题的语义），不能只改参数。"
            % (missing, sorted(g))
        )

    order_by_cam = sorted(g, key=lambda k: cam_dist(g[k]))
    nearest, second, farthest = order_by_cam[0], order_by_cam[1], order_by_cam[-1]
    label_of = {k: g[k]["label"] for k in g}

    def nearer(a: str, b: str) -> str:
        """两物体谁离相机更近 —— 直接比，不依赖"最近者恰好是其中之一"这种巧合。"""
        return label_of[a] if cam_dist(g[a]) < cam_dist(g[b]) else label_of[b]

    def farther(a: str, b: str) -> str:
        return label_of[a] if cam_dist(g[a]) > cam_dist(g[b]) else label_of[b]

    return [
        # -- abs：绝对米制，预期敏感 -----------------------------------------
        {"q": "桌子离相机多少米？", "short": "桌子到相机", "kind": "num",
         "truth": cam_dist(g["table_1"]), "cat": "abs"},
        {"q": "椅子离相机多少米？", "short": "椅子到相机", "kind": "num",
         "truth": cam_dist(g["chair_1"]), "cat": "abs"},
        {"q": "镜子离相机多少米？", "short": "镜子到相机", "kind": "num",
         "truth": cam_dist(g["mirror_1"]), "cat": "abs"},
        {"q": "沙发沿左右方向（相机坐标系的 x 轴）的跨度是多少米？", "short": "沙发x跨度", "kind": "num",
         "truth": g["sofa_1"]["extent_3d"][0], "cat": "abs"},
        {"q": "沙发和桌子之间的质心距离是多少米？", "short": "沙发↔桌子", "kind": "num",
         "truth": dist(g["sofa_1"], g["table_1"]), "cat": "abs"},
        {"q": "桌子和椅子之间的质心距离是多少米？", "short": "桌子↔椅子", "kind": "num",
         "truth": dist(g["table_1"], g["chair_1"]), "cat": "abs"},
        {"q": "场景里离相机最远的那个物体，离相机多少米？", "short": "最远物体到相机", "kind": "num",
         "truth": cam_dist(g[farthest]), "cat": "abs"},
        {"q": "镜子和沙发之间的距离是多少米？", "short": "镜子↔沙发", "kind": "num",
         "truth": dist(g["mirror_1"], g["sofa_1"]), "cat": "abs"},

        # -- rel：相对序，预期免疫 -------------------------------------------
        {"q": "哪个物体离相机最近？", "short": "最近者", "kind": "obj",
         "truth": label_of[nearest], "cat": "rel"},
        {"q": "哪个物体离相机最远？", "short": "最远者", "kind": "obj",
         "truth": label_of[farthest], "cat": "rel"},
        {"q": "离相机第二近的是哪个物体？", "short": "第二近者", "kind": "obj",
         "truth": label_of[second], "cat": "rel"},
        {"q": "椅子在桌子的左边还是右边？", "short": "椅子vs桌子·左右", "kind": "side",
         "truth": "左" if g["chair_1"]["centroid_3d"][0] < g["table_1"]["centroid_3d"][0] else "右",
         "cat": "rel"},
        {"q": "沙发在桌子的前面还是后面？", "short": "沙发vs桌子·前后", "kind": "side",
         # 相机系 z 越大越远 ⟹ z 小的是"前面"
         "truth": "前" if g["sofa_1"]["centroid_3d"][2] < g["table_1"]["centroid_3d"][2] else "后",
         "cat": "rel"},
        {"q": "镜子在沙发的左边还是右边？", "short": "镜子vs沙发·左右", "kind": "side",
         "truth": "左" if g["mirror_1"]["centroid_3d"][0] < g["sofa_1"]["centroid_3d"][0] else "右",
         "cat": "rel"},
        {"q": "桌子和椅子哪个离相机更近？", "short": "桌子vs椅子·谁近", "kind": "obj",
         "truth": nearer("table_1", "chair_1"), "cat": "rel"},
        {"q": "沙发和镜子哪个离相机更远？", "short": "沙发vs镜子·谁远", "kind": "obj",
         "truth": farther("sofa_1", "mirror_1"), "cat": "rel"},

        # -- cnt：与尺度无关的对照 -------------------------------------------
        {"q": "场景里一共有多少个物体？", "short": "物体总数", "kind": "num",
         "truth": float(len(g)), "cat": "cnt"},
        {"q": "场景里有几幅画？", "short": "画的数量", "kind": "num",
         "truth": float(sum(1 for v in g.values() if v["label"] == "picture")), "cat": "cnt"},
        {"q": "场景里有哪些不同类别？", "short": "类别清单", "kind": "set",
         "truth": sorted({v["label"] for v in g.values()}), "cat": "cnt"},
        {"q": "场景里有床吗？", "short": "有没有床", "kind": "yesno",
         "truth": "no", "cat": "cnt"},
    ]


def _assert_order_contract(truth: Sequence[dict], questions: Sequence[dict]) -> None:
    """题集文本/分组与 `build_truth` 必须逐题对齐 —— 错位会让"对错"完全反过来。

    这是 docstring 里"顺序即契约"那句话的**执行体**：没有它，那句只是安慰。
    """
    if len(truth) != len(questions):
        raise ValueError(
            "题数与真值表不一致：题集 %d 条、build_truth %d 条" % (len(questions), len(truth)))
    for i, (t, q) in enumerate(zip(truth, questions), 1):
        if t["q"] != q.get("question"):
            raise ValueError(
                "第 %d 题对不上（顺序即契约）：\n  题集 = %r\n  真值 = %r" % (i, q.get("question"), t["q"]))
        if t["cat"] != q.get("category"):
            raise ValueError(
                "第 %d 题分组对不上：题集 = %r、真值 = %r" % (i, q.get("category"), t["cat"]))


# ---------------------------------------------------------------------------
# 判定（尺子）
# ---------------------------------------------------------------------------


def as_number(v: Any) -> float | None:
    """宽松地把答案读成数（模型有时带单位/汉字）。读不出来返回 None —— 不是 0。"""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace("米", "").replace("m", "").replace("约", "").replace("，", "")
        try:
            return float(s)
        except ValueError:
            return None
    return None


def label_mentions(text: str) -> frozenset[str]:
    """文本里提到了哪些**规范标签**。用于「答的是哪一个物体」这类题。"""
    low = text.lower()
    return frozenset(
        canon for canon, alts in LABEL_ALIASES.items() if any(a.lower() in low for a in alts))


def judge(item: dict, answer: Any) -> dict[str, Any]:
    """判定单题。返回 `{"ok": True/False/None, ...}`；`None` = 尺子覆盖不到，**不算错**。"""
    kind, truth = item["kind"], item["truth"]

    if kind == "num":
        got = as_number(answer)
        if got is None:
            return {"ok": None, "rel_err": None, "note": "非数值/弃答"}
        return {"ok": abs(got - truth) <= 0.05 * abs(truth) if truth else got == 0,
                "rel_err": abs(got - truth) / abs(truth) if truth else abs(got), "got": got}

    if kind == "obj":
        if not isinstance(answer, str):
            return {"ok": None, "truth": truth, "note": "非字符串"}
        # 判据：答案提到的规范标签 ∩ 真值物体的规范标签 ≠ ∅。
        # ⚠ 已知偏宽：真值是 `sofa chair`（一个粘连的检测框）而模型答 `chair` 时算对。
        #   方向是**有利于模型**，所以不会把"模型不会做题"说过头。
        shared = label_mentions(answer) & label_mentions(str(truth))
        return {"ok": bool(shared), "truth": truth, "got": answer.strip().lower()}

    if kind == "side":
        if not isinstance(answer, str):
            return {"ok": None, "truth": truth}
        a = answer.strip().lower()
        return {"ok": any(w in a for w in SIDE_ALIASES[truth]), "truth": truth, "got": a}

    if kind == "yesno":
        if not isinstance(answer, str):
            return {"ok": None, "note": "非字符串"}
        says_no = any(x in answer for x in ("没有", "无", "no", "不存在"))
        says_yes = any(x in answer for x in ("有床", "yes", "是的"))
        return {"ok": says_no and not says_yes, "truth": "没有"}

    if kind == "set":
        if isinstance(answer, (list, tuple)):
            got = {str(x).strip().lower() for x in answer}
        elif isinstance(answer, str):
            # 模型回的是逗号分隔的字符串，不是 list —— 第一版尺子只认 list 就漏判了。
            got = {s.strip().lower() for s in answer.replace("、", ",").split(",") if s.strip()}
        else:
            return {"ok": None, "note": "非集合"}
        want = set(str(x).lower() for x in truth)
        return {"ok": want <= got or got == want, "truth": truth, "got": sorted(got)}

    return {"ok": None, "note": "未知 kind：%s" % kind}


# ---------------------------------------------------------------------------
# 第 0 段：提示词对齐
# ---------------------------------------------------------------------------


def check_hint_alignment(pred_scene: str, gt_scene: str) -> tuple[bool, str]:
    """两档的 `scene_hint` 是否逐字节相同。**这是可归因性的前提。**

    ⚠ 若你指向的是 `living_room_pred` / `living_room_gt`，它们 `scene_id` 不同
    ⟹ 这里会报差异。那**不是**误报：`scene_id` 确实进了提示词，两个提示词确实不同。
    干净的做法是用 `probe_instr_*`（已统一 `scene_id` 并去掉内参来源字段）。
    """
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from agents.prompts.system import scene_hint_for  # noqa: PLC0415
    from scene_graph.store import load_scene, scene_dir  # noqa: PLC0415

    a = scene_hint_for(load_scene(scene_dir(pred_scene)))
    b = scene_hint_for(load_scene(scene_dir(gt_scene)))
    ja = json.dumps(a, ensure_ascii=False, sort_keys=True)
    jb = json.dumps(b, ensure_ascii=False, sort_keys=True)
    if ja == jb:
        return True, ja
    return False, "pred档 = %s\n             参照档 = %s" % (ja, jb)


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="内参敏感度探针比对（口径见文件头 docstring）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--pred-run", type=Path, default=DEFAULT_PRED_RUN)
    ap.add_argument("--gt-run", type=Path, default=DEFAULT_GT_RUN)
    ap.add_argument("--pred-scene", default=DEFAULT_PRED_SCENE)
    ap.add_argument("--gt-scene", default=DEFAULT_GT_SCENE)
    ap.add_argument("--json-out", type=Path, default=None,
                    help="把机器可读的结果另存一份（给下游/报告用）")
    ap.add_argument("--allow-hint-drift", action="store_true",
                    help="⚠ 明知两档提示词不同还继续 —— 结论将不可归因到几何，只在排查时用")
    args = ap.parse_args(argv)

    # -- 第 0 段：输入与对齐 --------------------------------------------------
    print("=" * 108)
    print("第 0 段 · 前置校验（提示词对齐 + 题集契约）")
    print("=" * 108)
    for p in (args.questions, args.pred_run, args.gt_run):
        if not p.is_file():
            print("缺文件：%s" % p)
            return 2

    try:
        aligned, hint_repr = check_hint_alignment(args.pred_scene, args.gt_scene)
    except Exception as exc:                                   # noqa: BLE001
        print("⚠ 提示词对齐检查**没能执行**：%s: %s" % (type(exc).__name__, exc))
        print("  ⟹ 这不等于「已对齐」。请确认已在项目 venv 下、且场景存在。")
        if not args.allow_hint_drift:
            return 3
        aligned, hint_repr = False, "（未检查）"
    print("  两档 scene_hint 逐字节相同 ? %s" % ("是 ✓" if aligned else "**否 ✗**"))
    print("       %s" % hint_repr)
    if not aligned and not args.allow_hint_drift:
        print("  ⟹ 两档提示词不同，差异**无法**归因到几何。已中止（要硬跑加 --allow-hint-drift）。")
        return 3

    questions = json.loads(args.questions.read_text(encoding="utf-8"))["questions"]
    gpred, ggt = load_geom(args.pred_scene), load_geom(args.gt_scene)
    truth, struct = build_truth(ggt), build_truth(gpred)
    try:
        _assert_order_contract(truth, questions)
    except ValueError as exc:
        print(" ✗ %s" % exc)
        return 2
    print("  题集契约 %d 题逐题对齐 ✓（题目文本 + 分组）" % len(truth))

    pred = json.loads(args.pred_run.read_text(encoding="utf-8"))
    gt = json.loads(args.gt_run.read_text(encoding="utf-8"))
    rp, rg = pred["runs"], gt["runs"]
    if not (len(rp) == len(rg) == len(truth)):
        print("题数不一致：pred=%d gt=%d truth=%d" % (len(rp), len(rg), len(truth)))
        return 2

    # -- 第 1 段：自变量 -----------------------------------------------------
    print()
    print("=" * 108)
    print("第 1 段 · 几何真值本身在两档差多少（**这是自变量**）")
    print("=" * 108)
    print("%-22s %12s %12s %9s" % ("量", "pred档", "参照档", "倍数"))
    gauge = (
        ("桌子到相机(m)", lambda g: cam_dist(g["table_1"])),
        ("椅子到相机(m)", lambda g: cam_dist(g["chair_1"])),
        ("镜子到相机(m)", lambda g: cam_dist(g["mirror_1"])),
        ("沙发x跨度(m)", lambda g: g["sofa_1"]["extent_3d"][0]),
        ("沙发↔桌子(m)", lambda g: dist(g["sofa_1"], g["table_1"])),
        ("桌子↔椅子(m)", lambda g: dist(g["table_1"], g["chair_1"])),
        ("镜子↔沙发(m)", lambda g: dist(g["mirror_1"], g["sofa_1"])),
    )
    for nm, fn in gauge:
        a, b = fn(gpred), fn(ggt)
        print("%-22s %12.3f %12.3f %9.2f" % (nm, a, b, a / b))

    # -- 第 2 段：逐题 -------------------------------------------------------
    print()
    print("=" * 108)
    print("第 2 段 · 端到端答案：两档逐题对照（真值 = **参照档几何**，不是标定真值）")
    print("=" * 108)
    print("%-3s %-16s %-5s %-13s %-13s %-9s %s" % ("#", "题", "类", "pred档", "参照档", "相对误差", "判定"))
    tally = {c: [0, 0] for c in CATEGORIES}       # [有效题数, 对]
    rel_errs: dict[str, list[float]] = {c: [] for c in CATEGORIES}
    rows: list[dict[str, Any]] = []

    for i, (it, ra, rb) in enumerate(zip(truth, rp, rg), 1):
        ja, jb = judge(it, ra.get("answer")), judge(it, rb.get("answer"))

        def clip(x: Any) -> str:
            s = repr(x.get("answer"))
            return s if len(s) <= 12 else s[:11] + "…"

        re_ = ja.get("rel_err")
        mark = {True: "对", False: "错", None: "—"}[ja.get("ok")]
        if jb.get("ok") is False:
            mark += "  ⚠参照档也错"          # 那多半是尺子或真值构造的问题，不是模型的
        print("%-3d %-16s %-5s %-13s %-13s %-9s %s" % (
            i, it["short"], it["cat"], clip(ra), clip(rb),
            ("%.1f%%" % (re_ * 100)) if re_ is not None else "—", mark))
        if ja.get("ok") is not None:
            tally[it["cat"]][0] += 1
            tally[it["cat"]][1] += 1 if ja["ok"] else 0
        if re_ is not None:
            rel_errs[it["cat"]].append(re_)
        rows.append({"n": i, "question": it["q"], "category": it["cat"], "kind": it["kind"],
                     "truth": it["truth"], "pred_answer": ra.get("answer"),
                     "gt_answer": rb.get("answer"), "pred_ok": ja.get("ok"),
                     "gt_ok": jb.get("ok"), "rel_err": re_})

    # -- 第 3 段：分组 -------------------------------------------------------
    print()
    print("=" * 108)
    print("第 3 段 · 分组汇总（真值 = 参照档）")
    print("=" * 108)
    print("%-6s %8s %8s   %s" % ("类别", "pred对", "题数", "说明"))
    for c in CATEGORIES:
        n, ok = tally[c]
        print("%-6s %8d %8d   %s" % (c, ok, n, CATEGORY_NOTES[c]))
    print()
    for c in CATEGORIES:
        es = sorted(rel_errs[c])
        if es:
            print("  相对误差 %-4s 中位 %6.1f%%   最大 %6.1f%%   (n=%d)" % (
                c, es[len(es) // 2] * 100, es[-1] * 100, len(es)))

    # -- 第 4 段：换尺子 -----------------------------------------------------
    print()
    print("=" * 108)
    print("第 4 段 · 容差敏感性（**同一批答案，只换尺子** —— 所以结论会变）")
    print("=" * 108)
    print("  容差   " + "".join("%10s" % c for c in CATEGORIES) + "     说明")
    for tol in TOLERANCES:
        cells = []
        for c in CATEGORIES:
            es = rel_errs[c]
            cells.append("%9.0f%%" % (100 * sum(1 for e in es if e <= tol) / len(es)) if es else "%10s" % "—")
        note = "← 本项目惯用" if abs(tol - 0.05) < 1e-9 else ""
        print("  %-6s %s     %s" % ("%.1f%%" % (tol * 100), "".join(cells), note))
    print("  列含义（**别把三列读成同一件事**）：")
    print("    abs —— 8 题全是数值 ⟹ 容差直接决定结论（这就是本段存在的理由）")
    print("    rel —— 8 题全是「哪个物体 / 哪一边」⟹ **无相对误差**，容差对它无从作用")
    print("    cnt —— 4 题里 2 题是数值（计数，恒为精确值）、2 题是二元的；")
    print("           它恒为 100% 且**对容差完全不敏感**，这正是对照组该有的样子：")
    print("           计数与米制尺度无关 ⟹ 尺度错了它也不动。")
    print("  ⚠ 「—」是**没有可判的量**，不是 0%。写成 0% 会把「不适用」伪装成「全错」。")

    # -- 第 5 段：流程证据 ---------------------------------------------------
    print()
    print("=" * 108)
    print("第 5 段 · 状态与用量（不涉及对错，只看流程有没有异常）")
    print("=" * 108)
    for nm, runs in (("pred档", rp), ("参照档", rg)):
        st = Counter(r.get("status") for r in runs)
        tc = [r.get("tool_calls") for r in runs]
        # 有时是计数、有时是清单 —— 两种历史写法都吃
        tr = sum(x if isinstance(x, int) else len(x or []) for x in tc)
        att = sum((r.get("attempts") or 0) for r in runs)
        print("  %-6s status=%-28s 工具调用=%3d  重试合计=%d" % (nm, dict(st), tr, att))
    usage = {}
    for nm, doc in (("pred", pred), ("gt", gt)):
        u = doc.get("usage") or {}
        usage[nm] = {k: u.get(k) for k in
                     ("calls", "prompt_tokens", "completion_tokens", "cached_tokens", "cost_cny")
                     if k in u}
        print("  %-6s usage=%s" % (nm, usage[nm]))

    # -- 机器可读 -----------------------------------------------------------
    summary = {
        "_caveat": "真值 = 参照档（内参外部给定）几何，**不是标定真值**；本结果是相对偏离，不是准确率。",
        "pred_scene": args.pred_scene, "gt_scene": args.gt_scene,
        "questions_file": str(args.questions),
        "hint_aligned": aligned,
        "gauge": {nm: {"pred": fn(gpred), "gt": fn(ggt)} for nm, fn in gauge},
        "tally": {c: {"n": tally[c][0], "ok": tally[c][1]} for c in CATEGORIES},
        "rel_err": {c: {"median": (sorted(rel_errs[c])[len(rel_errs[c]) // 2]
                                  if rel_errs[c] else None),
                        "max": (max(rel_errs[c]) if rel_errs[c] else None),
                        "n": len(rel_errs[c])} for c in CATEGORIES},
        "tolerances": {("%.3f" % t): {c: (sum(1 for e in rel_errs[c] if e <= t) / len(rel_errs[c])
                                         if rel_errs[c] else None) for c in CATEGORIES}
                       for t in TOLERANCES},
        "rows": rows,
        "usage": usage,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        print()
        print("机器可读结果 → %s" % args.json_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
