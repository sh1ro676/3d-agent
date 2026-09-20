#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/probe_combination.py —— 组合题「可达性 + 证据契约」体检。**零成本、零 GPU、零 LLM。**

为什么要有这个脚本
==================
用户提的第一个想法是「增加回答问题的范围，目前来看功能还是比较单薄」。
但「功能单薄」至少有**四个互不相同**的原因，而且四者的修法完全不同：

    ① **提示词/模型行为**：工具够了，模型没想到用 `max(..., key=...)` 去组合。
       —— 修法：提示词 / 示例 / 换模型。**不要加工具。**
    ② **工具返回值太窄**：组合需要的那个字段**根本没有**出现在任何 `res.value` 里
       （例如二维 bbox：场景图里有 `bbox_2d`，但没有任何工具把它交出来）。
       —— 修法：**扩返回值**，不是加工具。加工具会让动作空间变长而信息量不变。
    ③ **真的缺算法**：需要点云上的 OBB / 主方向 / 体积 / 平面拟合 / 聚类。
       —— 修法：加工具（且这些工具必须有确定性，否则 `stats.py` 不可复现）。
    ④ **答案的派生量不被承认**：程序算出了正确答案，但 `verifier.numeric_backed`
       要求「答案必须落在 trace 出现过的数上」—— 而组合题的答案
       （均值、计数、最大值对）常常是**派生量**，天然不在任何返回值里。
       —— 修法：改证据契约（否则「加工具」和「改提示词」都白搭）。

把这四件事混在一起讨论，结论一定是「再加点工具吧」——而那是四个里最贵、
最可能做错的那个。所以这里用**零 API 成本**的方式把它们分开：

    · 每个题都**用真实工具**算一遍真值（GT）。GT 由工具产出，
      就意味着这条 trace 与「一个完美程序」的 trace 逐字段同构。
    · 把这份完美答案喂给真 `verifier.verify()` —— 于是「④」是可以被直接观测的，
      不需要花一分钱，也不需要模型配合。
    · 再做一次**字段清点**：把每个工具在真实场景上返回过的所有字段名收集起来，
      然后用「想要但可能没有」的字段清单去查。查不到的 → 属于「②」。
    · 最后做一次**自洽性对照**：把 trace 里所有**非量值键**（普查 `label_counts` /
      身份版本 `method`、`object_id` / 请求参数 `tol`、`k`）抹掉再验一次。
      如果结论翻转，说明刚才那个 `supported` 是**巧合命中**（答案撞上了某个与题目
      无关的常量），而不是真的被证据支持 —— 这是「④」里最隐蔽的一种，
      而且它**已经抓到了两个**（见 `_strip_non_quantity` 的 docstring）。

⚠ 本脚本**不判对错之外的任何事**，也**不调用模型**。它是「组合题诊断」的前半段：
决定「要不要加工具」。后半段（模型到底会不会组合）才需要真跑，见 `--plan` 的输出说明。

产物
====
`reports/combination_probe.md`（人读）+ `.json`（机读）。两个都写，因为本机
PowerShell 的 stdout 不回传 —— 脚本不落盘就等于没输出。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools as _tools  # noqa: E402

from agents.executor import QA_TOOLSET, Submission  # noqa: E402
from agents.verifier import verify  # noqa: E402
from scene_graph.store import load_scene, scene_dir  # noqa: E402
from tools.registry import TOOL_REGISTRY, ToolContext  # noqa: E402

# 与执行器同一条装配纪律：工具是**显式装载**的（见 tools/__init__.py 的说明）。
# 不装载的话 TOOL_REGISTRY 是空的，而报错会是一句看不懂的 KeyError。
_tools.load_tools()

DEFAULT_REPORT = ROOT / "reports" / "combination_probe.md"
DEFAULT_SCENE = "points_probe"

#: 改**口径与提示词之前**那一跑的数字（来源：`combination_run_20260919_212328.json`
#: 用修正后的归因口径复算出来的 `..._212608_reanalyzed.md`）。
#: 它是历史事实、不会再变，所以写在这里当基线 —— 而不是去猜磁盘上哪份 JSON 是「旧的」。
BASELINE_BEFORE = {
    "calls": 106,          # 10 题累计工具调用
    "min_calls": 26,       # 「最少几次就够」之和
    "ratio": 5.0,          # 冗余倍数中位数
    "class": '{"C": 3, "OK": 5, "OK*": 2}',
    "c_rejected": 3,       # 答对却被自己的证据契约拒收的题数
}


def _latest_run_summary(md_path: Path) -> dict[str, Any] | None:
    """从最新一次真跑的 `_reanalyzed.json` 里取几个数，让结论段落**自己更新**。

    为什么读文件而不是把数字抄进报告：手抄的数字在下一次真跑之后会变成一个
    **看起来正确**的旧数字 —— 那比缺失更难发现。读不到就返回 None（报告照出，
    只是少那两行）—— 报告不该因为一个统计文件缺失而跑不出来。
    """
    js = md_path.with_suffix(".json")
    try:
        data = json.loads(js.read_text(encoding="utf-8"))
        s = data["summary"]
        cls = s.get("class") or {}
        return {
            "file": js.name,
            "reverified": bool(data.get("reverified")),
            "calls": s.get("tool_calls_total"),
            "min_calls": s.get("tool_calls_min_total"),
            "ratio": s.get("call_ratio_median"),
            "class": json.dumps(cls, ensure_ascii=False),
            "c": cls.get("C", 0),
            "gt_ok": s.get("answer_matches_gt"),
            "n": s.get("n"),
            "cost": s.get("cost_cny"),
        }
    except Exception:  # noqa: BLE001  统计文件缺失/半写入都不该让报告失败
        return None

# ============================================================================
# 字段清点用的「想要清单」
# ============================================================================
#: 组合题常见需要、但可能没有任何工具交出来的字段。
#: 每项是 `(字段名, 为什么组合题需要它)`。
#: 判据只有一条：**它有没有出现在某个工具的 `res.value` 或 `res.evidence` 里**。
#: 只出现在 `SceneGraph`（`scene.json`）里不算 —— 模型看不到场景对象。
WANTED_FIELDS: tuple[tuple[str, str], ...] = (
    ("bbox_2d", "二维像素框 —— 「图像左半边」「画面上方」这类题目唯一的判据"),
    ("centroid_px", "物体的像素中心 —— 用二维方式表达「左上 / 右下」"),
    ("volume_m3", "体积 —— 只有点云算得出（AABB 乘积对所有非长方体都是错的）"),
    ("orientation", "朝向（主方向 / 偏航角）—— 点云 PCA，几何工具给不了"),
    ("obb", "有向包围盒 —— 细长物体的真实尺寸"),
    ("plane", "支撑平面（地面 / 桌面）—— 「在桌上」「在地上」"),
    ("n_points_per_object", "物体点数 —— 已落盘的 points.npy 支持，但没有任何工具读它"),
    ("delta_z", "关系判定的**实际差值** —— 没有它就无法判断答案是不是卡在容差死区上"
                "（仍留在 `evidence`：那是审计面，见第四节的口径说明）"),
    ("tol", "判定的容差 —— 已改成**文档化的参数**（`query_relation(tol=0)` 就是严格比较），"
            "但 `value` 里仍不出现它，理由见第五节末"),
)


# ============================================================================
# 真值（GT）—— 全部由**真实工具**算出来，不是脚本自己算的
# ============================================================================
#
# 这一点是整个脚本的可信度的地基：如果脚本自己用 scene.json 里的 centroid_3d
# 去算 GT，那么「程序能算出来吗」这个问题就答错了 —— 脚本手里的东西比程序多。
# 所以下面每一题都**只通过 TOOL_REGISTRY** 取数，与一个真实程序能拿到的完全一致。


class ToolError_(RuntimeError):
    """工具调用失败。脚本里出现它 = 这一题的「路线」写错了，不是模型的锅。"""


def call(ctx: ToolContext, name: str, **kwargs: Any) -> Any:
    """调一个真实工具并返回它的 `value`（失败直接抛 —— 那是脚本的 bug）。"""
    res = TOOL_REGISTRY[name](ctx, **kwargs)
    if not res.ok:
        raise ToolError_("%s(%s) 失败：[%s] %s"
                         % (name, kwargs, res.error.code.value, res.error.message))
    return res.value


def _dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


@dataclass
class Spec:
    """一个组合题。

    `needs` 是本脚本最重要的字段 —— 它就是「功能单薄」四因里的前三因：
        `current_values`  现有返回值够用（模型会不会用是另一回事 → 因①）
        `extra_field`     缺字段，返回值太窄（→ 因②）
        `new_algorithm`   缺点云算法（→ 因③）
    `blocked_by` 给出判据（缺哪个字段 / 缺什么算法），不能只写一个标签。
    """

    qid: str
    category: str
    question: str
    answer_type: str
    route: tuple[str, ...]
    needs: str
    blocked_by: str = ""
    solve: Callable[[ToolContext], tuple[Any, list[str], list[str]]] | None = None
    #: 「用现有工具能不能凑出这个量」的备注（通常是「能算但口径不对」）。
    caveat: str = ""
    #: 答案的**判定口径**。**只要某题的 GT 对某个有默认值的参数敏感，就必须在这里写明**
    #: —— 否则同一个问题会同时存在两个都合法的答案（见 C8）。
    #: 空串 = 该题的答案不依赖任何未声明的约定（绝大多数题）。
    convention: str = ""


# ---------------------------------------------------------------------------
# ① 现有返回值就够的题（因①：模型行为）
# ---------------------------------------------------------------------------


def _argmax_picture_area(ctx: ToolContext):
    pics = call(ctx, "find_object", label="picture")
    best = max(pics, key=lambda o: o["extent_m"]["w"] * o["extent_m"]["h"])
    return (best["object_id"],
            ["max(find_object(label='picture').value, key=extent_m.w*extent_m.h)"],
            [best["object_id"]])


def _nearest_picture_to_mirror(ctx: ToolContext):
    # 这一题有**现成工具**：find_nearest。它的存在说明动作空间里已经有
    # 「一类中的 argmin by distance」这个常见组合。
    near = call(ctx, "find_nearest", anchor="mirror_1", label="picture", k=1)
    return (near[0]["object_id"],
            ["find_nearest(anchor='mirror_1', label='picture', k=1).value[0].object_id"],
            [near[0]["object_id"]])


def _argmax_all_height(ctx: ToolContext):
    objs = call(ctx, "list_objects")
    best = max(objs, key=lambda o: o["extent_m"]["h"])
    return (best["object_id"],
            ["max(list_objects().value, key=extent_m.h)"],
            [best["object_id"]])


def _count_taller_than(ctx: ToolContext, thresh: float = 0.8):
    objs = call(ctx, "list_objects")
    kept = [o for o in objs if o["extent_m"]["h"] > thresh]
    return (len(kept),
            ["len([o for o in list_objects().value if o['extent_m']['h'] > %.1f])" % thresh],
            [o["object_id"] for o in kept])


def _mean_picture_width(ctx: ToolContext):
    pics = call(ctx, "find_object", label="picture")
    m = sum(o["extent_m"]["w"] for o in pics) / len(pics)
    return (round(m, 6),
            ["sum(o['extent_m']['w'] for o in find_object(label='picture').value) / %d" % len(pics)],
            [])


def _sort_pictures_left_to_right(ctx: ToolContext):
    pics = call(ctx, "find_object", label="picture")
    order = [o["object_id"] for o in sorted(pics, key=lambda o: o["centroid_m"][0])]
    return (",".join(order),
            ["sorted(find_object(label='picture').value, key=centroid_m[0])"],
            order)


def _nearest_any_to_sofa(ctx: ToolContext):
    # 注意这条路：`find_nearest` 要求**指定一个类别**，而这一题问的是「所有类别里的最近者」。
    # 所以程序必须自己遍历 —— 这正是「工具覆盖的是常见组合，不是任意组合」的具体体现。
    objs = call(ctx, "list_objects")
    sofa = next(o for o in objs if o["label"] == "sofa")
    best_id, best_d = None, float("inf")
    for o in objs:
        if o["object_id"] == sofa["object_id"]:
            continue
        d = call(ctx, "calculate_distance", a=sofa["object_id"], b=o["object_id"])
        if d < best_d:
            best_id, best_d = o["object_id"], d
    return (best_id,
            ["min over calculate_distance(a='%s', b=<every other>) — %d 次调用"
             % (sofa["object_id"], len(objs) - 1)],
            [best_id])


#: GT 用的容差。**必须显式传 0，不能吃工具的默认值。**
#:
#: 为什么：`query_relation` 的默认 `tol=0.05` 是一条**鲁棒性余量**（让近乎共面的两个物体
#: 不要被硬判成前后关系），它是**工具的实现细节**，不是「A 在 B 前面吗」这个问题的答案。
#: 问题问的是世界的样子，不是工具的输出。把 GT 定义成「工具默认值下的输出」，
#: 等于让工具的行为冒充真值 —— 于是「差 37.8 mm」这种**在物理上确实在前面**的物体
#: 会被记成「不在前面」，而任何按质心比较的程序都会被判成算错了。
#:
#: ⚠ 这不是事后迁就模型：C8 的一次调用解（`_one_call_count_front_of_table`）**一直**用
#: 严格比较，它的 docstring 早就写着「两个答案都对，取决于那个对程序不可见的容差」。
#: 也就是说，问题的真值**从一开始就没有被声明过** —— 这才是要修的东西。
GT_TOL = 0.0


def _count_in_front_of(ctx: ToolContext, anchor_label: str, tol: float = GT_TOL):
    objs = call(ctx, "list_objects")
    anchor = next(o for o in objs if o["label"] == anchor_label)
    kept = []
    for o in objs:
        if o["object_id"] == anchor["object_id"]:
            continue
        if call(ctx, "query_relation", relation="front_of",
                a=o["object_id"], b=anchor["object_id"], tol=tol):
            kept.append(o["object_id"])
    return (len(kept),
            ["count(query_relation(relation='front_of', a=<each>, b='%s', tol=%s) == True)"
             % (anchor["object_id"], tol)],
            kept)


def _count_in_front_of_table(ctx: ToolContext):
    return _count_in_front_of(ctx, "table")


def _count_in_front_of_sofa(ctx: ToolContext):
    return _count_in_front_of(ctx, "sofa")


#: 「**另一个口径下的答案**」—— 只对口径敏感的题登记。
#:
#: 键是 qid，值是「吃工具默认值」那一版的计算函数，也就是**修口径之前**那个 GT。
#: 有了它，「这题的 GT 会不会因为一个未声明的默认值而改」就变成一个**可计算**的问题，
#: 而不是靠人记得。C8 就是靠它被抓出来的 —— 见 `convention_audit()`。
#: 没登记的题 = 答案不依赖任何有默认值的参数。
_ALT_CONVENTION: dict[str, Callable[[ToolContext], Any]] = {}


def _alt_convention(qid: str):
    def deco(fn: Callable[[ToolContext], Any]) -> Callable[[ToolContext], Any]:
        _ALT_CONVENTION[qid] = fn
        return fn
    return deco


@_alt_convention("C8")
def _alt_count_front_of_table(ctx: ToolContext) -> Any:
    """吃 `query_relation` 的默认 `tol=0.05`。"""
    return _count_in_front_of(ctx, "table", tol=0.05)[0]


@_alt_convention("C8b")
def _alt_count_front_of_sofa(ctx: ToolContext) -> Any:
    return _count_in_front_of(ctx, "sofa", tol=0.05)[0]


def _farthest_pair(ctx: ToolContext):
    # 全局最近/最远「一对」：find_farthest 只能按类别找，所以这一题
    # 要么 36 次 calculate_distance，要么直接用 list_objects 的质心在纯 Python 里算。
    # 两条路都通 —— 但**调用次数**差 36 倍，这是「组合代价」而不是「能力缺失」。
    objs = call(ctx, "list_objects")
    best = (None, None, -1.0)
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            d = call(ctx, "calculate_distance", a=a["object_id"], b=b["object_id"])
            if d > best[2]:
                best = (a["object_id"], b["object_id"], d)
    return (",".join(sorted([best[0], best[1]])),
            ["max over all %d pairs of calculate_distance" % (len(objs) * (len(objs) - 1) // 2)],
            [best[0], best[1]])


# ---------------------------------------------------------------------------
# ③ 真的缺算法：真值存在（就在 points.npy 里），但没有任何工具交得出来
# ---------------------------------------------------------------------------


def _sofa_aabb_volume(ctx: ToolContext):
    """AABB 体积：**算得出来，但口径不对**。

    这是「因③」里最容易被忽略的一类：工具给了数，程序也算了，答案看起来很像样，
    但 `extent_3d` 是**轴对齐**包围盒的三边 —— 对一个斜放的沙发，
    三边乘积远大于真实体积。所以这一题的正确标注不是「能答」，
    而是「能算出一个**别的量**」。
    """
    objs = call(ctx, "list_objects")
    sofa = next(o for o in objs if o["label"] == "sofa")
    e = sofa["extent_m"]
    return (round(e["w"] * e["h"] * e["l"], 6),
            ["AABB: extent_m.w * extent_m.h * extent_m.l （口径 ≠ 真实体积）"],
            [sofa["object_id"]])


SPECS: tuple[Spec, ...] = (
    Spec("C1", "类内 argmax（按尺寸）", "四幅画里哪一幅面积最大？给出它的 object_id。",
         "str", ("find_object",), "current_values",
         solve=_argmax_picture_area),
    Spec("C2", "类内 argmin（按距离）", "哪幅画离镜子最近？给出它的 object_id。",
         "str", ("find_nearest",), "current_values",
         solve=_nearest_picture_to_mirror),
    Spec("C3", "全场景 argmax", "场景里最高的物体是哪个？给出它的 object_id。",
         "str", ("list_objects",), "current_values",
         solve=_argmax_all_height),
    Spec("C4", "筛选 + 计数", "有几个物体的高度超过 0.8 米？",
         "int", ("list_objects",), "current_values",
         solve=_count_taller_than),
    Spec("C5", "聚合（均值）", "四幅画的平均宽度是多少米？",
         "float", ("find_object",), "current_values",
         solve=_mean_picture_width),
    Spec("C6", "排序", "把四幅画按从左到右排序，给出 object_id 列表（逗号分隔）。",
         "str", ("find_object",), "current_values",
         solve=_sort_pictures_left_to_right),
    Spec("C7", "两跳组合（跨类别最近）", "离沙发最近的物体是哪个？给出它的 object_id。",
         "str", ("list_objects", "calculate_distance"), "current_values",
         solve=_nearest_any_to_sofa,
         caveat="find_nearest 必须指定类别，跨类别的最近者只能自己遍历"),
    Spec("C8", "关系计数（口径敏感）", "有多少个物体在桌子的前面？",
         "int", ("list_objects", "query_relation"), "current_values",
         solve=_count_in_front_of_table,
         convention="严格比较（`tol=0`）—— 见下方 caveat",
         caveat="GT **= 1**，用严格比较：chair_1 与 table_1 的 z 只差 **37.8 mm**，"
                "物理上确实在桌子前面。⚠ 但工具默认 `tol=0.05` 会把它判成不在前面 ⟹ "
                "**同一个问题存在两个都合法的答案（1 / 0）**。"
                "修口径之前这里记的是 0（「工具默认输出的」），于是任何按质心比较的程序"
                "都被错判成算错 —— 那是探针把「工具的行为」当成了「问题的真值」，"
                "与模型无关。第三节的受控对照给出 tol↔答案 的完整映射"),
    Spec("C8b", "关系计数（远离死区）", "有多少个物体在沙发的前面？",
         "int", ("list_objects", "query_relation"), "current_values",
         solve=_count_in_front_of_sofa,
         convention="严格比较（`tol=0`）—— 本题无物体落在容差死区内，两个口径答案相同"),
    Spec("C12", "全局极值对（O(n²)）", "场景里质心相距最远的是哪两个物体？给出两个 object_id（逗号分隔）。",
         "str", ("list_objects", "calculate_distance"), "current_values",
         solve=_farthest_pair,
         caveat="能力上够用，但 36 次调用；用 list_objects 的质心在纯 Python 里算只要 1 次调用"),

    Spec("C10", "体积", "沙发的体积大约是多少立方米？",
         "float", ("list_objects",), "new_algorithm",
         blocked_by="缺点云体积：extent_3d 是 AABB 三边，对斜放/不规则物体的乘积不是体积",
         solve=_sofa_aabb_volume),
    Spec("C11", "朝向", "沙发的朝向大致是哪个方向（+x / -x / +z / -z）？",
         "str", (), "new_algorithm",
         blocked_by="缺点云主方向（PCA）：质心与 AABB 都丢失了朝向信息"),
    Spec("C9", "二维区域", "图像左半部分（按像素 x < 宽/2）里有几个物体？",
         "int", (), "extra_field",
         blocked_by="缺 bbox_2d / 像素中心：scene.json 里有 bbox_2d，但没有任何工具的 value/evidence 交出来"),
)


# ============================================================================
# 体检
# ============================================================================


@dataclass
class Result:
    spec: Spec
    answer: Any = None
    evidence: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    #: 「完美程序」用的是几次工具调用 —— 用来量**冗余调用**（模型实际调了几次 / 最少几次）。
    #: 这个比值比答案对错更能说明「模型有没有看清返回值里已经有的东西」。
    n_tool_calls: int = 0
    error: str = ""
    verdict_level: str = ""
    verdict_matched_from: str = ""
    verdict_n_numbers: int = 0
    verdict_failed: list[str] = field(default_factory=list)
    #: 把 trace 里所有**非量值键**（普查 / 版本标记 / 请求参数）抹掉之后的结论 —— 用来抓
    #: 「巧合命中」。名字用 `pruned` 而不是列举某一类键：列举式的名字会随下一个同类缺陷过期。
    verdict_level_pruned: str = ""
    coincidence: bool = False

    @property
    def reachable(self) -> bool:
        return self.spec.needs == "current_values" and not self.error


def _strip_non_quantity(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """复制一份 trace，把所有**非量值键**（普查 + 身份/版本 + 请求参数）从证据里抹掉。

    这是一条**自洽性对照**，不是复查：把「与答案无关的字段」全部删掉再验一次，
    如果结论**翻转**，说明刚才那个 `supported` 是**凑巧命中**而不是真的有出处。

    抓过两类，都是同一个形状 —— 一个与题目无关的数字漏进池子：

      · `label_counts`：`{"sofa": 1, "picture": 4, ...}`，一组 1~4 的小整数。
        于是「有几个物体的高度超过 0.8 米 → 4」恒判 supported，「3 个」恒判 unsupported。
      · **`method`：`"geometry_v1"` 里的 `1`。** 它更隐蔽 —— 因为 `_ID_RE` 当时只认
        `名字_数字` 这一种形状（`chair_1`），认不出 `_v1` 这种**版本号后缀**，
        于是 `query_relation` 每调一次就往证据池塞一个 `('method', 1.0)`。
        C8「桌子前面有几个物体」的正确答案恰好是 `1`，出处却是**算法版本号**。

    判据的措辞很重要：这种检查的判断力来自数字的**大小**，不来自正确性 ——
    一个在回答「4 个」时恒为 supported、回答「3 个」时恒为 unsupported 的契约，
    量出来的不是「有没有证据」，而是「答案是不是恰好撞上了某个常量」。

    ⚠ 删的是 `verifier.PRUNABLE_KEYS`，**不是** `NON_QUANTITY_KEYS` —— 后者还含 `object_id`
    这类**身份键**，而字符串答案的 `text_backed` 正要靠它们找出处。第一版图省事
    删了全部非量值键，结果 C1/C2/C3/C7（答案形如 `picture_1`）从 supported 掉成 weak，
    被报成「巧合命中」——**四例全是假阳性**。删掉答案本身的出处不叫发现巧合。
    """
    from agents.verifier import PRUNABLE_KEYS

    drop = set(PRUNABLE_KEYS)
    out = json.loads(json.dumps(trace, ensure_ascii=False, default=str))

    def _prune(obj: Any) -> None:
        if isinstance(obj, dict):
            for k in [k for k in obj if str(k) in drop]:
                obj.pop(k, None)
            for v in obj.values():
                _prune(v)
        elif isinstance(obj, list):
            for v in obj:
                _prune(v)

    for row in out:
        _prune(row)
    return out


def run_spec(spec: Spec, scene: Any) -> Result:
    r = Result(spec=spec)
    if spec.solve is None:
        r.error = "未实现（本题按 needs=%s 判定为不可达）" % spec.needs
        return r

    ctx = ToolContext(scene=scene, record_trace=True)
    try:
        answer, evidence, targets = spec.solve(ctx)
    except Exception as exc:                     # noqa: BLE001  脚本自己的错，必须喊出来
        r.error = "%s: %s" % (type(exc).__name__, exc)
        return r

    r.answer = answer
    r.evidence = list(evidence)
    r.targets = list(targets)
    r.tools_used = sorted({row["tool"] for row in ctx.trace})
    r.n_tool_calls = len(ctx.trace)

    sub = Submission(answer=answer, answer_type=spec.answer_type,
                     target_ids=tuple(targets), evidence=tuple(evidence))
    v = verify(sub, trace=ctx.trace, scene=scene, answer_type=spec.answer_type)
    r.verdict_level = v.level
    r.verdict_matched_from = v.matched_from
    r.verdict_n_numbers = len(v.numbers)
    r.verdict_failed = list(v.failed())

    v2 = verify(sub, trace=_strip_non_quantity(ctx.trace), scene=scene,
                answer_type=spec.answer_type)
    r.verdict_level_pruned = v2.level
    r.coincidence = (v.level == "supported" and v2.level != "supported")
    return r


def dead_zone_probe(scene: Any, anchor_label: str = "table",
                    relation: str = "front_of") -> dict[str, Any]:
    """容差死区的**受控对照**：同一个问题、同一个场景，只改 `tol`，看答案怎么变。

    为什么值得单独做一次：`front_of(a, b)` 的判据是 `a.z < b.z - tol`，`tol` 默认 50 mm。
    于是「A 在 B 前面吗」在 50 mm 以内**一律返回 False** —— 不是「不确定」，是 `False`。
    而 `query_relation` 的 `res.value` 是一个**裸 bool**：`delta_z` 与 `tol` 只写在
    `evidence` 里，模型读不到。⟹ **程序拿到一个 False，从中无法知道是真没有、还是卡在死区。**

    这一条会直接改掉「计数类」组合题的答案：把 tol 设成 0，同一个问题可能从 0 变成 1。
    所以不能只报一个数 —— 必须报「这个数对哪个未声明的参数敏感」。
    """
    ctx = ToolContext(scene=scene, record_trace=True)
    objs = call(ctx, "list_objects")
    anchor = next(o for o in objs if o["label"] == anchor_label)

    rows: list[dict[str, Any]] = []
    for o in objs:
        if o["object_id"] == anchor["object_id"]:
            continue
        res_default = TOOL_REGISTRY["query_relation"](
            ctx, relation=relation, a=o["object_id"], b=anchor["object_id"])
        delta = (ctx.trace[-1]["result"]["evidence"] or {}).get("delta_z")
        # ⚠ 命名：`tol=0` 是**严格**比较，不是宽松比较 —— 之前那个名字叫 `res_loose`，
        # 正好叫反了，读表的时候极易把两列看颠倒。
        res_strict = TOOL_REGISTRY["query_relation"](
            ctx, relation=relation, a=o["object_id"], b=anchor["object_id"], tol=0.0)
        rows.append({
            "object_id": o["object_id"],
            "delta_z": None if delta is None else round(float(delta), 6),
            "default_tol": bool(res_default.value),
            "tol_0": bool(res_strict.value),
        })

    flipped = [r for r in rows if r["tol_0"] and not r["default_tol"]]
    return {
        "relation": relation,
        "anchor": anchor["object_id"],
        "count_default_tol": sum(1 for r in rows if r["default_tol"]),
        "count_tol_0": sum(1 for r in rows if r["tol_0"]),
        "rows": rows,
        "flipped_by_dead_zone": flipped,
    }


def convention_audit(scene: Any, results: list[Result]) -> list[dict[str, Any]]:
    """**GT 口径审计** —— 每一题的 GT 是否对一个「有默认值的参数」敏感。

    为什么这是一等公民而不是脚注：如果一个问题同时存在两个都合法的答案，
    那么这个探针**测的不是模型**，而是模型对某个没写出来的约定的猜测。
    这种缺陷的危害在于它**长得像模型出错**：答案是 1、GT 是 0 ⟹ 归到「算错了」，
    于是「加工具」「改提示词」这些修法会被一个根本不存在的失败牵着走。

    所以判据必须是**可计算的**，不能靠人记得：
    对登记在 `_ALT_CONVENTION` 里的题，用「工具默认值」那一版再算一次；
    两次答案不同 ⟹ 该题是**口径敏感**的，必须在 `spec.convention` 里写明口径，
    否则报 `ambiguous`。没登记的题 = 不敏感（默认值对它没有影响）。

    ⚠ 审计只查**已登记**的题。登记本身是人工的 —— 这是这个函数的已知边界，
    写在这里而不是留给人猜：**下一个同类缺陷仍可能不被它抓到**，
    除非有人先把「另一个口径」写进 `_ALT_CONVENTION`。
    但只要有登记，它就再也不会退化成静默的错标签。
    """
    rows: list[dict[str, Any]] = []
    for r in results:
        qid = r.spec.qid
        alt_fn = _ALT_CONVENTION.get(qid)
        if alt_fn is None:
            rows.append({
                "qid": qid, "category": r.spec.category,
                "gt": r.answer, "alt": None, "sensitive": False,
                "convention": r.spec.convention, "ambiguous": False,
            })
            continue
        ctx = ToolContext(scene=scene, record_trace=True)
        try:
            alt: Any = alt_fn(ctx)
        except Exception:                        # noqa: BLE001  审计不该让主流程挂掉
            alt = None
        sensitive = alt is None or str(alt) != str(r.answer)
        rows.append({
            "qid": qid, "category": r.spec.category,
            "gt": r.answer, "alt": alt, "sensitive": sensitive,
            "convention": r.spec.convention,
            # 敏感但没声明口径 = 探针本身无效，必须显式报出来。
            "ambiguous": bool(sensitive and not r.spec.convention),
        })
    return rows


#: 「**一次工具调用**就能答对」的解法。键是题号。
#:
#: 这一组存在的理由是本轮最反直觉的一个发现：
#: `list_objects` 的返回值里**已经带了全部 9 个物体的 `centroid_m` 与 `extent_m`**
#: ——也就是说，上面那 10 道题里有 9 道的全部所需几何量，**第一次调用就全拿到了**。
#: 那么「再调 9 次 get_3d_extent」不是能力不足，是**没看清返回值里已经有的字段**。
#:
#: 判据是可算的：只要 `{qid: 用一次调用算出的答案}` 与 GT 相等，
#: 就证明这道题的最少调用次数是 1。跑一遍就知道有多少题属于这种情况。
_ONE_CALL_SOLVES: dict[str, Callable[[ToolContext], Any]] = {}


def _one_call(qid: str):
    def deco(fn: Callable[[ToolContext], Any]) -> Callable[[ToolContext], Any]:
        _ONE_CALL_SOLVES[qid] = fn
        return fn
    return deco


@_one_call("C1")
def _one_call_argmax_picture_area(ctx: ToolContext) -> Any:
    objs = call(ctx, "list_objects")
    pics = [o for o in objs if o["label"] == "picture"]
    return max(pics, key=lambda o: o["extent_m"]["w"] * o["extent_m"]["h"])["object_id"]


@_one_call("C2")
def _one_call_nearest_picture_to_mirror(ctx: ToolContext) -> Any:
    objs = call(ctx, "list_objects")
    mirror = next(o for o in objs if o["label"] == "mirror")
    pics = [o for o in objs if o["label"] == "picture"]
    return min(pics, key=lambda o: _dist(mirror["centroid_m"], o["centroid_m"]))["object_id"]


@_one_call("C3")
def _one_call_argmax_all_height(ctx: ToolContext) -> Any:
    return max(call(ctx, "list_objects"), key=lambda o: o["extent_m"]["h"])["object_id"]


@_one_call("C4")
def _one_call_count_taller_than(ctx: ToolContext) -> Any:
    return len([o for o in call(ctx, "list_objects") if o["extent_m"]["h"] > 0.8])


@_one_call("C5")
def _one_call_mean_picture_width(ctx: ToolContext) -> Any:
    pics = [o for o in call(ctx, "list_objects") if o["label"] == "picture"]
    return round(sum(o["extent_m"]["w"] for o in pics) / len(pics), 6)


@_one_call("C6")
def _one_call_sort_pictures(ctx: ToolContext) -> Any:
    pics = [o for o in call(ctx, "list_objects") if o["label"] == "picture"]
    return ",".join(o["object_id"] for o in sorted(pics, key=lambda o: o["centroid_m"][0]))


@_one_call("C7")
def _one_call_nearest_any_to_sofa(ctx: ToolContext) -> Any:
    objs = call(ctx, "list_objects")
    sofa = next(o for o in objs if o["label"] == "sofa")
    others = [o for o in objs if o["object_id"] != sofa["object_id"]]
    return min(others, key=lambda o: _dist(sofa["centroid_m"], o["centroid_m"]))["object_id"]


@_one_call("C8")
def _one_call_count_front_of_table(ctx: ToolContext) -> Any:
    """用 `centroid_m[2]` 直接比大小（等价于 `tol=0`），答案 1。

    ⚠ 这一版**曾经与 GT 不同**（那时 GT 吃工具默认的 `tol=0.05`，是 0），
    报告里被写成「一次调用能算出**另一个**合法答案」。
    那个「另一个」的说法本身就是缺陷的自白：**一个问题的两个答案不能都叫合法。**
    GT 现在改成严格比较（见 `GT_TOL`），于是这一版**就是** GT ——
    本题因此从「一次调用算出别的答案」变成「一次调用就够」，
    冗余统计里也就不该再给它留情面。
    """
    objs = call(ctx, "list_objects")
    table = next(o for o in objs if o["label"] == "table")
    tz = table["centroid_m"][2]
    return len([o for o in objs
                if o["object_id"] != table["object_id"] and o["centroid_m"][2] < tz])


@_one_call("C12")
def _one_call_farthest_pair(ctx: ToolContext) -> Any:
    objs = call(ctx, "list_objects")
    best, best_d = None, -1.0
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            d = _dist(a["centroid_m"], b["centroid_m"])
            if d > best_d:
                best, best_d = (a["object_id"], b["object_id"]), d
    return ",".join(sorted(best))


def one_call_solutions(scene: Any) -> dict[str, Any]:
    """每题「只用一次工具调用」能算出的答案（用不了就记 None）。"""
    out: dict[str, Any] = {}
    for qid, fn in _ONE_CALL_SOLVES.items():
        ctx = ToolContext(scene=scene, record_trace=True)
        try:
            out[qid] = {"answer": fn(ctx), "n_calls": len(ctx.trace)}
        except Exception as exc:                 # noqa: BLE001
            out[qid] = {"answer": None, "n_calls": len(ctx.trace),
                        "error": "%s: %s" % (type(exc).__name__, exc)}
    return out


def field_census(scene: Any) -> dict[str, list[str]]:
    """字段清点：调每一个问答臂工具，把它 `value` / `evidence` 里出现过的**键名**收集起来。

    只收集**键名**、不收集值 —— 报告要回答的是「有没有这个字段」，
    把值也打出来会顺手泄露场景坐标，也会让报告长度失控。

    故意把 `value` 与 `evidence` 分开记：
    `get_object` 的 `bbox_3d_min/max` 只出现在 `evidence` 里而**不在 value 里**，
    而模型读不到 evidence（那是给人审计用的）。混在一起会把「有字段」误判成「能取到」。
    """
    collected: dict[str, set[str]] = {"value": set(), "evidence": set()}
    for name in QA_TOOLSET:
        # ⚠ `record_trace=True` 是必须的：`ToolContext.record()` 在 `record_trace=False`
        # 时**直接 return**，于是 trace 永远为空、清点结果永远为空表 ——
        # 而空表会被读成「所有字段都缺」。这正是「不报错、只改结论」的那类缺陷。
        ctx = ToolContext(scene=scene, record_trace=True)
        try:
            _probe_tool(ctx, name)
        except Exception:                        # noqa: BLE001  探针失败不该中断清点
            pass
        for row in ctx.trace:
            res = row.get("result") or {}
            for bucket in ("value", "evidence"):
                _collect_keys(res.get(bucket), bucket, collected)
    return {k: sorted(v) for k, v in collected.items()}


def _probe_tool(ctx: ToolContext, name: str) -> None:
    """给每个工具一组「尽量能成功」的实参，好让它真的返回一份有形状的 value。

    参数写错（`ToolArgumentError`）不会静默 —— 会冒泡到 `field_census` 的 try 里。
    那些写不出合法参数的探针（例如缺 label 时 `find_object` 报 NOT_FOUND）
    会在 `_translate` 里被翻成 ToolResult，于是 value 为空 —— 这是**预期**的：
    字段清点是「至少有一个工具交出来过」，不是「每个工具都交得出来」。
    """
    scene = ctx.scene
    assert scene is not None
    ids = [n.id for n in scene.nodes]
    labels = sorted({n.label for n in scene.nodes})
    first, second = (ids + [""])[0], (ids + [""])[1] if len(ids) > 1 else ""
    args: dict[str, dict[str, Any]] = {
        "list_objects": {},
        "get_object": {"object_id": first},
        "find_object": {"label": labels[0] if labels else ""},
        "single_object": {"label": labels[0] if labels else ""},
        "find_nearest": {"anchor": first, "label": labels[0] if labels else "", "k": 1},
        "find_farthest": {"anchor": first, "label": labels[0] if labels else "", "k": 1},
        "query_relation": {"relation": "left_of", "a": first, "b": second},
        "get_3d_position": {"object_id": first},
        "get_3d_extent": {"object_id": first},
        "calculate_distance": {"a": first, "b": second},
        "calculate_angle": {"a": first, "b": second, "c": first},
        "get_attributes": {"object_id": first},
    }
    TOOL_REGISTRY[name](ctx, **args.get(name, {}))


def _collect_keys(obj: Any, bucket: str, out: dict[str, set[str]]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            out[bucket].add(str(k))
            _collect_keys(v, bucket, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_keys(v, bucket, out)


# ============================================================================
# 报告
# ============================================================================


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "%.6g" % v
    if v is None:
        return "—"
    return str(v)


def build_report(scene: Any, results: list[Result], census: dict[str, list[str]],
                 dead_zone: dict[str, Any], one_call: dict[str, Any], scene_ref: str,
                 convention: list[dict[str, Any]] | None = None) -> str:
    if convention is None:      # 直接调 build_report 的调用方可以省掉这一步，避免签名变长
        convention = convention_audit(scene, results)
    ambiguous = [c for c in convention if c["ambiguous"]]
    sensitive = [c for c in convention if c["sensitive"]]

    L: list[str] = []
    L.append("# 组合题「可达性 + 证据契约」体检")
    L.append("")
    L.append("> 零 LLM 调用。真值全部由**真实工具**算出，所以下面这条 trace 与")
    L.append("> 「一个完美程序」的 trace 逐字段同构 —— 「程序能不能算出来」这个问题")
    L.append("> 在这里的答案不依赖任何模型行为。")
    L.append("")
    L.append("- 场景：`%s`" % scene_ref)
    L.append("- 物体数：%d ｜ 关系数：%d" % (len(scene.nodes), len(scene.edges)))
    L.append("- 动作空间：%d 个工具（%s）" % (len(QA_TOOLSET), ", ".join(QA_TOOLSET)))
    L.append("")

    reachable = [r for r in results if r.reachable]
    by_needs: dict[str, list[Result]] = {}
    for r in results:
        by_needs.setdefault(r.spec.needs, []).append(r)

    L.append("## 一、结论摘要")
    L.append("")
    L.append("| 原因 | 题数 | 题号 | 修法 |")
    L.append("|---|---|---|---|")
    L.append("| ① 现有返回值够用（差的是模型会不会组合） | %d | %s | 改提示词 / 看真跑 trace，**不加工具** |"
             % (len(by_needs.get("current_values", [])),
                ", ".join(r.spec.qid for r in by_needs.get("current_values", []))))
    L.append("| ② 缺字段（返回值太窄） | %d | %s | **扩现有工具的返回值**，不加工具 |"
             % (len(by_needs.get("extra_field", [])),
                ", ".join(r.spec.qid for r in by_needs.get("extra_field", []))))
    L.append("| ③ 缺算法（点云层） | %d | %s | 加工具，且必须**确定性**（`stats.py` 要可复现） |"
             % (len(by_needs.get("new_algorithm", [])),
                ", ".join(r.spec.qid for r in by_needs.get("new_algorithm", []))))
    L.append("")
    L.append("**因① 的具体数字**：%d/%d 题的答案在现有返回值下**可以精确算出来**。"
             % (len(reachable), len(results)))
    L.append("")
    # -- GT 口径（在信任任何数字之前先看这一段）--------------------------------
    L.append("**GT 口径审计**：%s" % ("⚠ **有问题**" if ambiguous else "✅ 通过"))
    L.append("")
    if ambiguous:
        for c in ambiguous:
            L.append("- ⚠ **%s**「%s」：GT 在「声明的口径」下是 `%s`，在「工具默认值」下是 `%s`，"
                     "**但这一题没有声明口径** ⟹ 这个探针测的是模型对未写出的约定的猜测，"
                     "不是模型会不会算。" % (c["qid"], c["category"], _fmt(c["gt"]), _fmt(c["alt"])))
        L.append("")
    elif sensitive:
        L.append("有 %d 题对某个**有默认值的参数**敏感，且都已在 `Spec.convention` 里写明口径："
                 % len(sensitive))
        L.append("")
        for c in sensitive:
            L.append("- **%s**「%s」：声明口径下 `%s`；工具默认值下 `%s`。口径：%s"
                     % (c["qid"], c["category"], _fmt(c["gt"]), _fmt(c["alt"]), c["convention"]))
        L.append("")
        L.append("⚠ 这两行**必须一起看**：只报 `%s` 而不报「另一个口径下是 `%s`」，"
                 "读者会以为答案没有歧义。完整映射见第五节。"
                 % (_fmt(sensitive[0]["gt"]), _fmt(sensitive[0]["alt"])))
        L.append("")
    else:
        L.append("没有任何题的 GT 对「有默认值的参数」敏感 —— 本批结论不依赖未声明的约定。")
        L.append("")

    # -- 证据契约 --------------------------------------------------------------
    unsupported = [r for r in results if r.verdict_level == "unsupported"]
    derived = [r for r in results if str(r.verdict_matched_from).startswith("derived:")]
    coincident = [r for r in results if r.coincidence]
    L.append("## 二、证据契约体检（因④）")
    L.append("")
    L.append("把上面那些**完美答案**喂给真 `verifier.verify()`：")
    L.append("")
    L.append("| 题号 | 答案 | 类型 | 结论 | 证据档位 | trace 数个数 |")
    L.append("|---|---|---|---|---|---|")
    for r in results:
        if r.spec.solve is None:
            continue
        L.append("| %s | `%s` | %s | **%s** | %s | %d |"
                 % (r.spec.qid, _fmt(r.answer), r.spec.answer_type,
                    r.verdict_level, r.verdict_matched_from or "—", r.verdict_n_numbers))
    L.append("")
    L.append("`证据档位` 这一列是 `Verdict.matched_from`，它把「对得上」分成三级明说的档：")
    L.append("")
    L.append("| 档位 | 含义 | 落在哪一级 |")
    L.append("|---|---|---|")
    L.append("| `value` / `evidence` | 答案**逐字**落在 trace 的数上 | `supported` |")
    L.append("| `derived:within` | 聚合量落在 trace 量值的 `[min, max]` 内 | `weak` |")
    L.append("| `derived:count` | 计数不超过 trace 里**显式的人口数** | `weak` |")
    L.append("| ——（都没有） | 凭空出现的数 | `unsupported` |")
    L.append("")
    L.append("⚠ 派生量**刻意判 `weak` 而不是 `supported`**：它只是「对得上」，不是「有出处」。")
    L.append("把两者混成一级，「supported 率」会随口径悄悄变宽而没人发现；分开之后，")
    L.append("想统计「有多少答案靠弱档撑着」读 `matched_from` 即可 —— 这就是它存在的理由。")
    L.append("")
    if unsupported:
        L.append("### ⚠ 完美答案被判 `unsupported` 的题")
        L.append("")
        for r in unsupported:
            L.append("- **%s**「%s」答案 `%s` —— 未通过：%s"
                     % (r.spec.qid, r.spec.question, _fmt(r.answer),
                        ", ".join(r.verdict_failed)))
        L.append("")
        L.append("这些答案**算对了**，但 `numeric_backed` 既不认「逐字出现」也不认派生量不变量。")
        L.append("**在改证据契约之前**，加多少工具都不会让这些题被算成「有支持的答案」。")
        L.append("")
    else:
        L.append("### ✅ 没有完美答案被判 `unsupported`")
        L.append("")
        L.append("上面 %d 个完美答案**全部被接受**（其中 %d 个靠派生量档位，落在 `weak`）。"
                 % (len([r for r in results if r.spec.solve is not None]), len(derived)))
        L.append("")
        L.append("这一条是 **2026-09-19 修好的**：改口径之前，本题集有 4 个完美答案被判 "
                 "`unsupported`（C4/C5/C8/C8b），根因是 `numeric_backed` 只认「逐字落在 trace 的数上」，")
        L.append("而组合题的答案天然是**派生量**（均值 / 计数 / 极值对）。")
        L.append("修法落在 `agents/verifier.py`：加了两级派生量档位，并把「只靠弱档通过」")
        L.append("降为软检查 `numeric_backed_exact`，于是它落在 `weak` 而不是冒充 `supported`。")
        L.append("")
        if derived:
            L.append("靠派生量档位通过的题：%s" % ", ".join(r.spec.qid for r in derived))
            L.append("")
    if coincident:
        L.append("### ⚠ 巧合命中的题（抹掉非量值键后结论翻转）")
        L.append("")
        for r in coincident:
            L.append("- **%s** 答案 `%s`：完整 trace → `%s`；抹掉非量值键 → `%s`"
                     % (r.spec.qid, _fmt(r.answer), r.verdict_level,
                        r.verdict_level_pruned))
        L.append("")
        L.append("删掉的是 `verifier.PRUNABLE_KEYS` = 普查（`label_counts`）/ 版本来源（`method`）")
        L.append("/ 请求参数（`tol`、`k`）。它们与答案无关却进过数字池，")
        L.append("于是判断力来自数字**大小**而不是正确性。")
        L.append("")
    else:
        L.append("### ✅ 没有巧合命中的题")
        L.append("")
        L.append("`list_objects.evidence[\"label_counts\"]` 与 `query_relation.evidence[\"method\"]`")
        L.append("现在都被 `verifier` **整棵子树跳过**（`_CENSUS_KEYS` / `_NON_QUANTITY_KEYS`）：")
        L.append("")
        L.append("- 普查副本删得掉：同一个量在 `returned` 里已按**本次查询**给了一份；")
        L.append("- `method` 是个**版本标记**（`\"geometry_v1\"`），它里面那个 `1` 不是任何东西的度量。")
        L.append("")
        L.append("⚠ 第二条是**本轮新抓到的**，与第一条同形：旧 `_ID_RE` 只抹 `chair_1` 那种")
        L.append("「下划线紧跟数字」的 id，抹不掉 `geometry_v1` 这种「下划线 + 字母 + 数字」的版本号。")
        L.append("后果一模一样 —— C8 的答案 `1` 被判 supported，出处是**算法版本号**。")
        L.append("")
        L.append("⚠ 本节的 `coincidence` 是「手工把 `verifier.PRUNABLE_KEYS` 抹掉再验一次」算出来的")
        L.append("（**可删**键，比 `NON_QUANTITY_KEYS` 窄 —— 后者还含 `object_id` 这类身份键，")
        L.append("而字符串答案的 `text_backed` 正要靠它们找出处，删了会报出**假阳性**）。")
        L.append("它现在是**回归探针**：一旦重新变 True，说明排除名单被改坏了。")
        L.append("它只能抓**已经知道名字**的那些键 —— 下一个同类缺陷要靠别的办法：")
        L.append("`scripts/show_answer_provenance.py` 打印每题的命中键，那才是查这个的工具。")
        L.append("")

    # -- 逐题 ----------------------------------------------------------------
    L.append("## 三、逐题明细")
    L.append("")
    L.append("| 题号 | 类别 | needs | 判据 | 答案 | 用到的工具 |")
    L.append("|---|---|---|---|---|---|")
    for r in results:
        L.append("| %s | %s | `%s` | %s | `%s` | %s |"
                 % (r.spec.qid, r.spec.category, r.spec.needs,
                    r.spec.blocked_by or (r.spec.caveat or "—"),
                    _fmt(r.answer), ", ".join(r.tools_used) or "—"))
    L.append("")

    # -- 字段清点 -------------------------------------------------------------
    L.append("## 四、字段清点（因② 的判据）")
    L.append("")
    L.append("下面这些字段，是「有没有任何工具的返回值交出来过它」。")
    L.append("")
    L.append("**口径**：`res.value` 是**契约面** —— 模型读得到、可以据此答题；")
    L.append("`res.evidence` 是**审计面** —— 给报告和人看的，本身不进提示词。")
    L.append("这条分界是刻意留的：evidence 里装着 `ranked` / `label_counts` 这类派生统计，")
    L.append("把它们升成契约面，「工具算出来的量」与「工具附带的元数据」就不再可分。")
    L.append("")
    L.append("⚠ `delta_z` / `tol` 仍只在 `evidence` 里，这是**有意保留**的：")
    L.append("死区问题的修法是把 `tol` 作为**参数**写进 `query_relation` 的文档")
    L.append("（程序可以自己传 `tol=0` 做严格比较），而不是把判定过程塞进 `value`。")
    L.append("原因是后者会**静默改掉**所有已经写好的程序 —— 见第五节末。")
    L.append("")
    L.append("⚠ 另一件事要分清：**「在 evidence 里」不等于「算数字证据」。**")
    L.append("`evidence` 里有一部分键（`method` / `object_id` / `tol` / `k`…）的值**不是对场景的度量**，")
    L.append("所以 `verifier` 现在按**键名**把它们整棵子树跳过数字池（`_NON_QUANTITY_KEYS`）。")
    L.append("审计仍然看得到它们，只是不再有「答案恰好等于某个常量」这种假出处 —— 见第二节末。")
    L.append("")
    L.append("| 字段 | 在 `res.value` 里 | 在 `res.evidence` 里 | 为什么组合题需要它 |")
    L.append("|---|---|---|---|")
    for fname, why in WANTED_FIELDS:
        in_v = "✅" if fname in census["value"] else "—"
        in_e = "✅" if fname in census["evidence"] else "—"
        L.append("| `%s` | %s | %s | %s |" % (fname, in_v, in_e, why))
    L.append("")
    L.append("动作空间里所有工具交出来过的键（`res.value`）：")
    L.append("")
    L.append("```")
    L.append(", ".join(census["value"]))
    L.append("```")
    L.append("")

    # -- 容差死区 -------------------------------------------------------------
    L.append("## 五、容差死区（一个对程序不可见的参数改了答案）")
    L.append("")
    L.append("同一个问题「有多少个物体在 `%s` 的 `%s` 方向？」——只改 `tol`，不改场景："
             % (dead_zone["anchor"], dead_zone["relation"]))
    L.append("")
    L.append("| tol | 答案 |")
    L.append("|---|---|")
    L.append("| 默认（%s） | **%d** |" % ("0.05", dead_zone["count_default_tol"]))
    L.append("| 0.0 | **%d** |" % dead_zone["count_tol_0"])
    L.append("")
    if dead_zone["flipped_by_dead_zone"]:
        L.append("被死区翻掉的对象：")
        L.append("")
        L.append("| object_id | delta_z (m) | 默认 tol 下 | tol=0 下 |")
        L.append("|---|---|---|---|")
        for r in dead_zone["flipped_by_dead_zone"]:
            L.append("| `%s` | %s | %s | %s |"
                     % (r["object_id"], _fmt(r["delta_z"]), r["default_tol"], r["tol_0"]))
        L.append("")
    L.append("`front_of(a, b)` 的判据是 `a.z < b.z - tol`，`tol` 默认 50 mm。")
    L.append("于是 50 mm 以内**一律返回 `False`** —— 不是「不确定」，是一个确定的 `False`。")
    L.append("而 `query_relation` 的 `res.value` 是一个**裸 bool**：`delta_z` 与 `tol`")
    L.append("只写在 `evidence` 里。")
    L.append("")
    L.append("⟹ 程序拿到一个 `False`，无法从中分辨「真的不在前方」与「卡在死区」。")
    L.append("计数类组合题的答案因此依赖一个**未声明的参数**。")
    L.append("")
    L.append("### ⚠ 关键：这两行不是「两个都对」，GT 必须选一个")
    L.append("")
    L.append("问题问的是**世界的样子**，不是工具的输出。`tol=0.05` 是一条**鲁棒性余量**")
    L.append("（让近乎共面的两个物体不要被硬判成前后关系），它是**工具的实现细节**。")
    L.append("把 GT 定义成「工具默认值下的输出」，等于让工具的行为冒充真值 —— 于是")
    L.append("「差 37.8 mm、物理上确实在桌子前面」的 `chair_1` 被记成「不在前面」，")
    L.append("而**任何按质心比较的程序**（那是完全正确的做法）都被判成算错。")
    L.append("")
    L.append("⟹ **本脚本的 GT 改为严格比较（`tol=0`）**，并把口径写进 `Spec.convention`。")
    L.append("这不是迁就模型：C8 的「一次调用解」一直用严格比较，它的 docstring 早就写着")
    L.append("「两个答案都对，取决于一个对程序不可见的容差」—— 那个「两个都对」的说法，")
    L.append("本身就是**问题没被定义清楚**的自白。")
    L.append("")
    reg = [c for c in convention if c["alt"] is not None]
    L.append("**口径审计**（登记在 `_ALT_CONVENTION` 的 %d 题 / 合计 %d 题）："
             % (len(reg), len(convention)))
    L.append("")
    if reg:
        L.append("| 题号 | 类别 | 声明口径下的 GT | 工具默认值下 | 两口径不同 | 声明的口径 |")
        L.append("|---|---|---|---|---|---|")
        for c in reg:
            L.append("| %s | %s | `%s` | `%s` | %s | %s |"
                     % (c["qid"], c["category"], _fmt(c["gt"]), _fmt(c["alt"]),
                        "**是**" if c["sensitive"] else "否", c["convention"] or "—"))
        L.append("")
    if ambiguous:
        L.append("⚠ **`ambiguous`**：下列题的 GT 对默认值敏感、却没有声明口径 —— "
                 "在修掉之前，这些题的「答错」不能算在模型头上：")
        L.append("")
        for c in ambiguous:
            L.append("- **%s**" % c["qid"])
        L.append("")
    else:
        L.append("✅ 没有 `ambiguous`：凡 GT 对默认值敏感的题都已声明口径。")
        L.append("")
    L.append("⚠ **这个审计的已知边界**：它只查**已登记**的题 —— 登记是人工的。")
    L.append("下一个同类缺陷（答案依赖某个没人想到的默认值）**仍可能不被它抓到**，")
    L.append("除非有人先把「另一个口径」写进 `_ALT_CONVENTION`。")
    L.append("但只要登记了，它就再也不会退化成「长得像模型出错」的静默错标签。")
    L.append("")
    L.append("### 修法：把 `tol` 从「隐含条件」变成「文档化的参数」（**已做**）")
    L.append("")
    L.append("`tol` 本来就是 `query_relation` 的**参数**，程序完全能传 `tol=0`。")
    L.append("问题从来不是「做不到」，而是**没人告诉模型它存在、默认值是多少** ——")
    L.append("所以修法是把默认值写进工具文档的**第一段**（= 逐字节进提示词的那一段）。")
    L.append("")
    L.append("⚠ **不采用**「把 `delta_z`/`tol` 塞进 `res.value`」那个更直接的做法：")
    L.append("它会把 `value` 从 `bool` 变成 `dict`，而 `bool(dict)` 恒为 `True` ——")
    L.append("已经写好的程序（例如 C8 那一版 `is_front = bool(val) if isinstance(val, bool) else val > 0`）")
    L.append("会**静默地把每个物体都数成在前方**。这是「改了却不报错」的坏改动，")
    L.append("而它换来的信息量，一行文档就能给。")
    L.append("")

    # -- 一次调用够不够 --------------------------------------------------------
    gt_by_qid = {r.spec.qid: r.answer for r in results}
    same = [q for q, v in one_call.items() if v.get("answer") == gt_by_qid.get(q)]
    diff = [q for q, v in one_call.items()
            if v.get("answer") is not None and v.get("answer") != gt_by_qid.get(q)]
    L.append("## 六、只用**一次**工具调用够不够")
    L.append("")
    L.append("`list_objects` 的返回值里**已经带了全部物体的 `centroid_m` 与 `extent_m`**。")
    L.append("于是上面有一批题的全部所需几何量，**第一次调用就拿到了**。")
    L.append("")
    L.append("| 题号 | 一次调用的答案 | 与 GT 的关系 | 几次调用 |")
    L.append("|---|---|---|---|")
    for q in [s.qid for s in SPECS if s.qid in one_call]:
        v = one_call[q]
        g = gt_by_qid.get(q)
        rel = ("**一致**" if v.get("answer") == g
               else ("**不同（见下）**" if v.get("answer") is not None else "算不出"))
        L.append("| %s | `%s` | %s | %d |" % (q, _fmt(v.get("answer")), rel, v.get("n_calls", 0)))
    L.append("")
    L.append("**%d 道题（%s）的全部所需几何量，一次 `list_objects` 就够。**"
             % (len(same), ", ".join(same) or "—"))
    if diff:
        L.append("")
        L.append("⚠ **%s 与 GT 不同，而差异是本质的**：GT 走 `query_relation`（容差 50 mm），"
                 "一次调用那一版用质心直接比大小（等价于 `tol=0`）。" % ", ".join(diff))
        L.append("两个答案都「对」，选哪个取决于那个**对程序不可见的容差** —— "
                 "见第五节的死区对照。")
    L.append("")
    L.append("这一节把两件事分开了：**「工具不够用」**（需要新字段/新算法）与"
             "**「返回值里已经有、但没被用上」**（纯提示词问题）。")
    L.append("")

    # -- 真问题 ---------------------------------------------------------------
    L.append("## 七、这批数字回答了什么、没回答什么")
    L.append("")
    L.append("**回答了**：在「所有空间数值只能来自工具返回值」这条约束下，")
    L.append("上面 %d 题里有 %d 题的正确答案可以精确算出来 —— 所以「功能单薄」"
             "至少有一大半**不是**缺工具。" % (len(results), len(reachable)))
    L.append("")
    L.append("**没回答**：模型**实际会不会**去组合。工具够用 ≠ 模型会用。")
    L.append("那一半由 `scripts/probe_combination_run.py` 回答（真跑，10 题约 ¥0.09），")
    L.append("判据是**程序源码上的组合算子**与**调用次数冗余倍数**，")
    L.append("而不是只看答案对错 —— 答案对错会把「答对了但被契约拒收」和「答错了」混成一件事。")
    L.append("")
    # -- 结论：按「修法」排序 -------------------------------------------------
    n_unsupported = len(unsupported)
    n_coincide = len(coincident)
    n_one_call = len(same)
    from llm.schema import docs_text as _docs_text

    _docs_chars = len(_docs_text(tools=QA_TOOLSET))
    _found = sorted(DEFAULT_REPORT.parent.glob("combination_run_*_reanalyzed.md"),
                    key=lambda p: p.stat().st_mtime)
    _run_ref = ("`reports/%s`" % _found[-1].name) if _found else "（尚无真跑产物）"
    _after = _latest_run_summary(_found[-1]) if _found else None
    L.append("## 八、结论与下一步（按**修法**排序，不按题目排序）")
    L.append("")
    L.append("「功能单薄」不是一个问题，是四个。上面每节的数字对应其中一个。")
    L.append("**①②③ 已于 2026-09-19 落地；④ 未做。**")
    L.append("")
    L.append("### ① 不加工具：让模型看见返回值里**已经有**的字段 —— ✅ 已做")
    L.append("")
    L.append("- 证据：%d/%d 题的全部所需几何量，**一次 `list_objects` 就够**"
             "（`list_objects.value` 里每条都带 `centroid_m` 与 `extent_m`）。" % (n_one_call, len(results)))
    L.append("- 修法（已落地）：每个读类工具的 docstring **第一段**补上 `res.value` 形状。")
    L.append("  第一段正是渲染进提示词的那一段（`llm/schema.py::first_paragraph`）——")
    L.append("  项目里已有先例（`get_attributes` 的形状被挪进第一段后，"
             "「模型把 value 猜成另一种形状」的错误立刻消失）。")
    L.append("- **已核实**：改前 12 行工具文档共 1605 字符，**只有 `get_attributes` 那一行**写了形状；")
    L.append("  `list_objects` 那行只写「列出场景中的物体」—— 一个字都没提 value 里带 `extent_m`。")
    L.append("  形状原本写在 system prompt 的规则 2 里，但**每条工具文档自己不说**；")
    L.append("  两条信息源具体程度不一致时，模型跟的是更具体的那条：它去调 `get_3d_extent`")
    L.append("  （那条文档明确承诺了 `(w, h, l)`），而不是相信 `list_objects` 顺手也给了同一个量。")
    L.append("  ⟹ 这不是「模型不听话」，是**我们把契约写在了模型不查的那一处**。")
    L.append("- 改后：12 条文档全部带形状，共 **%d 字符**（预算 2500，见 "
             "`tests/test_agent_memory_and_layering.py::test_docs_are_compact`）；" % _docs_chars)
    L.append("  system prompt 规则 2 里那份**重复的**形状表同时被删掉 ——")
    L.append("  两处各写一份必然漂移，而实测模型跟的是更具体的那一条。")
    L.append("- 防回归：`test_every_tool_documents_its_return_shape` 现在对每个工具断言")
    L.append("  「第一段里出现 `res.value`」，并有 `DOC_SHAPE_EXEMPT` 作为显式出口。")
    L.append("- 真跑对照（**改前**）：`reports/combination_run_20260919_212608_reanalyzed.md`")
    L.append("  —— 106 次调用 vs 最少 26 次，**冗余倍数中位数 5.0**，8/10 题答案是**对的**。")
    L.append("  ⟹ 它不缺能力，缺的是「列表里已经有坐标」这个认知。")
    L.append("- 真跑对照（**改后**）：%s —— 同一批题、同一个场景、同一个模型。" % _run_ref)
    if _after:
        L.append("  - 工具调用 **%s → %s** 次（最少 %s 次）｜ 冗余倍数中位数 **%s → %s**"
                 % (BASELINE_BEFORE["calls"], _after["calls"], _after["min_calls"],
                    BASELINE_BEFORE["ratio"], _after["ratio"]))
        L.append("  - 结局分布 **%s → %s** ｜ 与 GT 一致 **%s/%s** ｜ 成本 ¥%s"
                 % (BASELINE_BEFORE["class"], _after["class"], _after["gt_ok"],
                    _after["n"], _after["cost"]))
        L.append("  - 「答对却被契约拒收」（class C）**%s → %s**" % (BASELINE_BEFORE["c_rejected"], _after["c"]))
        L.append("")
        L.append("  ⚠ **%s 这个数不要读成「提示词让模型变差了」。**"
                 % ("改后 %s" % _after["ratio"]))
        L.append("  `%s → %s` 这个跨度里混着**两件事**，必须拆开："
                 % (BASELINE_BEFORE["ratio"], _after["ratio"]))
        L.append("    ① 提示词/文档修复：调用次数 106 → 32（这一步模型真的变省了）；")
        L.append("    ② **C8 的 GT 口径修正**（见第五节）：修之前 C8 的「最少几次够」记 9，")
        L.append("       因为它那个算错的 GT 看起来需要逐物体调 `query_relation`；")
        L.append("       口径修正后同一件事**一次 `list_objects` 就够** ⟹ 分母 9 → 1，")
        L.append("       于是 C8 自己的倍数从 1.11 跳到 10.0，把中位数整个抬起来。")
        L.append("  模型的**绝对**调用次数在这次口径修正里一次没动过（仍 32 次）。")
        L.append("  ⟹ 中间那句「5.0 → 1.11」是只有 ① 的时候的数字，它**包含一个算错的题目标准**；")
        L.append("  现在的 %s 才是 ①+② 都对了之后的数。两个数不能混着引用。" % _after["ratio"])
        L.append("")
        L.append("  ⚠ 口径定义（用的时候要带上）：这里的「中位数」是**排序后取第 n/2 位**")
        L.append("  （`_summarize` 里 `sorted(ratios)[len(ratios)//2]`）—— 偶数题数时取的是")
        L.append("  偏大那一个，不是两数均值。同一份定义横跨改前改后，所以可比；")
        L.append("  但它与「教科书中位数」不是一回事，引用时别省掉这句。")
        L.append("  （这些数字是**从那一份 JSON 里读出来的**，不是我抄进报告里的 —— "
                 "抄进来的数字在下次真跑之后会变成一个**看起来正确**的旧数字。）")
    L.append("- ⚠ 不要为此加工具。加工具会让动作空间变长、信息量不变，")
    L.append("  而动作空间变长本身会让「工具选择准确率」这个指标变差。")
    L.append("")
    L.append("### ② 扩返回值：`get_object` 补 `bbox_2d`，`tol` 改成文档化参数 —— ✅ 已做")
    L.append("")
    L.append("- `bbox_2d` 原本没有任何工具的 `value` 交出来（只在 `get_3d_position` 的 evidence 里），")
    L.append("  于是「图像左半边 / 画面上方」这类二维区域题在**信息层面**就答不了。"
             "现在 `get_object` 的 value 带上了它")
    L.append("  （图像像素系，与相机系米制的 `centroid_m` 分属两套坐标 —— 文档里写明，免得混用）。")
    L.append("- ⚠ 只放在**单物体路径**上：`list_objects` 一次返回 9~20 个物体，"
             "每个多 4 个像素数会把观察文本撑大近一倍。")
    L.append("- ⚠ 这一条**没有**让 `C9` 变成「可答」：C9 要遍历每个物体的 `bbox_2d`，")
    L.append("  而 `get_object` 是逐物体调用（9 个物体 = 9 次，与 `get_3d_extent` 同价）。")
    L.append("  它给的是**能力**，不是**便宜**。")
    L.append("- `tol` 死区：改成**文档化参数**，而不是把 `delta_z`/`tol` 塞进 `value` ——")
    L.append("  `value` 由 bool 变 dict 会让 `bool(dict)` 恒真，**静默改掉所有已写好的程序**")
    L.append("  （详见第五节末）。")
    L.append("")
    L.append("### ③ 改证据契约：派生量档位 + 排除非量值键 —— ✅ 已做")
    L.append("")
    L.append("- 前半段（完美答案）：`unsupported` **4 → 0**；巧合命中**抓到一个修掉一个**。")
    L.append("  当前值：`unsupported` = %d，巧合命中 = %d，靠派生量档位通过 = %d。"
             % (n_unsupported, n_coincide, len(derived)))
    L.append("  ⚠ 这里不能写「巧合命中 N → 0」了事：本轮又抓到**第二个**同形的（`method`），")
    L.append("  而它不是被原口径抓到的 —— 是**改 GT 之后 `matched_from` 从 `derived` 变成")
    L.append("  `evidence`，顺着「这个逐字出处是哪来的」查下去才现形**。")
    L.append("  这就是为什么「巧合命中」这一项的当前值永远不能当成「已清零」。")
    L.append("- 后半段（真跑，改前）：10 题全对，但 **3 题被自己的契约判 `unsupported`**、2 题判 `weak`")
    L.append("  ⟹ **30% 的正确答案会被系统自己扔掉**。这不是模型的锅，加工具也治不了。")
    L.append("- 根因：`numeric_backed` 只认「答案**逐字落在** trace 的数上」，")
    L.append("  而组合题的答案是**派生量**：均值落在两个数之间、计数是个新整数、极值对是两个 id 的拼接。")
    L.append("- 落地（`agents/verifier.py`）：两级派生量档位 + 一条软检查。")
    L.append("")
    L.append("  | 档位 | 不变量 | 级别 |")
    L.append("  |---|---|---|")
    L.append("  | `value` / `evidence` | 答案逐字落在 trace 的数上 | `supported` |")
    L.append("  | `derived:within` | `min(量值) ≤ 答案 ≤ max(量值)` | `weak` |")
    L.append("  | `derived:count` | 整数答案 ≤ trace 里**显式的人口数** | `weak` |")
    L.append("  | —— | 都没有 | `unsupported` |")
    L.append("")
    L.append("  既保住了原来的能力（VADAR 那种凭空出现的 `998.65` 仍被拒：它不在任何量值区间内、")
    L.append("  也不是合法计数），又不再把「对的派生量」当垃圾扔掉。")
    L.append("- ⚠ 两条刻意的**收窄**，都是为了让口径不悄悄变宽：")
    L.append("  ① 派生量判 `weak` 而不判 `supported`（软检查 `numeric_backed_exact`）——")
    L.append("     「supported 率」因此不随口径变宽而上升；想数弱档读 `matched_from` 即可。")
    L.append("  ② 计数档的上界取 trace 里的**显式人口字段**（`total_objects`/`returned`/`count`），")
    L.append("     **不是** `len(某个列表)` —— 后者会让「id 里那个 1」重新变成证据。")
    L.append("- ⚠ **仍有一个已知缺口**：`sum` 类派生量不在覆盖范围内。")
    L.append("  求和的上界是「最大值 × 个数」，宽到会放走差一个数量级的答案 ——")
    L.append("  那正是 `test_wrong_magnitude_is_not_forgiven` 钉住的那条线。")
    L.append("  要治它得让 `submit` 支持**声明式派生**（把被聚合的输入集合一起报上来、逐项核对），")
    L.append("  那是一次契约变更，且要重跑才能量出遵从率。**本轮不做。**")
    L.append("- ⚠ 顺带修掉**两个同形的假出处**（都是「与题目无关的常量漏进了数字池」）：")
    L.append("  ① `label_counts`（`{\"sofa\": 1, \"picture\": 4}` —— 一组 1~4 的小整数）：")
    L.append("     让「4 个」恒真、「3 个」恒假，判断力来自数字**大小**。")
    L.append("  ② `method`（`\"geometry_v1\"` —— **算法版本号**）：旧 `_ID_RE` 只抹 `名字_数字`，")
    L.append("     抹不掉 `_v1` 这种版本后缀，于是每调一次 `query_relation` 就往池里塞一个 `1.0`。")
    L.append("     C8 的答案恰好是 `1` ⟹ 被判 `supported`，出处是**版本号**。")
    L.append("  两者现在都按**键名**整棵子树跳过（`_CENSUS_KEYS` / `_NON_QUANTITY_KEYS`），")
    L.append("  并各自留了回归用例 —— 这一类缺陷的共性是**长得像「有证据」**，")
    L.append("  所以唯一的抓法是「把无关字段删掉，看结论会不会翻」。")
    L.append("")
    L.append("### ④ 加工具（真的缺算法时才加）—— ⬜ 未做")
    L.append("")
    L.append("- 本轮 13 题里只有 **2 题**真的缺算法：体积（`C10`）与朝向（`C11`）。")
    L.append("  两者都**只需要点云**，而 `points.npy` 已经落盘 —— 不需要新模型、不需要新显存。")
    L.append("- 待补的工具（都必须**确定性**，否则 `stats.py` 不可复现）：")
    L.append("  `get_object_volume`（OBB / 凸包体积）、`get_object_orientation`（PCA 主方向 + 上轴 → 偏航角）、")
    L.append("  `fit_support_plane`（RANSAC，**固定种子**）、`cluster_points`（固定 eps/min_samples）。")
    L.append("- 落地位置：`scene_graph/pointcloud.py` 已经是「零 torch」的重建层，")
    L.append("  算法放它旁边即可保住「builder 之外全程不需要 GPU」这条性质。")
    L.append("")
    L.append("### 一句话")
    L.append("")
    L.append("> ①②③ 都已落地，**动作空间一个工具都没加**（仍是 12 个，只有文档与校验器变了）。")
    L.append("> ④ 等确认「加工具确实能改变可回答问题的集合」时再做 ——")
    L.append("> 在那之前加任何工具，都只会改变动作空间长度。")
    L.append("")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="组合题可达性与证据契约体检（零成本）")
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--out", default=str(DEFAULT_REPORT))
    ap.add_argument("--json", default=None, help="机读产物（默认与 --out 同目录同名 .json）")
    args = ap.parse_args(argv)

    scene_path = Path(args.scene)
    if not scene_path.exists():
        scene_path = scene_dir(args.scene)
    scene = load_scene(scene_path)

    results = [run_spec(s, scene) for s in SPECS]
    census = field_census(scene)
    dead_zone = dead_zone_probe(scene)
    one_call = one_call_solutions(scene)
    convention = convention_audit(scene, results)

    text = build_report(scene, results, census, dead_zone, one_call, args.scene, convention)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    payload = {
        "scene": args.scene,
        "scene_path": str(scene_path),
        "n_objects": len(scene.nodes),
        "toolset": list(QA_TOOLSET),
        "field_census": census,
        "dead_zone": dead_zone,
        "one_call_solutions": one_call,
        "convention_audit": convention,
        "results": [
            {
                "qid": r.spec.qid, "category": r.spec.category,
                "question": r.spec.question, "answer_type": r.spec.answer_type,
                "route": list(r.spec.route), "needs": r.spec.needs,
                "blocked_by": r.spec.blocked_by, "caveat": r.spec.caveat,
                "convention": r.spec.convention,
                "answer": r.answer, "evidence": r.evidence, "targets": r.targets,
                "tools_used": r.tools_used, "n_tool_calls": r.n_tool_calls, "error": r.error,
                "verdict": {"level": r.verdict_level,
                            "matched_from": r.verdict_matched_from,
                            "n_numbers": r.verdict_n_numbers,
                            "failed": r.verdict_failed,
                            "level_pruned": r.verdict_level_pruned,
                            "coincidence": r.coincidence},
            }
            for r in results
        ],
    }
    jout = Path(args.json) if args.json else out.with_suffix(".json")
    jout.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[combination] %s" % out)
    print("[combination] %s" % jout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
