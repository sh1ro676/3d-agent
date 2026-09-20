"""L5 —— 场景级输出工具（★ 第一类输出：让中间表示本身成为交付物）。契约见 §11.1。

为什么把这一层单列出来，而不是让它混在 L2/L3 里：L1–L4 全是「面向单个问题」的工具 ——
输出要么是某道题的答案，要么是某个视图动作，场景图跑完一道题就被丢掉。
L5 反过来：**输入是场景，输出是结构化的三维场景描述**。于是：

  • 「1.42 m」这种数字第一次成为**可保存、可复查、可对比**的产物，而不是一个立即被丢弃的中间值；
  • 评测上多出**独立于问答准确率**的指标（描述召回率、关系边一致率，§16.3 指标 11–12），
    于是「工具库升级」这一个创新点能出两张表而不是一张；
  • 项目从「三维问答工具」抬到「三维场景理解系统」—— 这才是标题里 `3D Spatial` 的应有产出。

**零额外模型成本**：`objects` 与 `relations` 的全部字段在 `build_scene_graph` 时就已经算好，
本模块只做**序列化、稀疏化、归因**三件事，不碰 GPU、不加载任何权重。
这条性质决定了本文件可以完全脱 GPU 单测 —— 所以它进 `load_tools()`。

与文档 §11.1 的一处**有意偏离**（记在这里以免下次有人以为写漏了）：
文档把 `summarize_scene` 写成「L5 中唯一允许走 LLM 的工具」，且签名是 `(scene_id) -> str`。
实测需求是**默认必须确定性**（评测里同一场景要能产出逐字节相同的报告，否则指标不可比），
所以本实现给成 `summarize_scene(scene_id, use_llm=False)`：
    use_llm=False（默认）→ 纯模板，零 LLM，可复现、可单测；
    use_llm=True         → 走 `ctx.flags["summarizer"]`，未注入时返回 `CAPABILITY_DISABLED`。
把「允许走 LLM」保留成一个**显式开关**而不是默认行为，是消融可复现的前提（§13.3(7)）。

`answer_with_evidence` **本阶段不实现**，理由见文件末尾。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from scene_graph.relations import RELATIONS, pairwise
from scene_graph.schema import Node, SceneGraph
from tools.guards import need_scene
from tools.registry import ToolArgumentError, tool
from tools.result import ErrorCode, ToolResult
from tools.version import TOOLS_VERSION

__all__ = [
    "DEFAULT_MAX_PAIRS",
    "REPORT_TOOL_VERSION",
    "describe_scene",
    "summarize_scene",
    "diagnose_failure",
    "counterfactual",
]

#: 工具库版本（见 tools.version）。本模块是纯派生计算，跟随库版本即可。
REPORT_TOOL_VERSION = TOOLS_VERSION

#: `detail="full"` 时最多展开多少对关系。
#:
#: 为什么要设上限：20 个物体是 190 对 × 11 个关系 = 2090 条，序列化后约 125 KB。
#: 作为**落盘交付物**这不是问题，但同一个返回值也会出现在程序合成模型的观察里，
#: 而观察必须压缩（§18 Phase 7 风险②）。超限时**不静默截断**，而是设 `truncated`
#: 并把被截掉的对数写进 `caveats` —— 静默截断会让「关系边一致率」这个指标说谎。
#:
#: **上限的量化后果**（实测，不是估的；9 个物体 / 36 对 / 396 条 ≈ 76 KB）：
#: 200 对 × 11 条 ≈ 2200 条，报告 JSON 上界约 **0.4 MB**。
#: 按「对」而不是按「条」截断是有意的：半对被砍会让关系边一致率算错，
#: 而砍掉整对至少还是自洽的。
DEFAULT_MAX_PAIRS = 200

#: 报告里关系的输出顺序。`distance` 必须在最前：它是唯一带量纲、可直接引用的量，
#: 布尔关系只是对它的定性切分。
_REPORT_RELATIONS: tuple[str, ...] = ("distance",) + tuple(
    r for r in RELATIONS if r != "distance"
)

#: 尺寸越界阈值（米）。单张室内图的合理上限，与 scripts/inspect_scene.py 保持同值。
_SUSPICIOUS_EXTENT_M = 3.0

#: 点数下限。**阈值来自实测的自然分界，不是拍出来的**：
#: `dataset/scenes/living_room_gt` 的 9 个物体里，3 个伪物体（低分 picture）是
#: 809 / 893 / 965 点，其余 6 个真实物体是 16137 / 18015 / 18785 / 23841 / 29108 / 30660。
#: 两者之间有 **17 倍的空隙**，所以 1000 落在空隙里 —— 取 2000 也一样有效。
#: 这条判据之所以必须存在：`schema.py` 的 `Node.n_points` 注释写明「落在掩码内的点云点数，
#: `DEGENERATE` 的判定依据」，而 L5 早先只查了尺寸**上界**、把它漏了。
_MIN_POINTS = 1000


# ----------------------------------------------------------------------------
# 组件
# ----------------------------------------------------------------------------


def _object_entry(node: Node) -> dict[str, Any]:
    """单个物体的报告条目 —— 米制尺寸与质心都在这里。

    刻意**不**用 `model_dump()`：它会带上 `mask_ref`（一个磁盘路径）、`bbox_2d`（像素）、
    `frame`（每行都重复）这些对读者无用的字段。报告是给人看和给指标算的，
    不是数据库导出，字段要挑。
    """
    return {
        "id": node.id,
        "label": node.label,
        "centroid_m": [round(c, 4) for c in node.centroid_3d],
        "extent_m": {
            "w": round(node.extent_3d[0], 3),
            "h": round(node.extent_3d[1], 3),
            "l": round(node.extent_3d[2], 3),
        },
        "confidence": round(node.score, 3),
        # 以下三项**不是**装饰：`centroid_source` 是 `diagnose_failure` 把失败归因到
        # 「掩码」还是「检测」的依据；`n_points` 是 DEGENERATE 的判据；
        # `attributes` 是角色② 唯一允许写入场景图的东西（只有颜色/材质，没有空间量）。
        "centroid_source": node.centroid_source,
        "n_points": node.n_points,
        "attributes": dict(node.attributes),
    }


#: 各条关系里**可以安全删掉**的 metric 键。判据是「能否从报告别处无损还原」，
#: 而不是「值是否与某条相同」。
#:
#: 只删质心分量：`a_x/a_y/a_z` 就是 `objects[][centroid_m]` 里那个物体的质心，
#: `b_*` 同理 —— 而 `objects` 本来就在同一份报告里。删掉它们是无损压缩。
#:
#: ❌ **曾经写错、已实测否决**的规则：「值与该对 distance 条里的同名键相同就删」。
#: 两个理由，任一条都足以否决：
#:   ① 它在浮点上是**不确定的** —— `round(x, 6) == x` 只在少数数上成立，
#:      于是 `left_of` 在 (sofa,picture) 上保留了 `delta_x`、在 (door,sofa) 上把它删了。
#:      报告字段集随数值抖动，diff 与测试都无法稳定断言。
#:   ② 它把**判别量**一起删了。`left_of` 的判据正是 `delta_x < -tol`，
#:      删掉之后「为什么判定为左」就不再能从该条自身复算 —— 恰好毁掉 metric 存在的理由。
#:      更危险的是按**键名**删同样不行：`above` 的 `delta_y` 过了 `up_coord()` 归一，
#:      与 distance 的同名键**含义不同**，按名字删会静默丢掉真实信息。
_CENTROID_METRIC_KEYS = ("a_x", "a_y", "a_z", "b_x", "b_y", "b_z")


def _relation_entries(
    scene: SceneGraph,
    pairs: Sequence[tuple[Node, Node]],
) -> list[dict[str, Any]]:
    """把若干**无序对**展开成扁平的关系列表。

    输出形态对齐文档 §11.1 的示例：量纲量用 `value_m`，布尔量用 `value`，
    两者都带 `metric` —— 于是「为什么判定为左」可以脱离代码复算（`delta_x` 就在里面）。

    `distance` 条保留完整基础量（含质心分量），是该对的**参照条**；
    其余关系只去掉质心分量（见 `_CENTROID_METRIC_KEYS` 的无损论证）。
    """
    out: list[dict[str, Any]] = []
    for a, b in pairs:
        verdicts = pairwise(a, b, up=scene.up_axis)
        for rel in _REPORT_RELATIONS:
            v = verdicts.get(rel)
            if v is None:
                # 缺 `bbox_3d` 的节点会跳过 on/inside（pairwise 的约定：不因一项缺数据
                # 就丢掉整对关系）。这里如实不输出，而不是补一个 null 让下游误解析。
                continue
            entry: dict[str, Any] = {"a": a.id, "b": b.id, "type": rel}
            entry["value_m" if rel == "distance" else "value"] = (
                round(float(v.value), 4) if rel == "distance" else bool(v.value)
            )
            metric = {k: round(float(x), 6) for k, x in v.metric.items()}
            if rel != "distance":
                metric = {k: x for k, x in metric.items() if k not in _CENTROID_METRIC_KEYS}
            entry["metric"] = metric
            out.append(entry)
    return out


def _all_pairs(nodes: Sequence[Node]) -> list[tuple[Node, Node]]:
    """无序对枚举 `i < j`。顺序稳定（按 nodes 原序），所以报告可逐字节 diff。"""
    return [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]


def _relation_summary(scene: SceneGraph, nodes: Sequence[Node] | None = None) -> dict[str, Any]:
    """`detail="brief"` 用的关系压缩：不给逐对明细，只给计数与「最近邻」。

    「每个物体的最近邻距离」是压缩后的关键信息 —— 只保留它，
    布局类问题的可回答性几乎不损失，而条目数从 O(n²) 降到 O(n)。

    `nodes` 显式传入是为了让 `counterfactual` 能对**剪枝后**的子集重算：
    否则「移走沙发后谁离门最近」会返回一个已经被移走的物体。
    """
    nodes = list(scene.nodes if nodes is None else nodes)
    counts: dict[str, int] = {}
    nearest: dict[str, dict[str, Any]] = {}
    for a, b in _all_pairs(nodes):
        verdicts = pairwise(a, b, up=scene.up_axis)
        counts["distance"] = counts.get("distance", 0) + 1
        for rel in _REPORT_RELATIONS:
            if rel == "distance":
                continue
            v = verdicts.get(rel)
            if v is None:
                continue
            if bool(v.value):
                counts[rel] = counts.get(rel, 0) + 1
        d = float(verdicts["distance"].value)
        for src, dst in ((a, b), (b, a)):
            cur = nearest.get(src.id)
            if cur is None or d < cur["distance_m"]:
                nearest[src.id] = {"object_id": dst.id, "distance_m": round(d, 4)}
    return {
        "relation_true_counts": counts,
        "n_pairs": len(_all_pairs(nodes)),
        "nearest_neighbour": nearest,
    }


def _is_scale_calibrated(scene: SceneGraph) -> bool:
    """尺度是否已校正。

    两个来源，优先 `build_meta["scale_calibrated"]`（显式记录），
    否则退化成「`scale_factor` 不等于 1.0」。

    ⚠ 后者只是一个**启发式**：一个场景恰好用 1.0 校正过是可能的，于是会被误判成未校正。
    宁可误报「未校正」也不误报「已校正」—— 前者只是保守，后者会让不可信的米数被当成真值。
    这条取舍与 `describe_scene` 把 caveats 一并输出是同一个动机。
    """
    meta = dict(scene.build_meta)
    default = not math.isclose(float(scene.scale_factor), 1.0)
    return bool(meta.get("scale_calibrated", default))


def _quality(scene: SceneGraph) -> tuple[dict[str, Any], list[str]]:
    """报告的可信度自述 —— 返回 `(质量字段, 告诫语)`。

    这一段是本模块**最该存在**的部分。场景报告一旦落盘就会被人引用，
    而单目度量深度在零样本设置下有尺度漂移（§11.1 末尾那条注意）——
    如果报告不自己声明「这个米数可能整体偏了 k 倍」，读者会把 1.42 m 当成真值。
    """
    meta = dict(scene.build_meta)
    scale_factor = float(scene.scale_factor)
    calibrated = _is_scale_calibrated(scene)
    caveats: list[str] = []

    if not calibrated:
        caveats.append(
            "尺度未校正（scale_factor=1.0）：全部米制数字共享同一个未知比例因子，"
            "**相对关系可信、绝对数值不可采信**。要引用绝对值需先 calibrate_scale。"
        )

    src = meta.get("intrinsics_source", "unknown")
    if src == "predicted":
        caveats.append(
            "内参来自模型预测而非外部传入：横向米制尺度不可信（§21/§22 实测三维误差中位数 "
            "1.943 m vs 传入 GT 的 0.267 m）。"
        )
    elif src == "unknown":
        caveats.append("未记录内参来源（旧版 scene.json）—— 无法判断横向尺度是否可信。")

    if not meta.get("up_axis_reliable", True):
        caveats.append(
            f"重力方向不可靠（up_axis={scene.up_axis}, "
            f"reason={meta.get('up_axis_reason')}）：`above`/`below` 可能整体翻转。"
        )

    n_fallback = sum(1 for n in scene.nodes if n.centroid_source != "mask")
    if n_fallback:
        caveats.append(
            f"{n_fallback}/{len(scene.nodes)} 个物体的质心走了降级路径（bbox_fallback）："
            "与掩码质心的实测差为均值 83 mm / 最大 208 mm，而关系容差是 50 mm —— "
            "涉及这些物体的近邻关系应降权。"
        )

    # 伪物体必须在这里就说出来。报告是**给人看的交付物**，它会把 objects 直接列成
    # 「这个房间里有 4 幅画」；如果其中 3 幅其实是低分检测框圈出的几十个像素，
    # 而报告一声不吭，那报告就在说谎 —— 这比数字不准更严重。
    thin = sorted(
        ((n.id, n.n_points) for n in scene.nodes
         if n.n_points is not None and n.n_points < _MIN_POINTS),
        key=lambda t: t[1],
    )
    if thin:
        caveats.append(
            f"{len(thin)} 个物体的掩码点数少于 {_MIN_POINTS}（{thin[:4]}）—— "
            "很可能是伪物体而非真实存在：点云质心不可信，物体计数与关系边数都会因此偏高。"
            "实测分界为伪物体 809–965 点 vs 真实物体 16137–30660 点（17 倍空隙）。"
        )

    return (
        {
            "scale_calibrated": calibrated,
            "scale_factor": round(scale_factor, 6),
            "intrinsics_source": src,
            "up_axis": scene.up_axis,
            "up_axis_tilt_deg": meta.get("up_axis_tilt_deg"),
            "up_axis_reliable": bool(meta.get("up_axis_reliable", True)),
            "n_objects": len(scene.nodes),
            "n_low_point_objects": len(thin),
            "n_edges_precomputed": len(scene.edges),
        },
        caveats,
    )


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------


@tool("describe_scene")
def describe_scene(
    ctx,
    scene_id: str | None = None,
    detail: str = "full",
) -> ToolResult:
    """把当前场景导出成结构化三维场景描述（物体清单 + 米制尺寸 + 两两几何关系）。

    `detail="full"`  逐对关系全部展开（受 `DEFAULT_MAX_PAIRS` 约束，超限会声明截断）；
    `detail="brief"` 只给物体清单 + 关系计数 + 每个物体的最近邻 —— O(n) 而非 O(n²)。

    这是**唯一**一个返回值不是「某道题的答案」的工具：它返回的是场景本身。
    程序合成模型可以把它当成观察入口（省掉多次 `list_objects` 的往返），
    也可以把它整个落盘当作答辩时的交付物。
    """
    if detail not in ("full", "brief"):
        # 值域写错属于「程序写错了」，必须冒泡成 ToolArgumentError 而不是被翻成
        # 工具失败 —— 否则失败诊断会把责任算到工具头上（见 registry 的异常策略）。
        raise ToolArgumentError(f"detail 只能是 'full' 或 'brief'，收到 {detail!r}")

    scene = need_scene(ctx, scene_id)
    nodes = list(scene.nodes)
    quality, caveats = _quality(scene)

    report: dict[str, Any] = {
        "scene_id": scene.scene_id,
        "image_id": scene.image_id,
        "objects": [_object_entry(n) for n in nodes],
        "quality": quality,
    }

    pairs = _all_pairs(nodes)
    n_pairs = len(pairs)

    if detail == "brief":
        report["relation_summary"] = _relation_summary(scene)
        report["relations"] = []
        report["truncated"] = False
    else:
        truncated = n_pairs > DEFAULT_MAX_PAIRS
        used = pairs[:DEFAULT_MAX_PAIRS] if truncated else pairs
        report["relations"] = _relation_entries(scene, used)
        # 机器可读的截断标志。caveat 是给人看的散文，**不要**让下游去匹配中文来判断
        # 报告是否完整 —— 那种耦合在改文案时会静默失效。
        report["truncated"] = truncated
        if truncated:
            # 明确声明，不静默。截断会让「关系边一致率」这一指标偏乐观，
            # 必须让下游知道它算的不是全集。
            caveats.append(
                f"关系被截断：场景 {n_pairs} 对 > 上限 {DEFAULT_MAX_PAIRS}，"
                f"只展开了前 {DEFAULT_MAX_PAIRS} 对（按物体原序，i<j）。"
                "需要全集请提高 max_pairs 或改用 detail='brief'。"
            )
        report["relation_summary"] = {
            "n_pairs_total": n_pairs,
            "n_pairs_reported": len(used),
        }

    report["caveats"] = caveats

    return ToolResult.success(
        value=report,
        evidence={
            "scene_id": scene.scene_id,
            "detail": detail,
            "n_objects": len(nodes),
            "n_pairs_total": n_pairs,
            "method": "geometry_v1",
            "tools_version": REPORT_TOOL_VERSION,
            # 报告的「证据」是它自己的构造方式：节点数、对数、方法与版本。
            # 没有这一步，落盘的报告无法回答「它是哪一版工具算出来的」。
            "label_counts": scene.label_counts(),
        },
    )


#: 模板化摘要里的方位词表。**不引入 LLM 也能生成可读句子** ——
#: 这正是「空间关系由几何算出、不由 LLM 判断」这条原则在输出侧的体现：
#: 句子是几何结论的渲染，不是几何结论的来源。
_CN_REL = {
    "left_of": "左侧",
    "right_of": "右侧",
    "front_of": "前方",
    "behind": "后方",
    "above": "上方",
    "below": "下方",
    "on": "上面",
    "inside": "内部",
}


def _template_summary(report: dict[str, Any]) -> str:
    """确定性中文摘要。同一份 report 永远产出同一段文字。"""
    objs = report["objects"]
    quality = report["quality"]
    lines: list[str] = []

    counts: dict[str, int] = {}
    for o in objs:
        counts[o["label"]] = counts.get(o["label"], 0) + 1
    inventory = "、".join(f"{k} × {v}" for k, v in sorted(counts.items()))
    lines.append(f"场景 {report['scene_id']} 共 {len(objs)} 个物体：{inventory or '（空）'}。")

    # 最近邻：一句「谁离谁最近」比一整张关系表好读，而信息量损失极小。
    nn = report.get("relation_summary", {}).get("nearest_neighbour") or {}
    if nn:
        parts = [
            f"{oid}→{d['object_id']} {d['distance_m']:.2f} m"
            for oid, d in sorted(nn.items(), key=lambda kv: kv[1]["distance_m"])
        ][:3]
        lines.append("最紧凑的三组相邻关系：" + "；".join(parts) + "。")

    # 只陈述**为真**的方位关系。列出全部 false 会把摘要变成噪声。
    rels = report.get("relations") or []
    directional: dict[str, list[str]] = {}
    for r in rels:
        if r["type"] in _CN_REL and r.get("value") is True:
            directional.setdefault(r["type"], []).append(f"{r['a']} 在 {r['b']} 的{_CN_REL[r['type']]}")
    for rel_type in _CN_REL:
        if rel_type in directional:
            lines.append(f"方位（{_CN_REL[rel_type]}）：" + "；".join(directional[rel_type][:4]) + "。")

    if quality.get("scale_calibrated"):
        lines.append("尺度已校正，米制数值可直接引用。")
    else:
        lines.append("**尺度未校正**：相对关系可信，绝对米数不可引用。")
    for c in report.get("caveats", []):
        if "尺度未校正" in c:
            continue
        lines.append(f"注意：{c}")
    return "\n".join(lines)


@tool("summarize_scene")
def summarize_scene(
    ctx,
    scene_id: str | None = None,
    use_llm: bool = False,
) -> ToolResult:
    """把场景报告渲染成一段自然语言。

    `use_llm=False`（默认）走**确定性模板** —— 同一场景逐字节可复现，评测可 diff；
    `use_llm=True` 走 `ctx.flags["summarizer"]`（一个 `dict -> str` 的可调用对象）。

    为什么把 LLM 做成开关而不是默认：报告会进指标比较，而 LLM 输出不可复现 ——
    默认走 LLM 会让「描述质量」这个指标混入采样噪声。见模块 docstring 的偏离说明。
    """
    report_res = describe_scene(ctx, scene_id, detail="brief")
    if not report_res.ok:
        return report_res

    report = report_res.value
    # detail="brief" 没有逐对关系，而模板里要陈述方位 —— 所以这里补算一次。
    # 补算只发生在**被显式要求生成摘要**时，不影响 describe_scene 的返回体积。
    full_res = describe_scene(ctx, scene_id, detail="full")
    if full_res.ok:
        report = {**report, "relations": full_res.value.get("relations", [])}

    if not use_llm:
        return ToolResult.success(
            value=_template_summary(report),
            evidence={
                "scene_id": report["scene_id"],
                "mode": "template",
                "deterministic": True,
                "n_objects": len(report["objects"]),
            },
        )

    summarizer = ctx.flags.get("summarizer")
    if not callable(summarizer):
        # 这就是消融开关：不注入 summarizer 就等于「关掉自然语言渲染」，
        # 而「关掉视觉语义」走的是 ctx.vlm=None —— 两件事都只需一次配置（§13.3(7)）。
        return ToolResult.disabled(
            "summarize_scene",
            "summarizer",
            hint='要启用请注入 ToolContext(flags={"summarizer": <Callable[[dict], str]>})；'
                 "不注入时 use_llm=False 的确定性模板仍然可用。",
        )

    try:
        text = summarizer(report)
    except Exception as exc:  # noqa: BLE001 —— 外部模型的问题，必须变成可恢复的结果
        return ToolResult.failure(
            ErrorCode.DEGENERATE,
            f"summarizer 调用失败：{type(exc).__name__}: {exc}",
            context={
                "hint": "summarizer 必须是 dict -> str；检查它是否需要额外依赖或网络。",
                "recovery_note": "可先用 use_llm=False 拿到确定性模板版本，不要因此中断整条流水线。",
            },
            tool="summarize_scene",
        )

    return ToolResult.success(
        value=str(text),
        evidence={
            "scene_id": report["scene_id"],
            "mode": "llm",
            "deterministic": False,
            "n_objects": len(report["objects"]),
        },
    )


# ----------------------------------------------------------------------------
# 失败诊断
# ----------------------------------------------------------------------------

#: 失败环节。顺序即**归因优先级** —— 尺度问题会让尺寸/距离集体越界，
#: 所以它必须先被检查，否则后面那些「尺寸不合理」的发现全是它的症状。
_STAGES = ("尺度", "检测", "工具", "程序")


@tool("diagnose_failure")
def diagnose_failure(
    ctx,
    scene_id: str | None = None,
    question_id: str | None = None,
) -> ToolResult:
    """定位失败环节：**尺度 / 检测 / 工具 / 程序**，按归因优先级返回有序发现。

    判据全部来自已经存在的记录，不重跑任何模型：
      • 尺度 —— `build_meta.intrinsics_source` / 视场可信性 / `scale_factor`；
      • 检测 —— 低分节点、走 bbox_fallback 的质心、空场景、退化的三维尺寸；
      • 工具 —— `ctx.trace` 里 `ok=False` 的结果，按错误码聚合（错误码本身就是归因）；
      • 程序 —— `ctx.flags["program_error"]`。

    为什么把「程序」单列：`ToolArgumentError` 是刻意**不被**工具层捕获的
    （见 registry 的异常策略）—— 参数值域写错是模型写错了程序，不是工具坏了。
    如果把它算进「工具失败」，工具调用成功率这个指标就会说谎。

    `question_id` 只作为标签写进 evidence（trace 本身是每题一清空的，见
    `ToolContext.reset_trace`）—— 若 `ctx.flags["questions"]` 提供了题集，则一并回填题干。
    """
    scene = need_scene(ctx, scene_id)
    meta = dict(scene.build_meta)
    findings: list[dict[str, Any]] = []

    # ---- 1. 尺度（优先级最高：它是根因，其余多为症状）------------------------
    src = meta.get("intrinsics_source", "unknown")
    if src == "predicted":
        findings.append(
            {
                "stage": "尺度",
                "severity": 10.0,
                "reason": "内参来自模型预测（intrinsics_source=predicted）",
                "detail": "横向米制尺度不可信；实测三维误差中位数 1.943 m vs 传入 GT 的 0.267 m。",
                "fix": "传入已知内参（BuildConfig.known_intrinsics，或 --intrinsics exif）。",
            }
        )
    if not _is_scale_calibrated(scene):
        findings.append(
            {
                "stage": "尺度",
                "severity": 6.0,
                "reason": "尺度未校正（scale_factor=1.0）",
                "detail": "绝对米数含未知比例因子；相对关系不受影响。",
                "fix": "用场景内已知尺寸物体或数据集 GT 做一次 calibrate_scale。",
            }
        )
    if not meta.get("up_axis_reliable", True):
        findings.append(
            {
                "stage": "尺度",
                "severity": 7.0,
                "reason": f"重力方向不可靠（up_axis={scene.up_axis}）",
                "detail": f"reason={meta.get('up_axis_reason')} —— `above`/`below` 可能整体翻转。",
                "fix": "补多视角或显式给出重力方向；单图下该量本质上只能估。",
            }
        )

    # ---- 2. 检测 ------------------------------------------------------------
    if not scene.nodes:
        findings.append(
            {
                "stage": "检测",
                "severity": 9.0,
                "reason": "场景里没有任何物体",
                "detail": "检测或提示词阶段就已经失败。",
                "fix": "换查询词 / 放宽检测阈值后重建场景图；若图里确实无目标物，应走 abstain。",
            }
        )

    fallback = [n.id for n in scene.nodes if n.centroid_source != "mask"]
    if fallback:
        findings.append(
            {
                "stage": "检测",
                "severity": 4.0,
                "reason": f"{len(fallback)} 个物体的质心走了降级路径（bbox_fallback）",
                "detail": f"涉及 {fallback[:6]}；与掩码质心实测差均值 83 mm / 最大 208 mm，"
                          f"而关系容差 50 mm。",
                "fix": "检查这些物体的 SAM2 掩码（脚本 scripts/inspect_scene.py 有掩码占框比）。",
            }
        )

    # 点数不足 —— DEGENERATE 的判据。这是**伪物体**最可靠的一个信号：
    # 低分检测框在掩码阶段只圈到几十个像素，于是点云质心完全不可信，
    # 而它不会报错，只会安静地进入场景图去参与左右/前后判断。
    thin = [(n.id, n.n_points) for n in scene.nodes
            if n.n_points is not None and n.n_points < _MIN_POINTS]
    if thin:
        findings.append(
            {
                "stage": "检测",
                "severity": 4.5,
                "reason": f"{len(thin)} 个物体的掩码点数不足 {_MIN_POINTS}",
                "detail": f"{sorted(thin, key=lambda t: t[1])[:6]}。"
                          f"实测分界：伪物体 809–965 点 vs 真实物体 16137–30660 点（17 倍空隙）。",
                "fix": "这些多半是伪物体（低分检测框）：考虑按 score 或点数过滤后重建，"
                       "并把它们标成 LOW_CONFIDENCE 而不是当作确定物体。",
            }
        )

    low = [n.id for n in scene.nodes if n.score < 0.3]
    if low:
        findings.append(
            {
                "stage": "检测",
                "severity": 3.0,
                "reason": f"{len(low)} 个物体的检测置信度低于 0.30",
                "detail": f"涉及 {low[:6]}；低分框很可能不是目标物。",
                "fix": "按 score 过滤后重建，或对低分项报 LOW_CONFIDENCE 而不是取整成确定值。",
            }
        )

    oversized = [
        (n.id, max(n.extent_3d)) for n in scene.nodes if max(n.extent_3d) > _SUSPICIOUS_EXTENT_M
    ]
    if oversized:
        worst = sorted(oversized, key=lambda t: -t[1])[:4]
        findings.append(
            {
                "stage": "检测",
                "severity": 5.0,
                "reason": f"{len(oversized)} 个物体的三维尺寸超过 {_SUSPICIOUS_EXTENT_M} m",
                "detail": f"最大几项 {[(i, round(v, 2)) for i, v in worst]}。",
                "fix": "尺寸越界多半是**内参或掩码外溢的症状**，先查上面「尺度」那几条 —— "
                       "不要直接改尺寸公式。",
            }
        )

    # ---- 3. 工具（按错误码聚合，错误码本身就是归因）--------------------------
    failures: dict[str, list[str]] = {}
    for row in ctx.trace:
        res = row.get("result") or {}
        if res.get("ok"):
            continue
        code = ((res.get("error") or {}).get("code")) or "UNKNOWN"
        failures.setdefault(code, []).append(str(row.get("tool")))
    for code, tools in sorted(failures.items()):
        findings.append(
            {
                "stage": "工具",
                "severity": 2.0,
                "reason": f"trace 里有 {len(tools)} 次工具失败，错误码 {code}",
                "detail": f"涉及工具 {sorted(set(tools))}。",
                "fix": "错误码自带 recovery（见 tools.result.ErrorCode）；"
                       "NOT_IN_SCENE 高发说明模型在编造 object_id。",
            }
        )

    # ---- 4. 程序 ------------------------------------------------------------
    prog_err = ctx.flags.get("program_error")
    if prog_err:
        findings.append(
            {
                "stage": "程序",
                "severity": 2.5,
                "reason": "程序执行抛错（非工具失败）",
                "detail": str(prog_err)[:400],
                "fix": "参数值域/语法问题属于模型侧：把 error 回灌给模型重写程序，"
                       "不要计入工具调用成功率。",
            }
        )

    question_text = None
    questions = ctx.flags.get("questions")
    if question_id and isinstance(questions, dict):
        question_text = questions.get(question_id)

    # 主环节 = 严重度最高的那条的 stage。没有发现时如实说「未发现」，
    # 而不是硬挑一个 stage 出来 —— 后者会让诊断本身变成噪声。
    primary = max(findings, key=lambda f: f["severity"])["stage"] if findings else None
    findings.sort(key=lambda f: -f["severity"])

    return ToolResult.success(
        value={
            "scene_id": scene.scene_id,
            "question_id": question_id,
            "question": question_text,
            "stage": primary,
            "findings": findings,
        },
        evidence={
            "scene_id": scene.scene_id,
            "stage_order": list(_STAGES),
            "n_findings": len(findings),
            "stages_seen": sorted({f["stage"] for f in findings}),
            "trace_len": len(ctx.trace),
            "question_resolved": question_text is not None,
        },
    )


# ----------------------------------------------------------------------------
# 反事实
# ----------------------------------------------------------------------------


@tool("counterfactual")
def counterfactual(
    ctx,
    scene_id: str | None = None,
    remove: Iterable[str] | None = None,
) -> ToolResult:
    """移走若干物体后重算场景关系 —— **纯图操作，不重跑任何视觉模型**。

    这是 L5 性价比最高的演示项：场景图建好之后，「若移走沙发，哪些关系消失了」
    只需要在内存里做一次子集运算。它直接证明**中间表示本身有价值**，
    而不只是工程脚手架 —— 答辩时比任何并行数字都直观（§11.1）。

    返回值里的 `diff` 是重点：`removed_relations` / `kept_relations` 的差集说明
    「移走 A 会让 B、C 的关系全部失效」。而「移走沙发后哪把椅子最靠近门」这类问题
    由**程序**组合达成：`counterfactual(remove=["sofa_1"])` 之后再对返回的场景图调
    `find_nearest` —— 这正是「动作空间是 Python 程序而不是 JSON tool call」的实例。

    `remove` 里的 id 必须真实存在于场景中：编造的 id 直接 `NOT_IN_SCENE`（幻觉捕获点）。
    """
    scene = need_scene(ctx, scene_id)
    ids = list(remove or [])

    unknown = [i for i in ids if not scene.has(i)]
    if unknown:
        return ToolResult.not_in_scene("counterfactual", unknown[0], known=scene.ids())

    unique_removed = sorted(set(ids))
    kept_nodes = [n for n in scene.nodes if n.id not in set(unique_removed)]

    # 原场景与剪枝后场景都按**同一套关系定义**重算，差集才有意义。
    # 直接复用 scene.edges 是不行的：那里可能只存了构建时选定的关系子集。
    before_pairs = _all_pairs(list(scene.nodes))
    after_pairs = _all_pairs(kept_nodes)

    def _key(e: dict[str, Any]) -> tuple[str, str, str]:
        return (e["a"], e["b"], e["type"])

    before = _relation_entries(scene, before_pairs)
    after = _relation_entries(scene, after_pairs)

    before_map = {_key(e): e for e in before}
    after_map = {_key(e): e for e in after}

    # 「删除」= 原本存在的键没了（必然包含所有涉及被移除物体的对）；
    # 「翻转」= 键还在但布尔值变了 —— 这是反事实里最有意思的一类，
    # 因为它说明某个关系**依赖于第三个物体之外的东西**，而不是纯粹的局部量。
    removed = [before_map[k] for k in before_map.keys() - after_map.keys()]
    flipped = [
        {"before": before_map[k], "after": after_map[k]}
        for k in before_map.keys() & after_map.keys()
        if before_map[k].get("value") != after_map[k].get("value")
        and before_map[k]["type"] != "distance"
    ]

    report_res = describe_scene(ctx, scene_id, detail="brief")
    report = dict(report_res.value) if report_res.ok else {}
    report["objects"] = [_object_entry(n) for n in kept_nodes]
    # ⚠ 必须用 kept_nodes **重算**摘要，不能沿用 describe_scene 的那一份：
    # 后者是在全场景上算的，于是「移走沙发后谁离门最近」会返回沙发自己 ——
    # 一个看起来完全合理、实际已经被移走的答案。这类错最危险的地方在于它不报错。
    report["relation_summary"] = _relation_summary(scene, kept_nodes)
    report["quality"] = {
        **report.get("quality", {}),
        "n_objects": len(kept_nodes),
        "counterfactual_removed": unique_removed,
    }
    report["caveats"] = [
        *report.get("caveats", []),
        f"这是反事实视图（已移除 {unique_removed or '无'}），不是真实观测结果。",
    ]

    return ToolResult.success(
        value={
            **report,
            "diff": {
                "removed_objects": unique_removed,
                "n_relations_before": len(before),
                "n_relations_after": len(after),
                "removed_relations": removed,
                "flipped_relations": flipped,
            },
        },
        evidence={
            "scene_id": scene.scene_id,
            "removed": unique_removed,
            "n_objects_before": len(scene.nodes),
            "n_objects_after": len(kept_nodes),
            # 明确记录「没有重跑视觉模型」—— 这是本工具的核心主张，必须可核对。
            "models_rerun": 0,
            "method": "geometry_v1",
        },
    )


# ----------------------------------------------------------------------------
# 未实现：answer_with_evidence
# ----------------------------------------------------------------------------
#
# 文档 §11.1 的第四个工具 `answer_with_evidence(scene_id, question_id) -> {answer, evidence[]}`
# 需要**题集**才能把 question_id 解析成一道题，而题集属于 Phase 2（§18）。
# 现在实现它只有两种写法，两种都不诚实：
#   • 自己造一份假题集 —— 会在报告里留下一个「已实现」的假证据；
#   • 接受自由文本 question 再调 LLM —— 那就把「答案必须来自几何」这条地基拆了。
# 因此明确挂起，并把依赖写在这里：**Phase 2 的 questions.json 一到，本函数就是一层薄封装**
# （按 question_id 取题 → 用 RELATIONS 分发 → 把 verdict.metric 原样塞进 evidence）。
#
# 注意 `diagnose_failure` 已经能接受 question_id（作为 trace 标签），
# 所以这一条挂起不会阻塞失败诊断面板的落地。
