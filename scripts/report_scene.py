#!/usr/bin/env python
r"""把场景导出成 **L5 第一类输出**：`scene_report.json` + `scene_report.md`。

    D:\3D_Spatial_Agent\venvs\vision\Scripts\python.exe scripts\report_scene.py `
        --scene D:\3D_Spatial_Agent\dataset\scenes\living_room_gt --auto-counterfactual

为什么要有这个脚本（而不是直接在 REPL 里调工具）：
    §11.1 说 L5 的价值在于让中间表示**成为可保存、可复查、可对比的交付物**。
    「可保存」需要一个确定的落盘入口 —— 否则「报告」只是某个会话里闪过的一段输出，
    既没法进版本库，也没法和上一版 diff。

产出两个文件，理由是它们服务两类读者：
    `scene_report.json` —— 给指标与下游程序。结构与 `describe_scene` 的返回值**逐字段相同**，
                            所以「工具返回什么、落盘就是什么」，不存在第二套口径。
    `scene_report.md`   —— 给人。物体表 + 自然语言摘要 + 失败诊断 + 反事实 diff。

零 GPU、零联网：L5 只对已有 `SceneGraph` 做派生计算（§11.1 的「零额外模型成本」）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_graph.store import load_scene  # noqa: E402
from tools import load_tools  # noqa: E402
from tools.registry import ToolContext  # noqa: E402
from tools.scene_report import (  # noqa: E402
    counterfactual,
    describe_scene,
    diagnose_failure,
    summarize_scene,
)


def _md_objects(report: dict) -> list[str]:
    lines = [
        "| id | label | 质心 (x, y, z) m | 尺寸 w×h×l m | 置信度 | 质心来源 |",
        "|---|---|---|---|---|---|",
    ]
    for o in report["objects"]:
        c = ", ".join(f"{v:+.3f}" for v in o["centroid_m"])
        e = o["extent_m"]
        lines.append(
            f"| `{o['id']}` | {o['label']} | {c} | "
            f"{e['w']:.2f}×{e['h']:.2f}×{e['l']:.2f} | {o['confidence']:.3f} | "
            f"{o['centroid_source']} |"
        )
    return lines


def _md_diagnosis(diag: dict) -> list[str]:
    if not diag["findings"]:
        return ["未发现异常。"]
    lines = [f"主环节：**{diag['stage']}**（共 {len(diag['findings'])} 条发现，按严重度降序）", ""]
    for i, f in enumerate(diag["findings"], 1):
        lines.append(f"{i}. **[{f['stage']}]** {f['reason']}")
        lines.append(f"   - 依据：{f['detail']}")
        lines.append(f"   - 处理：{f['fix']}")
    return lines


def _md_counterfactual(cf: dict) -> list[str]:
    diff = cf["diff"]
    lines = [
        f"移除 `{', '.join(diff['removed_objects'])}`：",
        "",
        f"- 物体 {cf['quality']['n_objects'] + len(diff['removed_objects'])} → {cf['quality']['n_objects']}",
        f"- 关系 {diff['n_relations_before']} → {diff['n_relations_after']}"
        f"（消失 {len(diff['removed_relations'])} 条，翻转 {len(diff['flipped_relations'])} 条）",
        f"- 重跑视觉模型 **{cf.get('models_rerun', 0)}** 次（纯图操作）",
        "",
    ]
    nn = cf["relation_summary"]["nearest_neighbour"]
    if nn:
        lines.append("移除后各物体的最近邻：")
        lines.append("")
        for oid, d in sorted(nn.items(), key=lambda kv: kv[1]["distance_m"]):
            lines.append(f"- `{oid}` → `{d['object_id']}` {d['distance_m']:.3f} m")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="场景目录或 scene.json 路径")
    ap.add_argument("--out", default=None, help="输出目录，默认写到场景目录")
    ap.add_argument("--detail", default="full", choices=("full", "brief"))
    ap.add_argument("--remove", nargs="*", default=None,
                    help="反事实：移除这些 object_id 后重算（纯图操作）")
    ap.add_argument("--auto-counterfactual", action="store_true",
                    help="反事实：自动移除三维尺寸最大的物体")
    args = ap.parse_args()

    scene_path = Path(args.scene)
    scene = load_scene(scene_path)
    out_dir = Path(args.out) if args.out else (
        scene_path if scene_path.is_dir() else scene_path.parent
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = ToolContext(scene=scene)

    # ---- 1. 结构化报告（落盘的主产物）----------------------------------------
    desc = describe_scene(ctx, detail=args.detail)
    if not desc.ok:
        assert desc.error is not None
        sys.stderr.write(f"describe_scene 失败：{desc.error.code.value} {desc.error.message}\n")
        return 1
    report = desc.value

    # ---- 2. 自然语言摘要（确定性模板，不依赖 LLM）----------------------------
    summ = summarize_scene(ctx)

    # ---- 3. 失败诊断 ----------------------------------------------------------
    diag = diagnose_failure(ctx)

    # ---- 4. 反事实（可选）----------------------------------------------------
    remove = list(args.remove or [])
    if args.auto_counterfactual and not remove and scene.nodes:
        biggest = max(scene.nodes, key=lambda n: max(n.extent_3d))
        remove = [biggest.id]
    cf = counterfactual(ctx, remove=remove) if remove else None
    if cf is not None and not cf.ok and cf.error is not None:
        sys.stderr.write(f"counterfactual 失败：{cf.error.code.value} {cf.error.message}\n")
        cf = None

    # ---- 落盘 ----------------------------------------------------------------
    json_path = out_dir / "scene_report.json"
    payload = {
        # 与 describe_scene 的返回值逐字段相同 —— 不存在第二套口径。
        "report": report,
        "summary": summ.value if summ.ok else None,
        "diagnosis": diag.value if diag.ok else None,
        "counterfactual": cf.value if cf is not None else None,
        "provenance": {
            "scene_json": str(scene_path),
            "tools_version": desc.evidence["tools_version"],
            "method": desc.evidence["method"],
            "detail": args.detail,
            "n_objects": desc.evidence["n_objects"],
            "n_pairs_total": desc.evidence["n_pairs_total"],
            "gpu_used_mb": desc.meta.gpu_peak_mb if desc.meta else None,
            "latency_ms": {
                "describe_scene": round(desc.meta.latency_ms, 2) if desc.meta else None,
                "summarize_scene": round(summ.meta.latency_ms, 2) if summ.meta else None,
                "diagnose_failure": round(diag.meta.latency_ms, 2) if diag.meta else None,
                "counterfactual": round(cf.meta.latency_ms, 2) if cf and cf.meta else None,
            },
        },
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md: list[str] = [
        f"# 场景报告 — `{report['scene_id']}`",
        "",
        "> 由 `scripts/report_scene.py` 生成（L5 第一类输出）。",
        f"> 工具库 `{desc.evidence['tools_version']}` · 关系方法 `{desc.evidence['method']}` ·",
        f"> 物体 {desc.evidence['n_objects']} 个 · 关系对 {desc.evidence['n_pairs_total']} 对 ·",
        f"> **未使用 GPU**（L5 是零额外模型成本的一层）。",
        "",
        "## 1. 物体清单",
        "",
        *_md_objects(report),
        "",
        "## 2. 自然语言摘要",
        "",
        "```",
        (summ.value if summ.ok else "（摘要不可用）"),
        "```",
        "",
        "## 3. 可信度自述",
        "",
        f"- 尺度已校正：**{report['quality']['scale_calibrated']}**"
        f"（scale_factor={report['quality']['scale_factor']}）",
        f"- 内参来源：`{report['quality']['intrinsics_source']}`",
        f"- 重力方向：`{report['quality']['up_axis']}`"
        f"（tilt {report['quality']['up_axis_tilt_deg']}°，可靠={report['quality']['up_axis_reliable']}）",
        f"- 关系是否被截断：**{report['truncated']}**",
        "",
    ]
    if report["caveats"]:
        md.append("告诫：")
        md.append("")
        md.extend(f"- {c}" for c in report["caveats"])
        md.append("")
    md += ["## 4. 失败诊断", "", *_md_diagnosis(diag.value if diag.ok else {"findings": [], "stage": None}), ""]
    if cf is not None and cf.ok:
        md += ["## 5. 反事实（纯图操作）", "", *_md_counterfactual(cf.value), ""]
    md_path = out_dir / "scene_report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    # 控制台输出只做「文件写好了」的确认。真正的结论在文件里 ——
    # 这条约定来自本机金律②（PowerShell 的 stdout 不回传给 Agent）。
    print(f"[report_scene] wrote {json_path}  ({json_path.stat().st_size} bytes)")
    print(f"[report_scene] wrote {md_path}    ({md_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
