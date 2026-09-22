"""把场景图与实验产物导出成 `demo/` 前端能直接吃的静态数据。

前端的硬约束是「不联网也能演」，所以导出必须**一次性、可重复、产物自包含**：
跑完这条脚本，`demo/` 目录可以整个拷到 U 盘、或直接静态托管，
里面已经有 three.js 和全部图片 —— 拔网线照演。

导出的四件东西
==============
1. `demo/data/index.json`         —— 场景清单（前端启动时读它）
2. `demo/data/<id>.scene.json`    —— 场景图（`bbox_3d` / `centroid_3d` / `extent_3d` … 前端画三维就用它）
3. `demo/data/<id>.report.json`   —— 场景报告**摘要**（不搬 149 KB 的全量关系矩阵：进前端既慢又没人看）
4. `demo/assets/<id>/`            —— 原图 + 掩码 PNG

为什么不让前端直接读 `dataset/`：那样 HTML 一离开仓库就废（相对路径全断）。
导出多花一次拷贝，换来「一个目录 = 一个可交付物」，值得。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_graph.store import load_scene  # noqa: E402

#: 场景目录里出现这些后缀就当候选原图。
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

#: `build_meta` 进前端时只带这些键。
#:
#: 白名单而不是黑名单：`build_meta` 会随 builder 演进而长出新键（有些是内部调试用的大对象），
#: 黑名单意味着「新增一个键就悄悄变大」，白名单意味着「想给前端看就显式加一行」。
_META_KEYS = (
    "n_detections_raw", "n_detections_kept", "n_nodes", "n_edges", "n_dropped", "n_fallbacks",
    "mask_box_coverage_mean", "mask_box_coverage_min", "label_counts",
    "up_axis", "up_axis_tilt_deg", "up_axis_reliable", "up_axis_reason",
    "scale_calibrated", "depth_range_m", "image_hw", "timings_ms", "perception",
    # 2026-09-22 追加：加载/构建/落盘三项聚合 + 口径（`scope`）。
    # ⚠ 少了它会怎样：单场景文件里有 `timings_ms`（只有六段热态细分），
    #   而 index 的场景条目连那六段都拿不到 —— 前端无法回答「建图要多久」。
    "timing",
)

#: `build_meta["config"]` 里进前端的键（这几个解释「数字是怎么算出来的」）。
_CONFIG_KEYS = ("prompt", "box_threshold", "text_threshold", "tol_m", "near_m", "relation_policy")

#: 一批 run 里进前端的字段。
#:
#: ⚠ 白名单是**刻意的**：`llm_env` / `backend` 里有环境变量映射与端点配置，
#: 它们属于运行环境、不属于演示数据。前端要展示"这一臂用了什么"，看 `switches` 就够了。
_RUN_KEYS = (
    "question", "answer", "answer_type", "abstained", "evidence", "status", "target_ids",
    "verdict", "plan", "program", "program_fenced", "trace", "tool_calls", "attempts",
    "elapsed_s", "usage", "stages", "failure", "render", "switches", "scene_id", "tools_version",
    # 2026-09-22 追加：让前端能看出「上面那段 program 到底跑过没有」。
    # ⚠ 少了它会怎样：末轮 static_check 没过时，前端把一段**从未执行**的程序
    #   和一段属于别的程序的 trace 并排显示，两者看起来是配套的。
    "program_matches_trace", "executed_attempt",
)


def _fwd(path: Path) -> str:
    return path.as_posix()


def _find_image(scene: Any, scene_path: Path) -> Path | None:
    """按 4 级回退找原图 —— 与 `scripts/run_agent.py` 的 `_find_image` 同序。

    第 3 级（解析 `build_log.txt`）不是凑数：`living_room` 是在 builder 开始写
    `build_meta["image_path"]` **之前**建的，只能靠日志找回原图。
    """
    meta = dict(getattr(scene, "build_meta", None) or {})
    override = meta.get("image_path")
    if override and Path(override).exists():
        return Path(override)
    for cand in sorted(scene_path.parent.iterdir()):
        if cand.is_file() and cand.suffix.lower() in _IMAGE_SUFFIXES:
            return cand
    log = scene_path.parent / "build_log.txt"
    if log.exists():
        for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().startswith("image="):
                cand = Path(line.split("=", 1)[1].strip())
                if cand.exists():
                    return cand
    return None


def _node_payload(node: Any, scene_id: str) -> dict[str, Any]:
    mask_url = None
    if getattr(node, "mask_ref", None):
        mask_url = f"assets/{scene_id}/masks/{Path(node.mask_ref).name}"
    bbox_3d = getattr(node, "bbox_3d", None)
    return {
        "id": node.id,
        "label": node.label,
        "score": round(float(node.score), 4),
        "bbox_2d": None if node.bbox_2d is None else [round(float(v), 2) for v in node.bbox_2d],
        "centroid_3d": [round(float(v), 4) for v in node.centroid_3d],
        "extent_3d": [round(float(v), 4) for v in node.extent_3d],
        "bbox_3d": None if bbox_3d is None else {
            "min": [round(float(v), 4) for v in bbox_3d.min],
            "max": [round(float(v), 4) for v in bbox_3d.max],
        },
        "mask_url": mask_url,
        "n_points": getattr(node, "n_points", None),
        "centroid_source": getattr(node, "centroid_source", "mask"),
        "attributes": dict(getattr(node, "attributes", {}) or {}),
        "frame": getattr(node, "frame", "camera"),
    }


def _meta_payload(meta: dict[str, Any]) -> dict[str, Any]:
    out = {k: meta[k] for k in _META_KEYS if k in meta}
    cfg = meta.get("config") or {}
    out["config"] = {k: cfg[k] for k in _CONFIG_KEYS if k in cfg}
    return out


#: 旧场景（2026-09-22 之前建的）没有 `build_meta["timing"]` 时的 `scope` 文案。
#: ⚠ 它**不是**占位符：它如实说明「哪几项是真的缺测」，而不是把它们读成 0。
_LEGACY_TIMING_SCOPE = (
    "旧场景：本次改动之前建的。只有 builder 六段（热态细分）；"
    "model_load_ms / save_ms / total_wall_ms 当时未测，故为 null —— 是「没测到」，不是「测到 0」。"
)


def _timing_payload(meta: dict[str, Any]) -> dict[str, Any] | None:
    """把「建图要多久」投影进 `index.json` —— **纯函数，零 GPU 可单测**。

    ⚠ **`scope` 必须跟着数字一起出去**。这几个数不是同一个东西：
    `stages_ms` 是**热态**的 builder 内部六段（不含模型加载、不含落盘），
    `model_load_ms` 只在**进程内第一张图**上非零，`save_ms` 不含 scene.json 自身。
    只把数字摆给前端、不带口径，读者一定会拿 `stages_ms` 当"建图耗时" ——
    实测 `living_room` 那会**低报约 5 倍**（3.4 s vs 16 s 级）。

    旧场景返回 `stages_ms` + 三个 null + 一句说明；完全没有计时信息的场景返回 `None`
    （让它整个缺席，而不是编一个全 0 的块 —— 后者会被读成"建图不花时间"）。
    """
    stages = meta.get("timings_ms")
    block = meta.get("timing") or {}
    if not stages and not block:
        return None
    return {
        "stages_ms": stages,
        "model_load_ms": block.get("model_load_ms"),
        "build_wall_ms": block.get("build_wall_ms"),
        "save_ms": block.get("save_ms"),
        "total_wall_ms": block.get("total_wall_ms"),
        "scope": block.get("scope") or _LEGACY_TIMING_SCOPE,
    }


def export_scene(scene_id: str, scenes_root: Path) -> dict[str, Any]:
    scene_path = scenes_root / scene_id / "scene.json"
    scene = load_scene(scene_path)
    meta = dict(getattr(scene, "build_meta", None) or {})

    image = _find_image(scene, scene_path)
    image_url = None
    image_hw = None
    if image is not None:
        image_url = f"assets/{scene_id}/{image.name}"
        raw = meta.get("image_hw")
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            # ⚠ builder 落盘的 `image_hw` 是 [高, 宽]，而前端统一用 [宽, 高]（与图片标签一致）。
            # 弄反了不会报错，只会让 2D 叠加的宽高比错掉 —— 所以只在这里翻一次。
            image_hw = [int(raw[1]), int(raw[0])]
        else:
            image_hw = None

    payload = {
        "scene_id": scene.scene_id,
        "image_id": scene.image_id,
        "up_axis": scene.up_axis,
        "scale_factor": float(scene.scale_factor),
        "camera_intrinsics": scene.camera_intrinsics,
        "image_hw": image_hw,
        "image_url": image_url,
        "nodes": [_node_payload(n, scene_id) for n in scene.nodes],
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "relation": e.relation,
                "value": e.value,
                "metric": {k: round(float(v), 4) for k, v in (e.metric or {}).items()},
            }
            for e in scene.edges
        ],
        "build_meta": _meta_payload(meta),
    }
    return payload


def export_report(scene_id: str, scenes_root: Path) -> dict[str, Any] | None:
    """`scene_report.json` → 摘要。

    **不带全量 `relations`**：那是 36 个有序对的多关系矩阵，149 KB 里绝大部分是它，
    而场景图（第 2 件产物）里的 133 条边已经覆盖了同一批事实的紧凑版本。
    """
    path = scenes_root / scene_id / "scene_report.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    report = raw.get("report") or {}
    cf = raw.get("counterfactual") or {}
    return {
        "scene_id": scene_id,
        "summary": raw.get("summary"),
        "quality": report.get("quality"),
        "relation_summary": report.get("relation_summary"),
        "caveats": report.get("caveats"),
        "truncated": report.get("truncated"),
        "objects": report.get("objects"),
        "provenance": raw.get("provenance"),
        "diagnosis": raw.get("diagnosis"),
        "counterfactual": {"diff": cf.get("diff"), "relation_summary": cf.get("relation_summary")},
    }


def copy_assets(scene_id: str, scenes_root: Path, assets_root: Path) -> dict[str, int]:
    src_dir = scenes_root / scene_id
    dst_dir = assets_root / scene_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    n_img = 0
    n_mask = 0

    scene_path = src_dir / "scene.json"
    if scene_path.exists():
        scene = load_scene(scene_path)
        image = _find_image(scene, scene_path)
        if image is not None:
            shutil.copy2(image, dst_dir / image.name)
            n_img = 1

    src_masks = src_dir / "masks"
    if src_masks.is_dir():
        dst_masks = dst_dir / "masks"
        dst_masks.mkdir(parents=True, exist_ok=True)
        for p in sorted(src_masks.glob("*.png")):
            shutil.copy2(p, dst_masks / p.name)
            n_mask += 1
    return {"images": n_img, "masks": n_mask}


def export_runs(runs_dir: Path, limit: int) -> dict[str, Any]:
    """把最近的批处理产物裁成前端能回放的形状。"""
    batches: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return {"batches": batches}

    files = sorted(runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    for path in sorted(files, key=lambda p: p.stat().st_mtime):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(raw, dict) or "runs" not in raw:
            continue
        runs = []
        for run in raw.get("runs") or []:
            item = {k: run.get(k) for k in _RUN_KEYS if k in run}
            plan = item.get("plan")
            if isinstance(plan, dict):
                # `plan.reply` 里塞着整段模型回复和它的 usage，对前端无用。
                item["plan"] = {k: v for k, v in plan.items() if k != "reply"}
            runs.append(item)
        batches.append({
            "file": path.name,
            "mtime": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "scene_id": raw.get("scene_id"),
            "scene_hint": raw.get("scene_hint"),
            "switches": raw.get("switches"),
            "toolset": raw.get("toolset"),
            "tools_version": raw.get("tools_version"),
            "usage": raw.get("usage"),
            "summary": raw.get("summary"),
            "vlm": raw.get("vlm"),
            "runs": runs,
        })
    return {"batches": batches}


def discover_scenes(scenes_root: Path) -> list[str]:
    if not scenes_root.is_dir():
        return []
    return sorted(
        p.parent.name for p in scenes_root.glob("*/scene.json")
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="导出 demo/ 前端的静态数据（场景图 + 报告摘要 + 图片 + 历史回放）",
    )
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="要导出的场景 id；默认自动发现 dataset/scenes/*/scene.json")
    ap.add_argument("--scenes-root", default=str(ROOT / "dataset" / "scenes"))
    ap.add_argument("--runs-dir", default=str(ROOT / "logs" / "agent_runs"))
    ap.add_argument("--out", default=str(ROOT / "demo"))
    ap.add_argument("--runs-limit", type=int, default=8, help="最多回放几个历史批次")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenes_root = Path(args.scenes_root)
    out = Path(args.out)
    data_dir = out / "data"
    assets_root = out / "assets"
    data_dir.mkdir(parents=True, exist_ok=True)
    assets_root.mkdir(parents=True, exist_ok=True)

    scene_ids = args.scenes or discover_scenes(scenes_root)
    if not scene_ids:
        print(f"没有在 {scenes_root} 下找到任何 scene.json")
        return 1

    index: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        scene_path = scenes_root / scene_id / "scene.json"
        if not scene_path.exists():
            print(f"跳过 {scene_id}：{scene_path} 不存在")
            continue
        payload = export_scene(scene_id, scenes_root)
        (data_dir / f"{scene_id}.scene.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        report = export_report(scene_id, scenes_root)
        if report is not None:
            (data_dir / f"{scene_id}.report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        counts = copy_assets(scene_id, scenes_root, assets_root)
        meta = payload["build_meta"]
        index.append({
            "scene_id": scene_id,
            "image_id": payload["image_id"],
            "n_nodes": len(payload["nodes"]),
            "n_edges": len(payload["edges"]),
            "image_url": payload["image_url"],
            "image_hw": payload["image_hw"],
            "has_report": report is not None,
            "label_counts": meta.get("label_counts"),
            "up_axis": payload["up_axis"],
            "up_axis_reliable": meta.get("up_axis_reliable"),
            "up_axis_tilt_deg": meta.get("up_axis_tilt_deg"),
            "scale_calibrated": meta.get("scale_calibrated"),
            "timing": _timing_payload(meta),
            "assets": counts,
        })
        print(f"场景 {scene_id}: {len(payload['nodes'])} 节点 / {len(payload['edges'])} 边 / "
              f"图 {counts['images']} 掩码 {counts['masks']}")

    runs = export_runs(Path(args.runs_dir), int(args.runs_limit))
    (data_dir / "runs.json").write_text(
        json.dumps(runs, ensure_ascii=False, indent=1), encoding="utf-8")

    (data_dir / "index.json").write_text(
        json.dumps({
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenes": index,
            "n_run_batches": len(runs["batches"]),
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"回放批次：{len(runs['batches'])}")
    print(f"导出完成 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
