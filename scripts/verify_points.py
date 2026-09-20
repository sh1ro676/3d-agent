#!/usr/bin/env python
r"""核对一个场景的 `points.npy` 与 `scene.json` 是否**同源** —— 零 GPU、零模型。

    python scripts/verify_points.py --scene living_room
    python scripts/verify_points.py --all
    python scripts/verify_points.py --all --json out/points_audit.json

逐节点比对三件事，**全部精确相等**才算通过：

    点数    pts.shape[1] == node.n_points
    质心    centroid_of(pts) == node.centroid_3d
    尺寸    robust_extent(pts) == node.extent_3d

**为什么值得做成脚本、而不是一次性探针。** 这个检查是「点云级工具可信」的
前提：只要它成立，就保证「用落盘点云算出的朝向 / 体积 / 支撑面」与
「`scene.json` 里那几个数字」来自**同一份数据**。而这一环恰恰最容易悄悄坏掉、
又最难从结果里看出来 —— 坏了之后两组数字**都看起来合理**，只是不同。

它**不重跑任何模型**（只读 points.npy + masks/*.png + scene.json），
所以：秒级、可以进 CI、也可以在答辩现场当着评委跑一遍 —— 这正是把它从
临时探针提成正式脚本的理由，探针没法被引用，脚本可以。

返回码：0 = 全部通过；1 = 有场景存在不一致或点云缺失。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_graph.pointcloud import node_points, points_meta_summary  # noqa: E402
from scene_graph.store import (  # noqa: E402
    DEFAULT_SCENES_DIR,
    has_points,
    load_points,
    load_scene,
    scene_dir,
)
from vision.geometry import centroid_of, robust_extent  # noqa: E402


def _triple(v: Any) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


def verify_scene(sd: Path) -> dict[str, Any]:
    """核对一个场景，返回结构化结果（**不打印**，打印由调用方决定）。"""
    try:
        scene = load_scene(sd)
    except Exception as exc:                      # noqa: BLE001  坏文件要报出来
        return {"scene": sd.name, "status": "unreadable", "error": str(exc)}

    out: dict[str, Any] = {
        "scene": scene.scene_id,
        "path": str(sd),
        "n_nodes": len(scene.nodes),
    }

    if not has_points(sd):
        # 这是**正常状态**，不是错误：落盘能力是后加的，老场景本来就没有。
        # 但要显式区分于「点云存在却不一致」—— 两者的修法完全不同
        # （重跑建图 vs 查代码），混成一个 "fail" 会让人先去看代码。
        out["status"] = "no_points"
        return out

    cloud, meta = load_points(sd)
    out["points"] = {**points_meta_summary(meta), "n_bytes": (
        sd / "points.npy").stat().st_size}

    rows: list[dict[str, Any]] = []
    for node in scene.nodes:
        pts = node_points(node, sd, points_chw=cloud, meta=meta)
        got_n = int(pts.shape[1])

        row: dict[str, Any] = {
            "object_id": node.id,
            "centroid_source": node.centroid_source,
            "n_points_expected": node.n_points,
            "n_points_got": got_n,
        }

        if got_n == 0:
            # 掩码丢了 / 框退化。**不能**在这里就断言质心不一致 ——
            # 那会把病因说成症状，让人去查几何而不是查掩码文件。
            row["status"] = "empty"
            rows.append(row)
            continue

        centroid, _ = centroid_of(pts)
        extent, _, _, _ = robust_extent(pts)
        got_c, got_e = _triple(centroid), _triple(extent)

        row.update({
            "centroid_expected": list(node.centroid_3d),
            "centroid_got": list(got_c),
            "extent_expected": list(node.extent_3d),
            "extent_got": list(got_e),
            "max_centroid_diff_m": max(
                abs(got_c[i] - float(node.centroid_3d[i])) for i in range(3)),
            "max_extent_diff_m": max(
                abs(got_e[i] - float(node.extent_3d[i])) for i in range(3)),
        })
        row["n_points_ok"] = got_n == node.n_points
        row["centroid_exact"] = got_c == node.centroid_3d
        row["extent_exact"] = got_e == node.extent_3d
        row["status"] = (
            "ok" if (row["n_points_ok"] and row["centroid_exact"] and row["extent_exact"])
            else "mismatch"
        )
        rows.append(row)

    out["nodes"] = rows
    out["n_ok"] = sum(1 for r in rows if r["status"] == "ok")
    out["n_mismatch"] = sum(1 for r in rows if r["status"] == "mismatch")
    out["n_empty"] = sum(1 for r in rows if r["status"] == "empty")
    out["status"] = "ok" if (out["n_mismatch"] == 0 and out["n_empty"] == 0) else "failed"
    return out


def _print_report(res: dict[str, Any]) -> None:
    sid = res["scene"]
    st = res["status"]
    if st == "unreadable":
        print(f"  [{sid}] 读不出来：{res['error']}")
        return
    if st == "no_points":
        n = res["n_nodes"]
        print(f"  [{sid}] 没有 points.npy（老场景）—— {n} 个节点无法核对。"
              f"重跑 scripts/build_scene.py 即可补上。")
        return

    pm = res["points"]
    gh = pm.get("grid_hw") or [0, 0]
    mb = pm["n_bytes"] / 1024 / 1024
    src = pm.get("intrinsics_source") or "?"
    note = "已标定" if src == "provided" else "未标定（米制量只能当相对量）"
    print(f"  [{sid}] 点云 3×{gh[0]}×{gh[1]} {pm.get('dtype')}  {mb:.1f} MB  "
          f"内参来源={src} → {note}")

    mark = "  " if st == "ok" else "!!"
    print(f"{mark}    {res['n_ok']}/{res['n_nodes']} 节点逐位一致"
          + (f"（不一致 {res['n_mismatch']}，空点集 {res['n_empty']}）"
             if st != "ok" else ""))

    if st != "ok":
        for r in res["nodes"]:
            if r["status"] == "mismatch":
                print(f"      · {r['object_id']}: 点数 {r['n_points_got']}/"
                      f"{r['n_points_expected']}，"
                      f"质心最大差 {r['max_centroid_diff_m']:.3e} m，"
                      f"尺寸最大差 {r['max_extent_diff_m']:.3e} m")
            elif r["status"] == "empty":
                print(f"      · {r['object_id']}: 取不到点（掩码或框退化）")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", action="append", default=None,
                    help="场景 id，可重复。省略且未给 --all 时报错退出")
    ap.add_argument("--all", action="store_true", help="核对 root 下全部场景")
    ap.add_argument("--root", default=str(DEFAULT_SCENES_DIR))
    ap.add_argument("--json", default=None, help="同时写一份 JSON 报告")
    args = ap.parse_args(argv)

    if not args.scene and not args.all:
        ap.error("要么给 --scene <id>（可重复），要么给 --all")

    root = Path(args.root)
    if args.all:
        targets = sorted(p for p in root.iterdir() if (p / "scene.json").is_file())
    else:
        targets = [scene_dir(s, root) for s in args.scene]

    if not targets:
        print(f"没有找到任何场景（root={root}）")
        return 1

    print()
    print("=" * 74)
    print("  点云 / 场景图 同源性核对")
    print("  （不重跑任何模型；比对 点数 / 质心 / 尺寸 是否逐位相等）")
    print("=" * 74)
    print()
    results = [verify_scene(p) for p in targets]
    for res in results:
        _print_report(res)

    failed = [r for r in results if r["status"] == "failed"]
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_skip = sum(1 for r in results if r["status"] == "no_points")
    n_bad = sum(1 for r in results if r["status"] == "unreadable")

    print()
    print("-" * 74)
    print(f"  通过 {n_ok}    不一致 {len(failed)}    "
          f"无点云（跳过）{n_skip}    读不出 {n_bad}")
    if failed:
        print("  不一致的场景：" + "、".join(r["scene"] for r in failed))
    print("-" * 74)
    print()

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"results": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  报告已写入 {p}")
        print()

    return 1 if (failed or n_bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
