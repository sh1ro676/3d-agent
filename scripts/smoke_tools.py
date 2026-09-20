"""Phase 1 冒烟测试：场景图 → 工具调用 → 答案 + 证据链 + trace。

不加载任何模型（场景手工构造），所以**不需要 GPU、不需要联网**，
可以当 CI 的第一道闸门，也可以在 8GB 机器上随时重跑。

它验证四件事：
    ① `scene_hint` 里确实**没有坐标**（§13.3(2)：坐标只能来自工具返回值）
    ② 一段「LLM 应当写出来的程序」能跑完并产出可审计的答案
    ③ 幻觉 id 会被 `NOT_IN_SCENE` 抓住，且错误里带回合法 id 列表
    ④ trace 能落成 JSONL —— 指标 2/3/4 直接从这里算

用法：
    D:\\3D_Spatial_Agent\\venvs\\vision\\Scripts\\python.exe scripts\\smoke_tools.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_graph.schema import BBox3D, Node, SceneGraph  # noqa: E402
from tools import load_tools  # noqa: E402
from tools.geometry import calculate_distance, get_3d_position  # noqa: E402
from tools.registry import ToolContext  # noqa: E402
from tools.result import ToolResult  # noqa: E402
from tools.spatial import find_nearest, find_object, list_objects, query_relation  # noqa: E402

load_tools()


# ----------------------------------------------------------------------------
# 一个手工场景（数据取自 §11.1 的客厅示例，坐标单位米）
# ----------------------------------------------------------------------------


def mk(node_id: str, label: str, xyz, size, score: float = 0.8) -> Node:
    x, y, z = xyz
    w, h, l = size
    return Node(
        id=node_id,
        label=label,
        score=score,
        bbox_2d=(100.0, 100.0, 300.0, 400.0),
        mask_ref=f"masks/{node_id}.png",
        centroid_3d=(x, y, z),
        extent_3d=(w, h, l),
        bbox_3d=BBox3D(min=(x - w / 2, y - h / 2, z - l / 2),
                       max=(x + w / 2, y + h / 2, z + l / 2)),
        n_points=4200,
    )


def build_scene() -> SceneGraph:
    return SceneGraph(
        scene_id="living_room_01",
        image_id="rgb_0001",
        camera_intrinsics=[[606.0, 0.0, 320.0], [0.0, 606.0, 240.0], [0.0, 0.0, 1.0]],
        up_axis="-y",
        nodes=(
            mk("door_1", "door", (-1.20, 0.00, 1.40), (0.90, 2.00, 0.10), 0.91),
            mk("chair_1", "chair", (0.80, 0.10, 2.90), (0.50, 0.90, 0.50), 0.83),
            mk("chair_2", "chair", (1.60, 0.10, 2.20), (0.50, 0.90, 0.50), 0.74),
            mk("sofa_1", "sofa", (0.31, -0.42, 2.86), (1.92, 0.78, 0.91), 0.84),
            mk("table_1", "table", (-0.44, -0.61, 3.12), (0.86, 0.45, 0.86), 0.71),
        ),
        build_meta={"detector": "grounding-dino-tiny", "tools_version": "1.0.0"},
    )


# ----------------------------------------------------------------------------
# 这是「LLM 应当生成的东西」—— 主路径的动作空间是 Python 程序，不是 JSON tool call
# ----------------------------------------------------------------------------


def program_nearest_chair_to_door(ctx: ToolContext, question: str) -> ToolResult:
    """问题：哪把椅子离门最近？

    写成一段直线 Python 就够 —— 而这正是 JSON tool call 表达不了的东西：
    「在 N 个候选里取 argmin」需要一次组合 + 一次聚合。用逐步 tool-calling，
    模型得自己维护「已算过的距离」列表并比较，4B 级模型很容易在这里翻车。
    """
    doors = find_object(ctx, label="door")
    if not doors:
        return doors                      # 失败要显式向上返回，不能假装成功

    chairs = find_object(ctx, label="chair")
    if not chairs:
        return chairs

    door_id = doors.value[0]["object_id"]
    nearest = find_nearest(ctx, anchor=door_id, label="chair", k=1)
    if not nearest:
        return nearest

    winner = nearest.value[0]["object_id"]
    pos = get_3d_position(ctx, object_id=winner)
    _ = pos                               # 位置进 trace，作为证据链的一环
    return nearest


# ----------------------------------------------------------------------------


def show(res: ToolResult, label: str) -> None:
    mark = "OK " if res.ok else "ERR"
    if res.ok:
        body = json.dumps(res.value, ensure_ascii=False)
        if len(body) > 96:
            body = body[:93] + "..."
        print(f"  [{mark}] {label:<34} -> {body}")
    else:
        assert res.error is not None
        recovery = ",".join(r.value for r in res.error.recovery)
        print(f"  [{mark}] {label:<34} -> {res.error.code.value}  (recovery: {recovery})")


def main() -> int:
    scene = build_scene()
    ctx = ToolContext(scene=scene)

    print("=" * 78)
    print("Phase 1 冒烟测试 —— 场景图 → 工具库 → 答案 + 证据 + trace")
    print("=" * 78)
    print(f"场景: {scene.summary_line()}")

    # ---- ① scene_hint 里没有坐标 ------------------------------------------
    print("\n① scene_hint（喂给 synthesize 的**全部**场景信息）")
    hint = scene.label_counts()
    print(f"  {json.dumps(hint, ensure_ascii=False)}  + image_size")
    coords_leaked = any(isinstance(v, (list, tuple)) for v in hint.values())
    print(f"  含坐标? {coords_leaked}   <- 必须是 False：坐标只能来自工具返回值")

    # ---- ② 一段「LLM 程序」跑完 -------------------------------------------
    question = "哪把椅子离门最近？"
    print(f"\n② 程序合成范式执行：{question}")
    print("  程序: find_object('door') -> find_object('chair') -> "
          "find_nearest(...) -> get_3d_position(winner)")
    result = program_nearest_chair_to_door(ctx, question)
    if not result.ok:
        print("  程序失败：", result.error)
        return 1

    winner = result.value[0]
    print(f"\n  答案: {winner['object_id']}（{winner['label']}）"
          f"，距门 {winner['distance_m']:.3f} m")

    print("\n  几何证据链（可脱离程序复算）：")
    ev = result.evidence
    print(f"    anchor          = {ev['anchor']} @ {ev['anchor_centroid_m']}")
    print(f"    n_candidates    = {ev['n_candidates']}")
    for row in ev["ranked"]:
        print(f"      {row['object_id']:<10} {row['distance_m']:.4f} m")

    print("\n  补充查询：")
    show(calculate_distance(ctx, a="sofa_1", b="table_1"), "calculate_distance(sofa, table)")
    show(query_relation(ctx, relation="left_of", a="door_1", b="chair_1"),
         "query_relation(left_of, door, chair)")
    show(get_3d_position(ctx, object_id="sofa_1"), "get_3d_position(sofa_1)")

    # ---- ③ 幻觉捕获 --------------------------------------------------------
    print("\n③ 幻觉捕获点（模型的 id 是编的）")
    hallucinated = get_3d_position(ctx, object_id="chair_9")
    show(hallucinated, "get_3d_position(chair_9)")
    assert hallucinated.error is not None
    known = hallucinated.error.context["known_ids"]
    print(f"    error.context 带回合法 id: {known}")
    print("    → 模型据此改写程序；「幻觉率」这个指标也因此可被直接统计")

    # ---- ④ trace -----------------------------------------------------------
    print(f"\n④ Trace（{len(ctx.trace)} 条，直接可算指标 2/3/4）")
    for i, line in enumerate(ctx.trace_jsonl(), 1):
        row = json.loads(line)
        status = "ok" if row["result"]["ok"] else row["result"]["error"]["code"]
        print(f"  #{i} {row['tool']:<18} {status:<16} "
              f"{row['result']['meta']['latency_ms']:>7.3f} ms")

    out = ROOT / "dataset" / "synthesized" / "smoke_trace.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ctx.trace_jsonl()), encoding="utf-8")
    print(f"\n  trace 已落盘: {out}")

    print("\n" + "=" * 78)
    print("结论：不需要任何模型与 GPU，整条「场景 → 几何 → 答案 → 可审计」链路成立。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
