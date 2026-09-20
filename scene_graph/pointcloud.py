"""从落盘的整图点云里取回点云 —— 点云级算法的统一入口。

**为什么需要这一层。** 建图时点云是逐物体切好、用完即弃的：`builder.py`
取掩码内的点、算出质心/尺寸/包围盒，然后把整片点云丢掉。落盘之后，点的信息
变成「整图 `points.npy` + `masks/*.png`」两样东西，于是「某个物体的点云」
重新成为一个**可计算的派生量**，而不是一份要跟着主数据同步维护的副本。

**取点必须与 builder 逐字同源**（同一个 `resample_mask_to` + `select_points`，
同一个调用顺序）。理由很实际：当工具算出的质心与 `scene.json` 里记的
`centroid_3d` 对不上时，两个数字都「看起来合理」，肉眼分不出谁错，
而这类偏差一旦进了结果就无法追溯。所以这里**不重新实现一个更快的版本**，
只做组合。`scene_graph/tests/test_points_roundtrip.py` 就是拿这一点当验收标准的 ——
用落盘的点云重算质心与点数，必须与 `scene.json` 逐节点对上。

**零 torch。** 只依赖 `vision.geometry`（纯 NumPy）与 `scene_graph.schema`
（pydantic）。这条不是洁癖：`tools/` 与 `agents/` 全程零 torch 是硬约束
（问答路径不加载任何 GPU 模型），点云级工具必须能在这条路径上安全使用。
也正因为这条，`box_selector` 从 `builder.py` 下移到了 `vision.geometry`
—— builder 会经 `vision.grounding` 拉起 torch，绕不过去。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from scene_graph.schema import Node
from scene_graph.store import load_mask, load_points
from vision.geometry import box_selector, resample_mask_to, select_points

__all__ = ["points_in_mask", "points_in_box", "node_points", "points_meta_summary"]

#: 取不到点时的返回值形状。`(3, 0)` 而不是空 list：调用方拿到的永远是
#: `(3, N)` 的同一个契约，不必为"空"再写一个分支去猜维度。
_EMPTY = np.zeros((3, 0), dtype=np.float64)


def points_in_mask(
    points_chw: np.ndarray,
    mask_hw: np.ndarray | None,
    grid_hw: tuple[int, int],
) -> np.ndarray:
    """掩码内的点云 `(3, N)`。与 `builder.py` 的取点路径完全一致。

    `mask_hw` 为 None 或空时返回 `(3, 0)` —— 对应 builder 里「没拿到掩码 →
    空选择器」那条分支。**由调用方决定下一步**（提示重建场景、或改用检测框），
    不在这里替它偷偷选一条。
    """
    if mask_hw is None or np.size(mask_hw) == 0:
        return _EMPTY
    return select_points(points_chw, resample_mask_to(mask_hw, grid_hw))


def points_in_box(
    points_chw: np.ndarray,
    box_xyxy: Sequence[float],
    grid_hw: tuple[int, int],
    image_hw: tuple[int, int],
) -> np.ndarray:
    """检测框内的点云 `(3, N)` —— 与 builder 的降级路径同源。"""
    return select_points(points_chw, box_selector(box_xyxy, grid_hw, image_hw))


def node_points(
    node: Node,
    scene_dir: Path | str,
    *,
    points_chw: np.ndarray | None = None,
    meta: dict[str, Any] | None = None,
) -> np.ndarray:
    """取回某个节点在**建图当时**用的那片点云 `(3, N)`。

    `points_chw` / `meta` 不传就现读（`store.load_points`）。批量算多个物体时
    先读一次、再反复传进来，避免同一个 3.7 MB 数组被读 N 遍。

    ⚠️ 路径按 `node.centroid_source` 选，**不做自动兜底**：`"mask"` 就用掩码，
    `"bbox_fallback"` 就用检测框。这两个不是「两种等价做法」——
    builder 专门记下这个字段，正是因为二选一会改变质心（实测均值差 83 mm）。
    自动兜底等于把一条已经记录在案的事实重新变回随机事件。

    取不到点时返回 `(3, 0)`，而不是抛错：掩码文件丢失、框退化都可能发生，
    而「拿不到点」在不同调用方那里对应不同动作（有的该报 `DEGENERATE`，
    有的该换 anchor），交出去比在这里替他们决定更合适。
    """
    if points_chw is None:
        points_chw, meta = load_points(scene_dir)
    meta = meta or {}
    grid_hw = _grid_hw_of(meta)

    if node.centroid_source == "bbox_fallback":
        if node.bbox_2d is None:
            return _EMPTY
        return points_in_box(points_chw, node.bbox_2d, grid_hw, _image_hw_of(meta))

    return points_in_mask(points_chw, _mask_of(node, scene_dir), grid_hw)


def points_meta_summary(meta: dict[str, Any]) -> dict[str, Any]:
    """`points_meta.json` 的紧凑摘要 —— 给报告与演示台看，不含大字段。"""
    return {
        k: meta.get(k)
        for k in ("_format", "_written_at", "shape", "dtype",
                  "grid_hw", "image_hw", "intrinsics_source", "scale_calibrated")
        if k in meta
    }


# ----------------------------------------------------------------------------
# 内部
# ----------------------------------------------------------------------------


def _grid_hw_of(meta: dict[str, Any]) -> tuple[int, int]:
    """点云网格尺寸。**缺失时抛错而不是猜** —— 掩码重采样需要它，
    猜错会让取出来的点整体错位，而那种错位的结果看起来完全正常。"""
    gh = meta.get("grid_hw")
    if not gh:
        raise ValueError(
            "points_meta.json 里没有 grid_hw —— 掩码重采样需要它。"
            "这个文件可能来自更早的版本或被人改过，请重跑 scripts/build_scene.py。"
        )
    return (int(gh[0]), int(gh[1]))


def _image_hw_of(meta: dict[str, Any]) -> tuple[int, int]:
    """图像尺寸。只有降级路径（按 `bbox_2d` 取点）需要它，
    因为 `bbox_2d` 是**图像像素系**，换算到点云网格必须知道图像多大。"""
    ih = meta.get("image_hw")
    if not ih:
        raise ValueError(
            "points_meta.json 里没有 image_hw —— 按检测框取点需要它"
            "（bbox_2d 是图像像素系）。请重跑 scripts/build_scene.py。"
        )
    return (int(ih[0]), int(ih[1]))


def _mask_of(node: Node, scene_dir: Path | str) -> np.ndarray | None:
    """读回该节点的掩码。找不到返回 None（调用方退化成 `(3, 0)`）。

    优先按 `node.mask_ref` 解析 —— 它是**相对 `dataset/scenes/`** 的路径，
    而 `scene_dir` 的标准形态是 `<root>/<scene_id>`，所以取父目录当根。
    解析不出来时退回「按 id 猜文件名」：这能救回那些 `mask_ref` 前缀被改过、
    但掩码确实躺在 `masks/` 下的历史场景。**两条都不中才算真的没有。**
    """
    d = Path(scene_dir)
    if node.mask_ref:
        p = d.parent / node.mask_ref
        if p.is_file():
            return load_mask(p)
    fallback = d / "masks" / f"{node.id}.png"
    if fallback.is_file():
        return load_mask(fallback)
    return None
