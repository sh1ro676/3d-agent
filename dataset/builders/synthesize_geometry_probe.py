#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""synthesize_geometry_probe.py —— 解析合成几何夹具：自带真值的 3D 场景。

为什么需要它（而不是"用真实图片跑一遍"）
----------------------------------------
一档的目标是**绝对精度**：质心差几毫米、尺寸差几米。这需要真值。
但本仓库**没有真 GT**：`dataset/raw/omni3d-bench` 没有 GT 相机 / 深度 / 三维框
（§21 已记下这条），`living_room_gt` 也不是真 GT —— 它是同一张图、同一套模型，
唯一区别是内参外部给定（掩码 md5 全同）。拿它对不出绝对精度。

所以真值只能**合成**。本模块不做「渲染照片再跑三个模型」那一套
（那会把 UniDepth 的误差混进几何口径的验证里，且要 GPU、要 8 GB 显存、
要 CC BY-NC 许可下的模型权重）。它做一件更窄的事：

    构造**已知几何**的场景 → 解析渲染出深度图与掩码 → 交给**真实的下游代码**
    （`vision.geometry` 的 `centroid_of` / `robust_extent`、`scene_graph.relations`）
    算几何量 → 与真值比。

**它测的是「从点云/掩码到几何量」的那一段**，不含 UniDepth 的深度误差、
不含 GroundingDINO/SAM2 的检测分割误差。这是有意的切分：口径要能被归因，
就必须先把线段划在能分开的地方。深度模型那一段要靠另一套探针（§22 的内参探针
已经量了它的横向敏感度）。

⚠ **GT 必须是「该视角可见表面」的几何量，不是盒子的几何量**
----------------------------------------------------------
下游拿到的点云**只是可见表面**（z-buffer 之后的那些点）。如果 GT 用盒子本体
的质心与尺寸，测出来的差值里就混进了**遮挡与可见性**造成的固有偏差 ——
而那是任何方法都躲不掉的，不是估计误差。混在一起会让结论读反：
「尺寸偏小」会被误读成模型低估，实际是只看得见三个面。

所以本模块给出**两层真值**，让三件事能分开：

    gt_box      盒子本体（解析已知）      —— 上界/参照，不等于可达目标
    gt_visible  可见表面上的点（与下游同口径）—— **估计误差的基准**

于是：
    ① 自洽性    pred = gt_visible  → 误差必须恰好 0（尺子自己没错）
    ② 固有偏差  gt_visible vs gt_box → 可见性带来的、**不可归因于估计方法**的部分
    ③ 估计误差  pred vs gt_visible → 真正要量的

三层分开，才是「误差结构分解」。只报第 ③ 层而不报第 ② 层，读者会把
「只看得见三个面」当成「算法不准」。

⚠ **与下游同口径**是硬要求：GT 的质心用 `np.median`（不是 mean）、
包围盒用 `robust_extent`（不是裸 min/max）。否则第 ① 层（自洽性）不成立，
而第 ① 层是唯一的「尺子检查」。

渲染
----
针孔相机 + **射线与盒求交**（slab 法）+ z-buffer 取最近命中。
盒子可绕相机 y 轴旋转（`Box3D.yaw_rad`）：**把射线变换到盒的局部系**，
在那里它又是轴对齐盒，slab 法一行不用改（旋转是刚体变换，`t` 沿射线不变）。
`yaw_rad = 0` 走轴对齐分支、与未加朝向时**逐位一致** ⟹ 「朝向」是一个
**可开关的单一变量**，不是把整条链路换掉。

默认场景刻意保持**轴对齐**：下游 `extent_3d` 本身就是轴对齐包围盒的跨度，
轴对齐 GT 让「口径」与「物体朝向」两个误差源不纠缠 —— 这是**默认值的理由**，
不是**能力边界**。要看朝向本身的影响，就显式给 yaw（见
`scripts/run_geometry_probe.py` 的「朝向层」）。那时「测得尺寸 vs 盒自身尺寸」
的差值里同时含**轴对齐口径高估**（可解析算出，斜放必然高估）与估计误差；
本模块把前者单独暴露成 `gt_box[...]["aabb_extent"]`，
这样「尺寸偏大」才不会被又一次读成算法不准。

零依赖：只用 numpy + 标准库 + `vision.geometry`（后者也已确认零 torch）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "Box3D",
    "SyntheticScene",
    "render_scene",
    "visible_geometry",
    "default_probe_boxes",
    "default_background_boxes",
    "default_scene_boxes",
    "default_intrinsics",
    "visibility_report",
    "apply_perturbation",
    "PERTURBATIONS",
]

#: 射线求交时认为「命中」的最小正值（避免 t=0 的自命中）。
_T_EPS = 1e-9


# ----------------------------------------------------------------------------
# 场景定义
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Box3D:
    """一个相机系下的盒子。`min_xyz` / `max_xyz` 是**盒自身轴**（未旋转）的对角角点，单位米。

    `yaw_rad` 绕**相机 y 轴**旋转（本相机系 x 右 / y 下 / z 前 ⟹ y 就是竖直轴，
    所以 yaw 正是「家具朝哪边」这个自由度）。
    `min_xyz` / `max_xyz` 定义的是**旋转之前**的盒子，绕自身中心旋转；
    因此 `centre` 与 `extent`（盒自身轴向的三个边长）**不随 yaw 改变**。

    ⚠ **两种「尺寸」不是一回事，别混用**（这是加朝向带出来的最要紧的口径）
    -------------------------------------------------------------------
        box.extent        盒**自身轴**向的边长   → 与物体的"真实尺寸"对应
        box.aabb_extent   旋转之后**轴对齐**包围盒的跨度 → 与下游 `extent_3d` 同口径

    `yaw_rad = 0` 时二者相等（轴对齐盒子）。一旦斜放，`aabb_extent` **必然大于**
    `box.extent` —— 这是**轴对齐口径**的特性，不是误差。
    下游 `robust_extent` 返回的是**轴对齐**跨度，所以斜放物体的
    「测得尺寸 vs 真实尺寸」这个比较里，差值 = **口径高估**（可解析算出的确定量）
    ＋ 估计误差。把口径那部分单独报出来，「尺寸偏大」才不会又被读成算法不准。

    `is_background=True` 的盒子**只参与 z-buffer**（提供墙面/地板的深度），
    不进 `masks`、不进两层真值。为什么要它见 `default_background_boxes()`。
    """

    object_id: str
    label: str
    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]
    #: `True` = 背景表面（墙/地板）：只提供深度，不算物体。
    is_background: bool = False
    #: 绕相机 y 轴的旋转角（弧度）。0 = 轴对齐（与加朝向之前的行为逐位一致）。
    yaw_rad: float = 0.0

    def __post_init__(self) -> None:
        lo = np.asarray(self.min_xyz, dtype=np.float64)
        hi = np.asarray(self.max_xyz, dtype=np.float64)
        if lo.shape != (3,) or hi.shape != (3,):
            raise ValueError(f"{self.object_id}: min/max 必须是 3 元组")
        if (hi <= lo).any():
            raise ValueError(
                f"{self.object_id}: max 必须逐轴大于 min，收到 min={lo.tolist()} max={hi.tolist()}"
                "（其中一个轴退化成 0 会让该轴不可测，而不是得到一个「很薄的物体」）"
            )
        if not np.isfinite(self.yaw_rad):
            raise ValueError(f"{self.object_id}: yaw_rad 必须是有限实数，收到 {self.yaw_rad!r}")

    @property
    def centre(self) -> np.ndarray:
        """盒中心。绕自身中心旋转 ⟹ **与 yaw 无关**。"""
        return (np.asarray(self.min_xyz, dtype=np.float64)
                + np.asarray(self.max_xyz, dtype=np.float64)) / 2.0

    @property
    def extent(self) -> np.ndarray:
        """盒**自身轴**向的三个边长（不是轴对齐跨度；斜放时二者不同，见类 docstring）。"""
        return np.asarray(self.max_xyz, dtype=np.float64) - np.asarray(self.min_xyz, dtype=np.float64)

    @property
    def rotation(self) -> np.ndarray:
        """`(3,3)` 旋转矩阵，**列 = 盒自身轴在相机系里的方向**。

        绕 +y 的右手旋转。`yaw_rad = 0` 时严格等于单位阵，于是所有既有路径
        逐位退化（渲染、`aabb`、真值都不变）。
        """
        c, s = float(np.cos(self.yaw_rad)), float(np.sin(self.yaw_rad))
        eye = np.eye(3, dtype=np.float64)
        if self.yaw_rad == 0.0:
            return eye
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)

    def corners(self) -> np.ndarray:
        """旋转后的 8 个角点，`(8,3)`，相机系。"""
        lo = np.asarray(self.min_xyz, dtype=np.float64)
        hi = np.asarray(self.max_xyz, dtype=np.float64)
        signs = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)],
                         dtype=np.float64)
        local = np.where(signs > 0, hi - self.centre, lo - self.centre)   # (8,3) 盒局部系
        return local @ self.rotation.T + self.centre

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """旋转后角点的**轴对齐**包围盒 `(min, max)`。"""
        cs = self.corners()
        return cs.min(axis=0), cs.max(axis=0)

    @property
    def aabb_extent(self) -> np.ndarray:
        """旋转后**轴对齐**跨度 —— 与下游 `robust_extent` 同口径。

        解析形式（绕 y 旋转 θ）：`x = Lx·|cosθ| + Lz·|sinθ|`、`y = Ly`、
        `z = Lx·|sinθ| + Lz·|cosθ|`。斜放物体「测得尺寸偏大」的那部分就是这个量，
        **与估计误差无关**，必须分开报。
        """
        lo, hi = self.aabb()
        return hi - lo


@dataclass
class SyntheticScene:
    """一次合成渲染的产物。字段名刻意与真实链路对齐，便于换用。"""

    #: (3, H, W) 相机系点云 —— 与 `DepthField.points_chw` 同形同义。
    points_chw: np.ndarray
    #: object_id -> (H, W) bool 掩码 —— 与 `builder` 里的掩码同义。
    masks: dict[str, np.ndarray]
    #: 点云网格 (H, W)。**它等于图像网格**：合成渲染不经过模型的 padding/resize，
    #: 这一点与真实链路不同，写出来免得后来者以为可以直接对照 ms 级数字。
    grid_hw: tuple[int, int]
    #: 3×3 内参（实际用于生成 points 的那一份）。
    intrinsics: np.ndarray
    #: object_id -> 两层真值（详见模块 docstring）。
    gt_box: dict[str, dict[str, Any]]
    gt_visible: dict[str, dict[str, Any]]
    #: 渲染元信息（盒子清单、可见像素数、z 范围），供报告与排查。
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_pixels(self) -> int:
        h, w = self.grid_hw
        return h * w


def _pixel_rays(intrinsics: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    """逐像素射线方向 `(3, H, W)`（**未归一化**，z 分量为 1）。

    返回的是「方向」而不是「射线上距离 t 处的点」，因为 slab 求交算出的 t
    是**沿这个方向的倍数**；归一化过一次会让 t 的单位变得含糊，
    而本模块所有长度都要能指回米。
    """
    h, w = image_hw
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if fx <= 0 or fy <= 0:
        raise ValueError(f"fx/fy 必须为正，收到 fx={fx} fy={fy}")

    u = np.arange(w, dtype=np.float64) + 0.5      # 像素中心
    v = np.arange(h, dtype=np.float64) + 0.5
    uu, vv = np.meshgrid(u, v)
    dirs = np.empty((3, h, w), dtype=np.float64)
    dirs[0] = (uu - cx) / fx
    dirs[1] = (vv - cy) / fy
    dirs[2] = 1.0
    return dirs


def _intersect_box(dirs: np.ndarray, box: Box3D) -> tuple[np.ndarray, np.ndarray]:
    """射线（起点在原点）与**可绕 y 旋转的盒**求交。

    返回 `(t_hit, hit)`：`t_hit` 为 `(H,W)`（未命中为 inf），`hit` 为 bool `(H,W)`。

    slab 法：每个轴给出一个参数区间 `[t1, t2]`，交完三个轴后
    `t_enter = max(t1,t2,t3)`、`t_exit = min(...)`；命中 ⟺ `t_exit >= max(t_enter, 0)`。

    ⚠ 相机原点在盒内时 `t_enter < 0`，此时「第一个可见点」是**出射点** `t_exit`
    而不是 `t_enter` —— 少写这一个分支，盒子套住相机的那一帧会整片消失。

    朝向怎么处理：把**射线**变换到盒的局部系，在那里它又是一个轴对齐盒，
    slab 法一行不用改。旋转是刚体变换，`t` 沿射线不变 ⟹ `t` 的单位仍是「米」，
    `points = t × dirs` 照旧成立。

    ⚠ **轴对齐分支必须留着，不能统一成一个公式。** 直觉上可以写成
    「`lo ← min−c`、原点偏移 `o ← −c`」，那样 `yaw_rad = 0` 时也算得对 ——
    但浮点上 `(min − c) + c ≠ min`（一般情形），会把原本精确的数字
    （例如正对相机的近面 `z = 2.0000000000000000`）改成差几个 ulp 的值，
    于是「正对盒子可见质心恰好落在近面」这类**精确**断言会变成 `approx`。
    精确断言是尺子的自检，不能被一个"更优雅"的公式换掉 ⟹ 两条分支并存，
    `yaw_rad = 0` 走原路，**逐位一致**。
    """
    if box.yaw_rad == 0.0:
        lo = np.asarray(box.min_xyz, dtype=np.float64)[:, None, None]
        hi = np.asarray(box.max_xyz, dtype=np.float64)[:, None, None]
        origin = np.zeros((3, 1, 1), dtype=np.float64)
        dirs_local = dirs
    else:
        c = box.centre
        R = box.rotation
        lo = (np.asarray(box.min_xyz, dtype=np.float64) - c)[:, None, None]
        hi = (np.asarray(box.max_xyz, dtype=np.float64) - c)[:, None, None]
        origin = -(R.T @ c)[:, None, None]
        dirs_local = np.tensordot(R.T, dirs, axes=([1], [0]))

    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - origin) / dirs_local
        t2 = (hi - origin) / dirs_local
    t_lo = np.minimum(t1, t2)
    t_hi = np.maximum(t1, t2)
    t_enter = t_lo.max(axis=0)
    t_exit = t_hi.min(axis=0)

    hit = (t_exit >= np.maximum(t_enter, 0.0)) & (t_exit > 0.0)
    t_hit = np.where(t_enter > 0.0, t_enter, t_exit)
    t_hit = np.where(hit, t_hit, np.inf)
    return t_hit, hit


def render_scene(
    boxes: Sequence[Box3D],
    *,
    intrinsics: np.ndarray,
    image_hw: tuple[int, int] = (240, 320),
) -> SyntheticScene:
    """解析渲染：z-buffer 取最近命中，得到点云 `(3,H,W)` 与每物体掩码。

    `image_hw` 是小尺寸（默认 320×240）：合成场景没有纹理，
    分辨率只影响边界像素的比例，不影响几何结论，而小图让整个扫描秒级完成。
    真要复现真实场景的量级，把 `image_hw` 调大即可（纯 numpy，成本线性）。
    """
    intr = np.asarray(intrinsics, dtype=np.float64)
    if intr.shape != (3, 3):
        raise ValueError(f"intrinsics 必须是 3×3，收到 shape={intr.shape}")
    if not boxes:
        raise ValueError("boxes 为空 —— 没有物体就没有可测的几何量")

    h, w = image_hw
    dirs = _pixel_rays(intr, image_hw)

    # ---- z-buffer：逐盒子求交，保留最近的 ----
    # 背景（墙/地板）与物体**一起**参与求交：它们要能挡住物体后面的东西，
    # 也要让「物体之外」的像素有深度，而不是留成 nan。
    best_t = np.full((h, w), np.inf, dtype=np.float64)
    winner = np.full((h, w), -1, dtype=np.int32)

    for i, box in enumerate(boxes):
        t_hit, hit = _intersect_box(dirs, box)
        closer = hit & (t_hit < best_t)
        best_t = np.where(closer, t_hit, best_t)
        winner = np.where(closer, np.int32(i), winner)

    # ---- 点云：命中像素 = t × 方向 ----
    valid = winner >= 0
    t_safe = np.where(valid, best_t, 0.0)
    points_chw = dirs * t_safe[None, :, :]
    points_chw[:, ~valid] = np.nan
    # 未命中的像素给 nan，而不是 0：0 会被下游当成「相机处的有效点」，
    # 使一个本该空着的像素反而把中位数拉飞。nan 会被 `valid_point_selector` 剔掉。
    # （有了背景盒子之后，正常情况下画面内不该再有 nan 像素 —— 见 `no_hit_px`。）

    # ---- 掩码与两层真值：**只对物体**，背景不算 ----
    masks: dict[str, np.ndarray] = {}
    gt_box: dict[str, dict[str, Any]] = {}
    gt_visible: dict[str, dict[str, Any]] = {}
    box_meta: list[dict[str, Any]] = []

    for i, box in enumerate(boxes):
        if box.is_background:
            continue
        masks[box.object_id] = winner == i
        aabb_lo, aabb_hi = box.aabb()
        gt_box[box.object_id] = {
            "centroid_3d": box.centre,
            #: 盒**自身轴**向边长 —— 斜放时**不要**拿它去比下游的轴对齐跨度（见 Box3D）。
            "extent_3d": box.extent,
            #: 旋转后**轴对齐**跨度 —— 与下游 `robust_extent` 同口径，斜放时用它比。
            "aabb_extent": box.aabb_extent,
            "bbox_min": np.asarray(box.min_xyz, dtype=np.float64),
            "bbox_max": np.asarray(box.max_xyz, dtype=np.float64),
            "aabb_min": aabb_lo,
            "aabb_max": aabb_hi,
            "yaw_rad": float(box.yaw_rad),
        }
        box_meta.append({
            "object_id": box.object_id,
            "label": box.label,
            "min_xyz": [float(v) for v in box.min_xyz],
            "max_xyz": [float(v) for v in box.max_xyz],
            "yaw_deg": float(np.degrees(box.yaw_rad)),
            "n_visible_px": int(masks[box.object_id].sum()),
        })

    scene = SyntheticScene(
        points_chw=points_chw,
        masks=masks,
        grid_hw=(h, w),
        intrinsics=intr,
        gt_box=gt_box,
        gt_visible={},          # 下面填（需要 points 已经就绪）
        meta={
            "boxes": box_meta,
            "image_hw": [h, w],
            "intrinsics": intr.tolist(),
            "n_hit_px": int(valid.sum()),
            #: 没有任何表面命中的像素数。**有背景时它应当为 0** ——
            #: 不为 0 说明画面某些方向没被任何表面覆盖，那些像素在下游是空洞，
            #: 而「空洞」与「背景」对掩码外溢类扰动的影响完全不同。
            "no_hit_px": int((~valid).sum()),
            "background_ids": [b.object_id for b in boxes if b.is_background],
            #: 带朝向的盒子数。`0` ⟹ 走的是轴对齐分支（与加朝向之前逐位一致）。
            "n_rotated": int(sum(1 for b in boxes if b.yaw_rad != 0.0)),
            "renderer": "analytic-obb-raycast",
        },
    )
    # ---- 第二层真值：可见表面（必须与下游同口径）----
    for box in boxes:
        if box.is_background:
            continue
        gt_visible[box.object_id] = visible_geometry(
            scene.points_chw, scene.masks[box.object_id], scene.grid_hw
        )
    scene.gt_visible = gt_visible
    return scene


# ----------------------------------------------------------------------------
# 取几何量：与下游同口径
# ----------------------------------------------------------------------------


def visible_geometry(
    points_chw: np.ndarray,
    mask_hw: np.ndarray,
    grid_hw: tuple[int, int],
) -> dict[str, Any]:
    """在一个掩码内的点云上，用**真实下游函数**算质心与包围盒。

    为什么直接调 `vision.geometry` 而不是在本模块里重写一遍：
    重写就等于给了自己一把可以「独立地错」的尺子。这里的全部意义是
    「GT 与预测同口径」，那么口径只能有一份实现。
    """
    from vision.geometry import centroid_of, resample_mask_to, robust_extent, select_points

    mask = resample_mask_to(np.asarray(mask_hw), grid_hw)
    pts = select_points(np.asarray(points_chw), mask)
    if pts.shape[1] == 0:
        return {
            "centroid_3d": None,
            "extent_3d": None,
            "bbox_min": None,
            "bbox_max": None,
            "n_points": 0,
        }
    centroid, cmeta = centroid_of(pts)
    extent, lo, hi, emeta = robust_extent(pts)
    return {
        "centroid_3d": centroid,
        "extent_3d": extent,
        "bbox_min": lo,
        "bbox_max": hi,
        "n_points": int(pts.shape[1]),
        "reduce": cmeta.get("reduce"),
        "n_inliers": emeta.get("n_inliers"),
    }


# ----------------------------------------------------------------------------
# 扰动：把「某一种具体错误」注入进去
# ----------------------------------------------------------------------------


def _grow_shrink_mask(mask: np.ndarray, n_px: int) -> np.ndarray:
    """掩码按 8 邻域膨胀 / 腐蚀 `|n_px|` 次（n_px>0 膨胀）。

    手写而不引 scipy：口径层要能在没装科学计算栈的机器上跑，
    而 3×3 形态学用 numpy 的切片求和就够。
    """
    m = np.asarray(mask, dtype=bool)
    for _ in range(abs(int(n_px))):
        p = np.pad(m, 1, mode="constant", constant_values=False)
        if n_px > 0:      # 膨胀：任一邻居为真即为真
            m = (p[:-2, :-2] | p[:-2, 1:-1] | p[:-2, 2:] |
                 p[1:-1, :-2] | p[1:-1, 1:-1] | p[1:-1, 2:] |
                 p[2:, :-2] | p[2:, 1:-1] | p[2:, 2:])
        else:             # 腐蚀：全部邻居为真才为真
            m = (p[:-2, :-2] & p[:-2, 1:-1] & p[:-2, 2:] &
                 p[1:-1, :-2] & p[1:-1, 1:-1] & p[1:-1, 2:] &
                 p[2:, :-2] & p[2:, 1:-1] & p[2:, 2:])
    return m


#: 可注入的扰动清单。**每一项都必须能一句话说清它模拟现实里的什么**，
#: 否则它只是「随便改个数字」，得到的响应曲线无法解释。
PERTURBATIONS: dict[str, str] = {
    "none": "不扰动 —— 自洽性对照，所有误差必须恰好为 0",
    "intrinsics_scale": "反投影用错的内参：fx/fy 缩放 k 倍 ⟹ 横向按 k 缩放（纵深不动）",
    "depth_scale": "深度整体乘性偏置（单目深度的常见失效：整体尺度错）",
    "depth_noise": "深度径向乘性噪声（重尾：少数点被拉得很远）",
    "mask_grow": "掩码膨胀 n 像素（SAM2 边界外溢；会把背景点拉进质心）",
    "mask_shrink": "掩码腐蚀 n 像素（SAM2 边界收缩；会削掉物体的真实极值）",
    "object_translate_x": "某个物体横向平移（模拟横向米制尺度错，§22 实测占 99%）",
    "object_translate_z": "某个物体纵深平移（对照项：纵深错在该指标上是否可见）",
}


def apply_perturbation(
    scene: SyntheticScene,
    kind: str,
    *,
    k: float = 1.0,
    n_px: int = 0,
    sigma_rel: float = 0.0,
    dz_m: float = 0.0,
    dx_m: float = 0.0,
    object_id: Optional[str] = None,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """返回扰动后的 `(points_chw, masks)`。

    ⚠ **返回值直接喂给 `visible_geometry`**，不是喂给「答案」。
    扰动作用在**点云与掩码**上，因为那才是链路里真正被传递的东西 ——
    改答案会得到一个漂亮的曲线，却和系统行为无关。

    各参数只在对应 `kind` 下有意义（其余被忽略，不报错，因为脚本要按网格扫参）。
    """
    if kind not in PERTURBATIONS:
        raise ValueError(f"未知扰动 {kind!r}；可选：{sorted(PERTURBATIONS)}")

    pts = np.array(scene.points_chw, dtype=np.float64, copy=True)
    masks = {k2: np.array(v, copy=True) for k2, v in scene.masks.items()}

    if kind == "none":
        return pts, masks

    if kind == "intrinsics_scale":
        # 用 fx'=k·fx 反投影：横向偏移按 k 缩放，z 不变。
        # 这正是 §22 里「model 自预测内参 vs 外部给定内参」的机制。
        if k <= 0:
            raise ValueError("intrinsics_scale 的 k 必须为正")
        pts[0] = pts[0] * k
        pts[1] = pts[1] * k
        return pts, masks

    if kind == "depth_scale":
        if k <= 0:
            raise ValueError("depth_scale 的 k 必须为正")
        pts = pts * k
        return pts, masks

    if kind == "depth_noise":
        rng = np.random.default_rng(seed)
        gain = rng.normal(1.0, sigma_rel, size=pts.shape[1:])
        pts = pts * gain[None, :, :]
        invalid = ~np.isfinite(pts).all(axis=0)
        pts[:, invalid] = np.nan
        return pts, masks

    if kind in ("mask_grow", "mask_shrink"):
        if n_px < 0:
            raise ValueError("n_px 请给正数；膨胀/收缩由 kind 决定")
        sign = n_px if kind == "mask_grow" else -n_px
        for oid in masks:
            masks[oid] = _grow_shrink_mask(masks[oid], sign)
        return pts, masks

    if kind in ("object_translate_x", "object_translate_z"):
        if not object_id:
            raise ValueError(f"{kind} 需要 object_id —— 不指定物体就无法解释这条曲线")
        if object_id not in masks:
            raise ValueError(f"场景里没有物体 {object_id!r}；可选 {sorted(masks)}")
        delta = np.zeros(3, dtype=np.float64)
        delta[0 if kind == "object_translate_x" else 2] = dx_m if kind.endswith("_x") else dz_m
        sel = masks[object_id]
        # ⚠ 只平移**该物体掩码内**的点。整片平移会连背景一起挪，
        # 于是「物体相对背景动了」这件事在几何量上完全看不出来。
        pts[:, sel] = pts[:, sel] + delta[:, None]
        return pts, masks

    raise AssertionError(f"未处理的扰动 {kind!r}")   # pragma: no cover


# ----------------------------------------------------------------------------
# 默认场景
# ----------------------------------------------------------------------------


def default_probe_boxes() -> list[Box3D]:
    """一个客厅尺度的默认场景：9 个物体，2.3–5.1 m，按 u 带错开避免互相遮挡。

    尺寸取真实量级（椅子 0.5 m、茶几 0.9 m、沙发 1.0 m、书架 0.9 m），
    因为「误差 200 mm」这种数字只在有物理参照时才能被判断大小 ——
    一个 20 cm 的沙发看起来是灾难，一个 20 cm 的定位误差是另一回事。

    ⚠ **布局是量出来的，不是摆出来的。** 第一版按「像客厅」直觉摆，
    结果 `chair_1` 在 z=2 m 处横向偏了 1.45 m，**已经出了 53° 视场**，
    单独渲染只有 339 个像素（比它的投影面积小两个数量级）——
    而它看起来完全像「算法不准」。现在每个物体的位置都过了 `visibility_report`：

        可见像素 ≥ 742、可见率 ≥ 0.74（逐物体数字见报告里 `visibility` 段）

    ⚠ **遮挡没有被消灭，只是被隔离。** 剩余遮挡（沙发/挂画被前排挡住一部分）
    落在 `gt_visible` vs `gt_box` 那一层，也就是「可见性固有偏差」——
    它**不该**混进估计误差里。这正是模块要分两层真值的原因。
    要把遮挡也去掉，就用 `visibility_report` 逐个挪，或者减小物体数。
    """
    return [
        # 近层（z≈2.3–2.8，小物体，放在画面两侧）
        Box3D("chair_1",    "chair",    (-1.50,  0.30, 2.35), (-1.00,  0.95, 2.75)),
        Box3D("plant_1",    "plant",    ( 0.85, -0.30, 2.30), ( 1.15,  0.85, 2.60)),
        # 中层（z≈3.0–3.7）
        Box3D("table_1",    "table",    (-0.45,  0.55, 3.00), ( 0.45,  0.90, 3.70)),
        Box3D("lamp_1",     "lamp",     ( 0.55, -0.50, 3.30), ( 1.05,  0.95, 3.60)),
        Box3D("monitor_1",  "monitor",  (-0.10, -0.60, 3.60), ( 0.35, -0.20, 3.64)),
        # 远层（z≈4.3–5.1，大物体）
        Box3D("sofa_1",     "sofa",     (-1.70,  0.40, 4.30), (-0.70,  1.15, 5.00)),
        Box3D("shelf_1",    "shelf",    (-0.70, -0.80, 4.60), ( 0.20,  0.60, 5.00)),
        Box3D("cabinet_1",  "cabinet",  ( 2.15,  0.88, 4.40), ( 2.85,  1.60, 4.80)),
        Box3D("painting_1", "painting", (-1.20, -1.00, 5.10), (-0.70, -0.20, 5.14)),
    ]


def default_background_boxes() -> list[Box3D]:
    """后墙 + 地板 —— **只为提供深度**，不算物体。

    为什么必须有背景（这不是装饰）
    ------------------------------
    没有背景时，物体之外的像素是 `nan`（没有表面命中）。于是：

    * `mask_grow`（模拟 SAM2 边界外溢）**完全测不出效果** ——
      溢出的像素落在 nan 上，`select_points` 把它们剔掉，几何量纹丝不动。
      而真实照片里那些像素是**墙面/地板**，会把背景点拉进质心。
      这个扰动是本探针的核心项之一，缺背景等于把它静音了。
    * `no_hit_px > 0` 意味着画面里存在「没有任何东西」的方向，与真实相机不符
      （真实照片每个像素都有深度）。

    这条是被测试逼出来的：`test_grow_changes_the_measured_extent` 一开始失败，
    原因不是扰动写错，而是**场景缺了背景**。

    几何约束（改动时必须守住）
    --------------------------
    * 后墙 `z ∈ [6.6, 6.8]` ⟹ 比所有物体（`z_max = 5.14`）都远 ⟹ 不挡物体。
    * 地板 `y ∈ [1.66, 1.86]` ⟹ 比所有物体（`y_max = 1.60`）都低 ⟹ 不挡物体。
      因为射线沿 +y（向下）单向前进，`t_地板 > t_物体` 恒成立。
    * 两者都远大于视锥在各自深度处的半径（`|x| < 0.667·z`、`|y| < 0.5·z`），
      所以画面每个方向都有命中。
    """
    return [
        Box3D("__back_wall__", "wall", (-6.0, -5.0, 6.6), (6.0, 5.0, 6.8), is_background=True),
        Box3D("__floor__", "floor", (-6.0, 1.66, 0.1), (6.0, 1.86, 6.8), is_background=True),
    ]


def default_scene_boxes() -> list[Box3D]:
    """默认场景 = 9 个物体 + 背景。直接喂给 `render_scene`。"""
    return default_probe_boxes() + default_background_boxes()


def default_intrinsics(*, fx: float = 240.0, cx: float = 160.0, cy: float = 120.0) -> np.ndarray:
    """默认内参：`fx=fy=240`、主点居中（对应 320×240），HFoV ≈ 67°。

    ⚠ 这个 67° 落在 `check_fov` 的可信区间内 —— 本夹具刻意**不**复制
    `163.7 px`（HFoV 125.8°）那个已知不可信的预测内参（§20.3 口径框）。
    要测内参错误，用 `intrinsics_scale` **受控注入**，
    而不是从一个不可复现的来源继承一个不知道大小的错误。

    为什么是 240 而不是更窄的 320：窄视场下九个物体挤不进画面，
    只好让它们互相遮挡，而遮挡会污染几何量的口径。67° 是「装得下」与
    「物体不会小到离散化主导」之间的折中 —— 最小物体（monitor 薄板）
    仍有 810 个可见像素。
    """
    return np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


# ----------------------------------------------------------------------------
# 场景体检：让「物体出界 / 被遮挡」变成可检查的事实
# ----------------------------------------------------------------------------


def visibility_report(
    boxes: Sequence[Box3D],
    *,
    intrinsics: np.ndarray,
    image_hw: tuple[int, int] = (240, 320),
) -> dict[str, Any]:
    """每个**物体**的可见率 = 多物体场景里的像素数 ÷ 单独渲染时的像素数。

    背景盒子（`is_background=True`）不参与统计，但会在「单独渲染」那一侧一起带上 ——
    否则「无遮挡参照」会缺背景，与多物体场景不可比。

    为什么需要它
    ------------
    合成夹具最容易犯的错不是渲染错，而是**物体跑到画面外、或互相遮住**。
    那两种情况下算出来的「几何误差」其实是「只看见了一小块」——
    而它看起来**完全像算法误差**，不会报错。

    这不是假设的风险：第一版默认场景（按「像客厅」的直觉摆）里 `chair_1`
    只有 339 个可见像素、比它的投影面积小两个数量级，原因是它在 z≈2 m 处
    横向偏了 1.45 m，已经出了当时的 53° 视场。靠手算 `|x| < 0.5·z` 逐个核对
    九个物体既慢又容易漏，所以这里把它变成一个**每次都能跑的函数**。

    判据：`visible_frac` 明显小于 1 ⟹ 该物体的几何量是「局部可见面」的，
    报告中必须写明；`px_in_scene` 太小（几百像素级）⟹ 离散化会主导，
    这个物体的误差数字不该被当作精度结论。
    """
    objects = [b for b in boxes if not b.is_background]
    background = [b for b in boxes if b.is_background]
    if not objects:
        raise ValueError("visibility_report 需要至少一个非背景物体")
    alone = {
        b.object_id: int(render_scene([b, *background], intrinsics=intrinsics,
                                      image_hw=image_hw).masks[b.object_id].sum())
        for b in objects
    }
    together = render_scene(list(boxes), intrinsics=intrinsics, image_hw=image_hw)

    rows: list[dict[str, Any]] = []
    for b in objects:
        n_alone = alone[b.object_id]
        n_together = int(together.masks[b.object_id].sum())
        frac = (n_together / n_alone) if n_alone else 0.0
        rows.append({
            "object_id": b.object_id,
            "label": b.label,
            "px_alone": n_alone,
            "px_in_scene": n_together,
            "visible_frac": frac,
            "off_screen": n_alone == 0,
        })
    return {
        "image_hw": list(image_hw),
        "objects": rows,
        "worst_visible_frac": min((r["visible_frac"] for r in rows), default=0.0),
        "min_px_in_scene": min((r["px_in_scene"] for r in rows), default=0),
        "no_hit_px": together.meta["no_hit_px"],
        "all_fully_visible": all(abs(r["visible_frac"] - 1.0) < 1e-9 for r in rows),
    }

