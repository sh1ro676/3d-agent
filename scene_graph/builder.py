"""单遍构建：检测 → 分割 → 升维 → 关系。契约见方案文档 §12.3。

对应 VADAR 的 `predefined_modules.py`，但有三处**结构性**差别，
每一处都对应一条 Phase 0 的实测或裁决：

| | VADAR | 本文件 |
|---|---|---|
| 三维量 | 只取 `depth`，3D 尺寸 = `2D 像素 × depth`（**整个式子没有焦距**） | 取整片 `points`，质心/尺寸/包围盒全部由点云算出 |
| 质心来源 | 检测框内像素的中位数 | **SAM2 掩码**内点云的中位数（实测差均值 83 mm / 最大 208 mm） |
| 物体身份 | 每次 `loc()` 回裸 bbox，靠 `same_object(iou>0.92)` 反推是不是同一个 | `{label}_{idx}` **稳定 id**，问题在架构上消失 |

「单遍」是刻意选的（不是偷懒）：点云只算一次、掩码只调一次、关系只算一轮，
于是整条链路的耗时可以被加总验证，而不是散在若干次重复推理里。
实测代价见 `stats["timings_ms"]`。

**为什么 `build_scene_graph` 的感知部分只依赖 `PerceptionLike` 协议**：
真实跑一次要拉三个模型（本机约 5 秒加载 + 2–3 秒推理）。把感知换成 fake，
本文件的全部分支 —— 去重、降级、丢帧、关系生成、掩码存盘 —— 都能在 0.5 秒内测完，
且不需要 GPU。见 `scene_graph/tests/test_builder.py`。

⚠️ **内参来源是横向尺度的决定性因素 —— 这不是「尺度校正」。** builder 不做尺度
校正（`calibrate_scale` 属 L3 工具，Phase 5），`scale_factor` 恒为 1.0。但 Phase 0
Step 6 用仓库自带的 GT 深度逐像素实测：同一个模型、同一张图，
**让模型猜内参** vs **把内参告诉它**，三维误差中位数是 **1.943 m vs 0.267 m**
（差 7.3 倍），深度 ARel 19.8% vs 11.7%。

所以 builder 把内参当成**输入**而不是后处理：

  • `BuildConfig.known_intrinsics` 有值 → 透传给 `lift()`，来源记 `"provided"`
  • 没有 → 用模型预测值，来源记 `"predicted"`
  • 预测值且**视场不可信** → 多发一条警告。数字照出，但下游绝不会以为它标定过

⚠️ 并且**不要**试图用一个全局标量把尺寸缩放回去。内参错是**各向异性**的：
focal 小了 k 倍，横向坐标按 k 倍放大，而 z 几乎不动（`depth` 就是 `points` 的 z 列，
近轴处 `x≈y≈0`，focal 在那里不起作用）。一个 `scale_factor` 修不了它 ——
它会把本来已经对的 z 一起弄错。这是 `calibrate_scale` 这个设计被撤销的原因。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from scene_graph.relations import DEFAULT_NEAR_M, DEFAULT_TOL, pairwise
from scene_graph.schema import BBox3D, Edge, Node, SceneGraph
from vision.geometry import (
    DEFAULT_MIN_POINTS,
    PLAUSIBLE_HFOV_DEG,
    UpAxisEstimate,
    box_coverage,
    box_selector,
    centroid_of,
    check_fov,
    clip_box_xyxy,
    estimate_up_axis,
    intrinsics_matrix,
    iou_xyxy,
    resample_mask_to,
    robust_extent,
    select_points,
)
from vision.grounding import DEFAULT_BOX_THRESHOLD, DEFAULT_PROMPT, DEFAULT_TEXT_THRESHOLD
from vision.types import Detection, DepthField, PerceptionLike

__all__ = ["BuildConfig", "BuildResult", "build_scene_graph", "slugify"]

#: 关系生成策略。默认「全对距离 + 所有判定为真的布尔关系」。
#: - `distance_plus_true`：稀疏、信息密度高。190 对物体约产出 190 + 数十条边。
#: - `all`：所有布尔关系都建边（含假值）。边数约 10×，仅用于消融对照。
#: - `distance_only`：只建距离边。L5 的 `describe_scene` 需要全量关系，用这个会缺数据。
RelationPolicy = Literal["distance_plus_true", "all", "distance_only"]


@dataclass(frozen=True)
class BuildConfig:
    """一次构建的全部可调项 —— 固化成值对象，方便进实验臂指纹与落盘复现。"""

    prompt: str = DEFAULT_PROMPT
    box_threshold: float = DEFAULT_BOX_THRESHOLD
    text_threshold: float = DEFAULT_TEXT_THRESHOLD

    #: 掩码内至少多少点才采信其质心。低于此值退回检测框（并标记 degraded）。
    min_points: int = DEFAULT_MIN_POINTS
    #: 面积小于此值（像素²）的检测框直接丢弃 —— 通常是噪声。
    min_box_area_px: float = 64.0
    #: 去重阈值（IoU）。0 = 关闭。高置信度的框优先保留，**跨类别也去重**：
    #: 两个不同词命中同一块区域时，保留两个节点比保留一个更糟。
    dedupe_iou: float = 0.85
    #: 单张图最多保留多少个节点，按置信度排序截断。防止 prompt 写太宽导致场景爆炸。
    max_objects: int = 40

    #: 关系容差与阈值，透传给 `relations.pairwise`。
    tol: float = DEFAULT_TOL
    near_m: float = DEFAULT_NEAR_M
    relation_policy: RelationPolicy = "distance_plus_true"

    #: 是否估计重力方向（画面下部点云拟合平面）。关掉则直接用 x 轴对齐的 `-y`。
    estimate_up: bool = True
    up_band: float = 0.2
    max_tilt_deg: float = 25.0

    #: 已知内参 `(fx, fy, cx, cy)`，**像素单位**，且必须对应当前输入图像的分辨率。
    #:
    #: 有就一定要传 —— Phase 0 Step 6 实测三维误差中位数 1.943 m → 0.267 m（7.3×）。
    #: 只知 fx/fy 而不知道主点时，主点填几何中心 `(W/2, H/2)` 仍然远好于不传：
    #: 误差的主要来源是 focal（实测预测 fx=163.7 vs 真值 518.9，差 3.17 倍），
    #: 主点偏移的影响是小一个量级的。
    #:
    #: 用途举例：Omni3D-Bench 自带 GT 相机、相机标定文件、手机照片的 EXIF 等效焦距。
    known_intrinsics: tuple[float, float, float, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        """记录用的可读形态（进 `build_meta`，键名带单位更易读）。

        ⚠️ 它是给**落盘与比对**用的，键名与字段名并不一一对应（`tol` → `tol_m`），
        所以**不要**用它来重建 `BuildConfig` —— 那要用 `replace()`。
        """
        return {
            "prompt": self.prompt,
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
            "min_points": self.min_points,
            "min_box_area_px": self.min_box_area_px,
            "dedupe_iou": self.dedupe_iou,
            "max_objects": self.max_objects,
            "tol_m": self.tol,
            "near_m": self.near_m,
            "relation_policy": self.relation_policy,
            "estimate_up": self.estimate_up,
            "up_band": self.up_band,
            "max_tilt_deg": self.max_tilt_deg,
            "known_intrinsics": (
                [round(float(v), 2) for v in self.known_intrinsics]
                if self.known_intrinsics
                else None
            ),
        }

    def replace(self, **changes: Any) -> "BuildConfig":
        """按字段名派生一个新配置（只覆盖给定字段）。

        用 `dataclasses.replace` 而不是 `BuildConfig(**as_dict(), ...)`：
        后者的键名与字段名不一致（见 `as_dict`），会在改配置时炸掉，
        而那时通常正在做消融对比 —— 最不该被打断的时候。
        """
        import dataclasses

        return dataclasses.replace(self, **changes)


@dataclass
class BuildResult:
    """构建产物。`scene` 是权威结果，其余是诊断材料。"""

    scene: SceneGraph
    #: `object_id -> (H, W) bool 掩码`。由调用方决定存不存盘（见 `store.save_masks`）。
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    #: 人类可读的警告。空列表 = 一切正常。**不要**把它当装饰 —— 它是失败诊断的一手输入。
    warnings: list[str] = field(default_factory=list)
    #: 结构化统计（含耗时）。原样进 `scene.build_meta`。
    stats: dict[str, Any] = field(default_factory=dict)
    #: ★ 整图点云 `(3, H, W)`，模型网格。落盘由 `store.save_points` 做。
    #:
    #: 为什么 builder 要把它交出来：落进 `scene.json` 的只有 `centroid_3d` /
    #: `extent_3d` / `bbox_3d` 这几个**标量摘要**，点云本身拿不到 ——
    #: 于是任何点云级算法（有向包围盒、体积、主方向、平面拟合、聚类补漏检）
    #: 都无从下手。交出来之后，「某个物体的点云」可以由「整图点云 + 掩码」
    #: 重建（`scene_graph.pointcloud`），不需要多存一份副本。
    #:
    #: 它是**引用**不是副本：与 `field.points_chw` 指向同一块内存。
    points_chw: np.ndarray | None = None
    #: 点云的重建元信息（网格尺寸 / 图像尺寸 / 实际生效的内参…）。
    #: 与掩码一样，**builder 不写盘**，只交出来，落盘由调用方决定。
    points_meta: dict[str, Any] = field(default_factory=dict)


def slugify(label: str) -> str:
    """类别名 → 可放进 id 的形式。

    `"Coffee Table"` → `"coffee_table"`。只保留 `[0-9a-z_]`，避免 id 里出现
    空格或标点 —— 那些字符会让 LLM 生成程序时不得不加引号转义，实测是它写错
    参数的高发区。空标签退化成 `"object"` 而不是空串，免得 id 变成 `"_1"`。
    """
    s = re.sub(r"[^0-9a-z]+", "_", label.strip().lower())
    return s.strip("_") or "object"


def _dedupe(
    detections: list[Detection],
    image_hw: tuple[int, int],
    iou_threshold: float,
) -> tuple[list[Detection], list[dict[str, Any]]]:
    """按置信度降序贪心去重，返回 `(保留, 被丢弃的记录)`。

    **跨类别也去重**：两个不同的词命中几乎同一块区域时（例如同一个桌子上
    同时被 "table" 和 "desk" 命中），保留两个节点会让「有几个桌子」这种问题
    直接算错，而且会让关系图里出现一条 self-loop 式的假边。
    """
    ordered = sorted(
        detections,
        key=lambda d: (-float(d.score), d.label, float(d.box_xyxy[0]), float(d.box_xyxy[1])),
    )
    kept: list[Detection] = []
    dropped: list[dict[str, Any]] = []
    for det in ordered:
        box = clip_box_xyxy(det.box_xyxy, image_hw)
        clash = None
        if iou_threshold > 0.0:
            for k in kept:
                if iou_xyxy(box, k.box_xyxy) >= iou_threshold:
                    clash = k
                    break
        if clash is None:
            kept.append(
                Detection(label=det.label, score=det.score, box_xyxy=box)
            )
        else:
            dropped.append(
                {
                    **det.as_dict(),
                    "reason": "duplicate_box",
                    "duplicate_of": clash.label,
                    "iou": round(iou_xyxy(box, clash.box_xyxy), 4),
                }
            )
    return kept, dropped


def _assign_ids(detections: list[Detection]) -> list[tuple[str, Detection]]:
    """给检测结果分配稳定 id：`{slug}_{序号}`，序号按置信度降序、从 1 开始。

    「稳定」指的是：同一张图、同一套权重、同样的 prompt，必然得到同样的 id。
    这是相对 VADAR 的关键改进 —— 它每次 `loc()` 都回裸 bbox，物体身份要靠
    `same_object(iou>0.92)` 反推（`predefined_modules.py`），既不准又浪费一次工具调用。
    """
    counters: dict[str, int] = {}
    out: list[tuple[str, Detection]] = []
    for det in detections:
        slug = slugify(det.label)
        counters[slug] = counters.get(slug, 0) + 1
        out.append((f"{slug}_{counters[slug]}", det))
    return out


def build_scene_graph(
    image: Any,
    *,
    perception: PerceptionLike,
    scene_id: str = "scene",
    image_id: str | None = None,
    config: BuildConfig | None = None,
    mask_rel_prefix: str | None = None,
) -> BuildResult:
    """把一张图建成一张场景图。

    `mask_rel_prefix` 非空时，每个节点会带上 `mask_ref = f"{prefix}/{object_id}.png"`；
    builder **不写盘**（保持纯函数、可测），落盘由 `store.save_masks` 做。
    """
    cfg = config or BuildConfig()
    image_id = image_id or scene_id
    image_hw = (int(image.size[1]), int(image.size[0]))
    warnings: list[str] = []
    timings: dict[str, float] = {}

    def _t() -> float:
        return time.perf_counter() * 1000.0

    # ---- ① 升维（先做：它决定点云网格，后面所有换算都依赖它）-----------------
    # 内参是**逐图**输入：cx/cy 与分辨率绑定，所以它走 lift() 的形参，
    # 而不是挂在 PerceptionStack 上（registry 只管模型生命周期，不持有图像状态）。
    known_K = (
        intrinsics_matrix(*cfg.known_intrinsics) if cfg.known_intrinsics else None
    )
    t0 = _t()
    field: DepthField = perception.lift(image, camera_K=known_K)
    timings["depth_ms"] = _t() - t0

    # 视场体检放在最前面：它是**横向尺度**唯一的总开关。让它早于节点生成，
    # 警告的阅读顺序才与「错误从哪儿来」一致 —— 否则读者会先去怀疑分割或关系。
    fov = field.fov or check_fov(field.intrinsics, image_hw)
    if field.intrinsics_source == "predicted":
        if not fov.plausible:
            warnings.append(
                f"内参是模型预测的，且视场不可信（HFoV {fov.hfov_deg:.1f}°，"
                f"reason={fov.reason}，可信区间 {PLAUSIBLE_HFOV_DEG[0]:.0f}–"
                f"{PLAUSIBLE_HFOV_DEG[1]:.0f}°）—— **所有横向米制尺寸可能被整体"
                "放大**，本图的坐标只能当相对量用，报告里必须标注。"
                "有已知内参请设 BuildConfig.known_intrinsics。"
                "量化证据见 phase0/probe_depth_gt.py。"
            )
    elif fov.plausible is False:
        # 来源是 provided 但视场照样不可信 —— 大概率是内参与图像分辨率不匹配
        # （例如把 1280×960 的标定参数直接用在 640×480 的图上），
        # 或者传错了顺序。这种情况必须说出来，不能默认「调用方一定对」。
        warnings.append(
            f"传入的已知内参视场不可信（HFoV {fov.hfov_deg:.1f}°，"
            f"reason={fov.reason}）—— 请确认它对应的是当前图像分辨率 "
            f"{image_hw[1]}×{image_hw[0]}，以及 (fx, fy, cx, cy) 的顺序。"
        )

    # ---- ② 检测 -------------------------------------------------------------
    t0 = _t()
    raw = list(perception.detect(
        image, cfg.prompt,
        box_threshold=cfg.box_threshold,
        text_threshold=cfg.text_threshold,
    ))
    timings["detect_ms"] = _t() - t0

    # 过滤：面积太小 / 退化框。这些不进去重阶段 —— 它们连「一个物体」都不是。
    viable: list[Detection] = []
    dropped: list[dict[str, Any]] = []
    for det in raw:
        box = clip_box_xyxy(det.box_xyxy, image_hw)
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area < cfg.min_box_area_px:
            dropped.append({**det.as_dict(), "reason": "box_too_small",
                            "area_px": round(area, 1)})
            continue
        viable.append(det)

    kept, dupes = _dedupe(viable, image_hw, cfg.dedupe_iou)
    dropped.extend(dupes)

    if len(kept) > cfg.max_objects:
        for det in kept[cfg.max_objects:]:
            dropped.append({**det.as_dict(), "reason": "over_max_objects"})
        kept = kept[: cfg.max_objects]

    named = _assign_ids(kept)
    if not named:
        warnings.append(
            f"没有检测到任何物体（prompt={cfg.prompt!r}）。"
            "检查 prompt 是否是「小写 + 每个标签以句点结尾」，或降低 box_threshold。"
        )

    # ---- ③ 分割（★ 一次调用带全部框）----------------------------------------
    # 实测 9 个框：一次调用 200 ms，逐个调用 1512 ms —— 差 7.57×（§20 Step 0.5b）。
    # 这条约束由接口形状保证：`segment()` 只接受一批框，没有单框重载。
    t0 = _t()
    boxes = [d.box_xyxy for _, d in named]
    masks_img = perception.segment(image, boxes) if boxes else np.zeros((0, 0, 0), bool)
    timings["segment_ms"] = _t() - t0

    if len(named) and len(masks_img) != len(named):
        warnings.append(
            f"掩码数 {len(masks_img)} 与框数 {len(named)} 不一致，多余的框将走降级路径。"
        )

    # ---- ④ 升维到节点 -------------------------------------------------------
    t0 = _t()
    nodes: list[Node] = []
    masks_out: dict[str, np.ndarray] = {}
    fallbacks: list[dict[str, Any]] = []
    coverages: list[float] = []

    for i, (object_id, det) in enumerate(named):
        mask_img = masks_img[i] if i < len(masks_img) else None
        if mask_img is None:
            warnings.append(f"{object_id}: 没有拿到掩码，走检测框降级路径")

        # 掩码在**图像**分辨率，点云在**模型**网格 —— 必须显式重采样。
        mask_grid = (
            resample_mask_to(mask_img, field.grid_hw)
            if mask_img is not None and mask_img.size
            else np.zeros(field.grid_hw, dtype=bool)
        )

        pts = select_points(field.points_chw, mask_grid)
        source: Literal["mask", "bbox_fallback"] = "mask"

        if pts.shape[1] < cfg.min_points:
            # 降级：掩码太空（检测框打在白墙/天空上时很常见），退回框内点云。
            # 已知精度更差（均值 83 mm），但**丢掉物体**比定位不准更糟 ——
            # 丢掉会让 find_object 报 NOT_FOUND，模型会以为图里没这东西。
            box_pts = select_points(
                field.points_chw,
                box_selector(det.box_xyxy, field.grid_hw, image_hw),
            )
            if box_pts.shape[1] >= cfg.min_points:
                pts = box_pts
                source = "bbox_fallback"
                fallbacks.append({
                    "object_id": object_id,
                    "reason": "mask_too_small",
                    "n_mask_points": int(select_points(field.points_chw, mask_grid).shape[1]),
                    "n_box_points": int(box_pts.shape[1]),
                })
                warnings.append(
                    f"{object_id}: 掩码仅 {select_points(field.points_chw, mask_grid).shape[1]} 点，"
                    "已退回检测框（质心精度下降，centroid_source=bbox_fallback）"
                )
            else:
                # 掩码和框都没有有效点 —— 无法给出米制坐标。
                # 直接丢掉而不是编一个坐标：Node 的核心契约是「质心由几何算出」，
                # 凭空造一个会让整条证据链失效。
                dropped.append({
                    **det.as_dict(),
                    "reason": "no_valid_points",
                    "n_mask_points": int(select_points(field.points_chw, mask_grid).shape[1]),
                    "n_box_points": int(box_pts.shape[1]),
                })
                warnings.append(f"{object_id}: 掩码与框内都无有效点云，已从场景图剔除")
                continue

        centroid, cmeta = centroid_of(pts)
        assert centroid is not None  # pts.shape[1] >= min_points > 0
        extent, lo, hi, emeta = robust_extent(pts)

        if mask_img is not None and mask_img.size:
            coverages.append(box_coverage(mask_img, det.box_xyxy))

        mask_ref = f"{mask_rel_prefix}/{object_id}.png" if mask_rel_prefix else None
        if mask_img is not None and mask_img.size:
            masks_out[object_id] = mask_img

        nodes.append(
            Node(
                id=object_id,
                label=det.label,
                score=float(det.score),
                bbox_2d=tuple(float(v) for v in det.box_xyxy),
                mask_ref=mask_ref,
                centroid_3d=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                extent_3d=(float(extent[0]), float(extent[1]), float(extent[2])),
                bbox_3d=BBox3D(
                    min=(float(lo[0]), float(lo[1]), float(lo[2])),
                    max=(float(hi[0]), float(hi[1]), float(hi[2])),
                ),
                n_points=int(cmeta["n_points"]),
                centroid_source=source,
            )
        )
    timings["nodes_ms"] = _t() - t0

    # ---- ⑤ 重力方向 ---------------------------------------------------------
    up_axis = "-y"
    up_est: UpAxisEstimate | None = None
    if cfg.estimate_up:
        t0 = _t()
        up_est = estimate_up_axis(
            field.points_chw, band=cfg.up_band, max_tilt_deg=cfg.max_tilt_deg
        )
        timings["up_axis_ms"] = _t() - t0
        up_axis = up_est.axis
        if not up_est.reliable:
            warnings.append(
                f"重力方向估计不可靠（reason={up_est.reason}, "
                f"tilt={up_est.tilt_deg:.1f}°）—— above/below 类关系在本题上不可采信，"
                "报告里必须显式说明"
            )

    # ---- ⑥ 关系（几何算，不经过任何模型）------------------------------------
    # 边**只按 (i, j)、i<j 单方向枚举**，反向关系不重复存：
    # 「chair_2 在 chair_1 右边」与「chair_1 在 chair_2 左边」是同一件事，
    # 后者已经在图里。存两份等于让边数翻倍而信息量不变 —— 而边数是
    # L5 `describe_scene` 与指标 12（关系边一致率）的直接成本。
    # 反向查询由 `query_relation` 现算，不依赖存下来的边。
    t0 = _t()
    edges: list[Edge] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            # `pairwise` 已经把「哪些关系需要 bbox」处理掉了：
            # 缺 bbox_3d 的节点会跳过 on/inside，其余照常给出 ——
            # 不因为一项缺数据就丢掉整对关系。
            for relation, verdict in pairwise(
                a, b, up=up_axis, tol=cfg.tol, near_m=cfg.near_m
            ).items():
                if cfg.relation_policy == "distance_only" and relation != "distance":
                    continue
                # `far` 是 `near` 的补集，两条都存等于把信息翻倍而信息量不变。
                if relation == "far":
                    continue
                is_bool = verdict.is_bool
                if cfg.relation_policy == "distance_plus_true" and is_bool and not bool(verdict.value):
                    continue
                edges.append(
                    Edge(
                        source=a.id,
                        target=b.id,
                        relation=relation,  # type: ignore[arg-type]
                        **verdict.as_edge_fields(),  # type: ignore[arg-type]
                    )
                )
    timings["relations_ms"] = _t() - t0

    # ---- ⑦ 组装 -------------------------------------------------------------
    depth_lo, depth_hi = field.depth_range_m
    gsx, gsy = field.grid_scale_vs_image()

    # label_counts 必须在**构造 SceneGraph 之前**算好：Pydantic 会复制 dict 字段，
    # 构造之后再改 `stats` 不会反映到 `scene.build_meta` 里（这个坑很安静）。
    label_counts: dict[str, int] = {}
    for n in nodes:
        label_counts[n.label] = label_counts.get(n.label, 0) + 1

    stats: dict[str, Any] = {
        "config": cfg.as_dict(),
        "timings_ms": {k: round(v, 1) for k, v in timings.items()},
        "image_hw": list(image_hw),
        "grid_hw": list(field.grid_hw),
        "grid_scale_vs_image": [round(gsx, 4), round(gsy, 4)],
        "depth_range_m": [round(depth_lo, 4), round(depth_hi, 4)],
        "n_detections_raw": len(raw),
        "n_detections_kept": len(kept),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_dropped": len(dropped),
        "n_fallbacks": len(fallbacks),
        "dropped": dropped,
        "fallbacks": fallbacks,
        "mask_box_coverage_mean": round(float(np.mean(coverages)), 4) if coverages else None,
        "mask_box_coverage_min": round(float(np.min(coverages)), 4) if coverages else None,
        "label_counts": label_counts,
        "up_axis": up_axis,
        "up_axis_tilt_deg": round(up_est.tilt_deg, 2) if up_est else None,
        "up_axis_reliable": bool(up_est.reliable) if up_est else False,
        "up_axis_reason": up_est.reason if up_est else "not_estimated",
        # 显式声明「没做尺度校正」——避免下游把这批数字当成已标定的。
        "scale_calibrated": False,
        # 内参与视场：横向尺度的总开关。`intrinsics_source="predicted"` 时，
        # 下游（L5 报告、指标表）必须把它当成「相对量」而不是「绝对量」来用。
        "intrinsics_source": field.intrinsics_source,
        "intrinsics": [
            [round(float(v), 6) for v in row] for row in np.asarray(field.intrinsics).tolist()
        ],
        "fov": fov.as_dict(),
        "perception": (
            perception.stats()
            if hasattr(perception, "stats")
            else {"kind": type(perception).__name__}
        ),
    }

    scene = SceneGraph(
        scene_id=scene_id,
        image_id=image_id,
        camera_intrinsics=[
            [float(v) for v in row] for row in np.asarray(field.intrinsics).tolist()
        ],
        up_axis=up_axis,
        scale_factor=1.0,
        nodes=tuple(nodes),
        edges=tuple(edges),
        build_meta=stats,
    )

    return BuildResult(
        scene=scene,
        masks=masks_out,
        warnings=warnings,
        stats=stats,
        points_chw=field.points_chw,
        points_meta={
            "grid_hw": list(field.grid_hw),
            "image_hw": list(image_hw),
            # ★ 必须记**实际生效**的那一份内参，不是模型回传的那一份。
            #   传了已知内参时两者不等（实测：fx=518.9 传进去、回传仍是 163.7），
            #   而点云是用前者生成的。记错这一项，"用落盘的点云重算三维量"
            #   这件事就从"可复现"退化成"看起来对"。
            "intrinsics": [
                [round(float(v), 6) for v in row]
                for row in np.asarray(field.intrinsics).tolist()
            ],
            "intrinsics_source": field.intrinsics_source,
            # 与 build_meta 里那条一致：点云是**未标定**的，米制量只能当相对量。
            # 写在点云自己的元信息里，是为了让只读 points_meta.json 的下游
            # （例如某个点云工具）也不会误以为它标定过。
            "scale_calibrated": False,
        },
    )
