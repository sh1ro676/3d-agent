#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把合成夹具接到**真实** `build_scene_graph` 上的那一段桥。

它为什么存在
------------
`dataset/builders/synthesize_geometry_probe.py` 产出的是**解析真值**（点云、
掩码、两层盒子）；`scene_graph/builder.py` 要的是一个 `PerceptionLike`
（`detect` / `segment` / `lift` 三个方法）。两者之间此前**没有桥**，于是
builder 的 ④ 段（掩码 → 点云 → 质心/包围盒）只在 `tests/test_builder.py`
的 fake 上被验过 —— 而那个 fake 的点云是 `pinhole_cloud` 造的**解析常量深度**：
没有遮挡、没有背景、没有 nan 空洞。

这正是两条降级路径的触发条件，也是它们此前从没被量过的原因：

===========================  ====================================================
`bbox_fallback`               掩码太空（检测框打在白墙上）。真实照片里是**常态**
`no_valid_points`             掩码与框内都无有效点。真实照片里出现在镜面 / 天空 /
                              遮挡后 —— 此时物体**整个消失**，会让 `find_object`
                              报 `NOT_FOUND`，模型于是以为图里没这东西
===========================  ====================================================

它**不**声称什么（这一段比上面那段重要）
---------------------------------------
本桥的检测与分割都取夹具真值，所以它测的是「**聚合段 + 降级路径**」，
不是感知质量。逐条列清，免得报告里的数字被读大：

* **`detect` 按标签集精确过滤**，不是文本相似度匹配 ⟹ 覆盖「prompt 漏了某个
  类别」的**后果**（节点消失、`find_object` 报 NOT_FOUND），**不**覆盖
  「prompt 措辞如何影响召回率」。这一条的实测后果见
  `scripts/run_aggregation_probe.py` 的 `[⑤ prompt 召回]`。
* **`segment` 按框 IoU 查表返回 GT 掩码** ⟹ 不含 SAM2 的边界误差。
  边界误差走 `apply_perturbation` 的 `mask_grow` / `mask_shrink`（§23.3），
  不在本桥重复注入。
* **不做遮挡剔除**：被挡住的物体照样被报出来。真实 GroundingDINO 多半也会报
  （它看不见遮挡关系）。遮挡带来的偏差落在 `gt_visible` vs `gt_box` 那一层
  （§23.4 的"固有偏差"），不在本桥重复注入 —— 在聚合段注入会把两层的账混起来。
* **`score` 是固定值**。分数不是本层要测的东西，让它变化只会把「分数高低」
  误读成感知质量。
* **图像内容是占位灰图**：`detect` / `segment` 都不看像素，只有 `image.size`
  有意义（builder 用它当图像分辨率）。真实链路里这两个方法**全靠纹理**。

零 torch：只依赖 `vision.types` / `vision.geometry`（两者都刻意零 torch）
与 PIL（只为造一张尺寸正确的占位图）。整条探针秒级、零 API、零 GPU。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from dataset.builders.synthesize_geometry_probe import (
    Box3D,
    SyntheticScene,
    render_scene,
)
from vision.geometry import iou_xyxy
from vision.types import Detection, DepthField, box_xyxy_of

__all__ = [
    "FIDELITY_NOTES",
    "Degradation",
    "SyntheticIntrinsicsError",
    "SyntheticPerception",
    "parse_prompt_labels",
    "project_box_xyxy",
]


#: 本桥的保真度声明 —— **单一来源**：`stats()` 与模块 docstring 都说这一份。
#: 放在常量里而不是写在两个地方，是因为两处写法一旦漂移，
#: 读得最顺的那一处会被当成事实，而它恰好可能是漏写最多的那一处。
FIDELITY_NOTES: tuple[str, ...] = (
    "检测/分割取夹具真值：本桥测的是聚合段与降级路径，不是感知质量。",
    "detect 按标签集精确过滤，不是文本相似度匹配。",
    "segment 按框 IoU 查表返回 GT 掩码，不含 SAM2 的边界误差。",
    "不做遮挡剔除：被挡物体照样报出（遮挡偏差属 gt_visible 那一层）。",
    "score 为固定值：分数不是本层要测的量。",
    "图像内容为占位灰图，仅 image.size 有意义。",
)

#: 占位图的填充值。选 128 而不是 0：全黑的图容易在后续人工排查时被误当成
#: 「渲染失败」，而这个值不会与任何真实的渲染输出相像。
_PLACEHOLDER_RGB = (128, 128, 128)


class SyntheticIntrinsicsError(RuntimeError):
    """内参没有按预期到达感知层 —— **必须响**。

    用 `RuntimeError` 而不是 `ValueError`：这不是「参数不合法」，而是
    「一条此前只能靠人眼发现的契约被违反了」。builder 不捕获它，所以它会
    一路冒到调用方 —— 这正是想要的：内参错 3.17 倍是**静默**的
    （§20.3），能在合成数据上把它变成异常，是这块真值最值钱的用途之一。
    """


@dataclass(frozen=True)
class Degradation:
    """把一个物体的**感知输出**改成指定的失效形态。

    ⚠ 这不是「模拟真实感知的统计特性」—— 真实链路里这两种失效的**分布**
    没有被本模块刻画（那需要真实照片 + 真实模型）。它是一次**受控注入**，
    全部目的是把 builder 的两条降级分支搬到**有真值**的地方去量。

    两个开关各自诱发**一条**降级路径，而且根因不同 —— 这正是 builder 要区分
    它们的原因：

        mask_empty          分割失败：框对了，但框里分不出东西。
                            掩码取不到点、框内**有**点 ⟹ `bbox_fallback`
                            （质心中位数被框内的背景点拉向远处）。
                            注入区域 = 该物体的掩码，语义干净。

        nan_depth_in_box    深度失败：物体的可见像素没有有效深度
                            （镜面 / 天空 / 过曝 / 遮挡后）。
                            掩码内与框内**都**没有有效点 ⟹ `no_valid_points`，
                            该物体从场景图里消失 ⟹ 下游 `find_object` 报
                            `NOT_FOUND`，模型于是以为图里没这东西。

    ⚠ **我第一版把 `no_valid_points` 写成需要两个开关同时打开，那是错的。**
    实测（`scripts/run_aggregation_probe.py` 的 `[③ 深度空洞]`）：单独打开
    `nan_depth_in_box` 就足以触发 —— 注入区域用的是**投影框**
    （`box_selector` 会向外取整），而物体的掩码 ⊂ 投影框，所以掩码内的点
    必然一起变 nan。一个开关就能开出效果，比两个开关的组合少一种失败方式。

    同样由实测钉住的另一条：单独打开 `mask_empty` **不会**触发
    `no_valid_points`（框内还有 5082 个点）。两条路径不会互相冒充 ——
    否则「测的是哪条」这件事本身就说不清了。
    """

    #: 该物体的掩码整片抹空（模拟「框打在白墙上，SAM2 没分出东西」）。
    mask_empty: bool = False
    #: 该物体的**投影框**向外扩 `N` 像素（模拟检测框偏大 / 偏松）。
    #:
    #: ⚠ 它同时改两件事，两件都有意义：框内点云的多少（决定 `bbox_fallback`
    #: 代价的大小）、以及 `box_coverage`（框里有多少是背景）。真实
    #: GroundingDINO 的框几乎总比物体大一点，所以 `box_pad_px = 0` 是最乐观的一头。
    box_pad_px: float = 0.0
    #: 该物体的**投影框内**点云置 nan（模拟「这些方向的深度反投影失败」）。
    nan_depth_in_box: bool = False
    #: 自由文本，会进 `stats()`，用来在报告里区分同一组里的不同注入。
    note: str = ""


def parse_prompt_labels(prompt: str) -> set[str]:
    """把 `"sofa. chair. table."` 解析成 `{"sofa", "chair", "table"}`。

    规则来自 `vision/grounding.py` 写死的那条契约：**小写 + 每个标签以句点结尾**。
    这里按句点切分、去空白、转小写 —— 与那份契约同形。

    ⚠ 它是**精确集合**语义，不是相似度：真实 GroundingDINO 用文本编码器算
    相似度，`"sofa"` 也能命中一把椅子。所以本函数只用来问
    「**prompt 有没有问这个类别**」，不能用来预测真实召回率。
    """
    return {p.strip().lower() for p in prompt.split(".") if p.strip()}


def project_box_xyxy(
    box: Box3D,
    *,
    intrinsics: np.ndarray,
) -> tuple[float, float, float, float] | None:
    """把一个 3D 盒子投影成图像系 `(x1, y1, x2, y2)`（取 8 角点的外接框）。

    **不裁进画面**：裁是 builder 的 `clip_box_xyxy` 的职责，在这里先裁一次
    会让「这个物体有多少在画面外」这件事在中途被抹掉。`box_selector` 与
    `iou_xyxy` 都按「连续像素边界坐标」理解 xyxy，而 `_pixel_rays` 的像素中心
    是 `arange + 0.5` ⟹ 本函数的输出与它们同一套坐标，不需要再补半个像素。

    有角点落在相机后方（`z <= 0`）时返回 `None`：此时透视投影在数学上不成立，
    硬算会得到一个**镜像的**、看起来还挺合理的框。宁可返回 `None` 让调用方
    显式处理 —— 合成场景里不该出现这种情况，真出现了说明场景定义错了。

    `yaw` 由 `Box3D.corners()` 带进来（它已含旋转），所以带朝向的盒子
    投影天然正确；本桥的探针目前只跑 `yaw = 0` 的场景。
    """
    intr = np.asarray(intrinsics, dtype=np.float64)
    if intr.shape != (3, 3):
        raise ValueError(f"intrinsics 必须是 3×3，收到 shape={intr.shape}")

    corners = np.asarray(box.corners(), dtype=np.float64)  # (8, 3)
    z = corners[:, 2]
    if not bool((z > 0.0).all()):
        return None

    fx, fy = float(intr[0, 0]), float(intr[1, 1])
    cx, cy = float(intr[0, 2]), float(intr[1, 2])
    u = fx * corners[:, 0] / z + cx
    v = fy * corners[:, 1] / z + cy
    return (float(u.min()), float(v.min()), float(u.max()), float(v.max()))


def _match_object(
    box_xyxy: Sequence[float],
    projected: Mapping[str, tuple[float, float, float, float]],
) -> str | None:
    """找出与 `box_xyxy` IoU 最大的物体 id；完全无交集时返回 `None`。

    并列时取**先出现**的那个（`projected` 的顺序 = 渲染时 boxes 的顺序），
    于是同一个输入必得同一个结果 —— 遮挡会让两个物体的投影框重叠，
    而「今天匹配到 A、明天匹配到 B」这种不确定性会让整份报告无法复现。
    """
    best_id: str | None = None
    best_iou = 0.0
    for oid, pbox in projected.items():
        value = iou_xyxy(box_xyxy, pbox)
        if value > best_iou:
            best_id, best_iou = oid, value
    return best_id


class SyntheticPerception:
    """`PerceptionLike` 的合成实现：检测与分割取夹具真值，点云直接用解析渲染。

    用法（推荐走 `from_boxes`，它保证 `scene` 与 `boxes` **同源**）：

        per = SyntheticPerception.from_boxes(
            boxes, intrinsics=INTR, image_hw=(240, 320),
            degrade={"plant_1": Degradation(mask_empty=True)},
            require_camera_K=True,
        )
        result = build_scene_graph(per.image, perception=per, config=cfg)

    ⚠ `require_camera_K=True` 只是「要求调用方传」，真正的校验在 `lift()`：
    传进来的 `camera_K` 必须与合成点云赖以生成的内参**逐位相同**，否则抛
    `SyntheticIntrinsicsError`。真实链路拿不到这个校验（它不知道真值），
    但也不该因此在本桥上放行。
    """

    def __init__(
        self,
        scene: SyntheticScene,
        boxes: Sequence[Box3D],
        *,
        labels: Mapping[str, str] | None = None,
        degrade: Mapping[str, Degradation] | None = None,
        require_camera_K: bool = False,
        score: float = 0.90,
        image: Any | None = None,
    ) -> None:
        self.scene = scene
        self.boxes: tuple[Box3D, ...] = tuple(boxes)
        self.require_camera_K = bool(require_camera_K)
        self.score = float(score)

        h, w = (int(v) for v in scene.grid_hw)
        self._image_hw = (h, w)
        self._grid_hw = (h, w)

        if image is None:
            from PIL import Image

            image = Image.new("RGB", (w, h), _PLACEHOLDER_RGB)
        if (int(image.size[1]), int(image.size[0])) != (h, w):
            raise ValueError(
                f"image 尺寸 {image.size[1]}×{image.size[0]} 与夹具网格 {h}×{w} 不一致 —— "
                "builder 拿 image.size 当图像分辨率，不一致会让 box_selector 的"
                "图像→网格换算整体偏移，而它**不会报错**，只会安静地给错框内的点。"
            )
        self.image = image

        # ---- 与 scene 同源校验：合成场景 + 一份不同的盒子清单 = 一份无法追溯的报告 ----
        # ⚠ 只比**前景**盒子：背景盒子（墙/地板）刻意不进 `scene.gt_box`
        #   —— 它们只提供深度，不算物体。这一点漏掉会让这条守卫在**每一个**
        #   带背景的正常场景上误报，而误报的守卫会被人顺手关掉，连带关掉它
        #   真正要守的东西。
        by_id = {b.object_id: b for b in self.boxes}
        fg = {oid: b for oid, b in by_id.items() if not b.is_background}
        if set(fg) != set(scene.gt_box):
            raise ValueError(
                "boxes 与 scene 不是同一次渲染的结果（只比前景物体，背景已排除）：\n"
                f"  只在 boxes 里 {sorted(set(fg) - set(scene.gt_box))}\n"
                f"  只在 scene 里 {sorted(set(scene.gt_box) - set(fg))}"
            )
        for oid, box in fg.items():
            gt = scene.gt_box[oid]
            if not (
                np.array_equal(np.asarray(box.min_xyz, dtype=np.float64), gt["bbox_min"])
                and np.array_equal(np.asarray(box.max_xyz, dtype=np.float64), gt["bbox_max"])
            ):
                raise ValueError(
                    f"{oid}: boxes 里的盒子与 scene 真值不一致 "
                    f"（boxes {box.min_xyz}–{box.max_xyz} vs scene "
                    f"{gt['bbox_min'].tolist()}–{gt['bbox_max'].tolist()}）—— "
                    "用一份盒子渲染、又用另一份投影，得到的框和掩码会互相错位，"
                    "而两边的数字各自都看着很合理。"
                )

        # ---- 标签：夹具把 label 放在 meta['boxes'] 里，gt_box 里没有 ----
        meta_labels = {
            str(b["object_id"]): str(b["label"]) for b in scene.meta.get("boxes", [])
        }
        merged = dict(meta_labels)
        if labels:
            merged.update({str(k): str(v) for k, v in labels.items()})
        missing = sorted(set(fg) - set(merged))
        if missing:
            raise ValueError(
                f"这些物体没有 label：{missing} —— 需要 labels={{...}} 显式给出，"
                "否则它们会在 detect() 里被静默丢掉（name 都没有，无从匹配 prompt）"
            )
        self._labels: dict[str, str] = {oid: merged[oid] for oid in fg}

        self._degrade: dict[str, Degradation] = {
            str(k): v for k, v in (degrade or {}).items()
        }
        unknown = sorted(set(self._degrade) - set(fg))
        if unknown:
            raise ValueError(
                f"degrade 指向了不存在的物体 {unknown} —— 拼错一个 id 的后果是"
                "「注入没生效」，而报告里那一栏会显示「一切正常」，"
                "正是本模块要防的那种静默。"
            )

        # ---- 投影一次，三处共用（detect 的框、segment 的匹配、nan 注入的区域）----
        self._projected: dict[str, tuple[float, float, float, float]] = {}
        self._unprojectable: list[str] = []
        for oid, box in fg.items():
            pbox = project_box_xyxy(box, intrinsics=scene.intrinsics)
            if pbox is None:
                self._unprojectable.append(oid)
                continue
            pad = float(self._deg(oid).box_pad_px)
            if pad:
                # 外扩发生在**投影之后**：真实检测框偏大也是「画出来的框偏大」，
                # 而不是把物体本身放大。两者的区别在 `mask_empty` 那一档里
                # 才是可见的 —— 那时框内点云完全由这个框决定。
                pbox = (pbox[0] - pad, pbox[1] - pad, pbox[2] + pad, pbox[3] + pad)
            self._projected[oid] = pbox

        #: 前景物体数（背景盒子刻意不计 —— 它们不进掩码、不进真值、不进检测）。
        self.n_foreground = len(fg)
        #: 背景盒子数。落进 `stats()`，让「画面每个方向都有深度」这件事可追溯。
        self.n_background = len(by_id) - len(fg)
        self._pts: np.ndarray | None = None
        self.detect_calls: list[dict[str, Any]] = []
        self.segment_calls: list[list[tuple[float, ...]]] = []
        self.segment_matches: list[list[str | None]] = []
        self.lift_calls: list[np.ndarray | None] = []

    # ------------------------------------------------------------------
    # 工厂
    # ------------------------------------------------------------------

    @classmethod
    def from_boxes(
        cls,
        boxes: Sequence[Box3D],
        *,
        intrinsics: np.ndarray,
        image_hw: tuple[int, int] = (240, 320),
        **kwargs: Any,
    ) -> "SyntheticPerception":
        """渲染一份场景并用**同一份** `boxes` 构造本桥。

        存在的理由是消灭一种很容易犯、且完全不报错的错：用 A 组盒子渲染、
        又把 B 组盒子传进来投影。`__init__` 里的同源校验能拦住它，但
        「根本不需要传两次」比「传错会被拦住」更好。
        """
        box_list = list(boxes)
        scene = render_scene(box_list, intrinsics=intrinsics, image_hw=image_hw)
        return cls(scene, box_list, **kwargs)

    # ------------------------------------------------------------------
    # PerceptionLike —— 三个方法，一个不多
    # ------------------------------------------------------------------

    def detect(
        self,
        image: Any,
        prompt: str,
        *,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
    ) -> list[Detection]:
        """按 prompt 的标签集返回夹具物体（投影框 + 固定分数）。

        **不看 `image`**：合成场景没有纹理，检测的真值就是盒子清单。
        阈值被记录下来但**不参与筛选** —— 分数是固定的，用它筛等于用一个
        常数做判断，只会制造「阈值看起来生效了」的假象。
        """
        wanted = parse_prompt_labels(prompt)
        self.detect_calls.append(
            {
                "prompt": prompt,
                "labels": sorted(wanted),
                "box_threshold": float(box_threshold),
                "text_threshold": float(text_threshold),
            }
        )

        out: list[Detection] = []
        for oid in self._iter_ids():
            label = self._labels[oid]
            if label not in wanted:
                continue
            pbox = self._projected.get(oid)
            if pbox is None:
                continue
            out.append(
                Detection(label=label, score=self.score, box_xyxy=box_xyxy_of(pbox))
            )
        return out

    def segment(self, image: Any, boxes: Sequence[Sequence[float]]) -> np.ndarray:
        """按框 IoU 匹配到夹具物体，返回它的 GT 掩码。

        一次调用带全部框（`PerceptionLike` 的契约），返回 `(N, H, W)`。

        三种「拿不到掩码」的情形**区分开**，因为它们对下游的含义完全不同：

        * 框匹配不到任何物体 ⟹ 全空掩码。真实语义是「SAM2 在这个框里
          什么也没分出来」—— 例如框打在白墙上。**这正好是 `bbox_fallback`
          的触发条件**，所以它是被测对象，不是错误。
        * 匹配到了、但该物体被 `Degradation(mask_empty=True)` 注入 ⟹ 全空掩码。
        * 匹配到了、没注入 ⟹ 内容**逐位等于**夹具掩码。

        ⚠ 返回的数组**必然是一次拷贝**，不能是夹具掩码的视图：协议要求
        `(N, H, W)` 的堆叠结果，而各个物体的掩码在 `scene.masks` 里是分开的。
        所以这里的拷贝是 numpy 形状要求的必然，不是实现选择 ——
        但内容与 `scene.masks[oid]` 逐位相同（由测试钉住）。
        """
        self.segment_calls.append([tuple(float(v) for v in b) for b in boxes])
        h, w = self._image_hw
        out = np.zeros((len(boxes), h, w), dtype=bool)
        matched: list[str | None] = []
        for i, box in enumerate(boxes):
            oid = _match_object(box, self._projected)
            matched.append(oid)
            if oid is None or self._deg(oid).mask_empty:
                continue
            out[i] = self.scene.masks[oid]
        self.segment_matches.append(matched)
        return out

    def lift(self, image: Any, camera_K: np.ndarray | None = None) -> DepthField:
        """返回夹具的解析点云。

        `intrinsics_source` **恒为 `"provided"`**，理由必须写清楚，否则会被
        误读成「模型接受了条件并回传」：合成点云**就是用** `scene.intrinsics`
        逐像素反投影生成的，它没有「模型自己的相机头」这个中间环节。
        「提供的内参已生效」在合成数据上是**结构性成立**的事实，不是一次
        需要被信任的声明。真实链路里这一项要看 `unidepthv2.py` 的条件分支。

        ★ 两条守卫（本桥最值钱的部分）：

        1. `camera_K` 非空时必须与 `scene.intrinsics` **逐位相同**，否则抛
           `SyntheticIntrinsicsError`。这检验的是「`BuildConfig.known_intrinsics`
           真的走到了感知层」—— 真实链路里这条链断了是**静默**的（点云照样
           有、数字照样合理、只是横向整体错 3.17 倍）。
        2. `require_camera_K=True` 而 `camera_K is None` 时同样抛。它把
           「配置写了 known_intrinsics，但那条信息没到达 lift」这件事从
           「看着正常」变成「当场失败」。
        """
        self.lift_calls.append(
            None if camera_K is None else np.array(camera_K, dtype=np.float64, copy=True)
        )
        intr = np.asarray(self.scene.intrinsics, dtype=np.float64)

        if camera_K is None:
            if self.require_camera_K:
                raise SyntheticIntrinsicsError(
                    "lift() 没有收到 camera_K，但本桥被要求校验它"
                    "（require_camera_K=True）—— 说明 BuildConfig.known_intrinsics "
                    "没有到达感知层。内参是**输入**而不是后处理；这条链断了会让"
                    "所有横向米制尺寸整体错约 3.17 倍（§20.3），而且是静默的。"
                )
        else:
            given = np.asarray(camera_K, dtype=np.float64)
            if given.shape != (3, 3) or not np.array_equal(given, intr):
                raise SyntheticIntrinsicsError(
                    "传进来的 camera_K 与合成点云赖以生成的内参不是同一份：\n"
                    f"  传入 {given.tolist()}\n"
                    f"  真值 {intr.tolist()}\n"
                    "合成场景里真值已知，所以这条不一致**必须响** ——"
                    "真实链路拿不到这个校验，但也不该因此在本桥上放行。"
                )

        pts = self.points_chw()
        return DepthField(
            points_chw=pts,
            depth_hw=np.array(pts[2], dtype=np.float64, copy=True),
            intrinsics=intr.copy(),
            grid_hw=self._grid_hw,
            image_hw=self._image_hw,
            intrinsics_source="provided",
            # `fov` 故意留 None，让 builder 自己调 check_fov：
            # 同一个量有两处算法时，两处都可能各自漂移；只留一条路径最省事。
            raw={
                "source": "synthetic-geometry-probe",
                "renderer": self.scene.meta.get("renderer"),
                "n_hit_px": self.scene.meta.get("n_hit_px"),
                "no_hit_px": self.scene.meta.get("no_hit_px"),
            },
        )

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------

    def points_chw(self) -> np.ndarray:
        """点云 `(3, H, W)`，带 `nan_depth_in_box` 注入；**懒构造且缓存**。

        ⚠ 没有任何注入时返回的是**夹具自己的那一块内存**（不复制）。builder
        只读它，所以安全 —— 但调用方**不得原地修改**它，否则会污染夹具，
        让后续所有测量一起偏。这条约束与 `BuildResult.points_chw`
        （「它是引用不是副本」）一致。
        """
        if self._pts is None:
            holes = [oid for oid in self._iter_ids() if self._deg(oid).nan_depth_in_box]
            if not holes:
                self._pts = self.scene.points_chw
            else:
                pts = np.array(self.scene.points_chw, dtype=np.float64, copy=True)
                flat = pts.reshape(3, -1)
                for oid in holes:
                    sel = self.box_selector_for(oid)
                    flat[:, sel.reshape(-1)] = np.nan
                self._pts = pts
        return self._pts

    def projected_box(self, object_id: str) -> tuple[float, float, float, float] | None:
        """该物体**当前生效**的投影框（已应用 `box_pad_px`）；不可投影时 `None`。

        交出来是为了让探针能算 `box_coverage(mask, box)` —— 「框里有多少是背景」
        这个量必须用**同一个框**算，用未外扩的框去算会低估背景占比，
        而那个数字正是「降级代价有多大」的自变量。
        """
        return self._projected.get(object_id)

    def box_selector_for(self, object_id: str) -> np.ndarray:
        """该物体的**投影框**在点云网格上的 bool 选择器（用真实 `box_selector`）。

        与 builder 的降级路径调用的是**同一个函数**：在探针里重写一遍
        「框 → 点在不在框里」就等于给自己一把可以独立地错的尺子。
        """
        from vision.geometry import box_selector

        pbox = self._projected.get(object_id)
        if pbox is None:
            h, w = self._grid_hw
            return np.zeros((h, w), dtype=bool)
        return box_selector(pbox, self._grid_hw, self._image_hw)

    def stats(self) -> dict[str, Any]:
        """进 `build_meta["perception"]` —— 让落盘的场景**自证**它来自合成真值。

        builder 会调这个（`hasattr(perception, "stats")`）。没有它的话，
        一份由本桥建出的 `scene.json` 与一份真实照片建出的 `scene.json`
        在文件层面长得一模一样，而两者的可信范围差得很远。
        这正是"口径必须随数字一起写出来"那条纪律在本次落地里的形态。
        """
        return {
            "kind": type(self).__name__,
            "source": "dataset/builders/synthesize_geometry_probe.py（解析渲染真值）",
            "fidelity": list(FIDELITY_NOTES),
            "image_hw": list(self._image_hw),
            "grid_hw": list(self._grid_hw),
            "require_camera_K": self.require_camera_K,
            "score": self.score,
            "n_objects_in_fixture": self.n_foreground,
            "n_background_boxes": self.n_background,
            "unprojectable_objects": list(self._unprojectable),
            "degradations": {
                oid: {
                    "mask_empty": d.mask_empty,
                    "box_pad_px": d.box_pad_px,
                    "nan_depth_in_box": d.nan_depth_in_box,
                    "note": d.note,
                }
                for oid, d in self._degrade.items()
            },
            "detect_calls": list(self.detect_calls),
            "segment_matches": [list(m) for m in self.segment_matches],
            "lift_intrinsics_source": "provided",
        }

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _iter_ids(self) -> list[str]:
        return list(self._labels)

    def _iter_objects(self) -> list[tuple[str, Box3D]]:
        return [(b.object_id, b) for b in self.boxes]

    def _deg(self, object_id: str) -> Degradation:
        return self._degrade.get(object_id, Degradation())
