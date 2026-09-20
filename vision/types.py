"""L1 感知层的数据结构 —— **故意零 torch**（只要 NumPy + `vision.geometry`）。

为什么要单独一个文件（这不是拆分癖）：
    `scene_graph/builder.py` 与 `scene_graph/tests/test_builder.py` 都只需要
    「一个检测框长什么样」「一个深度场长什么样」这两个定义。如果它们住在
    `vision/grounding.py` / `vision/depth.py` 里，那么连测试 fake 感知栈都要
    先 `import torch`（3–5 秒），builder 的单元测试就永远跑不进 0.5 秒档。

    这个文件与 `vision/geometry.py` 一起，让「builder 逻辑可脱离 GPU 单测」
    成为一句可执行的话，而不是一句愿望。

`PerceptionLike` 是 builder 唯一依赖的接口 —— 它是 Protocol 而不是基类，
所以真实的 `PerceptionStack` 与测试里的 `FakePerception` 都不需要继承任何东西。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

import numpy as np

from vision.geometry import FovCheck

__all__ = ["Detection", "DepthField", "PerceptionLike", "box_xyxy_of"]


@dataclass(frozen=True, slots=True)
class Detection:
    """一个检测结果。

    `box_xyxy` 是**绝对像素**坐标（左上原点）—— 这是 transformers 原生
    GroundingDINO 的约定，与已被弃用的编译版 `groundingdino` 包不同
    （那个回归一化的 cxcywh，见 §20 Step 0.5 的对照表）。写死在这里，
    免得每个消费方各猜一次。

    `label` 必须小写：GroundingDINO 的 prompt 要求小写句点结尾
    （`"sofa. chair."`），检测回来的标签也就跟着小写。`builder.py` 会再
    正规化一次（slug），但那一步是为了生成合法 `object_id`，不是为了纠错。
    """

    label: str
    score: float
    box_xyxy: tuple[float, float, float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": round(float(self.score), 4),
            "box_xyxy": [round(float(v), 2) for v in self.box_xyxy],
        }


def box_xyxy_of(box: Sequence[float]) -> tuple[float, float, float, float]:
    """把任意 4 元序列统一成 `(x1, y1, x2, y2)` float 元组。"""
    x1, y1, x2, y2 = (float(v) for v in box)
    return (x1, y1, x2, y2)


@dataclass(frozen=True, slots=True)
class DepthField:
    """UniDepth 的一次输出。

    `points_chw` 的形状是 `(3, HP, WP)` —— **模型自己的分辨率**，可能不等于
    输入图像分辨率。这不是理论顾虑：`infer()` 内部会先 padding 到 ratio_bounds、
    再 resize 到 pixels_bounds、最后把点云裁回（`unidepthv2.py:282-336`），
    所以两个分辨率必须分别记录，让 `builder.py` 显式做换算
    （`vision.geometry.resample_mask_to`），而不是假设它们相等。

    `intrinsics` 只用于记录与可视化。**不要用它反投影点云**：它是独立预测头，
    与 `rays`（decoder 预测的方向场）差 3–5%，`points` 才是权威（§5）。

    ⚠️ `intrinsics` 的语义在 Phase 0 Step 6 之后被收紧了一条：它必须是
    **实际用来生成 `points` 的那一份**，而不是「模型回传的那一份」。两者在
    传入已知内参时**不相等** —— `infer(rgb, camera=...)` 走的是
    `unidepthv2.py:361-362`（把 camera 转成 rays 喂进 decoder 当条件），
    `out["intrinsics"]` 仍然是那个独立相机头的输出。实测：传入 GT 内参
    （fx=518.9）后，回传的 `intrinsics` 依然写着 fx=163.7。见 `intrinsics_source`。
    """

    points_chw: np.ndarray
    #: (H, W) 深度图，单位米。**它等于 points 的 z 列**，不是独立测量量。
    depth_hw: np.ndarray
    #: 3×3 内参 —— **实际用于生成 `points` 的那一份**（见类 docstring 的警告）。
    intrinsics: np.ndarray
    #: 点云网格 (HP, WP)，方便下游显式换算。
    grid_hw: tuple[int, int]
    #: 输入图像 (H, W)。
    image_hw: tuple[int, int]
    #: 其余原始输出键，原样保留供排查。
    raw: dict[str, Any] = field(default_factory=dict)
    #: `"provided"` = 调用方给了已知内参并已生效；`"predicted"` = 用的是模型相机头。
    #: 下游据它决定「米制数字能不能当绝对量用」—— `predicted` 且视场不合理时
    #: 只能当相对量，报告里必须标注。
    intrinsics_source: Literal["provided", "predicted"] = "predicted"
    #: 视场合理性检查结果。`predicted` 来源时尤其要看它（见 `vision/geometry.check_fov`）。
    fov: FovCheck | None = None

    @property
    def depth_range_m(self) -> tuple[float, float]:
        finite = self.depth_hw[np.isfinite(self.depth_hw)]
        if finite.size == 0:
            return (float("nan"), float("nan"))
        return (float(finite.min()), float(finite.max()))

    def grid_scale_vs_image(self) -> tuple[float, float]:
        """`(WP/WI, HP/HI)`。不等于 1 时下游必须换算 —— 不要静默假设相等。"""
        hp, wp = self.grid_hw
        hi, wi = self.image_hw
        return (wp / wi, hp / hi)


@runtime_checkable
class PerceptionLike(Protocol):
    """`builder.py` 对感知栈的全部要求 —— 三个方法，没有别的。

    刻意保持窄：故意不把 `PerceptionStack` 的模型句柄、显存统计、卸载能力
    编进接口。builder 不该知道有多少个模型、它们占多少显存 —— 那是 registry
    的事。窄接口的直接好处是测试里 30 行就能写一个 fake。
    """

    def detect(
        self,
        image: Any,
        prompt: str,
        *,
        box_threshold: float = ...,
        text_threshold: float = ...,
    ) -> list[Detection]:
        """返回检测列表。可以是空的（图里没有 prompt 里的东西）。"""
        ...

    def segment(self, image: Any, boxes: Sequence[Sequence[float]]) -> np.ndarray:
        """返回 `(N, H, W)` 的 bool 掩码，N 与 `boxes` 一一对应。

        ★ 实现必须**一次调用带全部框**：实测 9 个框一次调用 200 ms，
        逐个调用 1512 ms，差 7.57×（§20 Step 0.5b）。
        """
        ...

    def lift(self, image: Any, camera_K: np.ndarray | None = None) -> DepthField:
        """单帧升维，返回点云 + 深度 + 内参。

        `camera_K` 非空时表示**已知内参**（`(3,3)` 数组，或 `None`）：实现应当
        把它当作条件喂给模型，而不是让模型自己猜相机。实测差别是量级级的 ——
        `phase0/probe_depth_gt.py` 在同一张图上：不给内参时三维误差中位
        **1.943 m**，给 GT 内参后 **0.267 m**（降到 13.8%）。

        它是**可选**参数而不是必填：多数真实照片没有标定内参，此时实现只能
        用模型预测值，但必须把 `DepthField.intrinsics_source` 标成 `"predicted"`
        并做视场合理性检查 —— 让不确定性可传递，而不是静默。
        """
        ...
