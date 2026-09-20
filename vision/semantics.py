#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`vision/semantics.py` —— 角色②（视觉语义）与工具层之间的**中性契约**。

为什么需要这个文件（一句话：为了让一条分层禁令保持为"禁令"）
================================================================

本项目有一条硬性分层规则（`tests/test_agent_memory_and_layering.py::TestLayering`
用 AST 静态检查钉住）：

    `tools/` 与 `scene_graph/` **永不** import `llm` 或 `agents`。

它的理由不是洁癖：几何/工具层必须在**没有任何 LLM 存在**的前提下可构造、可单测、
可离线运行 —— 否则「关掉视觉语义」这个消融开关就只是改了一份提示词，
而不是真的少了一条依赖。

可是 `get_attributes` 这条工具有个天然的矛盾：它是**唯一**会调到 LLM 的工具。
把它的实现写在 `tools/attributes.py` 里、又让它 `from llm.vlm import ...`，
那条禁令就破了一个口 —— 于是「工具层不知道 LLM 存在」这句话从"结构保证"
退化成"我们没往那边加东西"。

修法不是放宽检查，而是**把被跨层引用的东西搬出来**：工具层真正需要的只有四样，
它们**本来就不是 LLM 概念**：

    ATTRS / DEFAULT_CONFIDENCE_THRESHOLD   一套**领域词表**与一个默认阈值
    Attribute                              一条语义属性的数据结构
    Region                                 一次"看哪一块"的几何范围
    SemanticBackendError                   一个**契约级**失败类型

四样都放在这里，于是依赖方向变成：

    llm/vlm.py ──import──▶ vision/semantics.py ◀──import── tools/attributes.py
                                   ▲
                                   └── agents/prompts/system.py（也要归一化图像尺寸）

工具层**不认识 `VLM` 类、不认识 `LLMClient`、不认识任何提示词**；
它在运行时通过 `ctx.vlm` 拿到一个鸭子类型的后端（见 `tools/attributes.py`）。
**注入是运行时的事，类型是编译期的事** —— 前者用不着后者破例。

`llm/vlm.py` 只是把这些名字**再导出**一遍，所以老的 `from llm.vlm import Region`
照旧可用；新代码应当从本模块导入。

本模块是**零依赖纯数据**（只有标准库），与 `vision/types.py` 同级。
绝不在这里 import torch / numpy 之外的重物，也绝不 import `llm`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "ATTRS",
    "SPATIAL_PARAM_TERMS",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "CROP_MARGIN",
    "SemanticBackendError",
    "Attribute",
    "Region",
    "image_size_from_meta",
]


# ============================================================================
# 1. 词表与阈值 —— 「空间词汇不在这里」是刻意的
# ============================================================================

#: ★ `attrs` 的白名单。**空间词汇不在这里，也永远不该加进来。**
#: 这个元组同时是：① 提示词里列给模型的取值列表 ② 工具层的参数校验依据
#: ③ 「角色② 不得输出空间判断」这条禁令的可执行边界。
ATTRS: tuple[str, ...] = ("color", "material", "texture", "state", "shape")

#: 违反「describe 不得涉空间」的形参名词根。`llm.vlm.check_signature()` 用它做子串匹配。
#: ⚠ 它服务的是**自检**（导入期断言），所以放在这里只是为了让词表与 `ATTRS` 相邻 ——
#: 真正执行自检的是 `llm/vlm.py`（那里才知道 `describe` 长什么样）。
#: 宁可宽一点：多拦一个无害的词，比放过一个 `distance` 便宜得多。
SPATIAL_PARAM_TERMS: tuple[str, ...] = (
    "left", "right", "above", "below", "under", "over", "near", "far",
    "nearest", "farthest", "distance", "depth", "relation", "position",
    "coord", "centroid", "location", "size", "extent", "bbox", "meter",
    "metre", "angle", "front", "behind", "inside", "orientation", "direction",
    "pixel", "height", "width", "volume",
)

#: 低于它就**必须上报**，禁止当成确定值用（§13.3(3)）。
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.6

#: 裁剪目标区域时向外扩的比例。留一点上下文，否则「黑色」这种属性会因为
#: 画面里只剩一块黑而更难判；但留太多就等于没裁，起不到放大目标的作用。
CROP_MARGIN: float = 0.15


# ============================================================================
# 2. 失败类型
# ============================================================================


class SemanticBackendError(RuntimeError):
    """**契约级**失败：语义后端不可用（缺配置 / 网络失败 / 返回不可解析）。

    ⚠ 为什么这个类必须住在中性层，而不是留在 `llm/vlm.py`：
    工具层要 `except` 它。若它定义在 `llm/` 里，`tools/attributes.py` 就得
    `from llm.vlm import VLMError` —— 分层禁令当场失效（这正是本文件存在的原因）。
    放在这里之后，`llm/vlm.py` 的 `VLMError` 只是它的**子类**，
    工具层 `except SemanticBackendError` 照旧能接住，而它完全不认识视觉后端。

    与 `LLMError` 的分工也在这里说清楚：`LLMError` 是**文本角色**的失败，
    会被 `loop.py` 记成 `llm_error`（整题失败）；而视觉角色是**可降级**的
    （§13.3(3)：主路径不依赖本角色），它的失败应当由调用方翻译成一次
    「能力不可用」的工具结果，而不是把整题打断。
    """


# ============================================================================
# 3. 数据结构
# ============================================================================


@dataclass(frozen=True)
class Attribute:
    """一条语义属性。

    **没有坐标字段，一个都没有。** 这不是省事 —— 是 `describe()` 只能返回
    「这是什么颜色」而不能返回「它在哪」的落地方式：类型里没有地方放位置。
    """

    name: str
    value: str
    confidence: float
    source: str = "vlm"
    #: 给定时该值是否落在闭集内。False 的上层含义是「必须上报不确定」。
    in_closed_set: bool = True
    #: 模型的未加工说法（审计用：能看出它是原样给出还是被归一化过）。
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "confidence": round(float(self.confidence), 4),
            "source": self.source,
            "in_closed_set": self.in_closed_set,
            "raw": self.raw,
        }


def image_size_from_meta(meta: Mapping[str, Any]) -> tuple[int, int] | None:
    """从 `build_meta` 里读出 `(宽, 高)`。**键名与顺序的唯一归一化点。**

    三种历史写法都认，其中 `image_hw` 是 `scripts/build_scene.py` 的**真实输出**：

        image_hw    = [高, 宽]   ← builder 落盘的（★ 是 H, W，不是 W, H）
        image_width / image_height            显式键名
        image_size  = [宽, 高]                 对外口径（与 PIL 一致）

    ⚠ 为什么值得单独一个函数并写这么多注释：`image_hw` 是 H,W 而 `image_size` 是 W,H，
    **弄反了不会报错** —— 它只是把所有裁剪的宽高比错掉，
    于是「目标物体被裁掉一半」看起来像「模型看颜色不准」。
    这类错误在报告里查不出来，只能靠"只有一处实现"来防。

    ⚠ 它从 `llm/vlm.py` 搬到这里，是因为**两个互不相干的调用方都要它**：
    角色②（裁剪要用宽高）与提示词层（`scene_hint.image_size`）。把归一点放在
    任意一侧都会让另一侧跨层 import —— 那正是这个文件要消掉的东西。
    """
    raw_hw = meta.get("image_hw")
    if isinstance(raw_hw, (list, tuple)) and len(raw_hw) == 2:
        return int(raw_hw[1]), int(raw_hw[0])
    if meta.get("image_width") and meta.get("image_height"):
        return int(meta["image_width"]), int(meta["image_height"])
    raw = meta.get("image_size")
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return int(raw[0]), int(raw[1])
    return None


@dataclass(frozen=True)
class Region:
    """看图的范围。由**工具层**从 `Node.bbox_2d` 组装，模型看不到它。

    带 `bbox` 是为了裁剪（放大目标物体），**不是为了把它写进提示词** ——
    提示词里只说「下面这张图是从一张房间照片里裁出来的、含一个椅子」，
    不放任何数字。于是「角色② 不输出坐标」在输入侧也是成立的。
    """

    bbox: tuple[float, float, float, float] | None = None
    mask_ref: str | None = None
    label: str = ""
    image_size: tuple[int, int] | None = None

    @classmethod
    def from_node(cls, node: Any, *, image_size: tuple[int, int] | None = None) -> "Region":
        bbox = getattr(node, "bbox_2d", None)
        return cls(
            bbox=None if bbox is None else tuple(float(v) for v in bbox),
            mask_ref=getattr(node, "mask_ref", None),
            label=str(getattr(node, "label", "") or ""),
            image_size=image_size,
        )

    @classmethod
    def for_node(cls, scene: Any, node: Any) -> "Region":
        """从场景图 + 节点直接组装 —— **`Region` 的正确入口**。

        做成一等入口的原因：图像尺寸的键名与顺序是这套代码里最容易错、
        又最不容易发现的一处（错了不报错，只是宽高比反了）。
        让所有调用方都走这里，那处逻辑就只存在一份。
        """
        return cls.from_node(node, image_size=image_size_from_meta(
            getattr(scene, "build_meta", None) or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_2d": None if self.bbox is None else [round(v, 2) for v in self.bbox],
            "mask_ref": self.mask_ref,
            "label": self.label,
            "image_size": None if self.image_size is None else list(self.image_size),
        }
