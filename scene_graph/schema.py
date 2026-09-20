"""Node / Edge / SceneGraph —— 3D 场景的中间表示。契约见方案文档 §12.2。

与 VADAR 的关键差别只有一条，但影响很大：**Node 有稳定 id**。
VADAR 每次 `loc()` 都返回裸 bbox，于是必须靠 `same_object(iou>0.92)` 反推
「这俩框是不是同一个东西」—— 既不准，又白白浪费一次工具调用。
有了 `{label}_{idx}` 这样的稳定 id，这个问题在架构上消失（§11.2 最后一行）。

坐标系约定（全项目统一，写死在这里，不要在别处再定义一遍）：

    frame = "camera"  —— 相机坐标系，单位米
        x : 向右为正（沿图像 u 方向）
        y : **向下为正**（沿图像 v 方向，与图像坐标一致）
        z : 向前为正（即深度，UniDepth 的 `points[:, -1]` 就是它）

    ⚠️ y 向下意味着「更高」= y 更小。这个反直觉之处是 `above`/`below` 最容易写错的根源，
    所以 `relations.py` 里所有竖直方向判断都过一个 `up_coord()` 归一层，
    而不是在每处手写 `y < y`。见 §18 Phase 5 风险②。

`up_axis` 默认 `"-y"`，表示重力方向是「负 y 朝上」。单张图下这个值是**估出来的**
（取画面下部大面积点云拟合地平面），是整个场景图最脆弱的一环，需要在报告里显式讨论
（§12.3 步骤 6）。
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

__all__ = [
    "FRAME",
    "Vec3",
    "Extent3",
    "BBox2D",
    "BBox3D",
    "Node",
    "RelationType",
    "Edge",
    "SceneGraph",
]

#: 当前只支持相机系。世界系要等有了多视角或重力对齐再引入。
FRAME = Literal["camera", "world"]


def _as_triple(value: Any) -> tuple[float, float, float]:
    """把 list / tuple / numpy 数组 / torch 张量统一成 3 元 float 元组。

    为什么要有它：上游是 numpy（点云、PCA 结果），而 Pydantic 默认不认 ndarray。
    在这里一次性吃掉这个差异，比在每个调用点手写 `tuple(map(float, ...))` 可靠。
    """
    if hasattr(value, "tolist"):
        value = value.tolist()
    seq = tuple(float(x) for x in value)
    if len(seq) != 3:
        raise ValueError(f"需要 3 个分量，收到 {len(seq)} 个")
    return seq


Vec3 = Annotated[tuple[float, float, float], BeforeValidator(_as_triple)]
Extent3 = Annotated[tuple[float, float, float], BeforeValidator(_as_triple)]
BBox2D = Annotated[tuple[float, float, float, float], BeforeValidator(
    lambda v: tuple(float(x) for x in (v.tolist() if hasattr(v, "tolist") else v))
)]


class BBox3D(BaseModel):
    """三维轴对齐包围盒。`min`/`max` 是相机系下的角点。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min: Vec3
    max: Vec3

    def size(self) -> tuple[float, float, float]:
        """(x 跨度, y 跨度, z 跨度)，米。"""
        return (
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        )

    def diagonal_m(self) -> float:
        dx, dy, dz = self.size()
        return (dx * dx + dy * dy + dz * dz) ** 0.5


class Node(BaseModel):
    """场景里的一个物体。

    `extra="forbid"` 是有意的：写错字段名时立刻报错，而不是静默丢掉一个字段
    然后在几十行之后表现为「怎么质心是零」。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: 稳定 id，形如 `"chair_1"`。全项目所有工具都用它指代物体。
    id: str
    label: str
    #: 检测置信度（GroundingDINO 的 score）。
    score: float = Field(default=0.0, ge=0.0, le=1.0)

    bbox_2d: BBox2D | None = None
    #: 掩码文件引用（不内联，省内存）。相对 `dataset/scenes/` 解析。
    mask_ref: str | None = None

    #: ★ 核心字段：相机系质心，单位米。**取掩码内点云的中位数**，
    #: 不是 bbox 内的中位数 —— 实测两者在 9 个物体上差均值 83 mm / 最大 208 mm，
    #: 而关系容差是 50 mm（见 §17 裁决）。
    centroid_3d: Vec3
    #: 真实三维尺寸 (w, h, l)，米。由点云 PCA 或轴对齐跨度得到。
    extent_3d: Extent3 = (0.0, 0.0, 0.0)
    bbox_3d: BBox3D | None = None

    #: 语义属性，来自角色② `describe()`。**只放颜色/材质这类非空间属性** ——
    #: 空间判断写在这里就等于把幻觉请回了场景图（§13.3(3)）。
    attributes: dict[str, str] = Field(default_factory=dict)

    frame: FRAME = "camera"
    #: 落在该物体掩码内的点云点数。`DEGENERATE` 错误码的判定依据。
    n_points: int | None = None
    #: 质心来源。`"mask"` = SAM2 掩码内点云的中位数（正常路径）；
    #: `"bbox_fallback"` = 掩码失效、退回检测框内点云的中位数。
    #:
    #: 为什么要记录而不是丢掉降级节点：两者的精度**已知不同**（实测均值差 83 mm、
    #: 最大 208 mm，§20 Step 0.5b）。下游据此决定要不要降低 `above`/`left_of`
    #: 这类近邻关系的可信度，L5 的 `diagnose_failure` 也靠它把失败归因到「掩码」而不是「检测」。
    centroid_source: Literal["mask", "bbox_fallback"] = "mask"

    # -- 便利读取 -------------------------------------------------------------

    @property
    def x(self) -> float:
        return self.centroid_3d[0]

    @property
    def y(self) -> float:
        return self.centroid_3d[1]

    @property
    def z(self) -> float:
        return self.centroid_3d[2]

    def extent(self, axis: Literal["w", "h", "l"]) -> float:
        return self.extent_3d[{"w": 0, "h": 1, "l": 2}[axis]]


RelationType = Literal[
    "distance",
    "near",
    "far",
    "left_of",
    "right_of",
    "front_of",
    "behind",
    "above",
    "below",
    "on",
    "inside",
]


class Edge(BaseModel):
    """一条由**几何算出**的关系边。

    注意 `metric` —— 它让每条边都带着自己的证据（`delta_x`、`gap_m`、`overlap_ratio`…），
    于是「为什么判定为左」可以脱离代码复算。§12.4 的 `method` 字段则让
    「换了关系定义」这件事可追溯、可 diff。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    target: str
    relation: RelationType
    value: bool | float
    metric: dict[str, float] = Field(default_factory=dict)
    #: 来源标记。目前恒为 `"geometry_v1"` —— 未来换点云版关系时改成 `geometry_v2`，
    #: 历史结果与新材料可直接 diff。
    method: str = "geometry_v1"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class SceneGraph(BaseModel):
    """一个场景（当前等于一张图）的完整表示。

    `nodes` / `edges` 用 tuple 而非 list：模型是 frozen 的，内部不该可变。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    scene_id: str
    image_id: str
    #: 3×3 内参。UniDepth 的 `intrinsics` 是独立预测头 —— 只用它做记录与可视化，
    #: **不要用它反投影点云**（§5：与 `points` 差 3–5%，`points` 才是权威）。
    camera_intrinsics: list[list[float]] | None = None
    #: 重力方向，`"-y"` = 负 y 朝上。单图下为估计值，是最脆弱的一环。
    up_axis: str = "-y"
    #: 尺度校正因子（`calibrate_scale` 的产物）。1.0 = 未校正。
    scale_factor: float = Field(default=1.0, gt=0.0)

    nodes: tuple[Node, ...] = ()
    edges: tuple[Edge, ...] = ()
    #: 构建元信息：模型名与版本、时间戳、prompt、阈值……保证可复现（§12.2）。
    build_meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("camera_intrinsics")
    @classmethod
    def _check_intrinsics(cls, v: list[list[float]] | None) -> list[list[float]] | None:
        if v is None:
            return None
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError("camera_intrinsics 必须是 3×3")
        return [[float(x) for x in row] for row in v]

    # -- 查询 ----------------------------------------------------------------

    def ids(self) -> list[str]:
        return [n.id for n in self.nodes]

    def has(self, object_id: str) -> bool:
        return any(n.id == object_id for n in self.nodes)

    def node(self, object_id: str) -> Node:
        """按 id 取节点。不存在时抛 `KeyError`。

        刻意**不**返回 None：调用方必须显式处理「不存在」，
        工具层再把它翻译成 `NOT_IN_SCENE`（幻觉捕获点）。
        """
        for n in self.nodes:
            if n.id == object_id:
                return n
        raise KeyError(object_id)

    def by_label(self, label: str) -> list[Node]:
        """按类别取节点。label 比较是**大小写不敏感**的 —— 上游 GroundingDINO
        的 prompt 必须小写（§4.1），但 LLM 写 `find_object("Chair")` 很常见，
        这里吃掉差异比让模型重试划算。"""
        key = label.strip().lower()
        return [n for n in self.nodes if n.label.strip().lower() == key]

    def label_counts(self) -> dict[str, int]:
        """`{"chair": 4, "door": 1, ...}`。

        ★ 这就是 `scene_hint` 的全部内容（§13.3(2)）。
        它只给清单和计数、**不给坐标** —— 于是「坐标只能来自工具返回值」
        从提示词要求升级为信息约束：模型想抄坐标也无从抄起。
        """
        counts: dict[str, int] = {}
        for n in self.nodes:
            counts[n.label] = counts.get(n.label, 0) + 1
        return counts

    def edges_of(self, object_id: str) -> list[Edge]:
        return [e for e in self.edges if e.source == object_id or e.target == object_id]

    def summary_line(self) -> str:
        """一行摘要，进日志与 Demo 的 Scene Report 面板。"""
        return (
            f"{self.scene_id}: {len(self.nodes)} objects / {len(self.edges)} edges "
            f"(up_axis={self.up_axis}, scale={self.scale_factor:.3f})"
        )
