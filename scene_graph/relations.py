"""纯几何关系函数 —— 场景图关系判定的**内部层**。契约见方案文档 §13.3(4)。

这个文件里没有 `ToolResult`、没有 object_id 校验、没有 evidence 组装、没有模型调用。
它只做数学：吃两个 `Node`，给出结论 + 度量。**`tools/spatial.py` 才是外部层**，
负责把 `KeyError` 翻成 `NOT_IN_SCENE`、把度量包成 evidence。

为什么坚持分两层（这不是洁癖）：
    外层要依赖 `SceneGraph` 与 trace，因此天然难测；内层是干净数学，可以脱离
    场景图单测。混成一层，「关系可单测」这条 Scene Graph 的核心收益就没了。
    → `scene_graph/tests/test_relations.py` 就是这个分层的存在意义。

坐标系（与 `schema.py` 一致，不要在这里重新定义）：
    camera 系，米。x 向右、**y 向下**、z 向前（深度）。
    ⚠️ y 向下 ⟹ 「更高」= y 更小。所有竖直判断都过 `UpAxis` 归一，
       绝不在业务代码里手写 `a.y < b.y`。

关于 `scale_factor`：尺度校正**不在这里做**。`builder.py` 读
`SceneGraph.scale_factor` 并在生成 `centroid_3d` 时缩放一次；本层拿到的坐标
已经是校正后的米。这样避免了 `scale` 参数在每个函数签名上传染。

容差语义（`tol`）：单位是米，且是**沿该轴的**容差 —— 不是比例。
`left_of(a, b, tol=0.05)` 的判据是 `a.x < b.x - tol`，即两者在 x 上至少错开 5 cm
才算「左」。理由是投影误差与掩码质心误差都在厘米量级（实测均值 83 mm、最大 208 mm），
用严格不等式会把噪声当关系。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

from scene_graph.schema import BBox3D, Node
from tools.version import RELATIONS_METHOD

__all__ = [
    "RelationVerdict",
    "MissingGeometry",
    "UpAxis",
    "DEFAULT_TOL",
    "DEFAULT_NEAR_M",
    "distance_m",
    "left_of",
    "right_of",
    "front_of",
    "behind",
    "above",
    "below",
    "on",
    "inside",
    "near",
    "far",
    "pairwise",
    "RELATIONS",
]

#: 沿轴容差默认值（米）。50 mm 这个数不是随手定的：它是实测「框内中位数 vs
#: 掩码质心」误差（均值 83 mm / 最大 208 mm）与「必须区分两个相邻物体」之间
#: 折中的结果，并且已经写进 §18 Phase 5 的风险条目。
DEFAULT_TOL = 0.05

#: `near` 的默认距离阈值（米）。
DEFAULT_NEAR_M = 1.0

_AXIS_INDEX: dict[str, int] = {"x": 0, "y": 1, "z": 2}


class MissingGeometry(ValueError):
    """所需几何字段缺失（例如 `on` 需要 `bbox_3d` 但该节点没有）。

    本层**不**决定这该对应哪个错误码 —— 抛出它，由 `tools/spatial.py` 翻成
    `ErrorCode.DEGENERATE` 并附上「换 anchor」的恢复建议。
    """


# ----------------------------------------------------------------------------
# 上下方向归一
# ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UpAxis:
    """把「重力朝哪边」这件事收敛到一处。

    没有它的话，每个竖直判断都要写一遍「因为 y 向下所以取负号」，
    而这是最容易在半年后看错的一行代码。

    `up_axis="-y"` 表示**负 y 朝上**（相机系默认），于是 `sign=-1`，
    `up_coord()` 越大表示越高。
    """

    index: int
    sign: float
    name: str

    @classmethod
    def parse(cls, up_axis: str) -> "UpAxis":
        s = up_axis.strip().lower()
        if s.startswith("-"):
            sign, axis = -1.0, s[1:]
        elif s.startswith("+"):
            sign, axis = 1.0, s[1:]
        else:
            sign, axis = 1.0, s
        if axis not in _AXIS_INDEX:
            raise ValueError(f"无法解析 up_axis={up_axis!r}，轴必须是 x/y/z 之一")
        return cls(index=_AXIS_INDEX[axis], sign=sign, name=s)

    def of(self, vec: Sequence[float]) -> float:
        """把这个向量投影到「高度」轴上：**值越大越高**。"""
        return self.sign * float(vec[self.index])

    def horizontal_axes(self) -> tuple[int, int]:
        """除高度轴以外的两个轴索引。"""
        return tuple(i for i in (0, 1, 2) if i != self.index)  # type: ignore[return-value]

    def reversed(self) -> "UpAxis":
        """翻转上下（用于验证「关系确实依赖 up_axis」的测试）。"""
        return UpAxis(index=self.index, sign=-self.sign, name=f"-({self.name})")


def _coerce_up(up: "UpAxis | str") -> UpAxis:
    return up if isinstance(up, UpAxis) else UpAxis.parse(up)


# ----------------------------------------------------------------------------
# 结论对象
# ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RelationVerdict:
    """一个关系判定 + 它的度量。

    为什么不直接返回 `bool`：§11 设计原则 2 要求「返回值必须是几何证据」。
    如果只回 bool，外层就没法在 `Edge.metric` 里留下依据，
    「为什么判定为左」这件事就丢了 —— 而这正是相对 VADAR 的卖点之一。

    `__bool__` 让它能直接写 `if left_of(a, b):`，只是要小心：
    `distance_m()` 的 value 是 float，此时 `bool()` 是「非零即真」。
    用 `v.is_bool` 可以判断类型；数值型请读 `v.value`。
    """

    value: bool | float
    metric: dict[str, float] = field(default_factory=dict)
    method: str = RELATIONS_METHOD
    confidence: float = 1.0

    def __bool__(self) -> bool:
        return bool(self.value)

    @property
    def is_bool(self) -> bool:
        return isinstance(self.value, bool)

    def as_edge_fields(self) -> dict[str, object]:
        """给 `Edge(**v.as_edge_fields(), source=..., target=..., relation=...)` 用。"""
        return {
            "value": bool(self.value) if self.is_bool else float(self.value),
            "metric": dict(self.metric),
            "method": self.method,
            "confidence": self.confidence,
        }


# ----------------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------------


def _require_bbox(node: Node) -> BBox3D:
    if node.bbox_3d is None:
        raise MissingGeometry(
            f"节点 {node.id!r} 没有 bbox_3d —— 该关系需要三维包围盒"
        )
    return node.bbox_3d


def _overlap_1d(lo1: float, hi1: float, lo2: float, hi2: float) -> float:
    return min(hi1, hi2) - max(lo1, lo2)


def _overlap_ratio(lo1: float, hi1: float, lo2: float, hi2: float) -> float:
    """重叠长度 / 较短者长度。完全错开为 0，完全包含为 1。"""
    inter = _overlap_1d(lo1, hi1, lo2, hi2)
    if inter <= 0.0:
        return 0.0
    shorter = min(hi1 - lo1, hi2 - lo2)
    if shorter <= 0.0:
        return 0.0
    return inter / shorter


def _up_range(bbox: BBox3D, up: UpAxis) -> tuple[float, float]:
    """返回 (低端, 高端) —— 按「高度」轴排序后的范围。

    因为 `sign` 可能是 -1，`min`/`max` 会翻转，所以这里重新排序而不是直接取
    `bbox.min[axis]` / `bbox.max[axis]`。这一行是 y 向下最容易出错的地方。
    """
    a = up.of(bbox.min)
    b = up.of(bbox.max)
    return (a, b) if a <= b else (b, a)


# ----------------------------------------------------------------------------
# 距离
# ----------------------------------------------------------------------------


def distance_m(a: Node, b: Node) -> RelationVerdict:
    """三维欧氏距离（米）。

    **不是 depth 相减** —— VADAR 的 `depth(image, bbox)` 返回单点深度，
    两个物体的「距离」只能是 `|depth_a - depth_b|`，这在两者有横向错位时
    完全错（正对相机的 A 与斜前方的 B 可能同深度）。
    """
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    d = (dx * dx + dy * dy + dz * dz) ** 0.5
    return RelationVerdict(
        value=d,
        metric={
            "distance_m": d,
            "delta_x": dx,
            "delta_y": dy,
            "delta_z": dz,
            "a_x": a.x, "a_y": a.y, "a_z": a.z,
            "b_x": b.x, "b_y": b.y, "b_z": b.z,
        },
    )


def near(a: Node, b: Node, thresh: float = DEFAULT_NEAR_M) -> RelationVerdict:
    v = distance_m(a, b)
    d = float(v.value)
    return RelationVerdict(
        value=d <= thresh,
        metric={**v.metric, "thresh_m": thresh},
    )


def far(a: Node, b: Node, thresh: float = DEFAULT_NEAR_M) -> RelationVerdict:
    v = near(a, b, thresh)
    return RelationVerdict(
        value=not bool(v.value),
        metric={**v.metric},
    )


# ----------------------------------------------------------------------------
# 水平方向：left / right / front / behind
# ----------------------------------------------------------------------------


def left_of(a: Node, b: Node, tol: float = DEFAULT_TOL) -> RelationVerdict:
    """a 是否在 b 的左边。

    相机系 x 向右，所以「更左」= x 更小。判据 `a.x < b.x - tol`。
    ⚠️ **显式假设**：相机没有 roll（图像上下的世界含义就是竖直）。
    相机有滚转时 `left_of` 不成立，这条要写进报告（§18 Phase 5 风险③）。
    """
    delta = a.x - b.x
    return RelationVerdict(
        value=delta < -tol,
        metric={"delta_x": delta, "tol": tol, "a_x": a.x, "b_x": b.x},
    )


def right_of(a: Node, b: Node, tol: float = DEFAULT_TOL) -> RelationVerdict:
    v = left_of(b, a, tol)
    return RelationVerdict(value=bool(v.value), metric={**v.metric})


def front_of(a: Node, b: Node, tol: float = DEFAULT_TOL) -> RelationVerdict:
    """a 是否在 b 的前方（离相机更近）。

    相机系 z 向前，所以「更近」= z 更小。判据 `a.z < b.z - tol`。
    """
    delta = a.z - b.z
    return RelationVerdict(
        value=delta < -tol,
        metric={"delta_z": delta, "tol": tol, "a_z": a.z, "b_z": b.z},
    )


def behind(a: Node, b: Node, tol: float = DEFAULT_TOL) -> RelationVerdict:
    v = front_of(b, a, tol)
    return RelationVerdict(value=bool(v.value), metric={**v.metric})


# ----------------------------------------------------------------------------
# 竖直方向：above / below
# ----------------------------------------------------------------------------


def above(a: Node, b: Node, tol: float = DEFAULT_TOL, *, up: "UpAxis | str" = "-y") -> RelationVerdict:
    """a 是否在 b 的上方。判据用高度差 > tol。

    ⚠️ 这是**最不可靠**的一组关系：`up_axis` 在单图下是估计出来的
    （§12.3 步骤 6），估错了 above/below 会整体翻转。报告里要显式讨论。
    """
    u = _coerce_up(up)
    ha, hb = u.of(a.centroid_3d), u.of(b.centroid_3d)
    delta = ha - hb
    return RelationVerdict(
        value=delta > tol,
        metric={"delta_up": delta, "tol": tol, "up_axis": float(u.index), "up_sign": u.sign},
    )


def below(a: Node, b: Node, tol: float = DEFAULT_TOL, *, up: "UpAxis | str" = "-y") -> RelationVerdict:
    v = above(b, a, tol, up=up)
    return RelationVerdict(value=bool(v.value), metric={**v.metric})


# ----------------------------------------------------------------------------
# 接触与包含：on / inside
# ----------------------------------------------------------------------------


def on(
    a: Node,
    b: Node,
    *,
    tol_v: float = DEFAULT_TOL,
    min_overlap: float = 0.5,
    up: "UpAxis | str" = "-y",
) -> RelationVerdict:
    """a 是否**放在** b 上面（接触 + 水平投影重叠）。

    判据两条同时成立：
      ① 竖直：a 的最低点贴近 b 的最高点（`|gap_v| <= tol_v`）。
         完全悬空（gap 大）或深陷（穿模）都不算。
      ② 水平：两个非高度轴上的投影重叠比例都 ≥ `min_overlap`。

    ⚠️ 这是 **bbox 近似**（`geometry_v1`）。真正的「放在上面」还要看接触面积，
    bbox 会把 L 形、镂空物体判错。升级路径是加载 `mask_ref` 的点云做包含率 ——
    换实现时把 `method` 改成 `geometry_v2`，历史结果可 diff（§12.4）。
    """
    u = _coerce_up(up)
    ab, bb = _require_bbox(a), _require_bbox(b)
    a_lo, _a_hi = _up_range(ab, u)
    _b_lo, b_hi = _up_range(bb, u)
    gap_v = a_lo - b_hi

    ratios = [
        _overlap_ratio(ab.min[i], ab.max[i], bb.min[i], bb.max[i])
        for i in u.horizontal_axes()
    ]
    ok = abs(gap_v) <= tol_v and all(r >= min_overlap for r in ratios)
    return RelationVerdict(
        value=ok,
        metric={
            "gap_v_m": gap_v,
            "tol_v": tol_v,
            "overlap_h1": ratios[0],
            "overlap_h2": ratios[1],
            "min_overlap": min_overlap,
        },
    )


def inside(
    a: Node,
    b: Node,
    *,
    min_contain: float = 0.9,
) -> RelationVerdict:
    """a 是否**在** b 内部（容器关系，如杯子在柜子里）。

    判据：三轴重叠比例之积 ≥ `min_contain`，且 a 体积小于 b。
    ⚠️ 同样是 bbox 近似。镂空容器（椅子腿之间、桌下）用 bbox 判会误报 ——
    这是已知局限，报告里要写明，升级路径同上。
    """
    ab, bb = _require_bbox(a), _require_bbox(b)
    ratios = [
        _overlap_ratio(ab.min[i], ab.max[i], bb.min[i], bb.max[i]) for i in range(3)
    ]
    contain = ratios[0] * ratios[1] * ratios[2]

    sa, sb = ab.size(), bb.size()
    vol_a = sa[0] * sa[1] * sa[2]
    vol_b = sb[0] * sb[1] * sb[2]
    vol_ratio = (vol_a / vol_b) if vol_b > 0.0 else float("inf")

    ok = contain >= min_contain and vol_ratio < 1.0
    return RelationVerdict(
        value=ok,
        metric={
            "contain_ratio": contain,
            "volume_ratio": vol_ratio,
            "min_contain": min_contain,
        },
    )


# ----------------------------------------------------------------------------
# 全关系计算（供 builder.py 与 L5 describe_scene 共用）
# ----------------------------------------------------------------------------

#: `query_relation` 的分发表。名字必须与 `schema.RelationType` 对得上。
RELATIONS: dict[str, Callable[..., RelationVerdict]] = {
    "distance": distance_m,
    "near": near,
    "far": far,
    "left_of": left_of,
    "right_of": right_of,
    "front_of": front_of,
    "behind": behind,
    "above": above,
    "below": below,
    "on": on,
    "inside": inside,
}


def pairwise(
    a: Node,
    b: Node,
    *,
    up: "UpAxis | str" = "-y",
    tol: float = DEFAULT_TOL,
    near_m: float = DEFAULT_NEAR_M,
) -> dict[str, RelationVerdict]:
    """算一对物体的**全部**关系。

    单图 20 个物体时是 190 对 × 11 个关系 —— 纯 NumPy/Python 运算，零 GPU，
    毫秒级。所以「把全部两两关系都算出来」在成本上完全可行，
    这正是 L5 场景级输出能「零额外模型成本」的原因（§11.1）。

    没有 `bbox_3d` 的节点会跳过 on/inside（那两项需要包围盒），
    其余关系照常给出 —— 不因为一项缺数据就丢掉整对关系。
    """
    out: dict[str, RelationVerdict] = {
        "distance": distance_m(a, b),
        "near": near(a, b, near_m),
        "far": far(a, b, near_m),
        "left_of": left_of(a, b, tol),
        "right_of": right_of(a, b, tol),
        "front_of": front_of(a, b, tol),
        "behind": behind(a, b, tol),
        "above": above(a, b, tol, up=up),
        "below": below(a, b, tol, up=up),
    }
    if a.bbox_3d is not None and b.bbox_3d is not None:
        out["on"] = on(a, b, tol_v=tol, up=up)
        out["inside"] = inside(a, b)
    return out
