#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""geometry_metrics.py —— 单目 3D 的几何专项指标 + 误差分解。

为什么要有这个模块
------------------
`metrics.py` 只回答一个问题：「一次问答算不算对」。它是全系统**唯一**的出口，
于是有三个后果：

1. **不可归因。** 答错了，你无法知道错在掩码、深度、内参，还是
   「从几何到答案」的映射。整条链路的中间量一个都没被度量。
2. **米制承诺是未验证的假设。** §2 记着「`points` 是真几何」，但那只在
   **表示层面**成立（`points` 的 z 列就是 `depth`，实测差 `0.000e+00`）——
   它没说明质心误差多少毫米、尺寸误差多少米。全项目至今只有一次
   掩码口径的对照（83 / 208 mm），那还不是绝对精度。
3. **错误在不同指标上的可见度差几个数量级。** 实测（§22 内参探针）：
   同一个内参错误，绝对米制题 **0/8**、相对序题 **8/8**、计数题 **4/4**。
   `metrics.py` 给不了这个结构，它只给一个数。

本模块把 3D 从「中间表示」变成「被测对象」。

必须分解成横向与纵深（这是本模块存在的核心理由）
------------------------------------------------
同一份实测给出了一个只看 3D 欧氏误差就会**读反**的事实：

    质心位移中位数 1.972 m，其中横向 1.952 m（99%）、纵深 0.366 m
    尺寸（extent）却系统性放大 2.9×

也就是说：**「尺寸错 2.9 倍」和「位置几乎没在纵深上动」同时成立**。
如果只报一个 `3D 误差 = 1.97 m`，读者会把它读成「整体尺度错了」，
而正确的读法是「横向尺度错了，纵深几乎没动，但尺寸又被放大了」。

所以 `centroid_error()` 恒定返回 total / lateral / depth 三项，**三个都要报**。
`docs/往3D视觉靠的路径建议` 里那句「只看 3D 欧氏误差会把这件事误判成整体尺度问题」
指的就是这一条。

口径（必须随数字一起写出来）
----------------------------
* 长度单位一律 **米**。报告里换算成 mm 时必须写明换算发生在哪一层。
* 相机系：x 向右、y 向下、z 向前。
  **横向 = `sqrt(dx² + dy²)`（轴 0、1）**，**纵深 = `|dz|`（轴 2）**。
* ⚠ **`total ≠ lateral + depth`**。三者是三个不同的范数：
  `total = ‖Δ‖₂`、`lateral = ‖Δ_xy‖₂`、`depth = |Δ_z|`。
  恒等式是 **`total² = lateral² + depth²`**，所以本模块提供 `decomposition_residual()`
  让调用方能**证明**分解自洽，而不是在文档里声称它自洽。
  把它写成加法会把一个几何恒等式说错 —— 这类错误不会报错，只会一直错下去。
* 尺寸误差用**逐轴绝对差 + L1**，不用相对误差：`extent_3d` 是**轴对齐包围盒**，
  某个轴会退化成接近 0（挂画、屏幕、薄板）。相对误差在那里除零且无意义，
  而绝对误差始终有定义。相对的版本由调用方**显式**要（`relative=True`）。
  ⚠ 轴对齐也意味着：斜放的物体其「3D 尺寸」会被**系统性高估**（包围盒比物体大）。
  这是口径的一部分，不是模型的错 —— 见 `docs/往3D视觉靠的路径建议` §诊断。

「缺几何」不许静默排除（本模块最硬的一条纪律）
----------------------------------------------
`metrics.py` 已经吃过一次这个教训：上游把无法解析的预测**跳过**（分子分母都不加），
结果是准确率被系统性高估 —— 一个只会输出乱码的模型在这些题上不扣分。
本模块沿用同一口径：

    单点函数（`centroid_error` 等）**要求输入是实数组**，`None` 直接抛错 ——
    「拿不到几何」是调用方的事实，不该在这里被悄悄变成 0。

    聚合函数（`summarize`）**要求显式传入 `n_expected`**，并把
    `n_missing = n_expected - len(rows)` 一起写进结果。
    **分母恒为 `n_expected`**，不是 `len(rows)`。

零点自洽性（使用前必做）
------------------------
任何指标上线前，先用「预测 = 真值」跑一遍，所有误差必须**恰好为 0**。
这不是形式主义：本项目已经遇到过一次「尺子错了但没人发现」
（正则把普通 `for` 当推导式，历史结论被误读）。`is_self_consistent()` 提供这个检查。

零依赖
------
本模块只用 numpy 与标准库，**不 import 项目内任何模块**（与 `metrics.py` 同规矩）。
这样它能在没有 torch、没有 GPU、没有场景数据的机器上跑 —— 口径层必须可独立复算。
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "LATERAL_AXES",
    "DEPTH_AXIS",
    "DELTA_THRESHOLDS",
    "centroid_error",
    "extent_error",
    "point_cloud_error",
    "decomposition_residual",
    "relation_prf",
    "summarize",
    "is_self_consistent",
]

#: 横向的两个轴（相机系 x 向右、y 向下）。
LATERAL_AXES: tuple[int, int] = (0, 1)
#: 纵深轴（相机系 z 向前）。
DEPTH_AXIS: int = 2

#: 深度估计文献惯用的 δ 阈值 `1.25^k`。用元组而不是散落的字面量：
#: 报告里写「δ1」时必须能指回一个具体数值。
DELTA_THRESHOLDS: tuple[float, ...] = (1.25, 1.25 ** 2, 1.25 ** 3)

#: 长度误差超过这个值就不参与「相对误差」类统计的分母（避免除到 0 附近的量级）。
_MIN_DENOM_M = 1e-9


# ----------------------------------------------------------------------------
# 基本量
# ----------------------------------------------------------------------------


def _as_vec3(value: Any, name: str) -> np.ndarray:
    """把入参收成 `(3,)` float64。

    长度不是 3 就报错，而不是「补零 / 截断」：坐标轴数量不对是**调用方的 bug**，
    静默补齐会让它在几十行之后表现成「怎么质心偏了」。
    """
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(f"{name} 必须是 3 维（相机系 xyz），收到 shape={np.shape(value)}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} 含 inf/nan：{arr!r} —— 这种值会无声污染所有下游统计")
    return arr


def centroid_error(pred_xyz: Any, gt_xyz: Any) -> dict[str, float]:
    """质心误差，**分解为 总 / 横向 / 纵深**（米）。

    返回的 `total_m / lateral_m / depth_m` 是三个独立范数，
    满足 `total_m² = lateral_m² + depth_m²`（见 `decomposition_residual`）。

    ⚠ `pred_xyz` 为 `None` 会抛 `TypeError` —— 见模块 docstring「缺几何不许静默排除」。
    调用方要处理「拿不到几何」，就把这一项排除在 `rows` 之外**并让 `n_expected` 覆盖它**。
    """
    if pred_xyz is None or gt_xyz is None:
        raise TypeError(
            "centroid_error 不接受 None —— 「这个物体没算出几何」必须由调用方显式记账"
            "（传给 summarize 的 n_expected 里），不能在这里被填成 0。"
            f"收到 pred={pred_xyz!r}, gt={gt_xyz!r}"
        )
    p = _as_vec3(pred_xyz, "pred_xyz")
    g = _as_vec3(gt_xyz, "gt_xyz")
    d = p - g

    lateral = float(np.linalg.norm(d[list(LATERAL_AXES)]))
    depth = float(abs(d[DEPTH_AXIS]))
    total = float(np.linalg.norm(d))
    return {
        "total_m": total,
        "lateral_m": lateral,
        "depth_m": depth,
        "dx_m": float(d[0]),
        "dy_m": float(d[1]),
        "dz_m": float(d[2]),
        "signed_dz_m": float(d[DEPTH_AXIS]),
    }


def extent_error(
    pred_extent: Any,
    gt_extent: Any,
    *,
    relative: bool = False,
) -> dict[str, float]:
    """尺寸误差：逐轴绝对差 + L1 和 + 最大轴。

    `pred_extent` / `gt_extent` 是 `(3,)` 的**跨度**（`max - min`），不是包围盒。

    `relative=True` 时另给 `rel_l1`——但**默认关**：`extent_3d` 是轴对齐包围盒，
    薄板物体的某个轴会趋近 0，相对误差在那里没有意义（`5 mm / 2 mm = 2.5`
    读起来像灾难，实际是毫米级）。要用就必须自己承担这个读法。
    """
    if pred_extent is None or gt_extent is None:
        raise TypeError(
            "extent_error 不接受 None —— 理由同 centroid_error（显式记账，不静默填 0）。"
        )
    p = _as_vec3(pred_extent, "pred_extent")
    g = _as_vec3(gt_extent, "gt_extent")
    if (p < 0).any() or (g < 0).any():
        # 负跨度只可能来自「min/max 传反」，不是真实尺寸。
        raise ValueError(f"跨度不能为负：pred={p!r} gt={g!r} —— 检查是否把 min/max 传反了")

    abs_per_axis = np.abs(p - g)
    out: dict[str, float] = {
        "l1_m": float(abs_per_axis.sum()),
        "max_axis_m": float(abs_per_axis.max()),
        "abs_dx_m": float(abs_per_axis[0]),
        "abs_dy_m": float(abs_per_axis[1]),
        "abs_dz_m": float(abs_per_axis[2]),
        # 有符号版本：系统性放大与系统性缩小必须能分开。
        # 「尺寸 2.9×」这类结论靠它，靠 `abs` 是看不出来的。
        "signed_dx_m": float(p[0] - g[0]),
        "signed_dy_m": float(p[1] - g[1]),
        "signed_dz_m": float(p[2] - g[2]),
        "gt_l1_m": float(np.abs(g).sum()),
    }
    if relative:
        denom = float(np.abs(g).sum())
        out["rel_l1"] = float(abs_per_axis.sum() / denom) if denom > _MIN_DENOM_M else float("nan")
    return out


def point_cloud_error(pred_pts: Any, gt_pts: Any) -> dict[str, float]:
    """点云级指标：`abs_rel` / `sq_rel` / `rmse` / `δ<1.25^k`。

    ⚠ **两个输入必须是同一批像素的点**，形状都是 `(3, N)`，且 `N` 相同。
    本函数不猜对应关系：逐点指标需要点对点比较，而「预测点云的第 i 个点」
    与「真值点云的第 i 个点」只有在**同一像素网格**上才有意义。
    形状不同就直接报错，不做最近邻匹配 —— 那是另一个方法（会引入它自己的误差，
    且没有真值可校）。

    `δ` 的定义沿用深度估计惯例：`max(pred/gt, gt/pred) < 1.25^k`。
    分母用 **gt**（真值），不是 pred —— 反了会让「整体放大」的预测看起来更好。
    所以这里对 `|gt|` 有下限检查：真值趋近 0 的点（噪声点、天空）不参与 δ。
    """
    p = np.asarray(pred_pts, dtype=np.float64)
    g = np.asarray(gt_pts, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != 3:
        raise ValueError(f"pred_pts 必须是 (3,N)，收到 shape={p.shape}")
    if g.shape != p.shape:
        raise ValueError(
            f"点云形状必须一致（同像素网格），收到 pred={p.shape} gt={g.shape}。"
            "形状不同时不做最近邻匹配 —— 那会引入一个没有真值可校的额外误差源。"
        )
    n = int(p.shape[1])
    if n == 0:
        # 空点云不是「误差 0」，是「没有可测的量」。给 nan 并写明 n=0，
        # 让聚合层去决定怎么记账（它手上有 n_expected）。
        return {"n": 0, "abs_rel": float("nan"), "sq_rel": float("nan"),
                "rmse_m": float("nan"), "delta1": float("nan"),
                "delta2": float("nan"), "delta3": float("nan")}

    diff = p - g
    dist = np.linalg.norm(diff, axis=0)
    rmse = float(np.sqrt(np.mean(dist ** 2)))

    # 逐点比较需要一个标量「距离」量。用**到相机原点的距离**而不是 z：
    # 本项目里 `points` 是 rays × radius 得到的，radius 就是它（§2）。
    # 用 z 会把横向错位算成 0。
    gp = np.linalg.norm(g, axis=0)
    pr = np.linalg.norm(p, axis=0)
    valid = gp > _MIN_DENOM_M
    if not valid.any():
        return {"n": n, "abs_rel": float("nan"), "sq_rel": float("nan"),
                "rmse_m": rmse, "delta1": float("nan"),
                "delta2": float("nan"), "delta3": float("nan"), "n_valid": 0}

    ratio = np.maximum(pr[valid] / gp[valid], gp[valid] / pr[valid])
    out: dict[str, float] = {
        "n": n,
        "n_valid": int(valid.sum()),
        "abs_rel": float(np.mean(np.abs(pr[valid] - gp[valid]) / gp[valid])),
        "sq_rel": float(np.mean(((pr[valid] - gp[valid]) ** 2) / gp[valid])),
        "rmse_m": rmse,
    }
    for i, thr in enumerate(DELTA_THRESHOLDS, start=1):
        out[f"delta{i}"] = float(np.mean(ratio < thr))
    return out


def decomposition_residual(err: dict[str, float]) -> float:
    """`|total² − (lateral² + depth²)|` —— 分解自洽性的**可计算**证明。

    物理上这个量恒为 0（勾股）。它不为 0 就说明：
    有人把某一项按别的口径算了（例如横向只取了 dx、纵深用了带符号值），
    或者有人**改成了加法**。写出来是为了让这件事可断言，而不是靠读文档记住。
    """
    total = err.get("total_m")
    lateral = err.get("lateral_m")
    depth = err.get("depth_m")
    if total is None or lateral is None or depth is None:
        raise ValueError("需要 centroid_error 的完整输出（total_m / lateral_m / depth_m）")
    return float(abs(total ** 2 - (lateral ** 2 + depth ** 2)))


# ----------------------------------------------------------------------------
# 关系
# ----------------------------------------------------------------------------


def relation_prf(
    pred: Iterable[tuple[str, str, str]],
    gt: Iterable[tuple[str, str, str]],
) -> dict[str, float]:
    """关系的 precision / recall / F1（`(source, target, relation)` 三元组集合）。

    ⚠ **对称关系不在这里展开**：`left_of(a,b)` 为真时 `right_of(b,a)` 也为真，
    展开规则属于关系层的定义（`scene_graph/relations.py`），
    本模块只比较**调用方给的两个集合**。要展开就在建集合时展开，
    在这里展开会让「口径」藏进一个指标函数里。

    `pred` / `gt` 为空集合时 precision/recall 给 `nan` 而不是 0 或 1：
    「没有预测」与「预测全错」不是同一件事，而 0/1 会把它们混起来。
    """
    p = set(pred)
    g = set(gt)
    tp = len(p & g)
    fp = len(p - g)
    fn = len(g - p)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and np.isfinite(precision) and np.isfinite(recall)
        else float("nan")
    )
    return {
        "tp": float(tp), "fp": float(fp), "fn": float(fn),
        "n_pred": float(len(p)), "n_gt": float(len(g)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
    }


# ----------------------------------------------------------------------------
# 聚合
# ----------------------------------------------------------------------------


def _stats(values: Sequence[float]) -> dict[str, float]:
    """中位数 / 均值 / P90 / 最大 —— 统一用 `float('nan')` 表示「无样本」。

    为什么四样都给：单目几何误差**长尾**是常态（掩码泄漏一次就几个像素的远景点）。
    只报均值会被尾部主导，只报中位数会隐藏最坏情况。`n` 一起给，否则读者无法判断
    这些数字有多可信。
    """
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"n": 0.0, "median": float("nan"), "mean": float("nan"),
                "p90": float("nan"), "max": float("nan")}
    return {
        "n": float(finite.size),
        "median": float(np.median(finite)),
        "mean": float(finite.mean()),
        "p90": float(np.percentile(finite, 90.0)),
        "max": float(finite.max()),
    }


#: 会被 `summarize` 自动聚合的键（前缀无关，按名字取）。
_SUMMARY_KEYS: tuple[str, ...] = (
    "total_m", "lateral_m", "depth_m",
    "l1_m", "max_axis_m",
    "abs_rel", "sq_rel", "rmse_m",
    "delta1", "delta2", "delta3",
    "precision", "recall", "f1",
)


def summarize(rows: Sequence[dict[str, float]], *, n_expected: int) -> dict[str, Any]:
    """把逐项结果聚合成一张可报告的统计表。

    ⚠ **`n_expected` 是必填的，且分母恒用它。** 这是本模块最容易被写错、
    又最容易被读错的地方：

        如果 8 道绝对题里有 3 道「没算出几何」，而 `n_missing` 被丢掉，
        读者看到的是「5 题、中位误差 0.2 m」，会读成「还挺准」。
        真相是「8 题里 3 题根本没出数」。

    `metrics.py` 的 strict 口径就是这么定的（解析失败记 0 分、分母不减），
    这里沿用同一条：`n_missing` 必须出现在结果里，且 `n_expected` 进分母。
    """
    if n_expected < len(rows):
        raise ValueError(
            f"n_expected={n_expected} 小于实际行数 {len(rows)} —— "
            "分母比分子还小，说明调用方把「应测总数」记错了"
        )
    out: dict[str, Any] = {
        "n_expected": int(n_expected),
        "n_measured": len(rows),
        "n_missing": int(n_expected - len(rows)),
        "missing_rate": (float(n_expected - len(rows)) / n_expected) if n_expected else float("nan"),
    }
    for key in _SUMMARY_KEYS:
        vals = [r[key] for r in rows if key in r]
        if vals:
            out[key] = _stats(vals)
    return out


def is_self_consistent(
    rows: Sequence[dict[str, float]],
    *,
    atol: float = 1e-12,
) -> tuple[bool, list[int]]:
    """检查每条质心误差是否满足 `total² = lateral² + depth²`。

    返回 `(全部通过?, 失败的行号)`。

    **任何几何指标上线前都要先跑一次「预测 = 真值」**，那时所有误差必须恰好为 0。
    这个函数是那条纪律的机械化版本：与其在文档里声称分解自洽，不如可断言。
    """
    bad: list[int] = []
    for i, row in enumerate(rows):
        try:
            r = decomposition_residual(row)
        except ValueError:
            continue  # 不是质心误差行，跳过
        if r > atol:
            bad.append(i)
    return (not bad), bad
