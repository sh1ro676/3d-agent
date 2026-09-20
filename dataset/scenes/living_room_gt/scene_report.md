# 场景报告 — `living_room_gt`

> 由 `scripts/report_scene.py` 生成（L5 第一类输出）。
> 工具库 `1.1.0` · 关系方法 `geometry_v1` ·
> 物体 9 个 · 关系对 36 对 ·
> **未使用 GPU**（L5 是零额外模型成本的一层）。

## 1. 物体清单

| id | label | 质心 (x, y, z) m | 尺寸 w×h×l m | 置信度 | 质心来源 |
|---|---|---|---|---|---|
| `sofa_1` | sofa | +0.797, +0.319, +3.175 | 2.34×0.82×1.26 | 0.836 | mask |
| `picture_1` | picture | +1.337, -1.109, +4.058 | 1.12×0.98×0.24 | 0.795 | mask |
| `mirror_1` | mirror | -0.300, -1.037, +3.949 | 0.88×1.28×0.27 | 0.676 | mask |
| `chair_1` | chair | -0.740, +0.653, +1.834 | 0.98×0.30×0.83 | 0.528 | mask |
| `picture_2` | picture | -1.166, -1.060, +4.015 | 0.26×0.23×0.04 | 0.501 | mask |
| `picture_3` | picture | -1.155, -0.837, +4.054 | 0.25×0.21×0.06 | 0.489 | mask |
| `sofa_chair_1` | sofa chair | -1.066, +0.328, +2.739 | 1.33×0.81×1.48 | 0.440 | mask |
| `table_1` | table | +0.589, +0.606, +1.871 | 1.06×0.54×1.03 | 0.399 | mask |
| `picture_4` | picture | -1.919, -0.822, +4.082 | 0.15×0.29×0.15 | 0.332 | mask |

## 2. 自然语言摘要

```
场景 living_room_gt 共 9 个物体：chair × 1、mirror × 1、picture × 4、sofa × 1、sofa chair × 1、table × 1。
最紧凑的三组相邻关系：picture_2→picture_3 0.23 m；picture_3→picture_2 0.23 m；picture_4→picture_3 0.76 m。
方位（左侧）：sofa_1 在 picture_1 的左侧；mirror_1 在 table_1 的左侧；chair_1 在 table_1 的左侧；picture_2 在 sofa_chair_1 的左侧。
方位（右侧）：sofa_1 在 mirror_1 的右侧；sofa_1 在 chair_1 的右侧；sofa_1 在 picture_2 的右侧；sofa_1 在 picture_3 的右侧。
方位（前方）：sofa_1 在 picture_1 的前方；sofa_1 在 mirror_1 的前方；sofa_1 在 picture_2 的前方；sofa_1 在 picture_3 的前方。
方位（后方）：sofa_1 在 chair_1 的后方；sofa_1 在 sofa_chair_1 的后方；sofa_1 在 table_1 的后方；picture_1 在 mirror_1 的后方。
方位（上方）：sofa_1 在 chair_1 的上方；sofa_1 在 table_1 的上方；picture_1 在 mirror_1 的上方；picture_1 在 chair_1 的上方。
方位（下方）：sofa_1 在 picture_1 的下方；sofa_1 在 mirror_1 的下方；sofa_1 在 picture_2 的下方；sofa_1 在 picture_3 的下方。
**尺度未校正**：相对关系可信，绝对米数不可引用。
注意：3 个物体的掩码点数少于 1000（[('picture_4', 809), ('picture_3', 893), ('picture_2', 965)]）—— 很可能是伪物体而非真实存在：点云质心不可信，物体计数与关系边数都会因此偏高。实测分界为伪物体 809–965 点 vs 真实物体 16137–30660 点（17 倍空隙）。
```

## 3. 可信度自述

- 尺度已校正：**False**（scale_factor=1.0）
- 内参来源：`provided`
- 重力方向：`-y`（tilt 11.95°，可靠=True）
- 关系是否被截断：**False**

告诫：

- 尺度未校正（scale_factor=1.0）：全部米制数字共享同一个未知比例因子，**相对关系可信、绝对数值不可采信**。要引用绝对值需先 calibrate_scale。
- 3 个物体的掩码点数少于 1000（[('picture_4', 809), ('picture_3', 893), ('picture_2', 965)]）—— 很可能是伪物体而非真实存在：点云质心不可信，物体计数与关系边数都会因此偏高。实测分界为伪物体 809–965 点 vs 真实物体 16137–30660 点（17 倍空隙）。

## 4. 失败诊断

主环节：**尺度**（共 2 条发现，按严重度降序）

1. **[尺度]** 尺度未校正（scale_factor=1.0）
   - 依据：绝对米数含未知比例因子；相对关系不受影响。
   - 处理：用场景内已知尺寸物体或数据集 GT 做一次 calibrate_scale。
2. **[检测]** 3 个物体的掩码点数不足 1000
   - 依据：[('picture_4', 809), ('picture_3', 893), ('picture_2', 965)]。实测分界：伪物体 809–965 点 vs 真实物体 16137–30660 点（17 倍空隙）。
   - 处理：这些多半是伪物体（低分检测框）：考虑按 score 或点数过滤后重建，并把它们标成 LOW_CONFIDENCE 而不是当作确定物体。

## 5. 反事实（纯图操作）

移除 `sofa_1`：

- 物体 9 → 8
- 关系 396 → 308（消失 88 条，翻转 0 条）
- 重跑视觉模型 **0** 次（纯图操作）

移除后各物体的最近邻：

- `picture_2` → `picture_3` 0.227 m
- `picture_3` → `picture_2` 0.227 m
- `picture_4` → `picture_3` 0.765 m
- `mirror_1` → `picture_2` 0.869 m
- `chair_1` → `sofa_chair_1` 1.016 m
- `sofa_chair_1` → `chair_1` 1.016 m
- `table_1` → `chair_1` 1.330 m
- `picture_1` → `mirror_1` 1.642 m
