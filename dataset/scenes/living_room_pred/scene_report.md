# 场景报告 — `living_room_pred`

> 由 `scripts/report_scene.py` 生成（L5 第一类输出）。
> 工具库 `1.1.0` · 关系方法 `geometry_v1` ·
> 物体 9 个 · 关系对 36 对 ·
> **未使用 GPU**（L5 是零额外模型成本的一层）。

## 1. 物体清单

| id | label | 质心 (x, y, z) m | 尺寸 w×h×l m | 置信度 | 质心来源 |
|---|---|---|---|---|---|
| `sofa_1` | sofa | +2.400, +1.032, +2.864 | 6.71×2.35×1.13 | 0.836 | mask |
| `picture_1` | picture | +3.883, -3.033, +3.631 | 2.92×2.69×0.21 | 0.795 | mask |
| `mirror_1` | mirror | -0.804, -2.922, +3.664 | 2.57×3.69×0.28 | 0.676 | mask |
| `chair_1` | chair | -1.996, +1.847, +1.584 | 2.72×0.94×0.79 | 0.528 | mask |
| `picture_2` | picture | -3.243, -2.908, +3.612 | 0.72×0.66×0.06 | 0.501 | mask |
| `picture_3` | picture | -3.240, -2.287, +3.678 | 0.72×0.61×0.08 | 0.489 | mask |
| `sofa_chair_1` | sofa chair | -2.888, +0.984, +2.374 | 3.41×2.24×1.91 | 0.440 | mask |
| `table_1` | table | +1.795, +1.813, +1.706 | 3.00×1.48×1.29 | 0.399 | mask |
| `picture_4` | picture | -5.346, -2.207, +3.639 | 0.38×0.81×0.18 | 0.332 | mask |

## 2. 自然语言摘要

```
场景 living_room_pred 共 9 个物体：chair × 1、mirror × 1、picture × 4、sofa × 1、sofa chair × 1、table × 1。
最紧凑的三组相邻关系：picture_2→picture_3 0.62 m；picture_3→picture_2 0.62 m；chair_1→sofa_chair_1 1.47 m。
方位（左侧）：sofa_1 在 picture_1 的左侧；mirror_1 在 table_1 的左侧；chair_1 在 table_1 的左侧；picture_2 在 sofa_chair_1 的左侧。
方位（右侧）：sofa_1 在 mirror_1 的右侧；sofa_1 在 chair_1 的右侧；sofa_1 在 picture_2 的右侧；sofa_1 在 picture_3 的右侧。
方位（前方）：sofa_1 在 picture_1 的前方；sofa_1 在 mirror_1 的前方；sofa_1 在 picture_2 的前方；sofa_1 在 picture_3 的前方。
方位（后方）：sofa_1 在 chair_1 的后方；sofa_1 在 sofa_chair_1 的后方；sofa_1 在 table_1 的后方；picture_1 在 chair_1 的后方。
方位（上方）：sofa_1 在 chair_1 的上方；sofa_1 在 table_1 的上方；picture_1 在 mirror_1 的上方；picture_1 在 chair_1 的上方。
方位（下方）：sofa_1 在 picture_1 的下方；sofa_1 在 mirror_1 的下方；sofa_1 在 picture_2 的下方；sofa_1 在 picture_3 的下方。
**尺度未校正**：相对关系可信，绝对米数不可引用。
注意：内参来自模型预测而非外部传入：横向米制尺度不可信（§21/§22 实测三维误差中位数 1.943 m vs 传入 GT 的 0.267 m）。
注意：重力方向不可靠（up_axis=-y, reason=band_not_horizontal）：`above`/`below` 可能整体翻转。
注意：3 个物体的掩码点数少于 1000（[('picture_4', 809), ('picture_3', 893), ('picture_2', 965)]）—— 很可能是伪物体而非真实存在：点云质心不可信，物体计数与关系边数都会因此偏高。实测分界为伪物体 809–965 点 vs 真实物体 16137–30660 点（17 倍空隙）。
```

## 3. 可信度自述

- 尺度已校正：**False**（scale_factor=1.0）
- 内参来源：`predicted`
- 重力方向：`-y`（tilt 63.04°，可靠=False）
- 关系是否被截断：**False**

告诫：

- 尺度未校正（scale_factor=1.0）：全部米制数字共享同一个未知比例因子，**相对关系可信、绝对数值不可采信**。要引用绝对值需先 calibrate_scale。
- 内参来自模型预测而非外部传入：横向米制尺度不可信（§21/§22 实测三维误差中位数 1.943 m vs 传入 GT 的 0.267 m）。
- 重力方向不可靠（up_axis=-y, reason=band_not_horizontal）：`above`/`below` 可能整体翻转。
- 3 个物体的掩码点数少于 1000（[('picture_4', 809), ('picture_3', 893), ('picture_2', 965)]）—— 很可能是伪物体而非真实存在：点云质心不可信，物体计数与关系边数都会因此偏高。实测分界为伪物体 809–965 点 vs 真实物体 16137–30660 点（17 倍空隙）。

## 4. 失败诊断

主环节：**尺度**（共 5 条发现，按严重度降序）

1. **[尺度]** 内参来自模型预测（intrinsics_source=predicted）
   - 依据：横向米制尺度不可信；实测三维误差中位数 1.943 m vs 传入 GT 的 0.267 m。
   - 处理：传入已知内参（BuildConfig.known_intrinsics，或 --intrinsics exif）。
2. **[尺度]** 重力方向不可靠（up_axis=-y）
   - 依据：reason=band_not_horizontal —— `above`/`below` 可能整体翻转。
   - 处理：补多视角或显式给出重力方向；单图下该量本质上只能估。
3. **[尺度]** 尺度未校正（scale_factor=1.0）
   - 依据：绝对米数含未知比例因子；相对关系不受影响。
   - 处理：用场景内已知尺寸物体或数据集 GT 做一次 calibrate_scale。
4. **[检测]** 4 个物体的三维尺寸超过 3.0 m
   - 依据：最大几项 [('sofa_1', 6.7), ('mirror_1', 3.69), ('sofa_chair_1', 3.41), ('table_1', 3.0)]。
   - 处理：尺寸越界多半是**内参或掩码外溢的症状**，先查上面「尺度」那几条 —— 不要直接改尺寸公式。
5. **[检测]** 3 个物体的掩码点数不足 1000
   - 依据：[('picture_4', 809), ('picture_3', 893), ('picture_2', 965)]。实测分界：伪物体 809–965 点 vs 真实物体 16137–30660 点（17 倍空隙）。
   - 处理：这些多半是伪物体（低分检测框）：考虑按 score 或点数过滤后重建，并把它们标成 LOW_CONFIDENCE 而不是当作确定物体。

## 5. 反事实（纯图操作）

移除 `sofa_1`：

- 物体 9 → 8
- 关系 396 → 308（消失 88 条，翻转 0 条）
- 重跑视觉模型 **0** 次（纯图操作）

移除后各物体的最近邻：

- `picture_2` → `picture_3` 0.624 m
- `picture_3` → `picture_2` 0.624 m
- `chair_1` → `sofa_chair_1` 1.471 m
- `sofa_chair_1` → `chair_1` 1.471 m
- `picture_4` → `picture_3` 2.108 m
- `mirror_1` → `picture_2` 2.441 m
- `table_1` → `chair_1` 3.793 m
- `picture_1` → `mirror_1` 4.688 m
