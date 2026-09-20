"""`vision` —— L1 感知层，对应 `vendor/VADAR/engine/predefined_modules.py`。

对外分工（方案文档 §17 的目录树）：

    types.py         数据结构（Detection / DepthField / PerceptionLike），**零依赖**
    geometry.py      ★ 点云 → 质心 / 尺寸 / 包围盒 / 重力方向，**纯 NumPy，零 torch**
    grounding.py     GroundingDINO  包装（检测）
    segmentation.py  SAM2.1         包装（分割，**必须批量一次调用**）
    depth.py         UniDepthV2     包装（points + depth + intrinsics）
    registry.py      PerceptionStack：懒加载、显式持有引用、显存记账、卸载

**为什么 `__init__` 里什么都不导入**（与 `tools/__init__.py` 同一理由）：
    本包一半的模块要 `import torch`，在本机是 3–5 秒的代价。而 `vision.geometry`
    与 `vision.types` 是纯 NumPy —— 单元测试只想要它们，却会被迫先加载 torch。
    于是「几何可单测」这个卖点会名不副实。所以一律显式导入子模块：

        from vision.geometry import centroid_of          # 轻，随手可用
        from vision.registry import PerceptionStack      # 重，会拉起三个模型

与 VADAR 的三处关键差别（写进报告用）：
    ① VADAR 的 `depth()` 只回一个标量；这里的 `lift()` 回**整片点云 + 内参**。
    ② VADAR 的分割与检测在 CLEVR/GQA 支线才用；这里 Omni3D 主路径也必须分割，
       因为掩码质心是正确性要求，不是可选优化（见 `geometry.py` 抬头）。
    ③ VADAR 的模型是模块级单例、无法卸载；这里的模型由 `PerceptionStack`
       显式持有并可 `unload()`，显存是被记账的（§5「常驻 vs 峰值」）。
"""

from __future__ import annotations

__all__: list[str] = []
