"""`vision/depth.py::resolve_intrinsics` 的单元测试 —— 纯 NumPy，零 GPU、零 torch。

## 为什么这个纯函数值得单独一个测试文件

它只有十行，但它守着一个**会被静默搞错**的契约：

    `infer(rgb, camera=Pinhole(K_gt))` 会让 geometry 按 K_gt 生成，
    但模型回传的 `out["intrinsics"]` 仍然是它**自己相机头**的输出。

Phase 0 Step 6 实测：传入 GT 的 `fx=518.9`，回传依然写 `fx=163.7`。
如果不做覆盖，`scene.camera_intrinsics` 就会记下一个「与点云不自洽」的 K ——
点云按 519 生成、K 却写着 163.7。下游拿它做可视化或反投影核对时，
每一步单独看都对，合起来自相矛盾。这是最难查的一类 bug。

`vision/depth.py` 在模块级只 import `numpy` 与 `vision.types`（torch 在函数内部
才 import），所以这个文件可以在**不拉起 torch** 的前提下跑完 ——
`test_module_does_not_import_torch_at_module_level` 就在守这条属性。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.depth import resolve_intrinsics  # noqa: E402
from vision.geometry import intrinsics_matrix  # noqa: E402

#: Phase 0 Step 6 实测的两份内参，直接用真实数字。
PREDICTED = intrinsics_matrix(163.74, 163.42, 322.00, 248.11)
PROVIDED = intrinsics_matrix(518.86, 519.47, 325.58, 253.74)


class TestResolveIntrinsics:
    def test_without_provided_uses_prediction(self):
        K, source = resolve_intrinsics(PREDICTED)
        assert source == "predicted"
        np.testing.assert_allclose(K, PREDICTED)

    def test_with_provided_overrides_the_prediction(self):
        """★ 核心用例：必须返回 provided，而不是模型回传的那一份。

        这两份在实测里差 3.17 倍（fx 163.7 vs 518.9）。返回错的那份不会报错 ——
        只会让所有横向尺寸安静地放大 3 倍，直到有人量出 6.7 m 宽的沙发。
        """
        K, source = resolve_intrinsics(PREDICTED, PROVIDED)
        assert source == "provided"
        np.testing.assert_allclose(K, PROVIDED)
        # 明确断言「不是」预测值，免得将来有人把覆盖写反了还能通过
        assert not np.allclose(K, PREDICTED)
        assert K[0, 0] == pytest.approx(518.86)

    def test_accepts_various_input_shapes(self):
        """列表 / 嵌套列表 / 摊平的 9 个数都要能吃 —— CLI 会传列表。"""
        flat = [518.86, 0.0, 325.58, 0.0, 519.47, 253.74, 0.0, 0.0, 1.0]
        for form in (PROVIDED, PROVIDED.tolist(), flat):
            K, source = resolve_intrinsics(PREDICTED, form)
            assert source == "provided"
            assert K.shape == (3, 3)
            assert K[0, 0] == pytest.approx(518.86)

    def test_non_finite_provided_raises(self):
        bad = PROVIDED.copy()
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="非有限值"):
            resolve_intrinsics(PREDICTED, bad)

    def test_non_positive_focal_raises(self):
        """fx<=0 的内参必须当场炸掉：`x = (u-cx)·z/fx` 会给出翻转的横向坐标。"""
        for bad_fx in (0.0, -500.0):
            with pytest.raises(ValueError, match="必须为正"):
                resolve_intrinsics(PREDICTED, intrinsics_matrix(bad_fx, 500.0, 320.0, 240.0))

    def test_wrong_size_raises(self):
        with pytest.raises(ValueError):
            resolve_intrinsics(PREDICTED, np.eye(4))

    def test_predicted_is_also_accepted_loosely(self):
        """模型回传值走同样的规整化路径 —— 不因为「是我们自己的」就少校验一次。"""
        K, source = resolve_intrinsics([[500.0, 0.0, 10.0],
                                        [0.0, 500.0, 20.0],
                                        [0.0, 0.0, 1.0]])
        assert source == "predicted"
        assert K[1, 2] == pytest.approx(20.0)

    def test_module_does_not_import_torch_at_module_level(self):
        """`vision.depth` 的 torch 必须在函数内 import。

        否则本文件所在的一整套纯 NumPy 测试都要先付 3–5 秒的 torch 导入成本，
        「builder 逻辑可脱离 GPU 单测」这条设计就破了。torch 是在 `__call__` /
        `from_pretrained` 内部才 import 的，所以模块对象上不该有 `torch` 属性。
        """
        import vision.depth as m

        assert not hasattr(m, "torch"), (
            "vision.depth 在模块级 import 了 torch —— 请移到函数内，"
            "否则纯 NumPy 测试会被拖慢数秒"
        )
