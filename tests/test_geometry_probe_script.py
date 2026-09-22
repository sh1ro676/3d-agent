#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`scripts/run_geometry_probe.py` 的口径与失败路径测试。

两条最重要：

* `test_translate_sweep_reports_the_injected_amount` —— 注入 10 mm 平移，
  指标必须报 **10.0 mm**。这是**响应增益 = 1.0** 的刻度检查：
  指标如果系统性偏 1.3 倍，误差分解表整体就偏 1.3 倍，而且不会有东西报错。

* `test_self_consistency_fails_loudly_on_tampered_gt` —— 把真值改坏之后，
  自洽性检查必须**变红**。一个永远不会红的检查比没有检查更糟：
  它会让「尺子没错」成为一条无人验证的假设。

另有一条守着聚合口径：平移类扰动只动一个物体，用全体统计会把 200 mm 印成 0.0
（第一版就是这样），所以 `target_summary` 必须存在且只含那一个物体。
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.builders.synthesize_geometry_probe import (  # noqa: E402
    default_intrinsics,
    default_probe_boxes,
    default_scene_boxes,
    render_scene,
)


def _load_script():
    """按路径加载 `scripts/` 下的模块 —— 该目录不是包，不走 import 语句。"""
    path = ROOT / "scripts" / "run_geometry_probe.py"
    spec = importlib.util.spec_from_file_location("run_geometry_probe", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


probe = _load_script()
HW = (120, 160)
#: 测试用小图省时间，但 `fx` 必须按宽度同比例缩 —— 否则视场被压窄、
#: 物体出界，自洽性会变红而原因看起来像「指标坏了」。
FX = probe.FOV_RATIO * HW[1]


@pytest.fixture(scope="module")
def scene():
    intr = default_intrinsics(fx=FX, cx=HW[1] / 2.0, cy=HW[0] / 2.0)
    return render_scene(default_scene_boxes(), intrinsics=intr, image_hw=HW)


class TestIntrinsicsAreCoupledToResolution:
    def test_fov_ratio_keeps_the_field_of_view(self):
        """`fx = FOV_RATIO × width` ⟹ 改分辨率不改视场。

        这条守着的是一个真实缺陷：脚本曾经无论宽度都给 `fx=240`，
        于是 `--width 160` 时视场只剩 37°，物体被挤出画面 ——
        而表现是「几何误差突然变大 / 拿不到几何」，看着像指标坏了。
        """
        for width in (160, 320, 640):
            intr = default_intrinsics(fx=probe.FOV_RATIO * width, cx=width / 2.0, cy=width * 0.375)
            hfov = 2.0 * math.degrees(math.atan((width / 2.0) / intr[0, 0]))
            assert hfov == pytest.approx(2.0 * math.degrees(math.atan(0.5 / probe.FOV_RATIO)),
                                         abs=0.01)


# ---------------------------------------------------------------------------
# 打印口径
# ---------------------------------------------------------------------------


class TestFormatting:
    def test_mm_converts_and_marks_unavailable(self):
        """米 → 毫米，且**没有值**必须打成 `—` 而不是 `0.0`。

        把「没测到」印成 `0.0` 会让它读起来像「完美」，这是本项目
        已经吃过一次的教训（`metrics.py` 的 strict 口径）。
        """
        assert probe._mm(0.1) == "100.0"
        assert probe._mm(0.0) == "0.0"
        assert probe._mm(None) == "—"
        assert probe._mm(float("nan")) == "—"
        assert probe._mm(float("inf")) == "—"


# ---------------------------------------------------------------------------
# 缺几何的记账
# ---------------------------------------------------------------------------


class TestMissingAccounting:
    def test_missing_object_enters_the_denominator(self, scene):
        """一个物体拿不到几何 ⟹ 进 missing，但 `n_expected` 不变。

        `n_missing` 被丢掉会让「9 个里 3 个失败」读成「6 个里还挺准」。
        """
        gt = dict(scene.gt_visible)
        pred = {oid: dict(g) for oid, g in gt.items()}
        victim = sorted(pred)[0]
        pred[victim] = {**pred[victim], "centroid_3d": None}

        crows, _erows, missing = probe._per_object_errors(pred, gt)
        assert missing == [victim]
        s = probe._summarize_errors(crows, _erows, n_expected=len(gt))
        assert s["n_expected"] == len(gt)
        assert s["n_measured"] == len(gt) - 1
        assert s["n_missing"] == 1

    def test_all_objects_present_gives_zero_missing(self, scene):
        crows, erows, missing = probe._per_object_errors(scene.gt_visible, scene.gt_visible)
        assert missing == []
        s = probe._summarize_errors(crows, erows, n_expected=len(scene.gt_visible))
        assert s["n_missing"] == 0
        assert s["centroid"]["total_m"]["median"] == 0.0


# ---------------------------------------------------------------------------
# 三层
# ---------------------------------------------------------------------------


class TestLayers:
    def test_self_consistency_is_exactly_zero(self, scene):
        res = probe.run_self_consistency(scene)
        assert res["all_zero"] is True
        assert res["max_centroid_total_m"] == 0.0
        assert res["max_extent_l1_m"] == 0.0

    def test_self_consistency_fails_loudly_on_tampered_gt(self, scene):
        """把真值改坏，检查器必须变红 —— 否则它是一条无人验证的假设。"""
        from dataset.builders.synthesize_geometry_probe import SyntheticScene

        broken = SyntheticScene(
            points_chw=scene.points_chw,
            masks=scene.masks,
            grid_hw=scene.grid_hw,
            intrinsics=scene.intrinsics,
            gt_box=scene.gt_box,
            gt_visible={
                **scene.gt_visible,
                sorted(scene.gt_visible)[0]: {
                    **scene.gt_visible[sorted(scene.gt_visible)[0]],
                    "centroid_3d": np.array([9.0, 9.0, 9.0]),
                },
            },
            meta=scene.meta,
        )
        res = probe.run_self_consistency(broken)
        assert res["all_zero"] is False
        assert res["max_centroid_total_m"] > 1.0

    def test_intrinsic_bias_is_nonzero_and_decomposed(self, scene):
        """可见性固有偏差必须非零（否则两层真值等于同一层），且横向/纵深分开。"""
        ib = probe.run_intrinsic_bias(scene)
        c = ib["centroid"]["centroid"]
        assert c["total_m"]["median"] > 0.0
        assert c["depth_m"]["median"] > 0.0        # 可见面在近面上
        assert "lateral_m" in c
        assert len(ib["per_object"]) == len(scene.gt_box)


# ---------------------------------------------------------------------------
# 刻度检查：注入多少，指标就该报多少
# ---------------------------------------------------------------------------


class TestScaleCheck:
    def test_translate_sweep_reports_the_injected_amount(self, scene):
        """注入 10 mm 平移 ⟹ 指标报 **10.0 mm**（响应增益 = 1.0）。

        指标若系统性偏 1.3 倍，整张误差分解表就整体偏 1.3 倍，且不会报错。
        所以这条断言的是精确值。
        """
        for kind, field in (("object_translate_x", "lateral_m"), ("object_translate_z", "depth_m")):
            r = probe.run_sweep(scene, kind, 0.010)
            assert r["target_object"] == "sofa_1"
            assert r["target_summary"]["centroid"][field]["median"] == pytest.approx(0.010)
            # 另一轴必须纹丝不动
            other = "depth_m" if field == "lateral_m" else "lateral_m"
            assert r["target_summary"]["centroid"][other]["median"] == pytest.approx(0.0)

    def test_translate_summary_uses_target_not_the_whole_scene(self, scene):
        """聚合口径必须匹配扰动语义：全体统计会把 200 mm 印成 0.0。

        这条钉住的是第一版真实发生过的报告缺陷 —— 它**看起来像指标失灵**。
        """
        r = probe.run_sweep(scene, "object_translate_x", 0.20)
        # 全体中位数是 0（其余 8 个没动）……
        assert r["summary"]["centroid"]["total_m"]["median"] == pytest.approx(0.0)
        # ……而被扰动的那个物体是 200 mm。两个都要留在 JSON 里。
        assert r["target_summary"]["centroid"]["total_m"]["median"] == pytest.approx(0.20)
        assert r["target_summary"]["n_expected"] == 1

    def test_intrinsics_scale_moves_lateral_only(self, scene):
        """内参缩放：横向涨、纵深恒为 0 —— 分解必须在数据上成立，不只是口头上。"""
        for k in (1.10, 1.50):
            r = probe.run_sweep(scene, "intrinsics_scale", k)
            c = r["summary"]["centroid"]
            assert c["lateral_m"]["median"] > 0.0
            assert c["depth_m"]["median"] == pytest.approx(0.0)

    def test_depth_scale_moves_both_axes(self, scene):
        """深度缩放：两轴都涨 —— 与内参缩放必须落在不同的格子里。

        这正是「诊断哪个指标看得见哪种错误」的判据：两种扰动如果响应相同，
        这两个指标就是重复的。
        """
        r = probe.run_sweep(scene, "depth_scale", 1.10)
        c = r["summary"]["centroid"]
        assert c["lateral_m"]["median"] > 0.0
        assert c["depth_m"]["median"] > 0.0

    def test_mask_grow_hits_extent_much_harder_than_centroid(self, scene):
        """掩码外溢主要在**尺寸**上可见，质心几乎不动。

        这是分解表最有用的一个格子：只做质心评估的系统，会完全看不见
        掩码外溢这种错误（实测：grow 2 px 时尺寸误差是质心误差的百倍量级）。
        """
        r = probe.run_sweep(scene, "mask_grow", 2.0)
        centroid = r["summary"]["centroid"]["total_m"]["median"]
        extent = r["summary"]["extent"]["l1_m"]["median"]
        assert extent > 10 * centroid
        # 掩码变化不改变纵深（它只动掩码边界，而边界在横向/竖直方向扩散）
        assert r["summary"]["centroid"]["depth_m"]["median"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 关系层
# ---------------------------------------------------------------------------


class TestRelationLayer:
    def test_relation_set_is_nonempty_and_uses_the_real_layer(self, scene):
        labels = {b.object_id: b.label for b in default_probe_boxes()}
        rels = probe._relation_set(scene.gt_visible, labels)
        assert rels
        assert all(len(t) == 3 for t in rels)
        assert any(r[2] == "left_of" for r in rels)

    def test_identity_case_has_perfect_f1(self, scene):
        """无扰动时预测关系集合与真值相同 ⟹ F1 必须恰好 1.0。"""
        labels = {b.object_id: b.label for b in default_probe_boxes()}
        rows = probe.run_relation_cases(scene, labels)
        assert rows[0]["kind"] == "none"
        assert rows[0]["f1"] == pytest.approx(1.0)

    def test_relation_is_insensitive_to_intrinsics_error(self, scene):
        """关系层**看不见**内参错误 —— §22 的 128/133 = 96% 是同一现象。

        这条不是「关系算错了」，而是一条关于**指标分辨力**的结论：
        几何误差涨到 100 mm 量级时 F1 只掉 2%。所以关系准确率
        **不能单独当护栏**，它必须与几何指标一起报。
        """
        labels = {b.object_id: b.label for b in default_probe_boxes()}
        rows = {r["kind"]: r for r in probe.run_relation_cases(scene, labels)}
        assert rows["intrinsics_scale"]["f1"] > 0.95
        # 而同一扰动下几何误差是可见的（对照）
        geo = probe.run_sweep(scene, "intrinsics_scale", 1.10)
        assert geo["summary"]["centroid"]["total_m"]["median"] > 0.05


# ---------------------------------------------------------------------------
# 入口与失败路径
# ---------------------------------------------------------------------------


class TestMain:
    def test_dry_run_writes_json_and_exits_zero(self, tmp_path):
        out = tmp_path / "geo.json"
        code = probe.main(["--dry-run", "--json-out", str(out)])
        assert code == 0
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["self_consistency"]["all_zero"] is True
        # 干运行不该有扫描段（也就没有 ②③ 的成本）
        assert "sweeps" not in data
        # 单位口径必须随结果一起落盘，否则读者无从判断数字量级
        assert data["units"]["length_in_json"] == "m"
        assert data["units"]["length_in_printed_table"] == "mm"

    def test_full_run_writes_all_layers(self, tmp_path):
        out = tmp_path / "geo_full.json"
        code = probe.main(["--width", "160", "--height", "120", "--json-out", str(out)])
        assert code == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["self_consistency"]["all_zero"] is True
        assert set(data["sweeps"]) == set(probe.SWEEPS)
        assert len(data["relation_cases"]) == len(probe.RELATION_CASES)
        # 每个网格点都带齐 n_expected / n_missing（缺几何的记账不能丢）
        for rows in data["sweeps"].values():
            for r in rows:
                assert r["summary"]["n_expected"] == len(data["visibility"]["objects"])
                assert "n_missing" in r["summary"]

    def test_every_swept_perturbation_is_documented(self):
        """扫描的每一项都必须在扰动清单里有解释，否则响应曲线无法解读。"""
        from dataset.builders.synthesize_geometry_probe import PERTURBATIONS

        for kind in probe.SWEEPS:
            assert kind in PERTURBATIONS, f"{kind} 不在 PERTURBATIONS 里"
        for kind, _kw in probe.RELATION_CASES:
            assert kind in PERTURBATIONS, f"关系用例 {kind} 不在 PERTURBATIONS 里"

    def test_json_lengths_are_finite(self, tmp_path):
        """JSON 里不能出现 nan/inf —— 它们不是合法 JSON，会让下游解析器炸。

        `json.dumps` 默认把它们写成 `NaN`/`Infinity`（Python 扩展），
        严格解析器（含本项目的评估层）会拒收。
        """
        out = tmp_path / "geo_finite.json"
        probe.main(["--width", "160", "--height", "120", "--json-out", str(out)])
        text = out.read_text(encoding="utf-8")
        assert "NaN" not in text, "JSON 里出现了 NaN —— 严格解析器会拒收"
        assert "Infinity" not in text
        data = json.loads(text)
        # 逐层扫一遍有限性
        def walk(o):
            if isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
            elif isinstance(o, float):
                assert math.isfinite(o)
        walk(data)
