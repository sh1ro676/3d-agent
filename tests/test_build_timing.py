#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""「建图要多久」的口径测试 —— 零 GPU、零 API、零联网。

为什么值得单独一个文件
====================
这个问题在本项目里**一直答不了**，而原因**不是没测**：builder 的六段细分
（`build_meta["timings_ms"]`）与每个模型的 `load_s`（`build_meta["perception"]["models"]`）
其实一直都在，9 个场景全有。真正缺的是三件**口径**上的事：

  ① 加载 / 构建 / 落盘三项**聚合**数没有进 `build_meta`，只打印进了人读的 `build_log.txt`
     —— 而 `build_scene.py` 自己就写着「`build_log.txt` 是给人看的过程记录、**不是契约**，
     程序去解析它等于把一份日志当接口用」（那句话本来是给 `image_path` 写的）；
  ② `save_ms` 根本没测过；
  ③ **没有任何一处写下「这几个数不是同一个东西」** —— 拿热态六段合计
     （`living_room` ≈3.4 s）回答「建图要多久」，会**低报约 5 倍**（实际 16 s 级）。

所以这个文件测的是**尺子**，不是数字：`None` ≠ `0`、聚合口径、以及「口径不许漏字段」。
实测耗时**不进单测** —— 它会随硬件与冷热态变，进单测只会变成"换机器就得改测试"。

导入方式：`scripts/` 不是包（没有 `__init__.py`），按路径加载，不动包结构。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


build_scene = _load("_bs_timing", "scripts/build_scene.py")
export_demo = _load("_ed_timing", "scripts/export_demo.py")


class TestTimingBlock:
    def test_total_is_the_sum_of_the_three_parts(self):
        t = build_scene.timing_block(
            model_load_ms=12580.0, build_wall_ms=3410.0, save_ms=115.0)
        assert t["model_load_ms"] == 12580.0
        assert t["build_wall_ms"] == 3410.0
        assert t["save_ms"] == 115.0
        assert t["total_wall_ms"] == 16105.0

    def test_missing_save_does_not_silently_become_zero(self):
        """★ 本文件最重要的一条：`--no-save` 时总量是**不知道**，不是 0。

        把 `None` 写成 `0` 会让「这一轮特别快」变成一个**假结论**，
        而它看起来完全正常 —— 这正是 `absent ≠ zero` 那条口径要挡的东西。
        """
        t = build_scene.timing_block(
            model_load_ms=12580.0, build_wall_ms=3410.0, save_ms=None)
        assert t["save_ms"] is None
        assert t["total_wall_ms"] is None, "缺一块时总量必须缺席，不能是 0"

    def test_zero_is_kept_as_zero(self):
        """反方向也要守住：真的测到 0（同进程第二张图不再加载模型）必须是 0，不是 None。

        两个方向都钉住，才叫「分得开」；只守一边等于没守。
        """
        t = build_scene.timing_block(
            model_load_ms=0.0, build_wall_ms=3410.0, save_ms=115.0)
        assert t["model_load_ms"] == 0.0
        assert t["total_wall_ms"] == 3525.0

    def test_values_are_rounded(self):
        t = build_scene.timing_block(
            model_load_ms=1.23456, build_wall_ms=2.34567, save_ms=3.45678)
        assert t["model_load_ms"] == 1.2
        assert t["save_ms"] == 3.5
        assert t["total_wall_ms"] == round(1.2 + 2.3 + 3.5, 1)

    def test_scope_documents_every_numeric_key(self):
        """★ 口径不许漏字段。

        加了新数却不写进口径，是「文档写了、实现没做」的**镜像版**：
        **数字出去了、说明没跟上**，读者只能自己猜它是什么。
        这条断言让「加字段忘改 scope」当场变红，而不是等报告被人读错。
        """
        t = build_scene.timing_block(
            model_load_ms=1.0, build_wall_ms=2.0, save_ms=3.0)
        scope = t["scope"]
        for key in t:
            if key == "scope":
                continue
            assert key in scope, "口径 `scope` 里没有交代 %s" % key

    def test_scope_warns_about_the_cold_start_trap(self):
        """`model_load_ms` 只在进程第一张图上非零 —— 不提这一点，它一定会被拿来平均。"""
        assert "第 2 张" in build_scene.TIMING_SCOPE

    def test_scope_admits_that_scene_json_write_is_excluded(self):
        """scene.json 自身的写入不计入（要闭包就得写两次）—— 这是个坑，必须写明。"""
        assert "scene.json" in build_scene.TIMING_SCOPE


class TestIndexProjection:
    def test_full_block_is_projected(self):
        meta = {
            "timings_ms": {"depth_ms": 2085.0},
            "timing": {
                "scope": "S", "model_load_ms": 12580.0, "build_wall_ms": 3410.0,
                "save_ms": 115.0, "total_wall_ms": 16105.0,
            },
        }
        p = export_demo._timing_payload(meta)
        assert p["stages_ms"] == {"depth_ms": 2085.0}
        assert p["total_wall_ms"] == 16105.0
        assert p["scope"] == "S"

    def test_legacy_scene_keeps_stages_and_marks_the_rest_unknown(self):
        """旧场景（本次改动前建的）只有六段 —— 其余三项必须 `None`，**不许编 0**。"""
        p = export_demo._timing_payload({"timings_ms": {"depth_ms": 1.0}})
        assert p["stages_ms"] == {"depth_ms": 1.0}
        assert p["model_load_ms"] is None
        assert p["save_ms"] is None
        assert p["total_wall_ms"] is None
        assert "没测到" in p["scope"]

    def test_scene_without_any_timing_is_absent_not_zeroed(self):
        """整个缺席比一个全 0 的块诚实 —— 后者会被读成「建图不花时间」。"""
        assert export_demo._timing_payload({}) is None

    def test_timing_key_reaches_the_scene_payload(self):
        """`_META_KEYS` 是白名单：忘了加一行，`timing` 就在进前端的载荷里静默消失。"""
        out = export_demo._meta_payload({"timing": {"total_wall_ms": 1.0}, "n_nodes": 3})
        assert "timing" in out
        assert out["n_nodes"] == 3
