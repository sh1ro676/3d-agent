"""`scene_graph/builder.py` 的单元测试 —— 用 fake 感知栈，零 GPU、零模型、零网络。

**这个文件本身就是一条论据。** 它证明 builder 的全部分支 —— 去重、掩码降级、
无点云剔除、关系生成、掩码存盘、确定性 —— 都可以在没有 GPU 的机器上秒级验证。
早期基线做不到这一点：它的对应能力是把区域裁出来交给 VLM，
一次模型调用，既不确定也没法写断言。这就是「关系可单测」这条架构收益的实测形态。

FakePerception 只有三个方法，正好是 `PerceptionLike` 协议的全部 ——
协议刻意的窄，就是为了让这里能这么短。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_graph.builder import BuildConfig, build_scene_graph, slugify  # noqa: E402
from vision.geometry import intrinsics_matrix  # noqa: E402
from vision.types import Detection, DepthField  # noqa: E402

W, H = 200, 160


# ----------------------------------------------------------------------------
# 造世界
# ----------------------------------------------------------------------------


def pinhole_cloud(
    depth_hw: np.ndarray,
    *,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
) -> np.ndarray:
    """解析点云。默认 `fx = W`、主点在像素中心 —— 于是常量深度下，
    对称区域的质心恰好是漂亮的值，断言可以写成精确量。"""
    h, w = depth_hw.shape
    fx = float(w if fx is None else fx)
    fy = float(w if fy is None else fy)
    cx = (w / 2 - 0.5) if cx is None else cx
    cy = (h / 2 - 0.5) if cy is None else cy
    u = np.broadcast_to(np.arange(w, dtype=np.float64), (h, w))
    v = np.broadcast_to(np.arange(h, dtype=np.float64)[:, None], (h, w))
    z = depth_hw.astype(np.float64)
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z])


def make_image(w: int = W, h: int = H) -> Image.Image:
    return Image.new("RGB", (w, h), (128, 128, 128))


class FakePerception:
    """`PerceptionLike` 的 30 行实现 —— 这是接口刻意的窄带来的直接好处。"""

    def __init__(
        self,
        detections: list[Detection],
        masks: np.ndarray,
        points_chw: np.ndarray,
        *,
        image_hw: tuple[int, int] = (H, W),
        intrinsics: np.ndarray | None = None,
        intrinsics_source: str = "predicted",
    ) -> None:
        self.detections = list(detections)
        self.masks = masks
        self.detect_calls: list[str] = []
        self.segment_calls: list[list[tuple[float, ...]]] = []
        self.lift_calls: list[object] = []
        hp, wp = int(points_chw.shape[1]), int(points_chw.shape[2])
        self._field = DepthField(
            points_chw=points_chw,
            depth_hw=points_chw[2].copy(),
            # 默认内参与 `pinhole_cloud` 造点云时用的那一套**一致**
            # （fx=fy=W、主点在像素中心）。以前这里图省事写成 `np.eye(3)`，
            # 结果 fx=1 ⟹ 视场 179°，会让 builder 的视场体检在**每一个**用例上
            # 都报「不可信」。fake 造假可以，但不能造出一个不该出现的相机 ——
            # 那会把新加的检查变成噪声，然后再被顺手关掉。
            intrinsics=np.asarray(
                intrinsics_matrix(W, W, W / 2 - 0.5, H / 2 - 0.5)
                if intrinsics is None
                else intrinsics,
                dtype=np.float64,
            ),
            grid_hw=(hp, wp),
            image_hw=image_hw,
            intrinsics_source=intrinsics_source,  # type: ignore[arg-type]
        )

    def detect(self, image, prompt, *, box_threshold=0.30, text_threshold=0.25):
        self.detect_calls.append(prompt)
        return list(self.detections)

    def segment(self, image, boxes):
        self.segment_calls.append([tuple(float(v) for v in b) for b in boxes])
        return self.masks

    def lift(self, image, camera_K=None):
        # 记录送进来的内参 —— `known_intrinsics` 是否真的透传到感知层，
        # 是「内参是输入而不是后处理」这条设计的可验证点。
        self.lift_calls.append(camera_K)
        return self._field


def two_chairs() -> tuple[list[Detection], np.ndarray, np.ndarray]:
    """左右两把椅子：同深度、同高度、x 互为相反数。

    于是所有量的解析值都是整数级别的：质心 `(∓0.6, -0.2, 2.0)`、距离 `1.2 m`。
    数字好算不是为了好看 —— 是为了断言能写成**精确值**而不是「大概多少」，
    后者会在实现悄悄漂移时继续通过。
    """
    cloud = pinhole_cloud(np.full((H, W), 2.0))
    masks = np.zeros((2, H, W), dtype=bool)
    masks[0, 20:100, 20:60] = True      # 左
    masks[1, 20:100, 140:180] = True    # 右
    dets = [
        Detection("chair", 0.90, (20.0, 20.0, 60.0, 100.0)),
        Detection("chair", 0.80, (140.0, 20.0, 180.0, 100.0)),
    ]
    return dets, masks, cloud


def build(perception: FakePerception, **kw):
    return build_scene_graph(
        make_image(), perception=perception, scene_id="t", **kw
    )


# ----------------------------------------------------------------------------
# 基本结构
# ----------------------------------------------------------------------------


class TestBasicStructure:
    def test_ids_are_stable_and_score_ordered(self):
        dets, masks, cloud = two_chairs()
        res = build(FakePerception(dets, masks, cloud))
        assert res.scene.ids() == ["chair_1", "chair_2"]
        # 高置信度的排在前 —— 这条约定让「第一把椅子」有确定含义
        assert res.scene.node("chair_1").score == pytest.approx(0.90)

    def test_centroid_is_metric_camera_frame(self):
        dets, masks, cloud = two_chairs()
        res = build(FakePerception(dets, masks, cloud))
        a = res.scene.node("chair_1")
        b = res.scene.node("chair_2")
        assert a.centroid_3d == pytest.approx((-0.6, -0.2, 2.0), abs=1e-9)
        assert b.centroid_3d == pytest.approx((0.6, -0.2, 2.0), abs=1e-9)
        assert a.centroid_source == "mask"
        assert a.n_points == 80 * 40           # 掩码像素数
        assert a.frame == "camera"

    def test_extent_and_bbox_from_cloud(self):
        dets, masks, cloud = two_chairs()
        res = build(FakePerception(dets, masks, cloud))
        a = res.scene.node("chair_1")
        assert a.bbox_3d is not None
        # 左椅：x 从 (20-99.5)·0.01 到 (59-99.5)·0.01，y 从 -0.595 到 0.195
        lo, hi = a.bbox_3d.min, a.bbox_3d.max
        assert lo[0] == pytest.approx(-0.795, abs=1e-9)
        assert hi[0] == pytest.approx(-0.405, abs=1e-9)
        assert hi[1] - lo[1] == pytest.approx(0.79, abs=1e-9)
        assert a.extent_3d[0] == pytest.approx(0.39, abs=1e-9)
        assert a.extent_3d[2] == pytest.approx(0.0, abs=1e-9)   # 常量深度 → 平面
        assert all(v > 0 for v in a.extent_3d[:2])

    def test_label_counts_and_build_meta(self):
        dets, masks, cloud = two_chairs()
        res = build(FakePerception(dets, masks, cloud))
        assert res.scene.label_counts() == {"chair": 2}
        meta = res.scene.build_meta
        assert meta["n_nodes"] == 2 and meta["n_detections_raw"] == 2
        assert meta["scale_calibrated"] is False        # 显式声明没做尺度校正
        assert meta["image_hw"] == [H, W]
        assert set(meta["timings_ms"]) >= {"depth_ms", "detect_ms", "segment_ms", "relations_ms"}
        # fake 没有 stats()，应优雅退化而不是炸掉
        assert meta["perception"] == {"kind": "FakePerception"}
        assert res.scene.camera_intrinsics is not None

    def test_edges_are_enumerated_in_one_direction_only(self):
        """边只按 `(i, j)`（i<j）单方向枚举，反向关系不重复存。

        「chair_2 在 chair_1 右边」与「chair_1 在 chair_2 左边」是同一件事，
        后者已在图里。存两份等于让边数翻倍而信息量不变 —— 而边数是
        `describe_scene` 与指标 12（关系边一致率）的直接成本。
        反向查询由 `query_relation` 现算，不依赖存下来的边。

        这条不变量的另一半是**符号**：假值不建边，真值必建边。
        """
        dets, masks, cloud = two_chairs()
        res = build(FakePerception(dets, masks, cloud))
        edges = {(e.source, e.target, e.relation): e.value for e in res.scene.edges}

        assert edges[("chair_1", "chair_2", "left_of")] is True
        assert ("chair_1", "chair_2", "right_of") not in edges      # 假值不建边
        assert not any(s == "chair_2" for s, _, _ in edges)         # 没有反向边
        # 于是反向关系必须能现算出来 —— 这才是「省掉一半边」的前提
        from scene_graph.relations import left_of

        assert bool(left_of(res.scene.node("chair_2"), res.scene.node("chair_1"))) is False

    def test_distance_edge_is_geometric(self):
        dets, masks, cloud = two_chairs()
        res = build(FakePerception(dets, masks, cloud))
        dist = [
            e for e in res.scene.edges
            if e.relation == "distance"
            and {e.source, e.target} == {"chair_1", "chair_2"}
        ]
        assert len(dist) == 1
        assert float(dist[0].value) == pytest.approx(1.2, abs=1e-9)
        # 证据链必须在：能脱离代码复算「为什么是 1.2」
        assert dist[0].metric["delta_x"] == pytest.approx(-1.2, abs=1e-9)
        assert dist[0].method == "geometry_v1"


# ----------------------------------------------------------------------------
# ★ 批量分割：这条约束靠接口形状保证，测试把它钉住
# ----------------------------------------------------------------------------


class TestSegmentationIsBatched:
    def test_segment_called_exactly_once_with_all_boxes(self):
        """9 个框一次调用 200 ms vs 逐个 1512 ms —— 差 7.57×（§20 Step 0.5b）。

        调用次数是可测的，所以「必须批量」这件事不该只写在注释里。
        """
        cloud = pinhole_cloud(np.full((H, W), 2.0))
        masks = np.zeros((3, H, W), dtype=bool)
        dets = []
        for i in range(3):
            c0 = 10 + i * 60
            masks[i, 20:100, c0:c0 + 40] = True
            dets.append(Detection(f"obj{i}", 0.9 - i * 0.1, (c0, 20, c0 + 40, 100)))
        fake = FakePerception(dets, masks, cloud)
        res = build(fake)
        assert len(fake.segment_calls) == 1
        assert len(fake.segment_calls[0]) == 3          # 三个框一起进去
        assert res.scene.ids() == ["obj0_1", "obj1_1", "obj2_1"]

    def test_no_boxes_means_no_model_call(self):
        fake = FakePerception([], np.zeros((0, 0, 0), dtype=bool), pinhole_cloud(np.full((H, W), 2.0)))
        res = build(fake)
        assert fake.segment_calls == []
        assert res.scene.nodes == ()
        assert any("没有检测到任何物体" in w for w in res.warnings)


# ----------------------------------------------------------------------------
# 掩码 / 点云分辨率换算
# ----------------------------------------------------------------------------


class TestGridResampling:
    def test_centroid_consistent_across_grid_resolutions(self):
        """同一张图、同一批掩码，点云网格减半后质心必须不变。

        这条抓的是「掩码与点云分辨率不同却直接按同尺寸用」那类 bug ——
        它不报错，只让所有质心系统性偏移。精确的像素对齐另由
        `tests/test_vision_geometry.py::TestResampleMask::test_pixel_centre_alignment` 锁住。
        """
        dets, masks, cloud_full = two_chairs()
        res_full = build(FakePerception(dets, masks, cloud_full))

        grid = (H // 2, W // 2)
        cloud_half = pinhole_cloud(np.full(grid, 2.0))
        res_half = build(FakePerception(dets, masks, cloud_half, image_hw=(H, W)))

        for oid in ("chair_1", "chair_2"):
            a = res_full.scene.node(oid).centroid_3d
            b = res_half.scene.node(oid).centroid_3d
            assert b == pytest.approx(a, abs=1e-2), f"{oid} 在不同网格下质心不一致"
        assert res_half.scene.build_meta["grid_hw"] == list(grid)

    def test_mask_ref_only_when_prefix_given(self):
        dets, masks, cloud = two_chairs()
        res = build(FakePerception(dets, masks, cloud))
        assert res.scene.node("chair_1").mask_ref is None
        assert set(res.masks) == {"chair_1", "chair_2"}

        res2 = build(FakePerception(dets, masks, cloud), mask_rel_prefix="t/masks")
        assert res2.scene.node("chair_1").mask_ref == "t/masks/chair_1.png"


# ----------------------------------------------------------------------------
# 降级与剔除
# ----------------------------------------------------------------------------


class TestDegradation:
    def test_mask_too_small_falls_back_to_box(self):
        """掩码几乎为空时退回检测框 —— 与 Phase 0 实测的「框内中位数」是同一条路。

        必须**保留**这个物体并如实标记 `centroid_source`：丢掉它会让
        `find_object` 报 NOT_FOUND，模型会以为图里没这个东西 —— 那比定位不准更糟。
        """
        dets, masks, cloud = two_chairs()
        masks = masks.copy()
        masks[0] = False
        masks[0, 30, 30:33] = True              # 只剩 3 个点，低于 min_points=30
        res = build(FakePerception(dets, masks, cloud))

        assert res.scene.ids() == ["chair_1", "chair_2"]
        n1 = res.scene.node("chair_1")
        assert n1.centroid_source == "bbox_fallback"
        assert res.scene.build_meta["n_fallbacks"] == 1
        assert res.scene.build_meta["fallbacks"][0]["reason"] == "mask_too_small"
        assert any("退回检测框" in w for w in res.warnings)
        # 框内中位数依然给出一个可用坐标（框是 20..60 × 20..100）
        assert n1.centroid_3d[0] < 0 and n1.centroid_3d[2] == pytest.approx(2.0)

    def test_node_dropped_when_no_valid_points(self):
        """掩码和框里都没有有效点云 → 剔除该检测，且**不编造坐标**。

        `Node` 的核心契约是「质心由几何算出」。凭空造一个会让整条证据链失效 ——
        宁可少一个物体，也不能有一个坐标来源不明的物体。
        """
        dets, masks, _ = two_chairs()
        nan_cloud = np.full((3, H, W), np.nan)
        res = build(FakePerception(dets, masks, nan_cloud))

        assert res.scene.nodes == ()
        assert res.scene.build_meta["n_dropped"] == 2
        reasons = {d["reason"] for d in res.scene.build_meta["dropped"]}
        assert reasons == {"no_valid_points"}
        assert any("无有效点云" in w for w in res.warnings)

    def test_nan_pixels_inside_mask_are_ignored(self):
        """掩码内的 NaN/负深度点必须被剔除，而不是把中位数拖飞。"""
        dets, masks, cloud = two_chairs()
        cloud = cloud.copy()
        cloud[:, masks[0]] = np.nan             # 左椅整个变 NaN
        res = build(FakePerception(dets, masks, cloud))
        # 左椅 mask 全 NaN → 降级到框；框也是同一片区域 → 同样 NaN → 应被剔除
        assert res.scene.ids() == ["chair_2"]
        assert res.scene.build_meta["n_dropped"] == 1

    def test_dedupe_drops_overlapping_detection(self):
        """两个词命中几乎同一块区域时，只保留高置信度那个。

        保留两个会让「有几个桌子」这类计数题直接算错，还会在关系图里
        造出一条近乎 self-loop 的假边。**跨类别也去重**正是为了这个。
        """
        cloud = pinhole_cloud(np.full((H, W), 2.0))
        masks = np.zeros((2, H, W), dtype=bool)
        masks[0, 20:100, 20:60] = True
        masks[1, 20:100, 20:60] = True
        dets = [
            Detection("table", 0.90, (20.0, 20.0, 60.0, 100.0)),
            Detection("desk", 0.70, (21.0, 21.0, 61.0, 101.0)),   # IoU > 0.85
        ]
        res = build(FakePerception(dets, masks, cloud))
        assert res.scene.ids() == ["table_1"]
        assert res.scene.build_meta["n_dropped"] == 1
        assert res.scene.build_meta["dropped"][0]["reason"] == "duplicate_box"
        assert res.scene.build_meta["dropped"][0]["duplicate_of"] == "table"

    def test_tiny_boxes_dropped(self):
        cloud = pinhole_cloud(np.full((H, W), 2.0))
        dets = [Detection("dust", 0.9, (10.0, 10.0, 12.0, 12.0))]   # 4 px²
        res = build(FakePerception(dets, np.zeros((1, H, W), dtype=bool), cloud))
        assert res.scene.nodes == ()
        assert res.scene.build_meta["dropped"][0]["reason"] == "box_too_small"

    def test_max_objects_truncates(self):
        cloud = pinhole_cloud(np.full((H, W), 2.0))
        masks = np.zeros((2, H, W), dtype=bool)
        dets = []
        for i in range(2):
            c0 = 10 + i * 80
            masks[i, 20:100, c0:c0 + 40] = True
            dets.append(Detection(f"o{i}", 0.9 - i * 0.1, (c0, 20, c0 + 40, 100)))
        res = build(FakePerception(dets, masks, cloud), config=BuildConfig(max_objects=1))
        assert res.scene.ids() == ["o0_1"]
        assert res.scene.build_meta["dropped"][0]["reason"] == "over_max_objects"


# ----------------------------------------------------------------------------
# 关系生成策略与重力方向
# ----------------------------------------------------------------------------


class TestRelationsAndUp:
    def test_distance_only_policy(self):
        dets, masks, cloud = two_chairs()
        res = build(
            FakePerception(dets, masks, cloud),
            config=BuildConfig(relation_policy="distance_only"),
        )
        assert {e.relation for e in res.scene.edges} == {"distance"}

    def test_far_never_materialized(self):
        """`far` 是 `near` 的补集，两条都存等于把信息翻倍而信息量不变。"""
        dets, masks, cloud = two_chairs()
        res = build(FakePerception(dets, masks, cloud))
        assert "far" not in {e.relation for e in res.scene.edges}

    def test_all_policy_includes_false_booleans(self):
        dets, masks, cloud = two_chairs()
        res = build(
            FakePerception(dets, masks, cloud),
            config=BuildConfig(relation_policy="all"),
        )
        rels = {e.relation for e in res.scene.edges}
        assert {"left_of", "right_of", "above", "below", "on", "inside"} <= rels

    def test_unreliable_up_axis_is_surfaced_not_hidden(self):
        """常量深度 → 画面下沿是正对相机的竖直面，不是地板。

        此时必须判为不可靠并报警告。硬用一个错的 up 会**整体翻转** above/below，
        而这是报告里要单独讨论的一环（§12.3 步骤 6）。
        """
        dets, masks, cloud = two_chairs()
        res = build(FakePerception(dets, masks, cloud))
        meta = res.scene.build_meta
        assert meta["up_axis_reliable"] is False
        assert meta["up_axis_reason"] == "band_not_horizontal"
        assert meta["up_axis"] == "-y"            # 退化时仍给一个可用默认值
        assert any("重力方向" in w for w in res.warnings)

    def test_estimate_up_can_be_disabled(self):
        dets, masks, cloud = two_chairs()
        res = build(
            FakePerception(dets, masks, cloud),
            config=BuildConfig(estimate_up=False),
        )
        assert res.scene.build_meta["up_axis_reason"] == "not_estimated"
        assert not any("重力方向" in w for w in res.warnings)


# ----------------------------------------------------------------------------
# 可复现性
# ----------------------------------------------------------------------------


class TestReproducibility:
    def test_two_builds_are_identical(self):
        """同图 + 同权重 + 同 prompt ⟹ 同 id、同质心、同边集。

        早期基线做不到这一条：它每次运行都随机抽样生成工具集（且无种子），
        于是工具集每次不同、结果不可比。
        """
        out = []
        for _ in range(2):
            dets, masks, cloud = two_chairs()
            out.append(build(FakePerception(dets, masks, cloud)).scene)
        assert out[0].ids() == out[1].ids()
        assert out[0].nodes == out[1].nodes
        assert out[0].edges == out[1].edges

    def test_slugify(self):
        assert slugify("Coffee Table") == "coffee_table"
        assert slugify("  TV  ") == "tv"
        assert slugify("壁挂式空调") == "object"      # 空 slug 退化，不产出 "_1"
        assert slugify("a-b/c") == "a_b_c"


# ----------------------------------------------------------------------------
# 内参来源：横向尺度的总开关（Phase 0 Step 6）
# ----------------------------------------------------------------------------


class TestIntrinsics:
    """`known_intrinsics` 的透传与视场体检。

    这是一组**回归测试**，对象是 Phase 0 Step 6 那个发现：
    同一个模型、同一张图，让模型猜内参与把内参告诉它，三维误差中位数是
    1.943 m 与 0.267 m。如果哪天有人把这个参数「顺手简化掉」，
    或者把警告删成注释，这些用例会失败。
    """

    HONEST = (200.0, 200.0, 99.5, 79.5)      # 与 `pinhole_cloud` 同一套
    ABSURD = (1.0, 1.0, 0.0, 0.0)            # fx=1 ⟹ 视场 179°，模型相机头的极端形态

    def test_known_intrinsics_is_forwarded_as_matrix(self):
        """给了 `known_intrinsics`，必须作为 3×3 矩阵送到感知层。"""
        dets, masks, cloud = two_chairs()
        fake = FakePerception(dets, masks, cloud)
        build(fake, config=BuildConfig(known_intrinsics=self.HONEST))
        assert len(fake.lift_calls) == 1
        K = fake.lift_calls[0]
        assert K is not None
        assert np.asarray(K).shape == (3, 3)
        assert float(np.asarray(K)[0, 0]) == pytest.approx(self.HONEST[0])
        assert float(np.asarray(K)[1, 2]) == pytest.approx(self.HONEST[3])

    def test_no_known_intrinsics_forwards_none(self):
        """没给就传 `None`，不能凭空造一个 K —— 那等于假装标定过。"""
        dets, masks, cloud = two_chairs()
        fake = FakePerception(dets, masks, cloud)
        build(fake)
        assert fake.lift_calls == [None]

    def test_meta_records_source_and_fov(self):
        dets, masks, cloud = two_chairs()
        res = build(FakePerception(dets, masks, cloud))
        meta = res.scene.build_meta
        assert meta["intrinsics_source"] == "predicted"
        assert meta["fov"]["plausible"] is True
        assert meta["fov"]["reason"] == "ok"
        # fx=fy=200、宽 200 ⟹ HFoV = 2·atan(0.5) = 53.13°
        assert meta["fov"]["hfov_deg"] == pytest.approx(53.13, abs=0.01)
        assert meta["scale_calibrated"] is False

    def test_implausible_predicted_fov_warns(self):
        """预测内参 + 视场不可信 ⟹ 必须出现警告，且警告要指向根因。"""
        dets, masks, cloud = two_chairs()
        res = build(FakePerception(dets, masks, cloud,
                                   intrinsics=intrinsics_matrix(*self.ABSURD)))
        meta = res.scene.build_meta
        assert meta["intrinsics_source"] == "predicted"
        assert meta["fov"]["plausible"] is False
        assert meta["fov"]["reason"] == "hfov_out_of_range"
        assert any("视场不可信" in w for w in res.warnings)
        # 警告里必须出现「怎么修」——只报错不给出口的警告会被忽略
        assert any("known_intrinsics" in w for w in res.warnings)

    def test_plausible_predicted_fov_does_not_warn(self):
        """可信区间内的预测值不该报警告，否则这个检查会变成噪声。"""
        dets, masks, cloud = two_chairs()
        res = build(FakePerception(dets, masks, cloud))
        assert not any("视场不可信" in w for w in res.warnings)

    def test_provided_but_implausible_warns_differently(self):
        """来源是 provided 却依然不可信 ⟹ 也不能沉默（多半是分辨率不匹配）。"""
        dets, masks, cloud = two_chairs()
        res = build(
            FakePerception(dets, masks, cloud,
                           intrinsics=intrinsics_matrix(*self.ABSURD),
                           intrinsics_source="provided"),
            config=BuildConfig(known_intrinsics=self.ABSURD),
        )
        warnings = res.warnings
        assert any("传入的已知内参视场不可信" in w for w in warnings)
        # 不能错误地说成「模型预测的」
        assert not any("内参是模型预测的" in w for w in warnings)

    def test_config_records_and_replaces_known_intrinsics(self):
        cfg = BuildConfig(known_intrinsics=self.HONEST)
        assert cfg.as_dict()["known_intrinsics"] == [200.0, 200.0, 99.5, 79.5]
        assert cfg.replace(prompt="cat.").known_intrinsics == self.HONEST
        assert BuildConfig().as_dict()["known_intrinsics"] is None
