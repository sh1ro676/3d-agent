"""`scene_graph/store.py` 的单元测试 —— 落盘格式的回归测试。

格式是会腐烂的东西：改了 schema 却忘了改读写，症状往往是「实验结果读不出来」，
而那批结果可能已经跑了一晚上。所以信封、版本号、向后兼容这三件事都要有断言。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_graph.schema import BBox3D, Edge, Node, SceneGraph  # noqa: E402
from scene_graph.store import (  # noqa: E402
    load_mask,
    load_scene,
    masks_dir,
    read_format_version,
    save_masks,
    save_scene,
    scene_dir,
)
from tools.version import SCENE_FORMAT_VERSION  # noqa: E402


def mk_scene() -> SceneGraph:
    a = Node(
        id="chair_1", label="chair", score=0.9,
        bbox_2d=(20.0, 20.0, 60.0, 100.0),
        centroid_3d=(-0.6, -0.2, 2.0), extent_3d=(0.39, 0.79, 0.0),
        bbox_3d=BBox3D(min=(-0.795, -0.595, 2.0), max=(-0.405, 0.195, 2.0)),
        n_points=3200, centroid_source="mask",
    )
    b = Node(
        id="chair_2", label="chair", score=0.8,
        centroid_3d=(0.6, -0.2, 2.0), n_points=3100,
        centroid_source="bbox_fallback",
    )
    e = Edge(source="chair_1", target="chair_2", relation="distance",
             value=1.2, metric={"delta_x": -1.2})
    return SceneGraph(
        scene_id="t", image_id="t", up_axis="-y",
        nodes=(a, b), edges=(e,),
        build_meta={"n_nodes": 2, "up_axis_reliable": False},
    )


class TestSceneRoundtrip:
    def test_save_load_preserves_everything(self, tmp_path: Path):
        scene = mk_scene()
        p = save_scene(scene, tmp_path / "scene.json")
        assert p.is_file()
        back = load_scene(p)
        assert back.nodes == scene.nodes
        assert back.edges == scene.edges
        assert back.build_meta == scene.build_meta
        assert back.up_axis == "-y"
        # 降级来源必须活过一轮序列化 —— 否则 L5 的失败诊断会瞎
        assert back.node("chair_2").centroid_source == "bbox_fallback"

    def test_envelope_carries_format_version(self, tmp_path: Path):
        p = save_scene(mk_scene(), tmp_path / "scene.json")
        raw = json.loads(p.read_text(encoding="utf-8"))
        assert raw["_format"] == SCENE_FORMAT_VERSION
        assert "scene" in raw and "_written_at" in raw
        assert read_format_version(p) == SCENE_FORMAT_VERSION

    def test_accepts_directory_and_creates_parents(self, tmp_path: Path):
        d = tmp_path / "a" / "b"
        p = save_scene(mk_scene(), d)
        assert p.name == "scene.json" and p.parent == d
        assert load_scene(d).scene_id == "t"

    def test_format_mismatch_is_flagged_not_rejected(self, tmp_path: Path):
        """版本不同时**照常加载**，只在 build_meta 里记一笔。

        直接拒绝会让旧实验数据全废 —— 而旧数据正是做对照实验时最不能丢的。
        向后兼容的格式演进（新增可选字段）远多于破坏性变更。
        """
        p = save_scene(mk_scene(), tmp_path / "scene.json")
        raw = json.loads(p.read_text(encoding="utf-8"))
        raw["_format"] = "0.1"
        p.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

        back = load_scene(p)
        assert back.build_meta["_format_mismatch"] is True
        assert back.build_meta["_loaded_format"] == "0.1"
        assert back.build_meta["_current_format"] == SCENE_FORMAT_VERSION
        assert back.build_meta["n_nodes"] == 2      # 原有内容没丢

    def test_bare_scene_without_envelope_still_loads(self, tmp_path: Path):
        """裸 SceneGraph（无信封）也要能读 —— 兼容手工导出的历史文件。"""
        p = tmp_path / "bare.json"
        p.write_text(
            json.dumps(mk_scene().model_dump(mode="json"), ensure_ascii=False),
            encoding="utf-8",
        )
        assert read_format_version(p) is None
        assert load_scene(p).ids() == ["chair_1", "chair_2"]

    def test_path_helpers(self):
        assert scene_dir("x", root="R") == Path("R") / "x"
        assert masks_dir("x", root="R") == Path("R") / "x" / "masks"


class TestMasks:
    def test_roundtrip_is_bit_exact(self, tmp_path: Path):
        masks = {
            "chair_1": np.zeros((16, 20), dtype=bool),
            "chair_2": np.ones((16, 20), dtype=bool),
        }
        masks["chair_1"][3:7, 5:9] = True

        written = save_masks(masks, masks_dir("t", root=tmp_path))
        assert set(written) == {"chair_1", "chair_2"}
        for oid, m in masks.items():
            back = load_mask(written[oid])
            assert back.shape == m.shape
            assert back.dtype == bool
            assert np.array_equal(back, m)

    def test_png_is_smaller_than_npy(self, tmp_path: Path):
        """1-bit PNG 而不是 .npy —— 体积是选它的原因（稀疏掩码压缩率极高）。

        640×480 的 .npy 恒为 307 KB；同样的掩码存 PNG 通常只有几十 KB。
        20 个物体就是 6 MB vs 1 MB 的差别，而且 PNG 能直接用看图软件打开 ——
        排查「这个掩码分错了吗」的时候，这一条比体积更值钱。
        """
        big = np.zeros((480, 640), dtype=bool)
        big[100:300, 200:400] = True
        p = save_masks({"o": big}, tmp_path / "m")["o"]
        assert p.stat().st_size < big.size // 10        # 远小于 1 byte/像素
        assert np.array_equal(load_mask(p), big)
