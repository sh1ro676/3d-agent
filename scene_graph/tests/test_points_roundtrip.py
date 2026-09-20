"""整图点云的落盘与取回 —— 「落盘的确实是当初那一份」的回归测试。

**本文件守的不变量只有一条，但很锋利**：用落盘的点云重算出来的三维量，
必须与 `scene.json` 里记的**精确相等**（`==`，不是 `approx`）。

为什么值得专门守它：如果落盘的点云与建图时用的不是同一份（掩码读错、
重采样规则变了、网格尺寸记错、默认存了个更窄的 dtype），症状是**一声不吭** ——
只是每个点云级工具算出的质心都偏一点点，而那个数字看起来完全正常。
两个「都合理」的数字之间的差异，是最难查的一类 bug。

写成 `==` 而不是「差不多」，是因为**同一份数据走同一条代码路径本就不该有
任何差异**。一旦放成 `approx`，就再也发现不了「这份点云被悄悄改了 dtype」
这类问题 —— 而那恰好是默认值最容易犯的错（本文件的
`test_keep_dtype_is_a_noop` 就是被这个想法逼出来的）。

**零 GPU**：借 `test_builder.py` 的 fake 感知栈跑**真 builder**，
所以这里测的是真实链路（含掩码重采样与取点顺序），不是手搓的近似。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_graph.builder import build_scene_graph  # noqa: E402
from scene_graph.pointcloud import node_points, points_in_mask  # noqa: E402
from scene_graph.schema import Node  # noqa: E402
from scene_graph.store import (  # noqa: E402
    POINTS_FORMAT_VERSION,
    has_points,
    load_points,
    save_masks,
    save_points,
)
from scene_graph.tests.test_builder import (  # noqa: E402
    FakePerception,
    make_image,
    two_chairs,
)
from vision.geometry import centroid_of, resample_mask_to, robust_extent, select_points  # noqa: E402

#: 场景目录名。`mask_ref` 形如 `<scene_id>/masks/<id>.png`，相对
#: `dataset/scenes/`；测试里把它摆在 `tmp_path/<scene_id>/`，于是
#: `_mask_of()` 的 `scene_dir.parent / mask_ref` 正好解析得到。
SCENE_ID = "t"


def build_two_chairs(**kw):
    """跑一次**真** builder，返回 `BuildResult`。

    `mask_rel_prefix` 必须传 —— 不传的话每个节点的 `mask_ref` 是 None，
    重建时就找不到掩码，而这恰恰是「落盘与重建」这条链路上最容易漏的一环：
    建图本身完全正常，只有下游取不到点。
    """
    dets, masks, cloud = two_chairs()
    return build_scene_graph(
        make_image(),
        perception=FakePerception(dets, masks, cloud),
        scene_id=SCENE_ID,
        mask_rel_prefix=f"{SCENE_ID}/masks",
        **kw,
    )


def persist(res, root: Path, *, dtype: str | None = None) -> Path:
    """复刻 `scripts/build_scene.py` 的落盘两步，返回场景目录。

    ⚠️ 这里用 `dtype=None`（不转换），**与 CLI 的默认 float32 不同**，
    这是有意的**隔离变量**：本夹具的点云是 `pinhole_cloud` 算出来的**真**
    float64，压成 float32 会量化（实测质心从 `-0.6` 变成 `-0.6000000238…`）。
    那会把「重建路径与建图路径是否同源」这个待测问题，和「精度变了」
    这个额外变量混在一起，失败时分不清是谁的锅。

    真实点云不受这条影响：它落盘的 float64 是**假精度**
    （`scripts/verify_points.py` 配套实测：float32 往返 `max diff = 0`），
    所以 CLI 默认 float32 对真实数据是无损的。两条事实不矛盾 ——
    **一条说的是数据的性质，另一条说的是默认值的适用范围。**
    """
    sd = root / SCENE_ID
    save_masks(res.masks, sd / "masks")
    save_points(res.points_chw, sd, meta=res.points_meta, dtype=dtype)
    return sd


def mk_cloud(h: int, w: int) -> np.ndarray:
    """一个处处可辨认的点云：x = 列号、y = 行号、z = 1。

    值可预测，断言才能写成精确等式，而不是「与另一个重算结果差不多」——
    后者在实现本身错的时候会一起错。
    """
    xs = np.broadcast_to(np.arange(w, dtype=np.float64), (h, w))
    ys = np.broadcast_to(np.arange(h, dtype=np.float64)[:, None], (h, w))
    return np.stack([xs, ys, np.ones((h, w))])


# ----------------------------------------------------------------------------
# store：落盘契约
# ----------------------------------------------------------------------------


class TestStorePoints:
    def test_roundtrip_preserves_array_and_meta(self, tmp_path: Path):
        cloud = mk_cloud(6, 8)
        p = save_points(
            cloud, tmp_path,
            meta={"grid_hw": [6, 8], "image_hw": [6, 8], "intrinsics_source": "predicted"},
        )
        assert p.name == "points.npy" and p.is_file()

        back, meta = load_points(tmp_path)
        assert np.array_equal(back, cloud)
        assert meta["grid_hw"] == [6, 8]
        assert meta["intrinsics_source"] == "predicted"
        assert meta["shape"] == [3, 6, 8]
        assert meta["_format"] == POINTS_FORMAT_VERSION
        assert "_written_at" in meta

    def test_default_precision_is_float32(self, tmp_path: Path):
        """默认 float32 —— 依据见 `store.DEFAULT_POINTS_DTYPE`（实测逐位无损）。

        `mk_cloud` 的值是小整数，float32 能精确表示，所以这里连**值**都要求
        逐位不变。真实点云同样满足（实测 `max |a - f32(a)| = 0`），
        区别只是真实数据需要那次实测来背书，而测试夹具能自证。
        """
        cloud = mk_cloud(4, 4).astype("float64")
        save_points(cloud, tmp_path, meta={"grid_hw": [4, 4]})
        back, meta = load_points(tmp_path)
        assert back.dtype == np.dtype("float32")
        assert meta["dtype"] == "float32"
        assert np.array_equal(back, cloud)

    def test_keep_dtype_is_opt_in(self, tmp_path: Path):
        """要「一个比特都不动」必须**显式**传 `dtype=None`。

        留着这个档位，是因为「当前数据是假精度」这个判断依据的是一次实测，
        而实测有适用范围。需要严格原样保存时，它必须是可表达的 ——
        否则默认值就从「一个决定」变成了「一个无法绕过的约束」。
        """
        cloud = mk_cloud(4, 4).astype("float64")
        save_points(cloud, tmp_path, meta={"grid_hw": [4, 4]}, dtype=None)
        back, meta = load_points(tmp_path)
        assert back.dtype == np.dtype("float64")
        assert meta["dtype"] == "float64"
        assert np.array_equal(back, cloud)

    def test_explicit_float16_is_recorded_not_silent(self, tmp_path: Path):
        """显式压缩要能被**读出来** —— meta 里必须记着实际存的是什么精度。"""
        save_points(mk_cloud(4, 4), tmp_path,
                    meta={"grid_hw": [4, 4]}, dtype="float16")
        back, meta = load_points(tmp_path)
        assert back.dtype == np.dtype("float16")
        assert meta["dtype"] == "float16"

    def test_rejects_wrong_shape(self, tmp_path: Path):
        with pytest.raises(ValueError, match=r"\(3, H, W\)"):
            save_points(np.zeros((6, 8)), tmp_path, meta={"grid_hw": [6, 8]})
        with pytest.raises(ValueError, match=r"\(3, H, W\)"):
            save_points(np.zeros((4, 6, 8)), tmp_path, meta={"grid_hw": [6, 8]})

    def test_rejects_meta_without_grid_hw(self, tmp_path: Path):
        """meta 缺 `grid_hw` 必须在**写入时**失败。

        没有网格尺寸的点云无法用于重建（掩码重采样需要目标网格）。
        如果这里放行，错误会推迟到某个工具取点时爆发 —— 那时算错的坐标
        已经混进结果里了。
        """
        with pytest.raises(ValueError, match="grid_hw"):
            save_points(mk_cloud(4, 4), tmp_path, meta={"image_hw": [4, 4]})
        with pytest.raises(ValueError, match="grid_hw"):
            save_points(mk_cloud(4, 4), tmp_path)          # 连 meta 都没给

    def test_missing_file_raises_with_actionable_hint(self, tmp_path: Path):
        """缺文件必须抛错，**不能返回 None**。

        「这个场景建于点云落盘之前」与「点云存在但是空的」需要完全不同的
        处理（重跑建图 vs 查数据）。返回 None 会把两者压进同一个分支，
        而由此给出的「点云为空」诊断是**错的**，会把人支到错误的方向。
        """
        with pytest.raises(FileNotFoundError) as ei:
            load_points(tmp_path / "nope")
        assert "build_scene" in str(ei.value)      # 提示要指向可执行的下一步

    def test_has_points_accepts_dir_and_file(self, tmp_path: Path):
        assert not has_points(tmp_path)
        p = save_points(mk_cloud(4, 4), tmp_path, meta={"grid_hw": [4, 4]})
        assert has_points(tmp_path)
        assert has_points(p)


# ----------------------------------------------------------------------------
# pointcloud：取点
# ----------------------------------------------------------------------------


class TestPointsInMask:
    def test_empty_mask_is_empty_not_an_error(self):
        """没有掩码 → `(3, 0)`，不抛错。

        「这题不需要点云」与「掩码丢了」在调用方那里落到同一个出口，
        由它决定是报 DEGENERATE 还是换 anchor —— 不在这里替它决定。
        """
        cloud = mk_cloud(4, 4)
        assert points_in_mask(cloud, None, (4, 4)).shape == (3, 0)
        assert points_in_mask(cloud, np.zeros((0, 0), dtype=bool), (4, 4)).shape == (3, 0)

    def test_matches_builder_recipe_exactly(self):
        """必须是「重采样 + 取点」这两步，**不加任何自己的规则**。"""
        cloud = mk_cloud(8, 10)
        mask = np.zeros((8, 10), dtype=bool)
        mask[2:5, 3:7] = True
        want = select_points(cloud, resample_mask_to(mask, (8, 10)))
        assert np.array_equal(points_in_mask(cloud, mask, (8, 10)), want)

    def test_resamples_when_grid_differs_from_image(self):
        """掩码在图像分辨率、点云在另一个网格时必须**显式重采样**。

        这里掩码是 4×4、点云是 2×2：最近邻按像素中心取样，只有
        `mask[1,1]` 落在唯一的网格单元上 —— 于是恰好选出 1 个点。
        如果实现假设两者同尺寸，会直接抛 shape 不匹配或选出 4 个点。
        """
        cloud = mk_cloud(2, 2)
        mask = np.zeros((4, 4), dtype=bool)
        mask[0:2, 0:2] = True
        assert points_in_mask(cloud, mask, (2, 2)).shape[1] == 1


# ----------------------------------------------------------------------------
# 端到端：建图 → 落盘 → 重建
# ----------------------------------------------------------------------------


class TestRebuildRoundTrip:
    def test_every_node_reproduces_exactly(self, tmp_path: Path):
        """★ 本文件的核心断言。

        同一条不变量同时管住三件事：落盘 dtype 没被改窄、掩码重采样用的
        是同一套规则、取点顺序没变。任何一件出错，点数或质心就对不上。
        """
        res = build_two_chairs()
        assert res.points_chw is not None
        assert res.points_meta["grid_hw"] == list(res.points_chw.shape[1:])

        sd = persist(res, tmp_path)

        for node in res.scene.nodes:
            pts = node_points(node, sd)
            assert pts.shape[1] == node.n_points, f"{node.id} 点数不一致"

            centroid, _ = centroid_of(pts)
            assert tuple(float(c) for c in centroid) == node.centroid_3d, \
                f"{node.id} 质心不一致"

            extent, _, _, _ = robust_extent(pts)
            assert tuple(float(e) for e in extent) == node.extent_3d, \
                f"{node.id} 尺寸不一致"

    def test_default_float32_quantizes_true_float64(self, tmp_path: Path):
        """把这条差异显式记下来，免得日后有人以为「默认 float32 永远无损」。

        真实点云（UniDepth 出来的）落成 float64 是**假精度**，所以 float32 无损；
        但这不是普遍规律 —— 任何**真** float64 数据压成 float32 都会量化。

        误差是 2.4e-8 m，对 50 mm 的判定容差毫无影响；这条测试的价值不在
        数字，而在于它钉住了「默认 float32 无损」这句话的**适用范围**
        —— 那是实测过的那类数据，不是所有数据。
        """
        res = build_two_chairs()
        sd = persist(res, tmp_path, dtype="float32")
        node = res.scene.node("chair_1")
        pts = node_points(node, sd)
        centroid, _ = centroid_of(pts)

        assert pts.shape[1] == node.n_points                     # 点数不受影响
        assert tuple(float(c) for c in centroid) != node.centroid_3d   # 但质心被量化
        assert abs(float(centroid[0]) - node.centroid_3d[0]) < 1e-6    # 差异在微米以下

    def test_reading_meta_once_then_reusing_gives_same_result(self, tmp_path: Path):
        """批量算多个物体时先读一次、再反复传进来 —— 结果必须与逐个现读一致。

        这条防的是"缓存版本与现读版本走了两条不同代码路径"这类漂移：
        两条路都能跑通，却在某个边界上给出不同答案。
        """
        res = build_two_chairs()
        sd = persist(res, tmp_path)
        cloud, meta = load_points(sd)

        for node in res.scene.nodes:
            assert np.array_equal(
                node_points(node, sd, points_chw=cloud, meta=meta),
                node_points(node, sd),
            )

    def test_bbox_fallback_uses_box_not_mask(self, tmp_path: Path):
        """降级节点按**检测框**取点，绝不回落到掩码。

        这不是「两条等价路径」：builder 专门记 `centroid_source`，正是因为
        二选一会改变质心（实测均值差 83 mm）。自动兜底等于把一条已经记录
        在案的差异重新变成随机事件。

        夹具刻意让框与掩码**不重叠**，于是走错路会得到明显不同的点数。
        """
        h, w = 8, 10
        save_points(mk_cloud(h, w), tmp_path, meta={"grid_hw": [h, w], "image_hw": [h, w]})
        mask = np.zeros((h, w), dtype=bool)
        mask[0:2, 8:10] = True                       # 4 个点（2×2），与框完全不相交
        save_masks({"o": mask}, tmp_path / "masks")

        node = Node(
            id="o", label="thing", centroid_3d=(0.0, 0.0, 1.0),
            bbox_2d=(0.0, 0.0, 4.0, 8.0),
            centroid_source="bbox_fallback",
        )
        pts = node_points(node, tmp_path)
        assert pts.shape[1] == 4 * 8                 # 来自框（列 0:4 × 行 0:8）
        assert set(np.unique(pts[0])) == {0.0, 1.0, 2.0, 3.0}

    def test_mask_path_ignores_box_when_source_is_mask(self, tmp_path: Path):
        """反向也要成立：`centroid_source="mask"` 时不许去用框。"""
        h, w = 8, 10
        save_points(mk_cloud(h, w), tmp_path, meta={"grid_hw": [h, w], "image_hw": [h, w]})
        mask = np.zeros((h, w), dtype=bool)
        mask[0:2, 8:10] = True
        save_masks({"o": mask}, tmp_path / "masks")

        node = Node(
            id="o", label="thing", centroid_3d=(0.0, 0.0, 1.0),
            bbox_2d=(0.0, 0.0, 4.0, 8.0),            # 给了框，但来源是 mask
            centroid_source="mask",
        )
        # 掩码只有 4 个点（2×2），而框有 32 个 —— 走错路会立刻得到 32
        assert node_points(node, tmp_path).shape[1] == 4

    def test_missing_grid_hw_in_meta_raises(self, tmp_path: Path):
        """meta 里的 `grid_hw` 缺失或被改坏时必须炸。

        猜一个网格尺寸会让取出来的点整体错位 —— 而错位后的结果
        看起来完全正常，这是它比"直接报错"糟得多的原因。
        """
        save_points(mk_cloud(4, 4), tmp_path, meta={"grid_hw": [4, 4]})
        (tmp_path / "points_meta.json").write_text('{"shape": [3, 4, 4]}', encoding="utf-8")

        node = Node(id="o", label="thing", centroid_3d=(0.0, 0.0, 1.0),
                    centroid_source="mask")
        with pytest.raises(ValueError, match="grid_hw"):
            node_points(node, tmp_path)

    def test_missing_mask_yields_empty_not_exception(self, tmp_path: Path):
        """掩码文件丢失 → 空点集，而不是异常。调用方据此走自己的降级。"""
        res = build_two_chairs()
        sd = persist(res, tmp_path)
        for p in (sd / "masks").glob("*.png"):
            p.unlink()

        for node in res.scene.nodes:
            assert node_points(node, sd).shape == (3, 0)
