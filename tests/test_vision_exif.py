"""`vision/exif.py` 的单元测试 —— 零 GPU、零模型、零网络、零真实照片。

这组测试要钉住的是 EXIF 这条路**唯一挡在「真实照片可测」前面的环节**。
Step 6/7 已经证明「有没有内参」会让横向误差差 118 倍；如果 EXIF 这条路上
有一个静默的换算错误，那 118 倍的收益会被一个 1.5× 或 6.4× 的错误抵消掉，
而输出的 K 看上去仍然是一组合理数字 —— 这正是本项目反复遇到的那类 bug。

所以测试的重点不是「能读出一个数」，而是**四条容易静默出错的性质**：

    ① 长边约定：竖构图必须用 H，否则 fx 按宽高比整体缩放
    ② 来源区分：只有 FocalLength 时必须是「假设传感器」，不能冒充真值
    ③ 尺寸不一致：EXIF 记录的像素尺寸与当前不符时必须报告
    ④ 坏标签退化：0 / "abc" / 缺失 都应当返回 None 或忽略，而不是抛

另外用 `_lookup` 的桩对象测试 Exif SubIFD 那条分支 —— 真机文件里
`FocalLengthIn35mmFilm` 多数住在 SubIFD，只测 IFD0 会漏掉主路径。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.exif import (  # noqa: E402
    SENSOR_WIDTH_MM_35MM,
    ExifIntrinsics,
    read_exif_intrinsics,
)

TAG_FOCAL_LENGTH = 0x920A
TAG_FOCAL_35MM = 0xA405
TAG_PIXEL_X = 0xA002
TAG_PIXEL_Y = 0xA003
EXIF_IFD = 0x8769


def _write_jpeg(
    path: Path,
    size: tuple[int, int] = (4000, 3000),
    *,
    focal_35: object = None,
    focal_mm: object = None,
    pixel_xy: tuple[int, int] | None = None,
    nested: bool = False,
) -> Path:
    """写一张带 EXIF 的 JPEG。

    `nested=True` 时把标签塞进 Exif SubIFD（真机的常见位置）；Pillow 的写法是
    先填进顶层再整体序列化，两种位置我们都要能读。
    """
    from PIL import Image

    im = Image.new("RGB", size, (13, 17, 23))
    exif = im.getexif()
    target = exif
    if nested:
        sub = exif.get_ifd(EXIF_IFD)
        sub.clear()
        target = sub
    if focal_35 is not None:
        target[TAG_FOCAL_35MM] = focal_35
    if focal_mm is not None:
        target[TAG_FOCAL_LENGTH] = focal_mm
    if pixel_xy is not None:
        target[TAG_PIXEL_X] = pixel_xy[0]
        target[TAG_PIXEL_Y] = pixel_xy[1]
    if nested:
        exif[EXIF_IFD] = sub
    im.save(path, exif=exif)
    return path


# ---------------------------------------------------------------------------
# ① 长边约定
# ---------------------------------------------------------------------------

def test_focal_35mm_converts_with_long_side(tmp_path: Path):
    """f_px = f_35 / 36 × 长边。4000×3000 时长边 4000。"""
    p = _write_jpeg(tmp_path / "a.jpg", (4000, 3000), focal_35=26)
    got = read_exif_intrinsics(p)

    assert got is not None
    assert got.source == "exif:35mm"
    assert got.focal_35mm_mm == pytest.approx(26)
    assert got.fx == pytest.approx(26 / 36 * 4000)
    assert got.fy == pytest.approx(got.fx)          # 方形像素
    assert got.cx == pytest.approx(2000.0)
    assert got.cy == pytest.approx(1500.0)
    assert got.image_hw == (3000, 4000)
    assert not got.assumed_sensor


def test_portrait_uses_height_as_long_side(tmp_path: Path):
    """竖构图（3000×4000）时长边是 H=4000 ⟹ fx 与横构图完全相同。

    这是本模块「不按 Orientation 换宽高」的设计所依赖的性质。若误用 W，
    竖构图的 fx 会缩到 3/4，横向尺度整体放大 1.33 倍 —— 而且看不出来。
    """
    landscape = read_exif_intrinsics(
        _write_jpeg(tmp_path / "l.jpg", (4000, 3000), focal_35=26))
    portrait = read_exif_intrinsics(
        _write_jpeg(tmp_path / "p.jpg", (3000, 4000), focal_35=26))

    assert landscape is not None and portrait is not None
    assert portrait.fx == pytest.approx(landscape.fx)
    # 但主点各按自己的 (W, H) 取中心
    assert portrait.cx == pytest.approx(1500.0)
    assert portrait.cy == pytest.approx(2000.0)
    assert any("竖构图" in n for n in portrait.notes)


def test_image_size_override_uses_given_pixels(tmp_path: Path):
    """图被下游改过尺寸时，f_px 必须按**实际推理用的**像素数算。

    缩放不改变视场角，所以等效焦距仍然有效 —— 但前提是公式里用的是当前长边。
    这里把尺寸减半，fx 应当正好减半（视场不变）。
    """
    p = _write_jpeg(tmp_path / "a.jpg", (4000, 3000), focal_35=26)
    full = read_exif_intrinsics(p)
    half = read_exif_intrinsics(p, image_size=(1500, 2000))

    assert full is not None and half is not None
    assert half.fx == pytest.approx(full.fx / 2)
    assert half.cx == pytest.approx(1000.0)
    assert half.cy == pytest.approx(750.0)
    # 视场角不变 —— 这才是「缩放不影响等效焦距」的可检验含义
    assert half.fov.hfov_deg == pytest.approx(full.fov.hfov_deg, abs=1e-9)


# ---------------------------------------------------------------------------
# ② 来源区分：不能把假设冒充真值
# ---------------------------------------------------------------------------

def test_focal_length_only_is_marked_as_assumed(tmp_path: Path):
    p = _write_jpeg(tmp_path / "b.jpg", (4000, 3000), focal_mm=4.25)
    got = read_exif_intrinsics(p)

    assert got is not None
    assert got.assumed_sensor
    assert got.source == "exif:focal+sensor_assumed"
    assert got.focal_35mm_mm is None
    assert got.quantisation_rel is None            # 无整数毫米量化
    assert got.fx == pytest.approx(4.25 / SENSOR_WIDTH_MM_35MM * 4000)
    assert any("全画幅假设" in n for n in got.notes)


def test_phone_photo_under_full_frame_assumption_fails_fov_check(tmp_path: Path):
    """全画幅假设用在手机照片上时，check_fov 必须判不可信 —— 不能静默通过。

    手机典型值：真实焦距 4.25 mm、传感器宽约 5.6 mm ⟹ 等效 27 mm。
    若按 36 mm 假设，等效焦距被当成 4.25 mm，视场被算成约 153°，远超
    PLAUSIBLE_HFOV_DEG 的上界 110°。这条测试把「错误假设会被下游拦住」钉住；
    它成立的前提是 `known_intrinsics` 路径上 check_fov 仍然生效 —— 接线时别拆掉。
    """
    p = _write_jpeg(tmp_path / "phone.jpg", (4000, 3000), focal_mm=4.25)
    got = read_exif_intrinsics(p)

    assert got is not None
    assert not got.fov.plausible
    assert got.fov.reason == "hfov_out_of_range"
    assert got.fov.hfov_deg > 110.0
    assert any("不可信" in n for n in got.notes)

    # 若用手机的真实传感器宽度，视场就回到合理区间 —— 说明责任在假设而非算法
    good = read_exif_intrinsics(p, sensor_width_mm=5.6)
    assert good is not None
    assert good.fov.plausible
    assert good.fx == pytest.approx(4.25 / 5.6 * 4000)


# ---------------------------------------------------------------------------
# ②b 降级路径的边界 —— 跨来源实测（phase0/probe_cross_source.py A 段）逼出来的
# ---------------------------------------------------------------------------

def test_mft_photo_escapes_fov_check_under_full_frame_assumption(tmp_path: Path):
    """**证伪测试**：全画幅假设对 MFT 不算「骗不过 check_fov」。

    素材来自真实相机 EXIF（Olympus E-P3）：`FocalLength=15 mm`、真实画幅 MFT
    （crop 2.0）⟹ 真实等效焦距 30 mm、HFoV 61.9°。若按 36 mm 假设，等效焦距被
    当成 15 mm，HFoV 算成 **100.4°** —— 仍然落在 30–110° 的窗口内。

    所以旧写法「这种粗暴假设骗不过 check_fov」只对手机（crop≈6.4，视场 153°）成立。
    窗口 30–110° 等价于 f_35 ∈ [12.6, 67.2] mm；误差要被兜住必须让 assumed 值
    **逃出**窗口，即 FocalLength 小于 12.6 mm —— APS-C(1.5×)/MFT(2×)/1 吋(2.7×)
    的常见焦距全都在窗口内。这条测试把「漏报」这一档钉住，防止它被改回去。

    责任归属：**不是算法错，是这条假设不该靠视场检查来担保**。修法是让调用方
    拿到 `assumed_sensor=True` 时无条件警觉（`scripts/build_scene.py` 已如此），
    或显式传 `sensor_width_mm`。
    """
    p = _write_jpeg(tmp_path / "e-p3.jpg", (80, 80), focal_mm=15)
    got = read_exif_intrinsics(p)

    assert got is not None
    assert got.assumed_sensor
    assert got.fx == pytest.approx(15.0 / SENSOR_WIDTH_MM_35MM * 80)
    # ⚠ 这一行就是那条漏报：视场错了 1.62 倍（100.4° vs 61.9°），却判为可信
    assert got.fov.plausible is True
    assert got.fov.reason == "ok"
    assert got.fov.hfov_deg == pytest.approx(100.4, abs=0.3)
    assert any("不可由 check_fov 担保" in n for n in got.notes)

    # 反证：给出真实传感器宽度（MFT 取 18.0 mm ⟹ 恰好 crop 2.0）视场就对上了
    good = read_exif_intrinsics(p, sensor_width_mm=18.0)
    assert good is not None
    assert good.fov.hfov_deg == pytest.approx(61.9, abs=0.2)
    assert good.fx / got.fx == pytest.approx(2.0, rel=1e-6)


def test_legitimate_telephoto_trips_plausibility_flag(tmp_path: Path):
    """**误报测试**：K 完全正确，但 check_fov 仍然告警。

    真实相机 EXIF（Panasonic DMC-L10，50 mm 镜 / MFT）：`FocalLengthIn35mmFilm=100`
    ⟹ HFoV 20.4°，低于 30° 下界。K 的算式没有任何问题 —— 100 mm 等效就是长焦。

    ⟹ `check_fov` 的语义是「**典型照片的合理性先验**」（大多数照片在 30–110°），
    不是「内参正确性的校验器」。把它当后者用，会在长焦照片上误报，
    也会在 MFT 广角上漏报（见上一条）。两条例外一起看，才是这个工具的真实边界。
    """
    p = _write_jpeg(tmp_path / "tele.jpg", (4000, 3000), focal_35=100, focal_mm=50)
    got = read_exif_intrinsics(p)

    assert got is not None
    assert not got.assumed_sensor                 # 有等效焦距，走的是安全路径
    assert got.fx == pytest.approx(100.0 / 36.0 * 4000)   # 算式无误
    assert got.fov.hfov_deg == pytest.approx(20.4, abs=0.2)
    assert got.fov.plausible is False             # 却仍被告警
    assert got.fov.reason == "hfov_out_of_range"


# ---------------------------------------------------------------------------
# ③ 尺寸不一致（缩放 vs 裁剪）
# ---------------------------------------------------------------------------

def test_size_mismatch_is_reported(tmp_path: Path):
    """EXIF 记录的像素尺寸与当前不符 ⟹ 缩放或裁剪过，必须报告。

    本模块无法区分两者（缩放无害、裁剪致命），所以只能把它变成一个显式字段。
    一个**不能区分**的事实更不该被藏起来 —— 藏起来就等于默认为无害。
    """
    p = _write_jpeg(tmp_path / "c.jpg", (4000, 3000), focal_35=26,
                    pixel_xy=(8000, 6000))
    got = read_exif_intrinsics(p)

    assert got is not None
    assert got.size_mismatch
    assert any("被缩放或裁剪过" in n for n in got.notes)


def test_matching_size_is_not_flagged(tmp_path: Path):
    p = _write_jpeg(tmp_path / "d.jpg", (4000, 3000), focal_35=26,
                    pixel_xy=(4000, 3000))
    got = read_exif_intrinsics(p)

    assert got is not None
    assert not got.size_mismatch


def test_absent_pixel_tags_are_not_flagged(tmp_path: Path):
    """没有 PixelX/Y 标签时不能误报 —— 缺失 ≠ 不一致。"""
    p = _write_jpeg(tmp_path / "e.jpg", (4000, 3000), focal_35=26)
    got = read_exif_intrinsics(p)

    assert got is not None
    assert not got.size_mismatch


# ---------------------------------------------------------------------------
# ④ 坏标签与缺失：退化，不抛
# ---------------------------------------------------------------------------

def test_no_exif_returns_none(tmp_path: Path):
    """无 EXIF 是**正常分支**而不是错误 —— 截图、聊天转存、PNG 都会走到这里。"""
    from PIL import Image

    p = tmp_path / "plain.png"
    Image.new("RGB", (640, 480), (0, 0, 0)).save(p)

    assert read_exif_intrinsics(p) is None


@pytest.mark.parametrize("bad", [0, "", None])
def test_bad_focal_values_degrade_to_none(tmp_path: Path, bad: object):
    """写进文件里的坏值（0 / 空串 / 缺失）都要退化成 None，不能抛。

    刻意只放**能被 Pillow 序列化**的坏值：-1 写进 SHORT 会在保存阶段就失败，
    那测的是 Pillow 而不是本模块。非写盘的边界值改由 `_as_float` 直接单测覆盖。
    """
    p = _write_jpeg(tmp_path / f"bad_{abs(hash(str(bad))) % 9973}.jpg",
                    (1000, 750), focal_35=bad)
    assert read_exif_intrinsics(p) is None


def test_as_float_rejects_junk_values():
    """`_as_float` 是唯一接触外部原始值的地方，边界要顶住。

    它必须把「不能用的值」与「能用的值」分干净：外部输入坏掉时的正确行为是
    **退化**（当成没有内参），不是抛异常 —— 抛出去会让整条流水线停在一个
    与几何无关的原因上。
    """
    from vision.exif import _as_float

    for junk in (None, 0, -1, float("nan"), float("inf"), "abc", "", (1, 0), [1]):
        assert _as_float(junk) is None, junk

    # 正常形态：float / int / rational 元组 / "num/den" 字符串
    assert _as_float(26) == pytest.approx(26.0)
    assert _as_float(26.5) == pytest.approx(26.5)
    assert _as_float((85, 20)) == pytest.approx(4.25)
    assert _as_float("85/20") == pytest.approx(4.25)

    # PIL 的 IFDRational 也要认（真机读出来的就是这个类型）
    from PIL.TiffImagePlugin import IFDRational

    assert _as_float(IFDRational(85, 20)) == pytest.approx(4.25)


def test_rational_tuple_form_survives_a_roundtrip(tmp_path: Path):
    """EXIF 的 rational 是真机读取路径上的常客，必须能端到端走一遍。"""
    from PIL import Image

    im = Image.new("RGB", (2000, 1500), (0, 0, 0))
    exif = im.getexif()
    exif[TAG_FOCAL_LENGTH] = (85, 20)                 # = 4.25 mm
    p = tmp_path / "rat.jpg"
    im.save(p, exif=exif)

    got = read_exif_intrinsics(p)
    assert got is not None
    assert got.focal_mm == pytest.approx(4.25)


def test_missing_file_raises_not_returns_none(tmp_path: Path):
    """路径不存在是**调用方的错误**，与「文件里没 EXIF」不同 —— 必须抛。

    把这两种情况都折成 None 会让「路径写错」变成一个静默的降级，
    然后在几个小时后的评测里表现为「内参没生效」。区分开，错误就不会漂移。
    """
    with pytest.raises(OSError):
        read_exif_intrinsics(tmp_path / "nope.jpg")


# ---------------------------------------------------------------------------
# Exif SubIFD 分支（真机的主路径）
# ---------------------------------------------------------------------------

class _StubExif:
    """只在 SubIFD 里有标签的桩。用来确定性覆盖 `_lookup` 的第二条分支。"""

    def __init__(self, sub: dict) -> None:
        self._sub = sub

    def get(self, tag: int, default: object = None) -> object:
        return default

    def get_ifd(self, tag: int) -> dict:
        return self._sub if tag == EXIF_IFD else {}


class _StubImage:
    def __init__(self, size: tuple[int, int], sub: dict) -> None:
        self.size = size
        self._exif = _StubExif(sub)

    def getexif(self) -> _StubExif:
        return self._exif


def test_reads_from_exif_subifd():
    """真机把 FocalLengthIn35mmFilm 放在 Exif SubIFD —— 只查 IFD0 会漏掉主路径。"""
    im = _StubImage((4000, 3000), {TAG_FOCAL_35MM: 26})
    got = read_exif_intrinsics(im)

    assert got is not None
    assert got.source == "exif:35mm"
    assert got.fx == pytest.approx(26 / 36 * 4000)


def test_ifd0_wins_over_subifd():
    """两处都有时以 IFD0 为准：那份通常由导出方写入，与当前像素尺寸更一致。"""
    class _Both(_StubImage):
        def getexif(self):                                   # type: ignore[override]
            sub = {TAG_FOCAL_35MM: 26}
            outer = _StubExif(sub)

            def get(tag: int, default: object = None) -> object:
                return 50 if tag == TAG_FOCAL_35MM else default

            outer.get = get                                   # type: ignore[method-assign]
            return outer

    got = read_exif_intrinsics(_Both((4000, 3000), {}))
    assert got is not None
    assert got.focal_35mm_mm == pytest.approx(50.0)
    assert got.fx == pytest.approx(50 / 36 * 4000)


# ---------------------------------------------------------------------------
# 结构与导出
# ---------------------------------------------------------------------------

def test_K_is_well_formed_and_fov_matches(tmp_path: Path):
    p = _write_jpeg(tmp_path / "k.jpg", (4000, 3000), focal_35=26)
    got = read_exif_intrinsics(p)

    assert got is not None
    K = got.K
    assert K.shape == (3, 3)
    assert K[2, 2] == pytest.approx(1.0)
    assert K[0, 1] == pytest.approx(0.0) and K[1, 0] == pytest.approx(0.0)
    assert np.isfinite(K).all()

    # 26 mm 等效在全画幅上是广角：水平视场应当落在 60–70° 附近
    assert 55.0 < got.fov.hfov_deg < 75.0
    assert got.fov.plausible
    # gfov 关系自洽：tan(hfov/2) = W/(2f)
    assert np.tan(np.radians(got.fov.hfov_deg / 2)) == pytest.approx(
        4000 / (2 * got.fx), rel=1e-6)


def test_quantisation_is_reported_for_integer_35mm(tmp_path: Path):
    """26 mm 的半步量化 = 0.5/26 = 1.92%。Step 7 的门槛是 6.6%（10 px）——
    EXIF 用掉约三分之一，所以这个数必须随 K 一起记录，而不是丢掉。"""
    p = _write_jpeg(tmp_path / "q.jpg", (4000, 3000), focal_35=26)
    got = read_exif_intrinsics(p)

    assert got is not None
    assert got.quantisation_rel == pytest.approx(0.5 / 26)
    assert got.quantisation_pct == pytest.approx(0.5 / 26 * 100)
    assert any("±6.6%" in n for n in got.notes)


def test_known_intrinsics_tuple_shape(tmp_path: Path):
    """`fx_fy_cx_cy` 直接喂给 `BuildConfig.known_intrinsics`，顺序不能错。"""
    p = _write_jpeg(tmp_path / "t.jpg", (4000, 3000), focal_35=26)
    got = read_exif_intrinsics(p)

    assert got is not None
    fx, fy, cx, cy = got.fx_fy_cx_cy
    assert (fx, fy, cx, cy) == pytest.approx(
        (got.K[0, 0], got.K[1, 1], got.K[0, 2], got.K[1, 2]))


def test_as_dict_is_json_serialisable(tmp_path: Path):
    """`as_dict()` 要能直接进 scene.json / build_log —— 不能夹 numpy 类型。"""
    p = _write_jpeg(tmp_path / "j.jpg", (4000, 3000), focal_35=26)
    got = read_exif_intrinsics(p)

    assert got is not None
    s = json.dumps(got.as_dict(), ensure_ascii=False)
    back = json.loads(s)
    assert back["source"] == "exif:35mm"
    assert back["image_hw"] == [3000, 4000]
    assert isinstance(back["notes"], list) and back["notes"]


def test_describe_mentions_source_and_quantisation(tmp_path: Path):
    p = _write_jpeg(tmp_path / "d2.jpg", (4000, 3000), focal_35=26)
    got = read_exif_intrinsics(p)

    assert got is not None
    text = got.describe()
    assert "fx=" in text and "HFoV" in text and "量化误差" in text


def test_describe_flags_implausible_fov(tmp_path: Path):
    p = _write_jpeg(tmp_path / "d3.jpg", (4000, 3000), focal_mm=4.25)
    got = read_exif_intrinsics(p)

    assert got is not None
    assert "视场不可信" in got.describe()


def test_result_is_frozen(tmp_path: Path):
    """结果对象不可变：它一旦生成就被当证据用，不该被下游就地改写。"""
    p = _write_jpeg(tmp_path / "f.jpg", (4000, 3000), focal_35=26)
    got = read_exif_intrinsics(p)

    assert got is not None
    with pytest.raises(Exception):
        got.fx = 1.0                                     # type: ignore[misc]


def test_exif_module_does_not_import_torch():
    """EXIF 是纯元数据解析，不该把 torch 拉进无 GPU 的测试路径。

    这条不是洁癖：`vision/exif.py` 会被 builder/CLI 之外的诊断脚本用到，
    只要有一处 import torch，那些脚本的启动就从 0.2 s 变成 5 s。
    """
    import sys as _sys

    for name in list(_sys.modules):
        if name == "torch" or name.startswith("torch."):
            pytest.skip("本进程已加载 torch，无法验证；由 CI 的纯 CPU 任务覆盖")
    assert "torch" not in _sys.modules
