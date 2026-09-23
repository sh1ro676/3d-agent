"""`scripts/inspect_jpeg_segments.py` 的单元测试 —— 钉住**这把尺子本身**。

为什么一个诊断脚本也需要测试
----------------------------
它的输出是**否定性证据**：「文件里没有 APP1 段 ⟹ EXIF 是被剥掉的」。
一个坏掉的扫描器对任何输入都会说「没有 APP1」—— 那时这条结论看起来完全成立，
但它是假的。这正是本项目反复遇到的那类缺陷：**不报错，只改判结果**。

所以这里的重点不是「能打印段」，而是三条容易静默出错的性质：

    ① 对**已知含 EXIF** 的文件必须报出 `APP1 ← Exif`（扫描器真的在找）
    ② 对不含 EXIF 的文件必须报 0（它不是在无脑报有）
    ③ 非 JPEG 必须被识别为 `NOT_JPEG`，而不是被当成「没有 EXIF」

第 ① 条是这组测试存在的理由：没有它，②③ 全都无意义。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TAG_FOCAL_35MM = 0xA405


def _load():
    path = ROOT / "scripts" / "inspect_jpeg_segments.py"
    spec = importlib.util.spec_from_file_location("_segs_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


SEG = _load()


def _jpeg(path: Path, *, focal_35: object = None) -> Path:
    from PIL import Image

    im = Image.new("RGB", (64, 48), (13, 17, 23))
    if focal_35 is None:
        im.save(path)
        return path
    exif = im.getexif()
    exif[TAG_FOCAL_35MM] = focal_35
    im.save(path, exif=exif)
    return path


def _names(rows) -> list[str]:
    return [k for k, _v in rows]


def _blob(rows) -> str:
    return "\n".join("%s %s" % (k, v) for k, v in rows)


# ---------------------------------------------------------------------------
# ① 阳性对照：扫描器真的会认 EXIF（这组测试存在的理由）
# ---------------------------------------------------------------------------

def test_finds_app1_exif_in_a_file_that_has_it(tmp_path: Path):
    rows = SEG.scan(_jpeg(tmp_path / "with_exif.jpg", focal_35=24))

    assert "APP1" in _names(rows), "对含 EXIF 的文件没报出 APP1 ⟹ 扫描器坏了"
    assert "← Exif" in _blob(rows)


def test_actually_reads_the_project_fixture_if_present():
    """项目自带的夹具是最接近真机的对照，扫它必须报出 APP1。

    夹具缺失时跳过而不是失败 —— 它是 `.cache/` 下的产物，不属于源码。
    """
    p = ROOT / ".cache" / "exif_fixture" / "rgb_exif.jpg"
    if not p.is_file():
        pytest.skip("夹具不在盘上（.cache 下，属可重建产物）")

    rows = SEG.scan(p)
    assert sum(1 for _k, v in rows if "Exif" in v) >= 1
    assert any(k == "APP1" for k in _names(rows))


# ---------------------------------------------------------------------------
# ② 反向：不含 EXIF 的文件必须报 0（否则「都报有」也能骗过 ①）
# ---------------------------------------------------------------------------

def test_reports_no_app1_for_a_file_without_exif(tmp_path: Path):
    rows = SEG.scan(_jpeg(tmp_path / "plain.jpg"))

    assert "APP1" not in _names(rows)
    assert not any("Exif" in v for _k, v in rows), "没有 EXIF 的文件不该出现 Exif 字样"


def test_sos_terminates_the_scan(tmp_path: Path):
    """扫到 SOS 就该停 —— 之后是熵编码数据，继续扫会把压缩数据当段头读。

    这条用**真 JPEG** 验证：随机字节里很容易撞出 `FF E1`，若不停在 SOS，
    不含 EXIF 的文件会被误报成含 EXIF。
    """
    rows = SEG.scan(_jpeg(tmp_path / "term.jpg"))

    assert _names(rows)[-1] == "SOS"
    assert sum(1 for k in _names(rows) if k == "SOS") == 1


# ---------------------------------------------------------------------------
# ③ 非 JPEG 不能被混成「没有 EXIF」
# ---------------------------------------------------------------------------

def test_non_jpeg_is_reported_as_not_jpeg(tmp_path: Path):
    from PIL import Image

    p = tmp_path / "shot.png"
    Image.new("RGB", (32, 32), (1, 2, 3)).save(p)

    rows = SEG.scan(p)
    assert _names(rows) == ["NOT_JPEG"], "PNG 必须被明确标成 NOT_JPEG，而不是「没有 EXIF」"
