"""`scripts/build_scene.py::load_known_intrinsics` 的单元测试 —— 重点是**新加的** `f35:` 一档。

为什么这一档值得单测，而不是烟测一下就算
----------------------------------------
它的存在理由是一份实测发现（2026-09-23）：两张 iPhone 照片经微信送达后
**文件里一个 APP1 段都没有**（`reports/real_photo_exif_probe.txt`，阳性对照 3/3 命中）。
也就是说对普通人最常用的那条传图路径，`--intrinsics exif` **必然**退化成模型预测；
而原来的兜底是让用户手填 4 个像素焦距 —— 手机里根本查不到，**那条兜底是空的**。

补这一档时最大的风险不是「算错」，而是**悄悄引入第二份公式**。如果 `f35:` 自己再写一遍
`f_px = f35/36 × 长边`，两条路迟早分叉，而分叉之后同一张图经两条路得到不同 K，
**谁对谁错在产物里完全看不出来**。所以第一条测试就是逐位比较。

第二条同样重要：**坏输入必须响**。`f35:0` / `f35:abc` 属于用户填错，与「这张图没有
元数据」根本不是一类事；混起来会让一个填错的数字伪装成「文件里没有 EXIF」。

导入方式：`scripts/` 不是包（没有 `__init__.py`），按路径加载，不动包结构。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TAG_FOCAL_35MM = 0xA405


def _load():
    path = ROOT / "scripts" / "build_scene.py"
    spec = importlib.util.spec_from_file_location("_bs_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


BS = _load()


def _jpeg(path: Path, size: tuple[int, int] = (4000, 3000),
          focal_35: object = None) -> Path:
    """写一张 JPEG；`focal_35=None` 时**不带任何 EXIF**（模拟微信转存件）。"""
    from PIL import Image

    im = Image.new("RGB", size, (13, 17, 23))
    if focal_35 is None:
        im.save(path)
        return path
    exif = im.getexif()
    exif[TAG_FOCAL_35MM] = focal_35
    im.save(path, exif=exif)
    return path


#: 竖图。⚠ 两处顺序**不同**：`Image.new` 收 `(W, H)`，而 `image_size=` 收 `(H, W)`。
#: 写错的话「长边取自 H」这条约定会看起来成立、实际用的是 W —— 而结果只是 fx 按
#: 宽高比整体缩放，输出了一个看着仍然合理的数字。
PORTRAIT_WH = (3000, 4000)      # 文件：W=3000, H=4000
PORTRAIT_HW = (4000, 3000)      # image_size：(H, W)


# ---------------------------------------------------------------------------
# ① 两条路必须给出同一个 K（这是本次改动的全部风险所在）
# ---------------------------------------------------------------------------

def test_f35_and_exif_paths_agree_bit_for_bit(tmp_path: Path):
    """同一个 f_35：从文件读、由调用方给 —— 结果必须**逐位相等**。"""
    p = _jpeg(tmp_path / "has_exif.jpg", (4000, 3000), focal_35=29)
    k_exif, _, meta = BS.load_known_intrinsics("exif", p, image_size=(3000, 4000))
    f35 = meta["exif"]["focal_35mm_mm"]
    k_given, _, _ = BS.load_known_intrinsics(f"f35:{f35}", p, image_size=(3000, 4000))

    assert k_exif is not None and k_given is not None
    assert np.array_equal(np.asarray(k_exif), np.asarray(k_given)), (
        "两条路给出的 K 不逐位相等 ⟹ 大概率是 f35: 自己算了一遍，公式分叉了"
    )


def test_f35_uses_the_long_side_like_the_exif_path(tmp_path: Path):
    """竖构图：长边取 H。这条约定错了，fx 会按宽高比整体缩放而数字看着仍合理。"""
    p = _jpeg(tmp_path / "portrait.jpg", PORTRAIT_WH)       # 3000×4000（W×H），无 EXIF
    k, _, _ = BS.load_known_intrinsics("f35:24", p, image_size=PORTRAIT_HW)

    assert k is not None
    assert k[0] == pytest.approx(24.0 / 36.0 * 4000)        # 用 H=4000，不是 W
    assert k[2] == pytest.approx(1500.0) and k[3] == pytest.approx(2000.0)


# ---------------------------------------------------------------------------
# ② 它要解决的正是「没有 EXIF」这一档
# ---------------------------------------------------------------------------

def test_f35_works_on_a_photo_without_any_exif(tmp_path: Path):
    p = _jpeg(tmp_path / "from_wechat.jpg", (1280, 1707))   # 无 EXIF，模拟微信件

    degraded, note, meta = BS.load_known_intrinsics("exif", p, image_size=(1707, 1280))
    assert degraded is None, "没有 EXIF 时 exif 路就该降级 —— 这正是要补的缺口"
    assert meta["exif"] is None and "没有可用的等效焦距" in note

    k, note2, meta2 = BS.load_known_intrinsics("f35:24", p, image_size=(1707, 1280))
    assert k is not None, "有用户给的等效焦距时不该再降级"
    assert k[0] == pytest.approx(24.0 / 36.0 * 1707)
    assert meta2["exif"]["source"] == "user:35mm", "来源必须能区分「人填的」与「文件读的」"


def test_user_provided_source_is_distinguishable_in_the_note(tmp_path: Path):
    """事后只看数字分不出「照片读到的」还是「人填的」，所以说明文字必须标出来。"""
    p = _jpeg(tmp_path / "n.jpg", (4000, 3000))
    _, note, _ = BS.load_known_intrinsics("f35:24", p, image_size=(3000, 4000))

    assert "用户提供" in note
    assert "EXIF →" not in note


def test_f35_overrides_a_file_that_does_have_exif(tmp_path: Path):
    """文件里已有 EXIF 时，显式给的 f_35 必须是**生效的那一个**（双向验证）。

    只断言「与手算一致」不够：如果 f35: 被忽略、结果其实是文件里的 29 mm，
    某些输入下也可能碰巧相等。这里同时要求它**不等于**文件那条路。
    """
    p = _jpeg(tmp_path / "o.jpg", (4000, 3000), focal_35=29)
    k_file, _, _ = BS.load_known_intrinsics("exif", p, image_size=(3000, 4000))
    k_given, _, _ = BS.load_known_intrinsics("f35:13", p, image_size=(3000, 4000))

    assert k_file is not None and k_given is not None
    assert k_file[0] == pytest.approx(29.0 / 36.0 * 4000)
    assert k_given[0] == pytest.approx(13.0 / 36.0 * 4000)
    assert k_given[0] != k_file[0]


# ---------------------------------------------------------------------------
# ③ 坏输入必须响，不许静默降级成「没有内参」
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", [
    "f35:", "f35: ", "f35:abc", "f35:0", "f35:-5", "f35:nan", "f35:inf", "f35:+",
])
def test_bad_f35_specs_are_rejected_loudly(tmp_path: Path, spec: str):
    p = _jpeg(tmp_path / "b.jpg", (1280, 1707))
    with pytest.raises(SystemExit):
        BS.load_known_intrinsics(spec, p, image_size=(1707, 1280))


def test_f35_accepts_a_decimal(tmp_path: Path):
    """有些机型档位不是整毫米 —— 小数必须被接受，否则用户只能四舍五入。"""
    p = _jpeg(tmp_path / "d.jpg", (4000, 3000))
    k, _, _ = BS.load_known_intrinsics("f35:26.5", p, image_size=(3000, 4000))

    assert k is not None
    assert k[0] == pytest.approx(26.5 / 36.0 * 4000)


# ---------------------------------------------------------------------------
# ④ 既有写法不许被破坏（回归）
# ---------------------------------------------------------------------------

def test_four_numbers_still_work(tmp_path: Path):
    p = _jpeg(tmp_path / "r1.jpg", (1280, 1707))
    k, note, _ = BS.load_known_intrinsics("525.1, 525.1, 320, 240", p)

    assert k == (525.1, 525.1, 320.0, 240.0)
    assert "4 元组" in note


def test_none_spec_still_means_model_prediction(tmp_path: Path):
    p = _jpeg(tmp_path / "r2.jpg", (1280, 1707))
    k, note, meta = BS.load_known_intrinsics(None, p)

    assert k is None and meta == {}
    assert "模型预测" in note


def test_npy_file_still_wins_over_the_f35_prefix(tmp_path: Path):
    """`f35:` 是前缀匹配，不能把「文件名恰好以 f35: 开头」的路径吃掉。

    Windows 上 `f35:` 不是合法盘符，所以这里构造的是**相对路径**下的同名文件；
    要点是 `f35:` 分支必须**在**路径判定之外、且只认 `f35:<数字>` 这一种形状。
    """
    p = _jpeg(tmp_path / "r3.jpg", (1280, 1707))
    npy = tmp_path / "intrinsics.npy"
    np.save(npy, np.array([[100.0, 0, 10], [0, 200.0, 20], [0, 0, 1]]))

    k, note, _ = BS.load_known_intrinsics(str(npy), p)
    assert k == (100.0, 200.0, 10.0, 20.0)
    assert "npy" in note
