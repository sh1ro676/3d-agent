#!/usr/bin/env python
r"""看一张（或一批）图片的 EXIF 能不能给出可用内参。

## 为什么这个脚本比它看起来重要

`scripts/build_scene.py --intrinsics exif` 是真实照片拿到正确横向尺度的**唯一**
零成本途径。但它有两个不太直观的失败模式：

  ① **静默降级**：图片没有 EXIF 时（截图、微信转存、部分 PNG），`--intrinsics exif`
     会退化成模型预测。而模型预测在同一张图上横向误差是 EXIF 的 **118 倍**
     （Step 7 实测：306.3 px vs 2.6 px）。也就是说「有没有 EXIF」这一件事，
     决定了 scene.json 里的米制数字能不能用。
  ② **看起来对但是错的**：只有 `FocalLength` 而没有 35 mm 等效值时，本工具会按
     全画幅假设换算。对手机照片这会让视场宽出约 6 倍 —— `check_fov` 会拦住它，
     但前提是你知道要去看那个判定。

所以拿到一批照片的正确第一步，是**先看元数据**，而不是直接跑流水线。
这个脚本就是那一步：把「能不能用」「用哪一条路」「误差多大」一次说清。

## 用法

    # 单张
    python scripts/inspect_exif.py --image D:\photos\room.jpg

    # 整个目录（只看常见图片后缀，不递归）
    python scripts/inspect_exif.py --dir D:\photos --raw

`--raw` 会把原始 EXIF 标签表也打出来 —— 当结论是「读不到」时，这是唯一能
告诉你是「真的没有」还是「有但位置/类型不常见」的东西。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic")


def _fmt_table(exif) -> list[str]:
    """把原始 EXIF 打成人能读的表。"""
    from PIL import ExifTags

    out: list[str] = []
    try:
        items = list(exif.items())
    except Exception as e:                      # 元数据损坏时不该让整个脚本倒下
        return [f"    <无法枚举 EXIF：{type(e).__name__}>"]
    if not items:
        return ["    <EXIF 为空>"]
    for tag, val in items:
        name = ExifTags.TAGS.get(tag, ExifTags.TAGS.get(tag & 0xFFFF, "?"))
        s = str(val)
        if len(s) > 70:
            s = s[:67] + "..."
        out.append(f"    {tag:#06x}  {name:<28} {s}")
    return out


def report(path: Path, raw: bool = False) -> bool:
    """打印一张图的判定。返回「是否拿到了可用内参」。"""
    from PIL import Image

    from vision.exif import read_exif_intrinsics

    print()
    print("=" * 74)
    print(f"  {path}")
    print("=" * 74)

    try:
        with Image.open(path) as im:
            size = im.size
            exif = im.getexif()
            n_tags = len(list(exif.items())) if exif else 0
    except Exception as e:
        print(f"  [FAIL] 打不开：{type(e).__name__}: {e}")
        return False

    print(f"  尺寸 {size[0]}×{size[1]}   EXIF 顶层标签数 {n_tags}")

    got = read_exif_intrinsics(path)
    if got is None:
        print()
        print("  ✗ **读不到可用的等效焦距** ⟹ `--intrinsics exif` 会退化为模型预测。")
        print("    后果不是「差一点」：横向误差约为有内参时的 118 倍（Step 7）。")
        print("    原因只可能是这三种：")
        print("      · 图是截图 / 聊天软件转存 / 被平台重新编码，元数据被剥掉了")
        print("      · 相机或 App 本来就不写 FocalLength / FocalLengthIn35mmFilm")
        print("      · 标签存在但类型不常见（本工具已容忍 rational 元组与 "
              "\"num/den\" 字符串）")
        print('    出路：手动给 `--intrinsics "fx,fy,cx,cy"`（例如用已知视场的标定物'
              "反算），或者接受模型预测并在结论里标注横向尺度不可信。")
        if raw:
            print()
            print("  原始 EXIF：")
            print("\n".join(_fmt_table(exif)))
        return False

    print()
    print(f"  ✓ 来源 {got.source}")
    print(f"    K  fx={got.fx:.1f}  fy={got.fy:.1f}  cx={got.cx:.1f}  cy={got.cy:.1f}")
    print(f"    视场 HFoV {got.fov.hfov_deg:.1f}°  VFoV {got.fov.vfov_deg:.1f}°"
          f"  {'可信' if got.fov.plausible else '**不可信**'} ({got.fov.reason})")
    if got.focal_35mm_mm is not None:
        print(f"    等效焦距 {got.focal_35mm_mm:g} mm"
              f"   量化误差 ±{got.quantisation_pct:.1f}%")
    if got.focal_mm is not None:
        print(f"    真实焦距 {got.focal_mm:g} mm"
              f"   传感器宽度 {got.sensor_width_mm:g} mm"
              f"{'（假设值！）' if got.assumed_sensor else ''}")

    print()
    print("  注意事项：")
    for n in got.notes:
        print(f"    · {n}")
    if not got.fov.plausible:
        print()
        print("  ⚠ 视场不可信 ⟹ 这个 K **不要**直接喂给流水线。")
        print("    若是「只有 FocalLength」的情形，用 `--sensor-width` 给出真实传感器")
        print("    宽度（手机常见 5.6–7.6 mm，APS-C 23.5 mm，全画幅 36 mm）再试。")
    if raw:
        print()
        print("  原始 EXIF：")
        print("\n".join(_fmt_table(exif)))

    print()
    print(f"  建议命令：")
    print(f"    --intrinsics exif     （来源可信时）")
    print(f"    --intrinsics \"{got.fx:.1f},{got.fy:.1f},{got.cx:.1f},{got.cy:.1f}\""
          f"   （想把这个值固定下来、让场景图可复现时）")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", action="append", default=[],
                    help="可重复给出多张图")
    ap.add_argument("--dir", default=None, help="扫这个目录（不递归）")
    ap.add_argument("--raw", action="store_true",
                    help="同时打印原始 EXIF 标签表")
    ap.add_argument("--sensor-width", type=float, default=None,
                    help="给「只有 FocalLength」的图指定真实传感器宽度（mm）后重算")
    args = ap.parse_args()

    targets: list[Path] = [Path(p) for p in args.image]
    if args.dir:
        d = Path(args.dir)
        if not d.is_dir():
            print(f"[FAIL] 目录不存在：{d}")
            return 2
        targets += sorted(
            p for p in d.iterdir()
            if p.is_file() and p.suffix.lower() in _SUFFIXES
        )
    if not targets:
        print("[FAIL] 没给 --image 或 --dir")
        return 2

    if args.sensor_width is not None:
        # 只影响「只有 FocalLength」那条路。就地替换函数默认值而不是加分支，
        # 是为了让这条重算走的**完全是**生产路径，不引入第二套算法。
        import vision.exif as _exif_mod

        orig = _exif_mod.read_exif_intrinsics

        def with_sw(source, *, image_size=None, sensor_width_mm=args.sensor_width):
            return orig(source, image_size=image_size,
                        sensor_width_mm=args.sensor_width)

        _exif_mod.read_exif_intrinsics = with_sw
        print(f"  [--sensor-width {args.sensor_width:g} mm 已生效]")

    ok = 0
    for p in targets:
        if not p.is_file():
            print(f"\n  [FAIL] 文件不存在：{p}")
            continue
        ok += int(report(p, raw=args.raw))

    print()
    print("=" * 74)
    print(f"  小结：{ok}/{len(targets)} 张图能给出可用内参")
    if ok < len(targets):
        print(f"  其余 {len(targets) - ok} 张会退回模型预测 —— 横向尺度不可信，"
              "结论里必须标注。")
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
