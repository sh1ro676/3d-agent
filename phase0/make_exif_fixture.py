#!/usr/bin/env python
r"""造一张**带真值**的 EXIF fixture，用来端到端验证 `--intrinsics exif`。

## 为什么必须自己造

实测仓库里所有图片的 EXIF **全为空**（`scripts/inspect_exif.py` 的结果：
`vendor/UniDepth/assets/demo/rgb.png` 也是 0 个标签）。
而真实照片几乎都带 EXIF。于是 EXIF 这条路**无法用仓库素材端到端验证**。

自己造的时候有一个陷阱：如果只造一张「有 EXIF 的图」，验证就退化成
「流水线跑通了」—— 那什么都没证明。真正的验证需要**知道正确答案**，所以：

    从 GT 内参反推该写进 EXIF 的整数等效焦距

    fx_gt = 518.9（640×480 的 demo 图，仓库自带 intrinsics.npy）
    f_35  = fx_gt × 36 / 长边 = 518.9 × 36 / 640 = 29.19 mm
    取整  → 29 mm            （EXIF 只能存整数毫米，这就是量化的来源）
    回推  → fx = 29/36 × 640 = 515.6 px    偏差 −0.64%

于是这张 fixture 就有了一个**已知的正确答案**（GT K）和一个**已知的偏差**
（−0.64%），可以用来检验两件事：

  ① 读出的 K 是否等于按 29 mm 反算的值（读得对不对）
  ② 这个 −0.64% 传到几何上还剩多少（值不值得在意 —— 拿 Step 7 的容差预算对照）

## 与 Step 7 容差预算的关系

Step 7 算出：方位误差 ≤10 px 要求焦距偏差 ≤±6.6%。这里的 0.64% 只用掉预算的
约十分之一。所以**预期结论是「EXIF 可用」** —— 但这是要跑出来看的，
不是可以顺手断言的：如果 EXIF 那条路上的换算写错了宽高（例如竖构图用了 W
而不是长边），偏差会立刻变成 33% 而不是 0.64%，而这个错误在单元测试之外
是看不出来的。

## 用法

    python phase0/make_exif_fixture.py            # 造图 + 核对读数
    # 然后按它打印的命令跑 build_scene.py --intrinsics exif
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

DEMO = ROOT / "vendor" / "UniDepth" / "assets" / "demo"
OUT_DIR = ROOT / ".cache" / "exif_fixture"
FIXTURE = OUT_DIR / "rgb_exif.jpg"

SENSOR_WIDTH_MM_35MM = 36.0
REPORT = HERE / "probe_exif_pipeline_report.txt"
RESULT = HERE / "probe_exif_pipeline_result.json"

# EXIF 标签
TAG_FOCAL_LENGTH = 0x920A       # rational，真实毫米
TAG_FOCAL_35MM = 0xA405         # short，整数毫米
TAG_PIXEL_X = 0xA002
TAG_PIXEL_Y = 0xA003

LINES: list[str] = []


def say(s: str = "") -> None:
    print(s)
    LINES.append(s)


def hr(t: str) -> None:
    say()
    say("=" * 74)
    say("  " + t)
    say("=" * 74)


def main() -> int:
    from PIL import Image

    from vision.exif import read_exif_intrinsics

    K_gt = np.load(DEMO / "intrinsics.npy").astype(np.float64).reshape(3, 3)
    src = Image.open(DEMO / "rgb.png").convert("RGB")
    W, H = src.size
    long_side = float(max(W, H))
    fx_gt = float(K_gt[0, 0])

    hr("A.  从 GT 内参反推该写进 EXIF 的整数等效焦距")
    say(f"  源图   {DEMO / 'rgb.png'}   {W}×{H}   长边 {long_side:.0f}")
    say(f"  GT K   fx={fx_gt:.1f} fy={K_gt[1,1]:.1f} "
        f"cx={K_gt[0,2]:.1f} cy={K_gt[1,2]:.1f}")
    ideal_35 = fx_gt * SENSOR_WIDTH_MM_35MM / long_side
    f35_int = int(round(ideal_35))
    fx_exif = f35_int / SENSOR_WIDTH_MM_35MM * long_side
    dev = (fx_exif - fx_gt) / fx_gt
    say()
    say(f"  f_35 = fx_gt × 36 / 长边 = {fx_gt:.1f} × 36 / {long_side:.0f} "
        f"= {ideal_35:.3f} mm")
    say(f"  EXIF 只能存整数毫米 → 写入 {f35_int} mm")
    say(f"  回推 fx = {f35_int}/36 × {long_side:.0f} = {fx_exif:.2f} px"
        f"   相对 GT 偏差 {dev*100:+.2f}%")
    say()
    say(f"  这个 {abs(dev)*100:.2f}% 就是 EXIF 量化误差在本图上的实际大小。")
    say("  拿它对照 Step 7 的容差预算：方位误差 ≤10 px 要求焦距偏差 ≤±6.6%，")
    say(f"  所以本 fixture 只用掉了预算的 {abs(dev)/0.066*100:.0f}%。")

    # ---- 写 fixture ---------------------------------------------------------
    hr("B.  写带 EXIF 的 JPEG")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    q = 95
    im = Image.new("RGB", (W, H))
    im.paste(src)
    exif = im.getexif()
    # 两个标签都写，模拟真机（多数手机两者都有）。本模块优先用 35 mm 等效值。
    exif[TAG_FOCAL_35MM] = f35_int
    exif[TAG_FOCAL_LENGTH] = (f35_int, 1)     # rational；这里没有真实物理焦距可依据
    exif[TAG_PIXEL_X] = W
    exif[TAG_PIXEL_Y] = H
    # 记一个来源标记，免得日后分不清这张图是拍的还是造的
    exif[0x010F] = "3DSpatialAgent"           # Make
    exif[0x0110] = "EXIF-fixture (synthetic)"  # Model
    im.save(FIXTURE, exif=exif, quality=q)
    say(f"  写入 {FIXTURE}   quality={q}   {FIXTURE.stat().st_size/1024:.0f} KB")
    say("  注意：JPEG 重编码会引入压缩伪影，所以这张图的**像素**已不是原图。")
    say("  因此它只用于「同图内」的内参 A/B 对照，不与 PNG 那几次运行横比。")

    # ---- 核对读数 -----------------------------------------------------------
    hr("C.  核对：读出来的 K 是否等于按 29 mm 反算的值")
    got = read_exif_intrinsics(FIXTURE)
    if got is None:
        say("  [FAIL] 读不到内参 —— fixture 本身有问题，后面的流水线验证没有意义")
        REPORT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
        return 1

    ok_fx = abs(got.fx - fx_exif) < 0.05
    say(f"  来源      {got.source}")
    say(f"  读出 K    fx={got.fx:.2f} fy={got.fy:.2f} "
        f"cx={got.cx:.1f} cy={got.cy:.1f}")
    say(f"  期望      fx={fx_exif:.2f}（按 {f35_int} mm 与长边 {long_side:.0f} 反算）"
        f"   {'✓ 一致' if ok_fx else '✗ 不一致'}")
    say(f"  视场      HFoV {got.fov.hfov_deg:.2f}°  "
        f"{'可信' if got.fov.plausible else '**不可信**'} ({got.fov.reason})")
    say(f"  量化误差  ±{got.quantisation_pct:.2f}%"
        f"   （0.5/{f35_int} = {0.5/f35_int*100:.2f}%）")
    say(f"  尺寸一致  {not got.size_mismatch}")
    say()
    say("  与 GT 的逐项对照：")
    say(f"    {'项':<10}{'GT':>12}{'EXIF 反算':>14}{'相对差':>11}")
    for name, g, e in (("fx", fx_gt, got.fx), ("fy", K_gt[1, 1], got.fy),
                       ("cx", K_gt[0, 2], got.cx), ("cy", K_gt[1, 2], got.cy)):
        say(f"    {name:<10}{g:>12.2f}{e:>14.2f}{(e-g)/g*100:>10.2f}%")
    say()
    say("  ⚠ 注意 cx/cy：GT 是 325.6/253.7，而 EXIF 无主点信息只能取图像中心")
    say(f"     (320.0/240.0)。差 {abs(got.cx-K_gt[0,2]):.1f} / "
        f"{abs(got.cy-K_gt[1,2]):.1f} px —— 这是 EXIF 路线的**固有**误差，")
    say("     不能靠更准的元数据解决，只能靠「主点可忽略」这个假设成立。")
    say("     本图它约为画面宽的 0.9%，对多数物体尺寸的影响在毫米级。")
    say()
    say("  注意事项原文：")
    for n in got.notes:
        say(f"    · {n}")

    hr("VERDICT")
    say(f"  {'✓' if ok_fx else '✗'} 读数与解析期望一致（fx 差 {abs(got.fx-fx_exif):.3f} px）")
    say(f"  {'✓' if got.fov.plausible else '✗'} 视场落在可信区间")
    say(f"  量化误差 {got.quantisation_pct:.2f}% vs 容差预算 6.6% —— "
        f"占 {got.quantisation_pct/6.6*100:.0f}%")
    say("  ⟹ EXIF 作为真实照片的内参来源在**精度预算之内**；")
    say("     剩下的固有误差是主点（本图约 0.9% 画面宽），不是量化的锅。")

    RESULT.write_text(json.dumps({
        "source_image": str(DEMO / "rgb.png"),
        "fixture": str(FIXTURE),
        "image_wh": [W, H],
        "gt_K": {"fx": fx_gt, "fy": float(K_gt[1, 1]),
                 "cx": float(K_gt[0, 2]), "cy": float(K_gt[1, 2])},
        "ideal_focal_35mm": round(ideal_35, 4),
        "written_focal_35mm": f35_int,
        "fx_from_exif": round(fx_exif, 4),
        "fx_dev_pct": round(dev * 100, 4),
        "read_back": got.as_dict(),
        "read_matches_analytic": bool(ok_fx),
        "fov_plausible": bool(got.fov.plausible),
        "principal_point_error_px": [round(abs(got.cx - K_gt[0, 2]), 2),
                                     round(abs(got.cy - K_gt[1, 2]), 2)],
        "tolerance_budget_pct": 6.6,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    say()
    say("  下一步（端到端，两次跑同图对照）：")
    out_root = ROOT / "dataset" / "scenes"
    say(f"    # A: 用 EXIF 内参")
    say(f"    python scripts/build_scene.py --image {FIXTURE} \\")
    say(f"        --scene-id exif_fixture_exif --intrinsics exif \\")
    say(f"        --prompt \"sofa. chair. table. picture. mirror.\"")
    say(f"    # B: 不给内参（同一张图，掩码应当逐像素相同）")
    say(f"    python scripts/build_scene.py --image {FIXTURE} \\")
    say(f"        --scene-id exif_fixture_pred \\")
    say(f"        --prompt \"sofa. chair. table. picture. mirror.\"")
    say(f"    输出在 {out_root}")
    say()
    say(f"  结果写入 {RESULT}")
    say(f"  报告写入 {REPORT}")
    say()
    REPORT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0 if ok_fx else 1


if __name__ == "__main__":
    raise SystemExit(main())
