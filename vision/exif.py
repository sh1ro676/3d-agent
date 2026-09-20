"""从 EXIF 读等效焦距 → 内参。**真实照片在无 GT 内参时的唯一来源。**

## 为什么必须有这个模块

Phase 0 Step 6/7 把「内参来源」测成了一条硬结论（`phase0/probe_depth_gt_report.txt`、
`probe_k_sweep_report.txt`）：同一张图、同一份权重，只因为让模型猜相机还是把相机
告诉它，**横向（方位）误差差 118 倍**（306.3 px vs 2.6 px）。

但那条结论建立在 `assets/demo/intrinsics.npy` 上 —— 那是数据集自带的 GT 相机。
真实照片没有这个文件。于是「有没有内参」这条杠杆要落地，就只剩三个来源：

   ① 标定物 / 棋盘格标定 —— 需要额外拍摄流程，课程作业场景不现实
   ② 深度数据自带相机（Omni3D-Bench 等）—— 只在用数据集时有
   ③ **EXIF** —— 手机/相机拍的照片里就有，零成本

所以这个模块不是「顺手加的便利功能」，它是让 §21 那条杠杆在真实照片上成立的
**必要条件**。缺了它，全部结论就只能停在 demo 图上。

## 三条实现上的判断（每条都有具体理由）

### ① 用长边 `max(W, H)` 换算，而不是先看 Orientation

`FocalLengthIn35mmFilm` 定义的是**对角视场等效**，其中 36 mm 指的是 35 mm 画幅的
**长边**，而传感器长轴在图像里对应的就是**图像的长边**。所以：

    f_px = focal_35mm / 36.0 × max(W, H)

这样写有一个附带好处：**结果对旋转不变**。竖拍照片无论 Orientation 写 6 还是 8，
`max(W, H)` 都不变，于是不需要先解析 Orientation、再决定要不要交换宽高 ——
那条路上错一步（把 5/6/7/8 四个值记错一个）就会静默地按宽高比整体缩放 fx，
而输出的数字看上去仍然合理。**能消掉一个出错可能就不要留着它。**

### ② 优先 `FocalLengthIn35mmFilm`，只有 `FocalLength` 时降级并告警

`FocalLengthIn35mmFilm` 已经把传感器尺寸折算进去了，拿到它就能直接算 f_px。
只有 `FocalLength`（真实毫米）时，必须知道传感器宽度。默认取 36 mm 是**全画幅**
假设 —— 对 APS-C 差 1.5 倍，对手机（传感器宽约 5.6 mm）差**约 6.4 倍**。
所以这条路径写出 `assumed_sensor=True`，并在 `notes` 里说清后果。

顺带一个**曾经写错、已被实测证伪**的推论（`phase0/probe_cross_source.py` A 段）：

    ❌ 旧写法：「这种粗暴假设**骗不过** `check_fov`」

    实测 5 台真实相机（Panasonic DMC-L10 / Olympus E-P3 / Ricoh GR /
    Sigma DP3 Merrill / Sony DSC-RX1R）：

      情形                                         出现次数   后果
      有 35mm 等效值 + 视场在窗口内                 2/5      正确、无告警  ✓
      有 35mm 等效值 + 焦距是长焦（视场 < 30°）      2/5      **误报**（K 是对的，却告警）
      缺 35mm 等效值 → 全画幅假设，crop=2.0          1/5      **漏报**（K 错 2 倍，却不告警）

    漏报那一例：Olympus E-P3 的 `FocalLength=15 mm`、真实画幅是 MFT（crop 2.0），
    真实 f_35 = 30 mm（HFoV 61.9°）；按全画幅假设得 f_35 = 15 mm（HFoV **100.4°**）——
    **仍落在 30–110° 窗口内**，`check_fov` 判 `plausible=True`。fx 错了整整 2 倍，
    而下游一个告警都收不到。

    根因是窗口范围：30–110° 只等价于 f_35 ∈ [12.6, 67.2] mm。全画幅假设下
    `f_35_assumed = FocalLength`，真值 `= FocalLength × crop`；误差要被兜住，
    必须让 assumed 值**逃出**窗口，也就是 `FocalLength` 小于 12.6 mm ——
    只有手机超广（crop≈5–7）够格。**APS-C（1.5×）/ MFT（2×）/ 1 吋（2.7×）
    的常见焦距全都在窗口内，一律逃不掉。**

    ⟹ 结论：`assumed_sensor=True` 是**不可由 check_fov 担保**的一档，
       必须原样透传到调用方（`build_scene.py` 已据此在 `assumed_sensor` 时无条件告警），
       或让用户显式给出传感器宽度（`sensor_width_mm=`）。
       同时 `check_fov` 的语义要收窄成「**典型照片的合理性先验**」，
       而不是「内参正确性的校验器」—— 长焦照片的误报就是这条边界的体现。

### ③ 副产物：EXIF 自带量化误差，必须一起记录

`FocalLengthIn35mmFilm` 是整数毫米。四舍五入到整数意味着半步误差 0.5 mm，对 26 mm
是 ±1.9%（按整步算是 ±3.8%）。Step 7 的容差预算给出的门槛是：**方位误差控制在
10 px 以内要求焦距偏差 ≤ ±6.6%**（3 m 处约 58 mm）。两者一比：EXIF 的量化误差
**在预算之内**，但已用掉约三分之一，不是可以被忽略的量。所以它作为
`quantisation_rel` 一起返回，随 K 进 `build_meta`。

## 做不到的事（写清楚，免得下游误以为它是真值）

- **没有主点。** EXIF 不记录 cx/cy，只能用图像中心。真实主点常偏 1–3%，本模块无法修正。
- **裁剪过的图：fx 仍对，但主点一定错。** 等效焦距编码的是**视场角**，所以
  ① **缩放**（整体 resize）不改变视场，`f_px = f_35/36 × 长边` 用的又是**当前**长边
  ⟹ fx 与主点（=当前图像中心）**都仍然正确**；
  ② **裁剪**只截取一块，视场变小，但 fx 按当前长边算**依然正确**（像素焦距与
  分辨率无关）；**错的是主点** —— 光轴不再落在裁剪图的几何中心，偏移量等于
  裁剪窗相对原图中心的位移，符号与大小都不可知。
  ⟹ `size_mismatch=True` 的准确含义是「**主点不可信**」，而不是「整份 K 失效」。
  `phase0/probe_cross_source.py` 用真实相机素材（80×80 裁切件、EXIF 记录 3648×2736）
  在 5/5 文件上触发了这条分支，证明它不是理论情形。
  本模块只能报警，无法判断是缩放还是裁剪 —— 但下游至少该知道「方向级精度在此失效」。
- **没有畸变、没有像素长宽比。** 广角镜头的桶形畸变会让边缘横向几何系统性偏，
  EXIF 不含畸变系数。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vision.geometry import FovCheck, check_fov, intrinsics_matrix

__all__ = [
    "ExifIntrinsics",
    "read_exif_intrinsics",
    "SENSOR_WIDTH_MM_35MM",
]

#: 35 mm 画幅长边。`FocalLengthIn35mmFilm` 的参考尺度就是它。
SENSOR_WIDTH_MM_35MM = 36.0

# --- EXIF 标签编号（写死数字而不是字符串名：PIL 的标签表随版本变动）----------
TAG_ORIENTATION = 0x0112       # 仅用于文档说明；本模块刻意不依赖它（见 ①）
TAG_EXIF_IFD = 0x8769          # Exif SubIFD 指针
TAG_FOCAL_LENGTH = 0x920A      # FocalLength，rational，真实毫米
TAG_FOCAL_35MM = 0xA405        # FocalLengthIn35mmFilm，short，整数毫米
TAG_PIXEL_X = 0xA002            # PixelXDimension
TAG_PIXEL_Y = 0xA003            # PixelYDimension


@dataclass(frozen=True, slots=True)
class ExifIntrinsics:
    """一份从 EXIF 推出的内参，连同它的来源与不确定度。

    `K` 是唯一给下游用的东西；其余字段是**为了让 K 不被误信**才一起传来的。
    这个结构的存在本身就是本项目的一条主张：一个米制数字如果不带它的来源与
    误差量级，就不该被下游当成真值使用。
    """

    K: np.ndarray
    source: str
    fx: float
    fy: float
    cx: float
    cy: float
    image_hw: tuple[int, int]
    focal_mm: float | None
    focal_35mm_mm: float | None
    sensor_width_mm: float
    assumed_sensor: bool
    quantisation_rel: float | None
    fov: FovCheck
    size_mismatch: bool
    notes: tuple[str, ...] = ()

    @property
    def fx_fy_cx_cy(self) -> tuple[float, float, float, float]:
        """`BuildConfig.known_intrinsics` 要的 4 元组。"""
        return (self.fx, self.fy, self.cx, self.cy)

    @property
    def quantisation_pct(self) -> float | None:
        return None if self.quantisation_rel is None else self.quantisation_rel * 100.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "fx": round(self.fx, 2),
            "fy": round(self.fy, 2),
            "cx": round(self.cx, 2),
            "cy": round(self.cy, 2),
            "image_hw": list(self.image_hw),
            "focal_mm": self.focal_mm,
            "focal_35mm_mm": self.focal_35mm_mm,
            "sensor_width_mm": self.sensor_width_mm,
            "assumed_sensor": self.assumed_sensor,
            "quantisation_rel": (None if self.quantisation_rel is None
                                 else round(self.quantisation_rel, 5)),
            "size_mismatch": self.size_mismatch,
            "fov": self.fov.as_dict(),
            "notes": list(self.notes),
        }

    def describe(self) -> str:
        """一行人类可读的说明，供 CLI 与 build_log 使用。"""
        parts = [f"EXIF → fx={self.fx:.1f} fy={self.fy:.1f} "
                 f"cx={self.cx:.1f} cy={self.cy:.1f}"]
        if self.focal_35mm_mm is not None:
            parts.append(f"等效焦距 {self.focal_35mm_mm:g} mm")
        elif self.focal_mm is not None:
            parts.append(f"焦距 {self.focal_mm:g} mm（传感器宽度按 "
                         f"{self.sensor_width_mm:g} mm 假设）")
        parts.append(f"HFoV {self.fov.hfov_deg:.1f}°")
        if self.quantisation_pct is not None:
            parts.append(f"量化误差 ±{self.quantisation_pct:.1f}%")
        if not self.fov.plausible:
            parts.append(f"[视场不可信:{self.fov.reason}]")
        return "  ".join(parts)


def _lookup(exif: Any, tag: int) -> Any:
    """在 IFD0 与 Exif SubIFD 两处找同一个标签。

    真机文件里 `FocalLengthIn35mmFilm` 的位置并不统一：多数相机放进 Exif SubIFD
    （0x8769），也有写在 IFD0 的，还有（部分安卓导出、微信转存）两处都有但值不同。
    **两处都查**比赌一处安全。顺序上先 IFD0：那份通常是导出方写入的，
    与当前像素尺寸更可能一致。
    """
    if exif is None:
        return None
    try:
        v = exif.get(tag)
    except Exception:               # 某些实现遇到未知标签会抛
        v = None
    if v not in (None, "", 0):
        return v
    try:
        sub = exif.get_ifd(TAG_EXIF_IFD)
    except Exception:
        return None
    if not sub:
        return None
    try:
        v = sub.get(tag)
    except Exception:
        return None
    return v if v not in (None, "", 0) else None


def _as_float(v: Any) -> float | None:
    """EXIF 数值 → 正 float。rational / IFDRational / "num/den" / 元组都要对付。

    对 0 与异常返回 `None` 而不是抛：EXIF 是**外部输入**，一张图里某个标签坏掉
    不该让整条流水线失败 —— 它应当退化成「没有内参」，由调用方按那条路径处理。
    """
    if v is None:
        return None
    f: float | None = None
    if isinstance(v, str) and "/" in v:
        num, _, den = v.partition("/")
        try:
            f = float(num) / float(den)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    else:
        try:
            f = float(v)
        except (TypeError, ValueError):
            try:                    # 有些写入方给 (分子, 分母)
                f = float(v[0]) / float(v[1])
            except Exception:
                return None
    if f is None or not np.isfinite(f) or f <= 0.0:
        return None
    return f


def _as_int(v: Any) -> int | None:
    f = _as_float(v)
    return None if f is None else int(round(f))


def _resolve_source(source: Any) -> tuple[Any, int, int, bool]:
    """→ `(句柄, W, H, 是否需要我们关掉它)`。"""
    if isinstance(source, (str, Path)):
        from PIL import Image

        im = Image.open(source)
        im.load()
        return im, int(im.size[0]), int(im.size[1]), True
    W, H = source.size
    return source, int(W), int(H), False


def read_exif_intrinsics(
    source: Any,
    *,
    image_size: tuple[int, int] | None = None,
    sensor_width_mm: float = SENSOR_WIDTH_MM_35MM,
) -> ExifIntrinsics | None:
    """`source` 为图片路径或已打开的 `PIL.Image`；读不到可用焦距时返回 `None`。

    `image_size` 显式给出 `(H, W)` 时以它为准 —— 用于「图片被下游改过尺寸、但
    EXIF 仍是原始尺寸」的情形：`f_px` 必须按**实际参与推理的那张图**的像素数算，
    否则 f_px 与像素网格不匹配，产生的错误与内参错一样隐蔽。

    **返回 `None` 而不是抛异常**是刻意的：EXIF 缺失太常见（截图、聊天软件转存、
    部分 PNG），它属于正常分支而非错误。调用方拿到 `None` 应当退到
    「模型预测 + `check_fov` 告警」那条路，并在记录里写明降级原因。
    """
    im, W_img, H_img, must_close = _resolve_source(source)
    try:
        if image_size:
            H, W = int(image_size[0]), int(image_size[1])
        else:
            W, H = W_img, H_img

        exif = im.getexif()
        f_35 = _as_float(_lookup(exif, TAG_FOCAL_35MM))
        f_mm = _as_float(_lookup(exif, TAG_FOCAL_LENGTH))
        if f_35 is None and f_mm is None:
            return None

        notes: list[str] = []
        long_side = float(max(W, H))

        if f_35 is not None:
            f_px = f_35 / SENSOR_WIDTH_MM_35MM * long_side
            src = "exif:35mm"
            assumed = False
            quant: float | None = 0.5 / f_35
            notes.append(
                f"由 FocalLengthIn35mmFilm={f_35:g} mm 换算："
                f"f_px = {f_35:g}/36 × max(W,H)={long_side:.0f} = {f_px:.1f}"
            )
        else:
            f_px = float(f_mm) / float(sensor_width_mm) * long_side
            src = "exif:focal+sensor_assumed"
            assumed = True
            quant = None
            notes.append(
                f"只有 FocalLength={f_mm:g} mm，无 35 mm 等效值；传感器宽度按 "
                f"{sensor_width_mm:g} mm 假设（**全画幅假设**）。若真实画幅是 APS-C（1.5×）"
                "或 MFT（2×），fx 会偏小 1.5–2 倍；若是手机（约 5.6 mm）则偏小约 6.4 倍。"
                "⚠ **本分支不可由 check_fov 担保**：实测（`phase0/probe_cross_source.py` A 段）"
                "MFT 的 15 mm 镜按全画幅假设算出 HFoV 100.4°，真值 61.9°，仍落在 30–110° "
                "窗口内而不告警。请显式传入 sensor_width_mm，或把此 K 当作 LOW_CONFIDENCE。"
            )

        if abs(long_side - float(W)) > 0.5:
            notes.append(
                f"竖构图：长边 {long_side:.0f} 取自 H（图像 {W}×{H}）。用长边换算"
                "使结果对旋转不变，因此**不**按 Orientation 交换宽高。"
            )

        # EXIF 自带的像素尺寸 vs 当前尺寸：不等 ⟹ 图被缩放或裁剪过。
        # 缩放**不影响**等效焦距（视场角不变，而公式里用的就是当前长边）；
        # **裁剪会**改变视场，此时这个内参已经失效。只能报警，不能修正。
        ex_px, ex_py = _as_int(_lookup(exif, TAG_PIXEL_X)), _as_int(_lookup(exif, TAG_PIXEL_Y))
        size_mismatch = False
        if ex_px and ex_py:
            ref_long = max(ex_px, ex_py)
            if abs(ref_long - long_side) / ref_long > 0.01:
                size_mismatch = True
                notes.append(
                    f"⚠ EXIF 记录的像素尺寸 {ex_px}×{ex_py} 与当前 {W}×{H} 不一致 "
                    f"（长边 {ref_long} vs {long_side:.0f}）⟹ 图片被缩放或裁剪过。"
                    "缩放不影响等效焦距，**但裁剪会改变视场** —— 若为裁剪，"
                    "下面的 K 已失效。本模块只能报警，无法自行判断是哪一种。"
                )

        notes.append("主点无 EXIF 来源，取图像中心；真实主点常偏 1–3%。")
        notes.append("EXIF 不含畸变系数；广角镜头的桶形畸变会让边缘横向几何系统性偏。")
        if quant is not None:
            notes.append(
                f"FocalLengthIn35mmFilm 为整数毫米 ⟹ 四舍五入半步 ±{quant*100:.1f}%"
                f"（整步 ±{quant*200:.1f}%）。Step 7 容差预算：方位误差 ≤10 px 要求"
                "焦距偏差 ≤±6.6% —— EXIF 在预算内，但用掉了约三分之一。"
            )

        K = intrinsics_matrix(f_px, f_px, W / 2.0, H / 2.0)
        fov = check_fov(K, (H, W))
        if not fov.plausible:
            notes.append(
                f"check_fov 判定不可信（HFoV {fov.hfov_deg:.1f}°，reason={fov.reason}）"
                " —— 很可能是传感器宽度假设错了。"
            )

        return ExifIntrinsics(
            K=K,
            source=src,
            fx=float(f_px),
            fy=float(f_px),
            cx=W / 2.0,
            cy=H / 2.0,
            image_hw=(H, W),
            focal_mm=f_mm,
            focal_35mm_mm=f_35,
            sensor_width_mm=float(sensor_width_mm),
            assumed_sensor=assumed,
            quantisation_rel=quant,
            fov=fov,
            size_mismatch=size_mismatch,
            notes=tuple(notes),
        )
    finally:
        if must_close:
            im.close()
