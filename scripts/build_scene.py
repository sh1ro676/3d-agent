#!/usr/bin/env python
r"""把一张图建成场景图并落盘 —— Phase 1c 的真实跑通入口。

    D:\3D_Spatial_Agent\venvs\vision\Scripts\python.exe scripts\build_scene.py `
        --image D:\3D_Spatial_Agent\vendor\UniDepth\assets\demo\rgb.png `
        --scene-id living_room `
        --prompt "sofa. chair. table. picture. mirror."

产出（默认落在 `dataset/scenes/<scene_id>/`）：

    scene.json    带信封的场景图（`_format` + `_written_at` + `scene`）
    masks/*.png   每个物体的 1-bit 掩码，可直接用看图软件打开核对
    points.npy    整图点云 (3, H, W) + points_meta.json —— 点云级算法的输入
    build_log.txt 人类可读的构建记录（警告 + 耗时 + 显存 + 物体清单）

**为什么要点云**（2026-09-19 加）：`scene.json` 里只有 `centroid_3d` /
`extent_3d` / `bbox_3d` 这几个**标量摘要**，点云本身此前算完即丢。
于是「物体朝向」「真实体积」「它靠在哪个面上」这类问题——
以及依赖它们的点云算法（有向包围盒、主方向、平面拟合、聚类补漏检）——
全都没有下手的地方。落盘之后，某个物体的点云可以由「整图点云 + 掩码」
重建（`scene_graph.pointcloud`），所以**不存每物体点云**：存两份等于给
同一个事实两个副本，而副本一定会漂移。体积代价见 `--points-dtype`。

这个脚本是 Phase 0 三个探针（`probe3d.py` / `probe_sam2.py`）的**合流点**：
探针各自验证了一个模型，这里第一次把三者串成一条流水线，
并且第一次产出真正可被下游消费的 `scene_graph.json`。

两条 Phase 0 的硬约束在这里生效（都不是「优化」，是正确性）：
  • 质心取 **SAM2 掩码**内点云的中位数，不是检测框内的中位数（差均值 83 mm / 最大 208 mm）
  • SAM2 **一次调用带全部框**（9 框：200 ms vs 1512 ms，快 7.57×）
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: `--intrinsics auto` 会在图片同目录按这个顺序找文件。
_AUTO_INTRINSIC_NAMES = ("intrinsics.npy", "camera.npy", "K.npy")

#: 「建图要多久」这个问题的**口径声明**。进 `build_meta["timing"]["scope"]`。
#:
#: ⚠ 为什么口径必须**落盘**而不是只写在文档里：这几个数字**不是同一个东西**，
#: 但它们看上去都叫「建图耗时」。实测 `living_room`：builder 六段合计 ≈3.41 s、
#: 三个模型加载 ≈12.58 s、总数 16 s 级 —— 拿 3.41 s 去回答「建图要多久」会**低报约 5 倍**，
#: 而报告读者从数字本身看不出这一点。（2026-09-22 补。）
TIMING_SCOPE = (
    "stages_ms = build_scene_graph 内部六段细分（热态：模型已在显存；不含落盘）。"
    " model_load_ms = 本进程首次加载三个模型的墙钟。"
    " build_wall_ms = build_scene_graph 这一次调用的整段墙钟（**含**上面那六段）。"
    " save_ms = masks 与 points 落盘的墙钟（**不含 scene.json 自身的写入** ——"
    " 那段耗时记在 build_log.txt，因为它要塞进 scene.json、逻辑上不可能闭包）。"
    " total_wall_ms = model_load_ms + build_wall_ms + save_ms。"
    " ⚠ 同一进程内建第 N 张图时 model_load_ms 从第 2 张起为 0 ⟹ total_wall_ms"
    " 对不同张数不能直接平均，必须先说明「第几张、进程冷热」。"
)


def timing_block(
    *,
    model_load_ms: float,
    build_wall_ms: float,
    save_ms: float | None,
) -> dict[str, object]:
    """组装 `build_meta["timing"]` —— **纯函数，零 GPU 可单测**。

    ⚠ `save_ms=None`（`--no-save`）时 `total_wall_ms` 也必须是 `None`，**不能是 0**：
    总量缺了一块就是不知道，把「不知道」写成 0 会让「这一轮特别快」变成一个**假结论**
    （与 `MEMORY-DETAIL` 的 `absent ≠ zero` 同一条口径：`None`＝没测到，`0`＝测到 0）。

    ⚠ builder 的六段细分**不在这里复制**：它们仍住在 `build_meta["timings_ms"]`。
    同一个事实两个副本一定会漂移，所以这里只放三项聚合数 + 口径。
    """
    load = round(float(model_load_ms), 1)
    build = round(float(build_wall_ms), 1)
    save = None if save_ms is None else round(float(save_ms), 1)
    total = None if save is None else round(load + build + save, 1)
    return {
        "scope": TIMING_SCOPE,
        "model_load_ms": load,
        "build_wall_ms": build,
        "save_ms": save,
        "total_wall_ms": total,
    }


def _intrinsics_from_exif(
    image_path: Path,
    image_size: tuple[int, int] | None,
    focal_35mm_mm: float | None = None,
) -> tuple[tuple[float, float, float, float] | None, str, dict]:
    """EXIF → 内参。读不到就返回 `(None, 降级说明, {})` —— 这是**正常分支不是错误**。

    截图、聊天软件转存、部分 PNG 都没有 EXIF。真实照片里这是常态，所以它必须
    走「降级 + 写清原因」而不是抛异常：一个与几何无关的原因（缺元数据）不该
    让整条流水线停下来，但它也绝不能被静默吞掉 —— 因为降级之后横向尺度会
    差 118 倍（Step 7 实测），那是必须在记录里看得见的事。

    2026-09-23 实测把「常态」升级成了「必丢」：两张 iPhone 照片经微信送达后文件里
    **一个 APP1 段都没有**（`reports/real_photo_exif_probe.txt`，阳性对照 3/3 命中）。
    于是多了一条 `focal_35mm_mm` 入口：让用户给出他真正查得到的那个数（等效焦距 mm），
    走**同一份换算**（见 `vision/exif.py` 模块 docstring ④）。**公式只有一份**是硬约束。
    """
    from vision.exif import read_exif_intrinsics

    got = read_exif_intrinsics(image_path, image_size=image_size,
                               focal_35mm_mm=focal_35mm_mm)
    if got is None:
        return None, ("EXIF 里没有可用的等效焦距（截图 / 转存 / 无元数据都会这样）"
                      " —— 退化为模型预测值"), {"exif": None}

    note = got.describe()
    if not got.fov.plausible:
        note += "   ⚠ 视场不可信 —— 详见 build_log 与下方 notes"
    if got.assumed_sensor:
        # ★ 不能靠 check_fov 兜住这一档：实测（phase0/probe_cross_source.py A 段）
        #   MFT 的 15 mm 镜按全画幅假设算出 HFoV 100.4°（真值 61.9°），仍落在
        #   30–110° 窗口内、plausible=True ⟹ fx 错 2 倍却零告警。
        #   所以这里**无条件**提示，不依赖 fov.plausible。
        note += ("   ⚠ 传感器宽度是假设值（无 35mm 等效焦距）—— fx 可能偏小 1.5–6 倍，"
                 "且**不能**靠视场检查发现；请传 `sensor_width_mm` 或改用 `--intrinsics <npy>`")
    if got.size_mismatch:
        # 这条的准确含义是「主点不可信」：缩放不改变视场但裁剪会（见 vision/exif.py）。
        note += "   ⚠ EXIF 像素尺寸与当前不一致 —— 若是裁剪，焦距仍对但**主点已错位**"
    return got.fx_fy_cx_cy, note, {"exif": got.as_dict()}


def load_known_intrinsics(
    spec: str | None,
    image_path: Path,
    image_size: tuple[int, int] | None = None,
) -> tuple[tuple[float, float, float, float] | None, str, dict]:
    """把 `--intrinsics` 的几种写法统一成 `(fx, fy, cx, cy)`。

    返回 `(4 元组或 None, 人类可读的来源说明, 供 build_log 的附加信息)`。
    第二项会被打印，第三项写进 `build_log.txt`。

    内参来源必须始终可见：它是横向尺度的总开关（Step 6：三维误差中位数
    1.943 m → 0.267 m；Step 7：横向误差 306.3 px → 2.6 px）。一个「没传内参」和
    一个「传了内参」的 scene.json 如果只差几个数字，事后根本没法判断哪个更可信。

    `exif` 这一路是给**真实照片**用的：没有 GT 文件时，EXIF 的
    `FocalLengthIn35mmFilm` 是唯一可得的内参来源。它带 ±2–4% 的量化误差，
    落在 Step 7 算出的容差预算（±6.6%，对应方位误差 10 px）之内，但用掉了约
    三分之一 —— 所以来源、量化误差、视场判定必须随 K 一起落到记录里。

    `f35:<mm>` 是给「**照片没有 EXIF、但用户知道自己用的哪颗镜头**」那一档
    （如 `f35:24`）：实测微信传图必丢 EXIF，而 `exif` 那一路会静默退化成模型
    预测，所以需要一个普通人真的填得出来的兜底。
    """
    if not spec:
        return None, "未提供 —— 将用模型预测值，并做视场合理性检查", {}

    if spec == "exif":
        return _intrinsics_from_exif(image_path, image_size)

    if spec.startswith("f35:"):
        # 「用户知道自己用的哪颗镜头」这一档。为什么需要它：实测两张 iPhone 照片
        # 经微信送达后 EXIF **必丢**，而原来的兜底是让用户填 4 个像素焦距 ——
        # 手机里根本查不到，等于兜底是空的。
        # 这里只做解析与校验；换算一律走 `_intrinsics_from_exif`（同一份公式）。
        raw = spec[4:].strip()
        try:
            mm = float(raw)
        except ValueError:
            raise SystemExit(
                f"--intrinsics f35: 后面要跟一个毫米数（等效焦距），例如 f35:24 "
                f"—— 收到 {raw!r}"
            ) from None
        if not (math.isfinite(mm) and mm > 0.0):
            raise SystemExit(f"--intrinsics f35: 需要正的有限数，收到 {raw!r}")
        return _intrinsics_from_exif(image_path, image_size, focal_35mm_mm=mm)

    if spec != "auto":
        p = Path(spec)
        if p.is_file():
            K = np.load(p).astype(np.float64).reshape(3, 3)
            return (
                (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])),
                f"npy 文件 {p}",
                {"npy": str(p)},
            )
        parts: list[float] = []
        for chunk in spec.replace(";", ",").split(","):
            chunk = chunk.strip()
            if chunk:
                try:
                    parts.append(float(chunk))
                except ValueError:
                    raise SystemExit(
                        f"--intrinsics 既不是存在的文件，也不是 4 个数，"
                        f"也不是 exif/auto：{spec!r}"
                    ) from None
        if len(parts) != 4:
            raise SystemExit(
                f"--intrinsics 需要 4 个数 (fx,fy,cx,cy)，收到 {len(parts)} 个：{spec!r}"
            )
        return (parts[0], parts[1], parts[2], parts[3]), "命令行 4 元组", {}

    for name in _AUTO_INTRINSIC_NAMES:
        cand = image_path.parent / name
        if cand.is_file():
            K = np.load(cand).astype(np.float64).reshape(3, 3)
            return (
                (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])),
                f"auto → {cand}",
                {"auto_stage": "sidecar_npy", "auto_hit": str(cand)},
            )

    # auto 的第二优先：EXIF。顺序刻意是「数据集/标定文件 → EXIF → 模型预测」，
    # 因为越靠前越接近真值：npy 是标定值或数据集真值，EXIF 带 2–4% 量化误差，
    # 而模型预测在同一张图上的横向误差是前者的 118 倍。把三者当等价的候选，
    # 会让「恰好没找到 npy」这种无关紧要的事悄悄换掉整个横向尺度。
    k, note, meta = _intrinsics_from_exif(image_path, image_size)
    if k is not None:
        return k, f"auto → {note}", {**meta, "auto_stage": "exif"}
    return None, (
        f"auto 没找到 {' / '.join(_AUTO_INTRINSIC_NAMES)}，图片也没有可用的 EXIF 焦距"
        " —— 将用模型预测值，并做视场合理性检查（横向尺度可能整体偏大）"
    ), {"auto_stage": "predicted"}


def setup_hf_env(offline: bool = False) -> dict[str, str]:
    r"""把 HF 相关环境变量指向 D 盘的隔离缓存。

    本机实测：`huggingface.co` 直连超时，必须走 `hf-mirror.com`；
    `HF_HUB_DISABLE_XET=1` 把下载速度从 0.53 MB/s 提到 1.3 MB/s
    （xet 后端在国内链路上是慢的，不是快）。C 盘只剩 20 余 GB，
    所以缓存必须落在 `D:\3D_Spatial_Agent\.cache\huggingface`。

    只 `setdefault`、不覆盖：允许用临时环境变量做一次性实验，
    而不必改脚本 —— 消融对比时这个差别很实用。
    """
    env = {
        "HF_HOME": str(ROOT / ".cache" / "huggingface"),
        "HF_ENDPOINT": "https://hf-mirror.com",
        "HF_HUB_DISABLE_XET": "1",
    }
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    for k, v in env.items():
        os.environ.setdefault(k, v)
    return {k: os.environ[k] for k in env}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--scene-id", default=None,
                    help="默认取图片文件名（不含扩展名）")
    ap.add_argument("--intrinsics", default=None,
                    help="已知内参：(a) .npy 路径（3×3）(b) \"fx,fy,cx,cy\" "
                         "(c) exif = 从图片 EXIF 的等效焦距换算（真实照片用这个）"
                         "(d) f35:24 = 用户直接给出等效焦距 mm —— 照片经微信/社交软件"
                         "转存后 EXIF 必丢，这时用这个（demo 上传框里有对应输入）"
                         "(e) auto = 依次试 同目录 intrinsics.npy → EXIF → 模型预测。"
                         "强烈建议提供：实测横向误差 306.3 px → 2.6 px（118 倍）")
    ap.add_argument("--prompt", default=None,
                    help='小写标签、每个以句点结尾，如 "sofa. chair. table."')
    ap.add_argument("--out-root", default=str(ROOT / "dataset" / "scenes"))
    ap.add_argument("--relation-policy", default=None,
                    choices=["distance_plus_true", "all", "distance_only"])
    ap.add_argument("--min-points", type=int, default=None)
    ap.add_argument("--max-objects", type=int, default=None)
    ap.add_argument("--no-up-estimate", action="store_true",
                    help="跳过重力方向估计（直接用 -y）")
    ap.add_argument("--offline", action="store_true",
                    help="HF_HUB_OFFLINE=1，只用本地缓存的权重")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--no-points", action="store_true",
                    help="不落盘整图点云（改动前的旧行为）。"
                         "该场景届时无法用于任何点云级工具")
    ap.add_argument("--points-dtype", default="float32",
                    choices=["float32", "float16", "float64", "keep"],
                    help="点云保存精度。float32（默认）= 3.5 MB/场景（480×640），"
                         "对实测点云**逐位无损**（float64 里一半是假精度，见 "
                         "store.DEFAULT_POINTS_DTYPE）；float16 体积再减半，"
                         "但**真有损**（实测 1.95 mm @4.4 m）；"
                         "float64 / keep = 完全不转换，体积翻倍且不增加信息")
    ap.add_argument("--gdino", default=None)
    ap.add_argument("--sam2", default=None)
    ap.add_argument("--unidepth", default=None)
    args = ap.parse_args()

    env = setup_hf_env(args.offline)

    # 这两个 import 会拉起 torch + transformers，几秒钟，所以放在参数解析之后。
    from PIL import Image

    from scene_graph.builder import BuildConfig, build_scene_graph
    from scene_graph.store import (
        masks_dir,
        save_masks,
        save_points,
        save_scene,
        scene_dir,
    )
    from vision.grounding import DEFAULT_PROMPT
    from vision.registry import PerceptionStack

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"[FAIL] 图片不存在：{image_path}")
        return 2
    image = Image.open(image_path).convert("RGB")
    scene_id = args.scene_id or image_path.stem

    cfg = BuildConfig()
    overrides: dict = {}
    # image_size 按 **(H, W)** 传入，且刻意用**已打开的那张图**的尺寸：
    # f_px 必须按实际参与推理的像素网格算，否则 K 与点云不自洽。
    known_K, intrinsics_note, intrinsics_meta = load_known_intrinsics(
        args.intrinsics, image_path, image_size=(image.size[1], image.size[0])
    )
    if known_K:
        overrides["known_intrinsics"] = known_K
    if args.prompt:
        overrides["prompt"] = args.prompt
    if args.relation_policy:
        overrides["relation_policy"] = args.relation_policy
    if args.min_points is not None:
        overrides["min_points"] = args.min_points
    if args.max_objects is not None:
        overrides["max_objects"] = args.max_objects
    if args.no_up_estimate:
        overrides["estimate_up"] = False
    if overrides:
        cfg = cfg.replace(**overrides)

    print()
    print("=" * 74)
    print(f"  构建场景图  scene_id={scene_id}")
    print("=" * 74)
    print(f"  image   : {image_path}  {image.size[0]}x{image.size[1]}")
    print(f"  prompt  : {cfg.prompt!r}")
    print(f"  intrinsics: {intrinsics_note}")
    if known_K:
        print(f"              fx={known_K[0]:.1f} fy={known_K[1]:.1f} "
              f"cx={known_K[2]:.1f} cy={known_K[3]:.1f}")
    _exif_meta = intrinsics_meta.get("exif")
    if _exif_meta:
        # EXIF 那条路的注意事项（量化误差、传感器假设、尺寸不一致…）也要打出来。
        # 只打一个 fx 数字会让「EXIF 内参」看起来比它实际更可靠 ——
        # 它带 ±2–4% 的量化误差，正好压在容差预算的三分之一上。
        for _n in _exif_meta["notes"]:
            print(f"              · {_n}")
    print(f"  HF_HOME : {env['HF_HOME']}")
    print(f"  offline : {args.offline}")

    stack = PerceptionStack(
        gdino_id=args.gdino, sam2_id=args.sam2, unidepth_repo=args.unidepth
    )
    if not stack.is_cuda:
        print("  [warn] 没有 CUDA —— 推理会非常慢，数字不代表 4060 表现")

    # 显式预热：懒加载会把「首次用到某模型」的那一题多出几秒加载时间，
    # 污染延迟数据的尾部分布。预热之后再计时才干净。
    print()
    print("  加载模型 ...")
    t_load = time.perf_counter()
    stack.load_all()
    load_wall_s = time.perf_counter() - t_load
    for role, spec in stack.specs.items():
        print(f"    {role:<13} {spec.source:<6} {spec.load_s:6.2f} s  "
              f"{spec.n_params_m:7.1f} M  {spec.resident_mb:8.1f} MB")
    stats = stack.stats()
    print(f"    合计常驻 {stack.resident_mb():.0f} MB / "
          f"{stats['device_total_mib']:.0f} MiB "
          f"({stats['resident_pct']}%)   加载墙钟 {load_wall_s:.1f} s")

    t0 = time.perf_counter()
    res = build_scene_graph(
        image,
        perception=stack,
        scene_id=scene_id,
        image_id=image_path.name,
        config=cfg,
        mask_rel_prefix=f"{scene_id}/masks" if not args.no_save else None,
    )
    wall_s = time.perf_counter() - t0

    scene = res.scene
    # ★ 源图像路径必须进 **build_meta**，而不只是进人读的 `build_log.txt`。
    #   为什么这是必须的：角色②（视觉语义）要按物体的 bbox 去裁图，
    #   它得先知道「这张场景图是从哪张图建的」。而 `build_log.txt` 是给人看的
    #   过程记录、不是契约 —— 程序去解析它等于把一份日志当接口用。
    #   旧场景（本次改动之前建的）没有这个字段，`run_agent._find_image` 会
    #   退化到解析 build_log.txt 并**在产物里标明来源**，不会静默。
    scene = scene.model_copy(update={
        "build_meta": {**scene.build_meta, "image_path": str(image_path)},
    })
    tim = scene.build_meta["timings_ms"]

    print()
    print("-" * 74)
    print(f"  结果：{len(scene.nodes)} 个物体 / {len(scene.edges)} 条关系")
    print("-" * 74)
    print(f"  检测 {scene.build_meta['n_detections_raw']} → 保留 "
          f"{scene.build_meta['n_detections_kept']} → 建节点 {len(scene.nodes)}"
          f"（丢弃 {scene.build_meta['n_dropped']}，降级 {scene.build_meta['n_fallbacks']}）")
    print(f"  深度范围 {scene.build_meta['depth_range_m']} m"
          f"   点云网格 {scene.build_meta['grid_hw']}"
          f"   图像 {scene.build_meta['image_hw']}")
    fovmeta = scene.build_meta.get("fov", {})
    print(f"  内参来源 {scene.build_meta.get('intrinsics_source')}"
          f"   HFoV {fovmeta.get('hfov_deg')}°  VFoV {fovmeta.get('vfov_deg')}°"
          f"  可信={fovmeta.get('plausible')} ({fovmeta.get('reason')})")
    print(f"  重力方向 up_axis={scene.build_meta['up_axis']}"
          f"  tilt={scene.build_meta['up_axis_tilt_deg']}°"
          f"  reliable={scene.build_meta['up_axis_reliable']}"
          f"  ({scene.build_meta['up_axis_reason']})")
    print(f"  掩码占框比 均值 {scene.build_meta['mask_box_coverage_mean']}"
          f"  最低 {scene.build_meta['mask_box_coverage_min']}")
    print()
    print(f"  分段耗时 (ms): " + "  ".join(f"{k}={v:.0f}" for k, v in tim.items()))
    print(f"  构建墙钟 {wall_s:.2f} s   "
          f"（含模型加载共 {load_wall_s + wall_s:.2f} s）")

    print()
    print("  物体清单（相机系，米）")
    print(f"    {'id':<16}{'label':<12}{'score':>6}"
          f"{'X':>9}{'Y':>9}{'Z':>9}   {'尺寸 w×h×l':<20}{'点数':>8}  来源")
    for n in scene.nodes:
        e = n.extent_3d
        print(f"    {n.id:<16}{n.label:<12}{n.score:>6.2f}"
              f"{n.x:>9.3f}{n.y:>9.3f}{n.z:>9.3f}   "
              f"{e[0]:.2f}×{e[1]:.2f}×{e[2]:.2f}{'':<8}{n.n_points:>8}  {n.centroid_source}")

    if scene.edges:
        print()
        print("  关系边（前 20 条）")
        for e in scene.edges[:20]:
            val = f"{float(e.value):.3f}" if isinstance(e.value, float) else str(e.value)
            print(f"    {e.source:<16} --{e.relation:<10}--> {e.target:<16} {val}")
        if len(scene.edges) > 20:
            print(f"    ... 其余 {len(scene.edges) - 20} 条见 scene.json")

    if res.warnings:
        print()
        print("  警告")
        for w in res.warnings:
            print(f"    ! {w}")

    # ---- 落盘 ---------------------------------------------------------------
    if args.no_save:
        print()
        print("  --no-save：未写盘")
        return 0

    out_dir = scene_dir(scene_id, args.out_root)
    # ⚠ 顺序有讲究：先写 masks/points（受时），把它们量到的 `save_ms` 注入 build_meta，
    #   最后才写 scene.json。反过来写的话，scene.json 里就永远没有 save_ms。
    #   `save_masks` / `save_points` 各自 mkdir(parents=True)，所以不必先建目录。
    t_save = time.perf_counter()
    written = save_masks(res.masks, masks_dir(scene_id, args.out_root))
    pts_p: Path | None = None
    if args.no_points:
        pass
    elif res.points_chw is None:
        # 不该发生 —— DepthField 一定带点云。真发生了就说出来，别让
        # 「为什么点云级工具在这个场景上不可用」变成一件需要考古的事。
        print("  [warn] 感知栈没有返回点云 —— 跳过 points.npy")
    else:
        pts_p = save_points(
            res.points_chw, out_dir, meta=res.points_meta,
            # `keep` → None：保持原 dtype。默认路径上不做任何有损转换。
            dtype=None if args.points_dtype == "keep" else args.points_dtype,
        )
    save_ms = (time.perf_counter() - t_save) * 1000.0

    # ★ 把「建图要多久」的另外两块写进 `build_meta` —— 而不只是 `build_log.txt`。
    #   与上面 `image_path` 同一条理由（那段注释就是本项目的判据）：
    #   `build_log.txt` 是给人看的过程记录、**不是契约**，程序去解析它等于把日志当接口。
    #   在本次改动之前，`build_wall_s` / `model_load_wall_s` **只**打印进了日志，
    #   于是 `demo/data/index.json`（它读 scene.json）根本拿不到这两项。
    timing = timing_block(
        model_load_ms=load_wall_s * 1000.0,
        build_wall_ms=wall_s * 1000.0,
        save_ms=save_ms,
    )
    scene = scene.model_copy(update={
        "build_meta": {**scene.build_meta, "timing": timing},
    })

    # ⚠ scene.json **自己**的写入耗时不进 `timing`：要把它写进 scene.json 就得先知道它，
    #   那需要写两次文件。它只进 `build_log.txt`，且**不参与任何指标**。
    t_json = time.perf_counter()
    p = save_scene(scene, out_dir)
    scene_json_ms = (time.perf_counter() - t_json) * 1000.0

    print()
    print(f"  已写入 {p}")
    print(f"  掩码 {len(written)} 个 → {masks_dir(scene_id, args.out_root)}")
    if pts_p is not None:
        gh = res.points_meta.get("grid_hw") or [0, 0]
        mb = pts_p.stat().st_size / 1024 / 1024
        # 打印**落盘后**的 dtype，不是内存里的那一个 —— 两者默认不相等
        # （内存是 float64、落盘是 float32）。写错这一项，读日志的人会
        # 照着 "float64" 去理解那 3.5 MB，然后怀疑是不是没压成功。
        dt = res.points_chw.dtype if args.points_dtype == "keep" else args.points_dtype
        print(f"  点云 {int(gh[0])}×{int(gh[1])}×3 {dt}"
              f" → {pts_p}  ({mb:.1f} MB)")
    elif args.no_points:
        print("  点云 --no-points：未写盘（该场景不支持点云级工具）")

    print(f"  耗时  加载 {timing['model_load_ms']:.0f} + 构建 {timing['build_wall_ms']:.0f}"
          f" + 落盘 {timing['save_ms']:.0f} = {timing['total_wall_ms']:.0f} ms"
          f"   （scene.json 自身 {scene_json_ms:.0f} ms 未计入）")

    log = [
        f"# 场景图构建记录  scene_id={scene_id}",
        f"image={image_path}",
        f"built_at={scene.build_meta.get('config', {}).get('prompt')!r}",
        "",
        "## 耗时 (ms)",
        json.dumps(tim, ensure_ascii=False),
        f"build_wall_s={wall_s:.2f}",
        f"model_load_wall_s={load_wall_s:.2f}",
        # ↓ 2026-09-22 追加。上面三行**原样保留**：它们已经存在于 9 份旧 build_log.txt 里，
        #   删掉只会让"新旧日志字段不一致"这件事凭空多出来，而没有任何收益。
        f"model_load_ms={timing['model_load_ms']}  build_wall_ms={timing['build_wall_ms']}"
        f"  save_ms={timing['save_ms']}  total_wall_ms={timing['total_wall_ms']}",
        f"scene_json_write_ms={scene_json_ms:.1f}  （**不计入** total，见 build_meta.timing.scope）",
        f"口径: {TIMING_SCOPE}",
        "",
        # 内参来源必须进记录：它是横向尺度的总开关，两个 scene.json 之间
        # 只有几个数字的差别时，没有这一段就事后无法判断哪个更可信。
        "## 内参与视场",
        f"intrinsics_arg={intrinsics_note}",
        json.dumps(
            {k: scene.build_meta.get(k) for k in
             ("intrinsics_source", "intrinsics", "fov", "scale_calibrated")},
            indent=2, ensure_ascii=False,
        ),
        # CLI 侧的来源细节（--intrinsics 如何解析、EXIF 的量化误差与假设）。
        # 与上面的 build_meta 分开记：那份是 builder **实际用的** K，这份是它
        # **从哪来、那个来源有多可信**。混在一起会让「K 是多少」与「K 多可信」
        # 难以分开读，而这恰恰是内参问题的全部要害。
        json.dumps(intrinsics_meta, indent=2, ensure_ascii=False),
        "",
        "## 感知栈",
        json.dumps(scene.build_meta["perception"], indent=2, ensure_ascii=False),
        "",
        # 点云的元信息必须进记录：它是「这个场景能不能回答点云级问题」的唯一
        # 判据，而判据如果只存在于一个二进制文件里，事后排查就只能靠猜。
        "## 点云",
        json.dumps(
            {
                **(res.points_meta or {}),
                "written": pts_p is not None,
                "file": str(pts_p) if pts_p else None,
                "bytes": pts_p.stat().st_size if pts_p else None,
            },
            indent=2, ensure_ascii=False,
        ),
        "",
        "## 警告",
        *[f"- {w}" for w in res.warnings],
        "",
        "## 物体",
        json.dumps(
            [
                {
                    "id": n.id, "label": n.label, "score": n.score,
                    "centroid_m": list(n.centroid_3d), "extent_m": list(n.extent_3d),
                    "n_points": n.n_points, "centroid_source": n.centroid_source,
                }
                for n in scene.nodes
            ],
            indent=2, ensure_ascii=False,
        ),
        "",
        "## 被丢弃的检测",
        json.dumps(scene.build_meta["dropped"], indent=2, ensure_ascii=False),
        "",
        "## 降级记录",
        json.dumps(scene.build_meta["fallbacks"], indent=2, ensure_ascii=False),
        "",
    ]
    (out_dir / "build_log.txt").write_text("\n".join(log), encoding="utf-8")
    print(f"  构建记录 {out_dir / 'build_log.txt'}")
    print()
    print("  下一步：把这个 scene.json 交给工具层（tools/spatial.py），")
    print("          或者用 scripts/smoke_tools.py 的套路直接对它提问。")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
