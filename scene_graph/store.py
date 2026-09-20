"""场景图的落盘与读取 —— JSON 缓存 + 掩码 PNG + 整图点云。

对应方案文档 §17 目录树里的 `scene_graph/store.py`。

落盘格式是一个**信封**而不是裸的 SceneGraph：

    {
      "_format": "1.0",
      "_written_at": "2026-09-16T20:00:00",
      "scene": { ...SceneGraph... }
    }

为什么要信封：`SceneGraph` 是 `extra="forbid"` 的，把 `_format` 塞进它的顶层
会直接校验失败。而版本号必须能独立于模型定义被读到 —— 否则将来改了 schema，
你连「这个 JSON 是哪一版」都读不出来，迁移无从谈起。

掩码存成 **1-bit PNG**（PIL 的 `mode="1"`）。选它而不是 `.npy` 的理由是体积：
一个 640×480 的掩码在 `.npy` 里恒为 307 KB，PNG 因为大片连续 0 而常有几十 KB。
20 个物体就是 6 MB vs 1 MB 的差别，而且 PNG 能直接用看图软件打开 ——
排查「这个掩码分错了吗」的时候，这一条比体积更值钱。

点云存成 **整图 `points.npy`（`(3, H, W)`），刻意不存每物体点云**。
理由不是省空间，是**避免副本漂移**：某个物体的点云 = 整图点云 ∩ 该物体掩码，
而掩码本来就要落盘。存两份等于给同一个事实两个副本，而两份一定会漂移
（改了取点规则、换了重采样方式，只有一份会被更新）。掩码 PNG 与
整图点云合起来是**完备且零冗余**的表示。

代价写在明处：任何点云级工具都要先做一次「掩码重采样 + 取点」
（`scene_graph.pointcloud`），而不是直接 load 一个现成的数组。
这一步是毫秒级的，而且它**复用 builder 的同一个函数** ——
所以工具算出的质心能和 `scene.json` 里记的逐位对上（见 `scene_graph/tests/test_points_roundtrip.py`）。

⚠️ 老场景（建于本改动之前）**没有点云文件**。`load_points` 会明确抛
`FileNotFoundError` 而不是静默返回空 —— 「没有这个文件」和「点云是空的」
需要完全不同的处理（重跑建图 vs 查数据），压成同一个分支会让后者被前者掩盖。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from scene_graph.schema import SceneGraph
from tools.version import SCENE_FORMAT_VERSION

__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_SCENES_DIR",
    "DEFAULT_POINTS_DTYPE",
    "POINTS_FILENAME",
    "POINTS_FORMAT_VERSION",
    "POINTS_META_FILENAME",
    "scene_dir",
    "masks_dir",
    "points_path",
    "save_scene",
    "load_scene",
    "save_masks",
    "load_mask",
    "save_points",
    "load_points",
    "has_points",
    "read_format_version",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
#: 默认的落盘根目录：`dataset/scenes/`。
DEFAULT_SCENES_DIR = PROJECT_ROOT / "dataset" / "scenes"

#: 整图点云文件名，与 `scene.json`、`masks/` 同级。
POINTS_FILENAME = "points.npy"
#: 点云的**描述文件**，与数据分开存。选 JSON 而不是 `.npz` 的内嵌字符串：
#: JSON 能直接打出来看（排查时最常做的就是确认「这份点云的网格尺寸是多少」），
#: 而 `.npz` 必须先在 Python 里加载才知道里面有什么。
POINTS_META_FILENAME = "points_meta.json"
#: 点云格式版本，**与 `scene.json` 的 `_format` 独立**。点云可以单独演进
#: （换 dtype、加降采样），不该拖着场景图 schema 一起升版本 ——
#: 否则一次纯点云的改动会让所有历史 scene.json 被判成"格式不匹配"。
POINTS_FORMAT_VERSION = "1.0"

#: 默认保存精度 **float32**。这个默认值是**实测选出来的，不是拍脑袋**：
#:
#: `points_probe` 场景（480×640）落盘的 float64 点云做 float32 往返，
#: `max |a - f32(a)| = 0.000e+00` —— **逐位无损**。也就是说那份 7.0 MB 里
#: 有一半是**假精度**（低位全 0），float32 是真正的无损压缩（7.0 → 3.5 MB）。
#:
#: 退一步，就算将来上游真给出 float64 精度，压成 float32 的绝对误差在 4.4 m
#: 处也只有约 0.5 µm —— 比本项目任何在意的容差（关系判定 50 mm）小 5 个数量级。
#: 所以这不是「拿精度换空间」的赌注，是**把不携带信息的位去掉**。
#:
#: float16 则是**真有损**的（同一次实测：max 1.95 mm），所以它不是「更省的默认」，
#: 而是一个需要显式选择的档位。要严格原样保存就传 `dtype=None`。
DEFAULT_POINTS_DTYPE: str | None = "float32"


def scene_dir(scene_id: str, root: Path | str | None = None) -> Path:
    """`dataset/scenes/<scene_id>/`（不创建）。"""
    return Path(root or DEFAULT_SCENES_DIR) / scene_id


def masks_dir(scene_id: str, root: Path | str | None = None) -> Path:
    """`dataset/scenes/<scene_id>/masks/`（不创建）。"""
    return scene_dir(scene_id, root) / "masks"


def points_path(scene_id: str, root: Path | str | None = None) -> Path:
    """`dataset/scenes/<scene_id>/points.npy`（不创建）。"""
    return scene_dir(scene_id, root) / POINTS_FILENAME


# ----------------------------------------------------------------------------
# 场景图 JSON
# ----------------------------------------------------------------------------


def _json_path(path: Path | str) -> Path:
    """把「目录或文件」统一成一个具体的 `.../scene.json` 路径。

    为什么要这条规则：`save_scene(scene, dataset/scenes/living_room)` 里那个目录
    **通常还不存在**（第一次跑的时候必然不存在），于是 `Path.is_dir()` 判不出来，
    简单实现就会把 `living_room` 当成一个文件名、写出一个叫 `living_room` 的 JSON 文件。
    规则定为：**后缀不是 `.json` 就当目录**。这条规则简单、可预测，
    并且不会误伤任何正常用法（没人会给 JSON 起别的后缀）。
    """
    p = Path(path)
    if (p.exists() and p.is_dir()) or p.suffix.lower() != ".json":
        return p / "scene.json"
    return p


def save_scene(scene: SceneGraph, path: Path | str) -> Path:
    """写 `scene.json`（自动建父目录）。`path` 可以是目录或 JSON 文件路径。"""
    p = _json_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "_format": SCENE_FORMAT_VERSION,
        "_written_at": datetime.now().isoformat(timespec="seconds"),
        "scene": scene.model_dump(mode="json"),
    }
    p.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return p


def read_format_version(path: Path | str) -> str | None:
    """只读版本号，不解析整个场景图。无信封（裸 SceneGraph）时返回 None。"""
    raw = json.loads(_json_path(path).read_text(encoding="utf-8"))
    return raw.get("_format") if isinstance(raw, dict) else None


def load_scene(path: Path | str) -> SceneGraph:
    """读 `scene.json`。同时接受带信封与裸 SceneGraph 两种历史格式。

    版本不匹配时**不抛异常**，只在 `build_meta` 里记一笔 —— 因为大多数
    格式演进是向后兼容的（新增可选字段），直接拒绝加载会让旧实验数据全废，
    而旧数据正是做对照实验时最不能丢的东西。
    """
    raw = json.loads(_json_path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "scene" in raw and "_format" in raw:
        fmt: str | None = str(raw.get("_format"))
        data = raw["scene"]
    else:
        fmt = None
        data = raw

    scene = SceneGraph.model_validate(data)
    if fmt is not None and fmt != SCENE_FORMAT_VERSION:
        meta = dict(scene.build_meta)
        meta["_loaded_format"] = fmt
        meta["_current_format"] = SCENE_FORMAT_VERSION
        meta["_format_mismatch"] = True
        scene = scene.model_copy(update={"build_meta": meta})
    return scene


# ----------------------------------------------------------------------------
# 掩码
# ----------------------------------------------------------------------------


def save_masks(
    masks: dict[str, np.ndarray],
    directory: Path | str,
) -> dict[str, Path]:
    """把 `{object_id: bool(H,W)}` 写成 1-bit PNG。返回 `{object_id: 路径}`。"""
    from PIL import Image

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for object_id, mask in masks.items():
        p = d / f"{object_id}.png"
        # bool ndarray 直接被 PIL 认成 mode="1"（1-bit），正是我们要的。
        Image.fromarray(np.asarray(mask, dtype=bool)).save(p, optimize=True)
        out[object_id] = p
    return out


def load_mask(path: Path | str) -> np.ndarray:
    """读回掩码为 `(H, W)` bool。"""
    from PIL import Image

    with Image.open(Path(path)) as im:
        arr = np.asarray(im.convert("1"), dtype=np.uint8)
    return arr.astype(bool)


# ----------------------------------------------------------------------------
# 整图点云
# ----------------------------------------------------------------------------


def save_points(
    points_chw: np.ndarray,
    directory: Path | str,
    *,
    meta: dict[str, Any] | None = None,
    dtype: str | None = DEFAULT_POINTS_DTYPE,
) -> Path:
    """写整图点云，返回 `points.npy` 路径。

    `meta` **刻意不给默认值、不去猜**：一份没有 `grid_hw` 的点云在重建物体
    点云时根本没法用（掩码重采样需要目标网格），而"缺字段"必须在**写入时**
    就失败，而不是等某个工具算出错误坐标时才暴露 —— 那时错误已经落进结果里了。
    """
    arr = np.asarray(points_chw)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"points_chw 必须是 (3, H, W)，收到 {arr.shape}")
    mp: dict[str, Any] = dict(meta or {})
    if not mp.get("grid_hw"):
        raise ValueError(
            "meta 必须含 grid_hw（例如 {'grid_hw': [H, W]}）—— 没有它的点云"
            "无法用于重建物体点云（掩码重采样需要目标网格尺寸）。"
            "这个校验必须发生在写入时：等到某个工具取点时才炸，"
            "那时错误的坐标已经混进结果里了。"
        )
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    # `dtype=None` → 原样保存（这是一个**显式**选择，不是默认）。
    # 默认 float32 的依据见 `DEFAULT_POINTS_DTYPE`：对实测点云逐位无损。
    saved = arr if dtype is None else arr.astype(dtype, copy=False)
    p = d / POINTS_FILENAME
    np.save(p, saved)
    payload: dict[str, Any] = {
        "_format": POINTS_FORMAT_VERSION,
        "_written_at": datetime.now().isoformat(timespec="seconds"),
        "shape": list(saved.shape),
        "dtype": str(saved.dtype),
        **mp,
    }
    (d / POINTS_META_FILENAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return p


def load_points(path: Path | str) -> tuple[np.ndarray, dict[str, Any]]:
    """读回 `(points_chw, meta)`。`path` 可以是场景目录，也可以是 npy 本身。

    ⚠️ **缺文件时抛 `FileNotFoundError`，不返回 `None`。** 调用方必须能区分
    「这个场景建于点云落盘之前」与「点云存在但是空的」—— 前者要重跑建图，
    后者才是数据问题。返回 None 会把两者压进同一个分支，而由此得出的
    「点云为空」诊断是**错的**，它会把人支到错误的方向上去查。
    """
    p = Path(path)
    if p.is_dir():
        p = p / POINTS_FILENAME
    if not p.is_file():
        raise FileNotFoundError(
            f"没有点云文件 {p}。这个场景可能建于点云落盘之前 —— "
            f"用 scripts/build_scene.py 重建一次即可拿到 points.npy。"
        )
    arr = np.load(p)
    mp = p.parent / POINTS_META_FILENAME
    meta: dict[str, Any] = (
        json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else {}
    )
    return arr, meta


def has_points(path: Path | str) -> bool:
    """点云文件是否存在 —— 用于**不加载那 3.7 MB 数组**就判断这件事。"""
    p = Path(path)
    if p.is_dir():
        p = p / POINTS_FILENAME
    return p.is_file()
