#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vadar_compat.py —— 让 vendor/VADAR 在 Windows 原生跑起来的兼容层。

设计原则：**vendor/VADAR 一个字节都不改。**
--------------------------------------------------------------------------
§17 已经定了这条规矩（原仓库原样保留，改动走薄适配器 + PATCHES.md 记录）。
本模块把「缺什么」补成「VADAR 原本期望的样子」，而不是去改 VADAR 的调用点。

缺的三样东西，以及为什么不该去装它们
------------------------------------
① `groundingdino`（编译版）：`setup.sh` 要 clone 源码 + `pip install -e .`
   现场编译 CUDA 扩展，**没有官方 Windows 支持**，且 VADAR 钉的是
   Python 3.10 而本机 venv 是 3.12。
   本项目已用 transformers 的 `grounding-dino-tiny`（= SwinT-OGC 的官方移植，
   同源权重）替代，所以这里提供一个**功能性** stub：接口、坐标语义、
   返回值形状与编译版一致，内部转发给 `vision/grounding.py`。

② `sam2`：omni3d 路径下 `ModulesList` 只会 import、不会实例化
   （`build_sam2`/`SAM2ImagePredictor` 只在 clevr/gqa 分支被调用）。
   给惰性 stub 即可 —— 一旦真被调用就抛错，绝不静默返回假结果。

③ `openai`：全仓库只有 `engine_utils.py:2,57` 用到（`Generator.__init__`），
   而 `Generator` 已被 `03_vadar_llm_bridge.py` 整体替换。
   给一个「一用就抛」的 stub，比装一个我们不需要的 SDK 更干净
   （也不用碰 VADAR 钉死的 `openai==1.51.2`）。

另外两处不是「缺包」而是「平台差异」：
④ `signal.SIGALRM` —— 见 `win_alarm.py`。
⑤ `LocateModule` 拿到的 grounding_dino 句柄来自 `load_model(cfg, weights)`，
   我们让它返回一个**惰性句柄**（首次 predict 才加载权重），
   避免 `ModulesList.__init__` 一上来就把三个模型全塞进显存。

坐标契约（最容易错、且错了不会报错的地方）
------------------------------------------
编译版 `groundingdino.util.inference.predict` 返回的 boxes 是
**归一化 cxcywh**；而本项目 `vision/grounding.py` 返回的是
**绝对像素 xyxy**（`grounding.py` docstring 坑 ② 写明了）。
VADAR 的 `_parse_bounding_boxes` 按归一化 cxcywh 消费，
所以这里做一次显式换算 —— 这是整层里唯一有语义风险的一行。

归一化坐标天然与尺度无关，因此「resize 过再算」不影响最终像素框：
`_parse_bounding_boxes` 会把归一化值乘回**原图**的 width/height。
"""

from __future__ import annotations

import os
import re
import sys
import types
from typing import Any, Optional

__all__ = [
    "ensure_stub_packages",
    "install_vadar",
    "install_program_sanitizer",
    "sanitize_program_text",
    "sanitize_program_file",
    "SANITIZER_STATS",
    "VADAR_REPO_ROOT",
    "DEFAULT_GDINO_DIR",
]

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
VADAR_REPO_ROOT = os.path.join(PROJECT_ROOT, "vendor", "VADAR")
DEFAULT_GDINO_DIR = os.path.join(PROJECT_ROOT, ".cache", "models", "grounding-dino-tiny")


# =====================================================================
# 0. 小工具
# =====================================================================
def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    return v if (v is not None and v != "") else default


def _module(name: str, **attrs: Any) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    m.__doc__ = "vadar_compat 提供的兼容 stub（见 evaluation/vadar_compat.py）"
    sys.modules[name] = m
    return m


def _attach(parent_name: str, child_full_name: str) -> None:
    """让 `import a.b` / `import a.b as x` 能解析到子模块属性。

    `child_full_name` 是**完整模块名**（`groundingdino.datasets`），
    挂到父模块上的属性名取最后一段（`datasets`）——
    写错的话 `import groundingdino.datasets.transforms as T`
    会在 `groundingdino.datasets` 上找不到 `transforms` 属性。
    """
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, child_full_name.rsplit(".", 1)[-1],
                sys.modules[child_full_name])


def _attach_all(full_names) -> None:
    for full in full_names:
        _attach(full.rsplit(".", 1)[0], full)


# =====================================================================
# 1. groundino 变换的忠实复刻
# =====================================================================
# 为什么自己写而不是 stub 成空操作：
# `LocateModule.transform_image()` 真的会调用它们，产出 `img_gd`。
# 我们要在 predict 里把 `img_gd` 反变换回图像内容，所以必须知道
# 它到底做了什么 —— 「假装什么都没做」会让反变换变成无源之水。
# 这四段与上游 groundingdino/datasets/transforms.py 行为一致。
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def _make_transforms_module() -> types.ModuleType:
    """构造 `groundingdino.datasets.transforms` 的兼容模块。

    这里**不在模块级 import torch**：项目约定单测零 torch
    （见 tests/test_vision_depth.py 的 `test_module_does_not_import_torch_at_module_level`），
    而本函数会在测试里被调用（验证 stub 注册正确）。
    torch 只在真正做张量运算的那两个函数内部 import。
    """
    import random

    from PIL import Image

    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, img, target=None):
            for t in self.transforms:
                img, target = t(img, target)
            return img, target

        def __repr__(self):
            return "Compose(%s)" % (self.transforms,)

    class RandomResize:
        def __init__(self, sizes, max_size=None):
            self.sizes = list(sizes)
            self.max_size = max_size

        def __call__(self, img, target=None):
            size = random.choice(self.sizes)
            return self.resize(img, target, size, self.max_size)

        def resize(self, img, target, size, max_size=None):
            w, h = img.size
            if max_size is not None and w > 0 and h > 0:
                min_o, max_o = float(min(w, h)), float(max(w, h))
                # 上游原式：max_size * min / max 只在超限时才压 size
                if max_o / min_o * size > max_size:
                    size = int(round(max_size * min_o / max_o))
            if (w <= h and w == size) or (h <= w and h == size):
                return img, target
            if w < h:
                ow, oh = size, int(size * h / w)
            else:
                oh, ow = size, int(size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), target

    class ToTensor:
        def __call__(self, img, target=None):
            # 等价 torchvision.transforms.functional.to_tensor：
            # uint8 HWC → float CHW ∈ [0,1]。手写是为了不引入 torchvision 依赖
            # （本 venv 只装了 torch，而 VADAR 的 requirements 里也没有它）。
            return _pil_to_tensor(img), target

    class Normalize:
        def __init__(self, mean, std):
            self.mean = list(mean)
            self.std = list(std)

        def __call__(self, image, target=None):
            if isinstance(image, list):
                for img in image:
                    for t, m, s in zip(img, self.mean, self.std):
                        t.sub_(m).div_(s)
                return image, target
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)
            return image, target

        def __repr__(self):
            return "Normalize(mean=%s, std=%s)" % (self.mean, self.std)

    def _pil_to_tensor(img):
        """等价 torchvision.transforms.functional.to_tensor：uint8 HWC → float CHW [0,1]。"""
        import numpy as np
        import torch

        arr = np.asarray(img.convert("RGB"), dtype="uint8")
        return torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous().float().div_(255.0)

    m = _module(
        "groundingdino.datasets.transforms",
        Compose=Compose,
        RandomResize=RandomResize,
        ToTensor=ToTensor,
        Normalize=Normalize,
    )
    return m


def tensor_to_pil(img_gd: Any) -> Any:
    """把 `LocateModule.transform_image()` 的输出反变换回 PIL 图像。

    `im_t = Normalize(ToTensor(RandomResize(img)))`，三步都可逆：
        CHW float * std + mean  →  [0,1]  →  *255 round  →  uint8 HWC  →  PIL

    反变换的确切性只需到「视觉等价」：随后的 HF processor 会自己做
    归一化与 resize，真正决定检测结果的只有像素内容与长宽比，
    而这两者在 RandomResize 下都不变。所以归一化坐标乘回原图尺寸仍然成立。
    """
    import numpy as np
    import torch
    from PIL import Image

    if isinstance(img_gd, Image.Image):
        return img_gd
    t = img_gd
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    t = t.detach().float().cpu()
    if t.ndim == 4:          # 容忍带 batch 维
        t = t[0]
    if t.shape[0] != 3 and t.shape[-1] == 3:
        t = t.permute(2, 0, 1)
    mean = torch.tensor(_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(_STD, dtype=torch.float32).view(3, 1, 1)
    t = t * std + mean                      # 逆 Normalize
    arr = (t.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(np.ascontiguousarray(arr), mode="RGB")


# =====================================================================
# 2. groundingdino 兼容面
# =====================================================================
class _GdinoHandle:
    """`load_model()` 的返回值：惰性持有 GroundingDetector。

    惰性是有原因的：`ModulesList.__init__` 里 `load_model(...)` 紧跟着
    `UniDepthV2.from_pretrained(...)`，如果这里立刻加载 GDINO，
    两个模型的峰值显存就叠在一起了 —— 而 §5 的显存实测是分开量的。
    首次 predict 才加载，可让显存曲线与既有测量口径一致。
    """

    def __init__(self, model_dir: str, device: str):
        self.model_dir = model_dir
        self.device = device
        self._detector = None

    @property
    def detector(self):
        if self._detector is None:
            from vision.grounding import GroundingDetector

            self._detector = GroundingDetector.from_pretrained(self.model_dir, self.device)
        return self._detector

    def __repr__(self):
        return "_GdinoHandle(model_dir=%r, device=%r, loaded=%s)" % (
            self.model_dir, self.device, self._detector is not None)


def _caption_for_hf(caption: str) -> str:
    """VADAR 的 caption 是 `"sofa-table ."`（词间连字符 + 句点前有空格）。

    `vision/grounding.py` 的坑 ① 说明这种写法会被 HF tokenizer 切错。
    默认**保持 VADAR 原样**（保真优先，臂 A 的意义就在这里），
    但留一个开关供对照：`VADAR_GDINO_CAPTION=normalized` 会转成
    `"sofa table."` 形式。两者的差异必须用实测数字说话，不能靠断言。
    """
    mode = _env("VADAR_GDINO_CAPTION", "vadar")
    if mode != "normalized":
        return caption
    s = caption.strip().rstrip(".").replace("-", " ").strip().lower()
    return (s + ".") if s else s


def _gdino_predict(model, image, caption, box_threshold=0.25,
                   text_threshold=0.25, device="cuda"):
    """编译版 `groundingdino.util.inference.predict` 的兼容实现。

    返回值形状与上游一致：`(boxes, logits, phrases)`，
    其中 boxes 是 **归一化 cxcywh** 的 (N,4) 张量。
    """
    # ★ import torch 必须在参数校验**之后**。放在最前面会让这个函数
    # 一被调用就把 torch 拉进进程 —— 而项目里有一条哨兵用例
    # （tests/test_vision_exif.py）靠「本进程是否加载过 torch」来验证
    # 某模块不在模块级 import torch。顺序错了会让那条用例变成 skip，
    # 表现为「测试变绿了但少了一个断言」。先校验、再 import。
    handle = model if isinstance(model, _GdinoHandle) else None
    if handle is None:
        raise TypeError("predict() 的 model 参数必须是 load_model() 返回的句柄，"
                        "收到 %r（见 evaluation/vadar_compat.py）" % (type(model),))

    import torch

    pil = tensor_to_pil(image)
    hf_caption = _caption_for_hf(caption)
    dets = handle.detector(
        pil, hf_caption,
        box_threshold=float(box_threshold),
        text_threshold=float(text_threshold),
    )

    w, h = pil.size
    if not dets:
        return (torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.float32),
                [])

    boxes, logits, phrases = [], [], []
    for d in dets:
        x1, y1, x2, y2 = d.box_xyxy
        # 绝对像素 xyxy → 归一化 cxcywh（上游语义）
        boxes.append([
            (x1 + x2) / 2.0 / w,
            (y1 + y2) / 2.0 / h,
            abs(x2 - x1) / w,
            abs(y2 - y1) / h,
        ])
        logits.append(float(d.score))
        phrases.append(str(d.label))
    return (torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(logits, dtype=torch.float32),
            phrases)


def _gdino_load_model(config_file, checkpoint_file=None, device="cuda"):
    """编译版 `load_model` 的兼容实现。

    忽略 .py config 与 .pth 权重路径（那是编译版独有的），改为读
    `VADAR_GDINO_DIR`（默认 `.cache/models/grounding-dino-tiny`）。
    """
    model_dir = _env("VADAR_GDINO_DIR", DEFAULT_GDINO_DIR)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            "找不到 GroundingDINO 权重目录 %r。设置 VADAR_GDINO_DIR 指向 "
            "transformers 版 grounding-dino-tiny 的快照目录。" % model_dir)
    dev = device
    try:
        import torch

        if not torch.cuda.is_available():
            dev = "cpu"
    except Exception:
        dev = "cpu"
    return _GdinoHandle(model_dir, dev)


def _make_gdino_modules() -> None:
    _module("groundingdino")
    _module("groundingdino.datasets")
    _make_transforms_module()
    _module("groundingdino.util")
    _module("groundingdino.util.inference",
            predict=_gdino_predict, load_model=_gdino_load_model)
    _attach_all((
        "groundingdino.datasets",
        "groundingdino.util",
        "groundingdino.datasets.transforms",
        "groundingdino.util.inference",
    ))


# =====================================================================
# 3. sam2 / openai 的惰性 stub
# =====================================================================
class _NotInstalled:
    """一被实例化就报错的占位类（而不是静默返回 None）。"""

    _pkg = "?"
    _why = ""

    def __init__(self, *a, **k):
        raise RuntimeError(
            "%s 未安装，且本臂（dataset=omni3d）不应该用到它。"
            "如果你确实走到了这条路径，说明这个实验臂的假设不成立：%s"
            % (self._pkg, self._why))


def _make_stub_modules() -> None:
    class _Sam2Predictor(_NotInstalled):
        _pkg = "sam2"
        _why = ("SAM2 只在 clevr/gqa 分支被实例化；omni3d 分支用不到它。"
                "若需要 SAM2，请用 vision/segmentation.py（HF 版 sam2.1-hiera-base-plus）。")

    def _build_sam2(*a, **k):
        raise RuntimeError(
            "sam2.build_sam.build_sam2 不可用：本机未安装编译版 sam2。"
            "omni3d 路径不会调用它。")

    class _OpenAI(_NotInstalled):
        _pkg = "openai"
        _why = ("Generator 已被 03_vadar_llm_bridge.py 整体替换，"
                "真正的调用走 LLMConfig + urllib。")

    _module("sam2")
    _module("sam2.build_sam", build_sam2=_build_sam2)
    _module("sam2.sam2_image_predictor", SAM2ImagePredictor=_Sam2Predictor)
    _attach_all(("sam2.build_sam", "sam2.sam2_image_predictor"))

    _module("openai", OpenAI=_OpenAI)


# =====================================================================
# 4. 总装
# =====================================================================
def ensure_stub_packages(which: str = "all", verbose: bool = True) -> dict:
    """注册缺失包。**必须在 import engine.* / agents.* 之前调用。**

    `which="all"`（默认）注册 groundingdino + sam2 + openai。
    已存在于 sys.modules 的名字不会覆盖 —— 所以如果你在装过真包的环境里跑，
    这个函数是空操作，会自动用真包（这是我们想要的：同一层适配器在
    Linux/WSL 与 Windows 上都成立）。
    """
    before = set(sys.modules)
    if which == "all":
        _make_gdino_modules()
        _make_stub_modules()
    elif which == "gdino":
        _make_gdino_modules()
    else:
        raise ValueError("which 只支持 'all' / 'gdino'，收到 %r" % (which,))
    added = sorted(set(sys.modules) - before)
    info = {"added": added, "count": len(added)}
    if verbose:
        print("[vadar_compat] 注入兼容包: %s" % ", ".join(added))
    return info


def _ensure_paths() -> dict:
    """把 vendor/ 与 vendor/VADAR 放上 sys.path。

    - `predefined_modules.py:17` 是 `from VADAR.prompts... import`，
      要求 **VADAR 这一层目录名必须是 VADAR**（§19 风险项），
      所以 `vendor/` 必须在 sys.path 上。
    - `agents.agents` / `engine.engine` 是顶层包名，所以 `vendor/VADAR/`
      自己也必须在 sys.path 上。
    """
    info = {"added": [], "repo_root": None}
    if not os.path.isfile(os.path.join(VADAR_REPO_ROOT, "engine", "predefined_modules.py")):
        raise FileNotFoundError("找不到 VADAR 仓库：%r" % VADAR_REPO_ROOT)
    if os.path.basename(VADAR_REPO_ROOT) != "VADAR":
        raise RuntimeError("目录名必须是 VADAR（predefined_modules.py:17 硬编码），实际 %r"
                           % os.path.basename(VADAR_REPO_ROOT))
    for p in (os.path.dirname(VADAR_REPO_ROOT), VADAR_REPO_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
            info["added"].append(p)
    info["repo_root"] = VADAR_REPO_ROOT
    # 项目根也放上，供 `import vision.*`（桥接与 runner 都需要）
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
        info["added"].append(PROJECT_ROOT)
    return info


# =====================================================================
# 5. 生成程序里的路径字面量消毒（Windows 上必然踩的 VADAR bug）
# =====================================================================
# 现象（2026-09-18 首次端到端冒烟实测，不是推演）：
#   VADAR 把输出路径**插进要执行的源码**里（`agents.py:465`，engine 里同款模板）：
#       with open("{result_file}", "w+") as result_file:
#   Linux 上无害（路径形如 `/a/b/c`）；Windows 上路径是 `D:\...`，
#   于是 Python 把这段字面量**按转义序列解析**：
#       `D:\3D_Spatial_Agent` → `\3`   = \x03  （八进制）
#       `\2026-09-18`         → `\202` = \x82
#       `\api_generator`      → `\a`   = BEL
#       `\trace.html`         → `\t`   = TAB
#   ⟹ 生成程序打不开自己的结果文件，`open()` 抛 `[Errno 22] Invalid argument`，
#      VADAR 的 5 次重试全部复现同一个错，该题直接记 0 分。
#
# 为什么不在 VADAR 里改：规矩是 vendor 一字不改（见本模块开头）。
# 为什么不在「路径本身」上绕开：绕不过去 ——
#   * `os.path.join` 在 Windows 用 `\` 拼，只要有一段的段名/文件名以
#     a b f n r t v x N u U 或数字 0–7 开头就中招，而 VADAR 自己的目录名里
#     就有 `api_generator`、`program_execution`，文件名里就有 `trace.html`；
#   * 换盘符、加 subst、改短路径都只是碰运气，迟早被某个新段名咬到。
# ⟹ 唯一稳的位置是**边界**：程序落盘之后、`runpy.run_path` 之前，
#   把带盘符的绝对路径字面量整体改成 **raw 字符串**。
#   语义完全等价 —— 这不是「把错误藏起来」，而是把 VADAR 的原意
#   （那个字符串就是一条路径）如实兑现。
#
# 可关闭（`VADAR_FIX_PATH_ESCAPE=0`），用于做「保真 vs 可用」的对照臂。
_PATH_LITERAL_RE = re.compile(
    r"""(?<![rRbBfF])(?P<q>['"])(?P<p>[A-Za-z]:\\[^'"\n]*)(?P=q)"""
)

#: 消毒计数 —— 由 runner 在收尾时读进实验记录（「修了几个文件」是证据，不是日志）。
SANITIZER_STATS: dict = {"files": 0, "replacements": 0, "namespace_coercions": 0,
                         "samples": []}

#: VADAR 的写盘过滤行（`agents.py` 模板里逐字出现）。
#: 它只留 `json.dumps` 过得去的值 —— 而 numpy/torch 的**标量**过不去。
_NAMESPACE_LINE = ("serializable_globals = "
                   "{k: v for k, v in globals().items() if is_serializable(v)}")

#: 替换成「先剥 0 维标量、再照原样过滤」。
#: 只剥 `shape == ()` 的标量（`.item()` 是语义等价的解包），
#: 数组/张量仍然照旧被过滤 —— 不要把行为放大成「什么都往里塞」。
#:
#: ⚠⚠ **必须保持「字典推导」这一形式，不能改写成 `x = {}` + `for` 循环。**
#: 2026-09-18 实测踩坑：写成
#:     serializable_globals = {}
#:     for _k, _v in list(globals().items()): ...
#: 时，`serializable_globals = {}` **先把名字绑好了**，于是 `globals()` 里
#: 已经有这个键，`_v` 拿到的就是**它自己**，循环体写下
#: `serializable_globals["serializable_globals"] = 自己` ⟹ 自引用。
#: 后果不是报错，而是 `json.dump` 写到最后一个键时抛
#: `ValueError: Circular reference detected`，把 `result.json` **截断在
#: 半路**（实测 9/9 题的文件都停在 `"serializable_globals": ` 处）⟹
#: VADAR 读不出来 → 判为执行失败 → 重试整个程序生成（每题多烧 5 轮 LLM
#: + 5 轮模型推理，整轮慢 4 倍）。
#: 字典推导不会踩这个坑：名字**在整个推导求值完之后**才绑定，所以
#: `globals()` 里根本没有它。这是 VADAR 原版的一个隐含前提，别弄丢。
_NAMESPACE_REPLACEMENT = '''def _unbox_scalar(v):
    """0 维 numpy/torch 标量 → Python 标量。其余原样返回。"""
    if getattr(v, "shape", None) == ():
        item = getattr(v, "item", None)
        if callable(item):
            try:
                return item()
            except Exception:
                return v
    return v


def is_serializable(obj):
    """VADAR 原本只捕 TypeError/OverflowError —— 漏了 `ValueError`。

    `json.dumps` 撞上**循环引用**时抛的是 `ValueError("Circular reference
    detected")`，没被捕获 → 整段程序崩掉、VADAR 记
    `Error in executing <方法>: Circular reference detected`（2026-09-18 实测）。
    一个循环引用的中间变量不该能废掉整道题：本意只是「这个值不写进结果」。
    """
    try:
        json.dumps(obj)
    except (TypeError, OverflowError, ValueError):
        return False
    return True


serializable_globals = {_k: _unbox_scalar(_v) for _k, _v in list(globals().items())
                        if is_serializable(_unbox_scalar(_v))}
'''


def repair_namespace_writer(text: str):
    """把「静默丢掉 numpy 标量」的写盘过滤换成「先解包再过滤」。

    为什么必须修（2026-09-18 实测，11/11 数值题空答案的根因）
    ------------------------------------------------------
    VADAR 用 `json.dumps` 能不能过来决定哪些变量写进 `result.json`。
    纯 Python 的 `float`/`int` 过得去，但 **`np.float32` / `np.int64` /
    torch 0 维张量过不去** —— 于是经过深度推出来的中间量与 `final_result`
    **一起消失**。VADAR 的兜底是「没有 `final_result` 就记空答案」：

        else:
            execution_data[execution_type]["answer"] = ""

    ⟹ **正确算出来的答案被静默变成空字符串，不报错、不重试、直接 0 分。**
    这比报错难查得多：`result.json` 里剩下的 `*_2d` 变量看起来一切正常。

    为什么改在这里而不是改上层：改动点必须落在「值的类型被丢掉」的那一行，
    否则只是把同一个静默失败往下游推。

    返回 `(新文本, 解包处数)`。
    """
    if _NAMESPACE_LINE not in text:
        return text, 0
    return text.replace(_NAMESPACE_LINE, _NAMESPACE_REPLACEMENT.strip()), 1


def sanitize_program_file(path: str, fix_paths: bool = True,
                          fix_coercion: bool = True) -> dict:
    """就地修复一个生成程序。返回 `{"paths": n, "namespace_coercions": n}`。

    读不到文件返回全 0（不抛）—— 消毒不能变成新的失败点。
    """
    out = {"paths": 0, "namespace_coercions": 0}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return out

    original = text
    if fix_paths:
        text, out["paths"] = sanitize_program_text(text)
    if fix_coercion:
        text, out["namespace_coercions"] = repair_namespace_writer(text)

    if text == original:
        return out

    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)

    SANITIZER_STATS["files"] += 1
    SANITIZER_STATS["replacements"] += out["paths"]
    SANITIZER_STATS["namespace_coercions"] += out["namespace_coercions"]
    if len(SANITIZER_STATS["samples"]) < 5:
        SANITIZER_STATS["samples"].append({"file": path, "n": out["paths"]})
    return out


def sanitize_program_text(text: str):
    """把 `"D:\\a\\b"` 改成 `r"D:\\a\\b"`。返回 `(新文本, 替换次数)`。

    只动**带盘符的绝对路径**字面量，且跳过已经是 raw 的（防止二次加 `r`）。
    单引号、双引号都覆盖 —— VADAR 的模板里两种都可能出现。
    """
    def _fix(m):
        return "r" + m.group("q") + m.group("p") + m.group("q")

    return _PATH_LITERAL_RE.subn(_fix, text)


def _wrap_execute_file(orig):
    """包一层「执行前消毒」。

    VADAR 有两套签名，都得管：
        `APIAgent._execute_file(self, program_executable_path)`   （agents.py:570）
        `Engine._execute_file(self)`                              （engine.py:592，路径藏在 self 上）
    """
    import inspect

    takes_path = len(inspect.signature(orig).parameters) >= 2

    if takes_path:
        def patched(self, program_executable_path, *a, **k):
            sanitize_program_file(program_executable_path)
            return orig(self, program_executable_path, *a, **k)
    else:
        def patched(self, *a, **k):
            path = getattr(self, "program_executable_path", None)
            if path:
                sanitize_program_file(path)
            return orig(self, *a, **k)

    patched.__vadar_path_sanitized__ = True
    patched.__name__ = getattr(orig, "__name__", "_execute_file")
    patched.__doc__ = getattr(orig, "__doc__", None)
    return patched


def install_program_sanitizer(verbose: bool = True) -> dict:
    """按「谁定义了这个方法」动态发现类，不写死类名。

    写死 `APIAgent` / `Engine` 的话，VADAR 一旦改名或加子类，这里会
    **静默失效** —— 而失效的表现是「实验分数变成 0」，不是报错。
    """
    enabled = str(_env("VADAR_FIX_PATH_ESCAPE", "1")).strip().lower() \
        not in ("0", "false", "no", "off")
    report = {"enabled": enabled, "patched": []}
    if not enabled:
        return report

    import agents.agents as _agents
    import engine.engine as _engine

    for mod, mod_name in ((_agents, "agents.agents"), (_engine, "engine.engine")):
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if not isinstance(obj, type):
                continue
            fn = obj.__dict__.get("_execute_file")
            if fn is None or getattr(fn, "__vadar_path_sanitized__", False):
                continue
            setattr(obj, "_execute_file", _wrap_execute_file(fn))
            report["patched"].append("%s.%s" % (mod_name, attr))

    if verbose and report["patched"]:
        print("[compat] 路径字面量消毒已装到: %s" % ", ".join(report["patched"]))
    return report


def install_vadar(light: bool = False, verbose: bool = True) -> dict:
    """一键：注册兼容包 → 接上 LLM 桥接 → 替换 Windows alarm。

    返回的报告可直接放进实验记录的 `environment` 段。
    调用后即可：
        from agents.agents import SignatureAgent, APIAgent, ProgramAgent
        from engine.engine import Engine
    """
    report = {"steps": {}}
    report["steps"]["paths"] = _ensure_paths()
    report["steps"]["stubs"] = ensure_stub_packages("all", verbose=verbose)

    # 桥接模块按文件名带数字前缀，不是合法包名，只能按路径导入
    bridge_path = os.path.join(PROJECT_ROOT, "phase0", "03_vadar_llm_bridge.py")
    import importlib.util

    spec = importlib.util.spec_from_file_location("vadar_llm_bridge", bridge_path)
    bridge = importlib.util.module_from_spec(spec)
    sys.modules["vadar_llm_bridge"] = bridge
    spec.loader.exec_module(bridge)
    report["steps"]["bridge_install"] = bridge.install(
        repo_root=VADAR_REPO_ROOT, light=light, verbose=verbose)

    # 导入 agents/engine —— 此刻 stub 已就位，import 才能成功
    import agents.agents  # noqa: F401
    import engine.engine  # noqa: F401

    # 相对 import 只在「作为包被导入」时成立；直接 `python evaluation/vadar_compat.py`
    # 自检时是 __main__，得走绝对 import。两条路都要能跑，否则自检本身没法用。
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    try:
        from .win_alarm import install_alarm_shim
    except ImportError:
        from win_alarm import install_alarm_shim

    report["steps"]["alarm"] = install_alarm_shim(verbose=verbose)
    # shim 对象不可 JSON 序列化，这里只留一个取值器说明
    report["steps"]["alarm_note"] = "计数请在实验结束时调用 report['steps']['alarm']['shim'].describe()"

    # 必须在 agents/engine 导入之后 —— 它要包在真类的方法上
    report["steps"]["program_sanitizer"] = install_program_sanitizer(verbose=verbose)

    text_cfg, vision_cfg = bridge.load_config()
    report["model_fingerprint"] = {
        "text": text_cfg.describe(),
        "vision": vision_cfg.describe(),
    }
    report["gdino_dir"] = _env("VADAR_GDINO_DIR", DEFAULT_GDINO_DIR)
    report["gdino_caption_mode"] = _env("VADAR_GDINO_CAPTION", "vadar")
    return report


if __name__ == "__main__":
    import json

    rep = install_vadar()
    rep["steps"]["alarm"].pop("shim", None)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
