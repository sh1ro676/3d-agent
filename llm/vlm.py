#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""llm/vlm.py —— 角色② 视觉语义 `describe()`。

**唯一能看图说话的角色，签名里没有任何空间参数。** 这是本节全部价值所在，
所以先说清楚它凭什么是一条**架构保证**而不是一句提示词祈祷。

`attrs` 是 `Literal` 白名单（`color` / `material` / `texture` / `state` / `shape`），
`left_of` / `distance` / `above` **根本不在枚举里** —— 空间幻觉在**类型层面**就写不出来。
在提示词里写「请不要判断空间关系」是祈祷；把空间词汇从签名里删掉是保证。
两者的区别在答辩时会被追问，所以这里还多做了一步：

    **`llm/vlm.py` 在导入时就自检 `describe()` 的形参名**（`check_signature()`），
    一旦有人加了 `left_of=` 或 `distance=` 这样的参数，**导入本模块直接失败**。
    这比把约束写在文档里可靠 —— 文档会过期，import 不会。

契约在本文件，**类型在中性层**（`vision/semantics.py`）
=====================================================
`ATTRS` / `Attribute` / `Region` / `image_size_from_meta` / 失败类型的**基类**
定义在 `vision/semantics.py`，本文件只是把它们**再导出**（老写法照旧可用）。
这么搬的理由只有一个：工具层（`tools/attributes.py`）也要用它们，
而本项目禁止 `tools/` import `llm/`（那条禁令保证「关掉视觉语义」是真的少一条依赖，
而不是改了一版提示词）。**契约放中性层、实现留在这里** —— 详见该文件抬头。

三个刻意的设计决定
================

1. **不硬编码模型名。** 端点走 `SPATIAL_VISION_*`（回退 `VADAR_VISION_*`）五个键，
   与文本端点共用同一套 `LLMSettings`/`LLMClient`。本地 Qwen3.5 多模态与云端
   `qwen3-vl-flash` 都只是换环境变量 —— 换模型不需要改一行代码，也不需要改实验记录格式。
   视觉端点完全没配时回落到文本端点（deepseek-flash 自带 Vision），
   这与 `phase0/03_vadar_llm_bridge.py` 同口径，两个臂因此可比。

2. **闭集优先，而且违规要能被看见。** 给了 `candidates` 时，模型答的必须落在闭集内。
   落不进去**不是**悄悄取个近似值，而是把该条标记为 `in_closed_set=False`
   并把置信度压到 0 —— 上层据此上报 `LOW_CONFIDENCE`。
   「闭集」的价值就在于它让「模型瞎编」变成一个**可计数的数字**，悄悄兜回来等于把度量扔掉。

3. **缺 confidence 不等于 confidence=1。** 模型没给置信度时默认 **0.0**（视作不确定），
   而不是 1.0。默认成 1 会让「模型忘了给」伪装成「模型很确定」，
   而 §13.3(3) 明确要求低置信度必须上报、禁止静默取最高分。

`describe()` 只看图，**不碰 SceneGraph、不碰点云**。它也**不写**任何坐标进返回值：
`Attribute` 里只有 `name/value/confidence/source` 四个字段，没有位置可填。
"""

from __future__ import annotations

import base64
import inspect
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from llm.adapter import LLMClient, LLMError, LLMSettings
# ⚠ 这几个名字的**定义**在 `vision/semantics.py`（中性契约层），这里只是再导出。
#   原因：工具层（`tools/attributes.py`）也要用它们，而分层禁令禁止 `tools/` import `llm/`。
#   契约放中性层、实现留在这里 —— 详见 `vision/semantics.py` 抬头。
#   老写法 `from llm.vlm import Region` 因此照旧可用，但新代码应从 `vision.semantics` 导入。
from vision.semantics import (
    ATTRS,
    CROP_MARGIN,
    DEFAULT_CONFIDENCE_THRESHOLD,
    SPATIAL_PARAM_TERMS,
    Attribute,
    Region,
    SemanticBackendError,
    image_size_from_meta,
)

__all__ = [
    "ATTRS",
    "SPATIAL_PARAM_TERMS",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "CROP_MARGIN",
    "VLMError",
    "SemanticBackendError",
    "Attribute",
    "Region",
    "DescribeCall",
    "VLM",
    "check_signature",
    "image_size_from_meta",
    "to_data_url",
]


class VLMError(SemanticBackendError):
    """视觉后端不可用（缺配置 / 网络失败 / 返回不可解析）。

    刻意与 `LLMError` 分开：`LLMError` 是**文本角色**的失败，会被 `loop.py` 记成
    `llm_error`（整题失败）；而视觉角色是**可降级**的（§13.3(3)：主路径不依赖本角色），
    它的失败应当由调用方翻译成一次「能力不可用」的工具结果，而不是把整题打断。

    ★ 基类 `SemanticBackendError` 住中性层（`vision/semantics.py`）：
    这样工具层 `except SemanticBackendError` 就能接住它，而**不必 import `llm`**。
    """


# ============================================================================
# 1. 返回值
# ============================================================================


@dataclass(frozen=True)
class DescribeCall:
    """一次 `describe()` 的调用侧事实 —— 进 tool evidence，供审计与成本归因。"""

    model: str = ""
    base_url: str = ""
    endpoint_label: str = "vision"
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    requested: tuple[str, ...] = ()
    returned: tuple[str, ...] = ()
    cropped: bool = False
    image_chars: int = 0
    prompt_chars: int = 0
    elapsed_s: float = 0.0
    usage: Mapping[str, Any] = field(default_factory=dict)
    #: 闭集违规与低置信度的属性名 —— 「必须上报」的具体清单。
    violations: tuple[str, ...] = ()
    low_confidence: tuple[str, ...] = ()
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "endpoint_label": self.endpoint_label,
            "threshold": self.threshold,
            "requested": list(self.requested),
            "returned": list(self.returned),
            "cropped": self.cropped,
            "image_chars": self.image_chars,
            "prompt_chars": self.prompt_chars,
            "elapsed_s": round(self.elapsed_s, 3),
            "usage": dict(self.usage),
            "violations": list(self.violations),
            "low_confidence": list(self.low_confidence),
            "error": self.error,
        }


# ============================================================================
# 2. 图像 → data URL（零硬依赖：PIL 缺席时退化成"只送原文件字节"）
# ============================================================================


_MIME_BY_SUFFIX: Mapping[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def _pil_module() -> Any:
    """惰性拿 PIL。

    惰性有两个理由：① `llm/` 的导入路径上不该出现重依赖；
    ② 单测要在**没装 PIL 的机器上**也能跑（裁剪会退化成不裁，而不是报错）。
    """
    try:
        from PIL import Image  # noqa: F401

        return Image
    except Exception:                          # noqa: BLE001
        return None


def to_data_url(image: Any) -> str:
    """把 `ImageRef` 统一成 `data:image/...;base64,...`。

    接受：文件路径（str / PathLike）/ `PIL.Image` / numpy 数组 / 已经是 data URL 的字符串 /
    `{"path": ...}`、`{"data_url": ...}`、`{"bytes": ...}` 这类字典型引用。

    统一的理由很实际：**图像从哪来不该影响角色②的契约**。今天它来自
    `dataset/scenes/*/image.jpg`，明天可能来自 viewer 传进来的内存位图；
    让 `describe()` 在两种情况下长得一样，它才是一个可替换的角色。
    """
    if image is None:
        raise VLMError("没有图像可看（image=None）")

    if isinstance(image, str):
        if image.startswith("data:"):
            return image
        return _file_to_data_url(image)

    if isinstance(image, Mapping):
        for key in ("data_url", "url"):
            v = image.get(key)
            if isinstance(v, str) and v.startswith("data:"):
                return v
        if image.get("path"):
            return _file_to_data_url(str(image["path"]))
        if image.get("bytes") is not None:
            return _bytes_to_data_url(bytes(image["bytes"]), str(image.get("mime") or "image/png"))
        raise VLMError("无法识别的图像引用：%r" % (dict(image),))

    if isinstance(image, (bytes, bytearray)):
        return _bytes_to_data_url(bytes(image), "image/png")

    if hasattr(image, "save"):                 # PIL.Image
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return _bytes_to_data_url(buf.getvalue(), "image/png")

    if hasattr(image, "shape") and hasattr(image, "dtype"):   # numpy / torch 数组
        pil = _pil_module()
        if pil is None:
            raise VLMError("收到数组图像但没有 PIL 可把它编码成图片")
        arr = image
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        if getattr(arr, "ndim", 0) == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (3, 4):
            arr = arr.transpose(1, 2, 0)       # CHW → HWC（UniDepth/SAM2 的常见排布）
        return to_data_url(pil.fromarray(arr))

    if hasattr(image, "__fspath__"):
        return _file_to_data_url(os.fspath(image))

    raise VLMError("无法识别的图像引用：%r" % (type(image).__name__,))


def _file_to_data_url(path: str) -> str:
    if not os.path.isfile(path):
        raise VLMError("图像文件不存在：%s" % path)
    mime = _MIME_BY_SUFFIX.get(os.path.splitext(path)[1].lower(), "image/png")
    with open(path, "rb") as f:
        return _bytes_to_data_url(f.read(), mime)


def _bytes_to_data_url(raw: bytes, mime: str) -> str:
    return "data:%s;base64,%s" % (mime, base64.b64encode(raw).decode("ascii"))


def _crop_for_region(image: Any, region: Region) -> tuple[Any, bool]:
    """按 `region.bbox` 裁出目标并外扩 `CROP_MARGIN`。返回 `(图像, 是否真的裁了)`。

    **裁不了就退化成整图**（PIL 缺席 / bbox 缺失 / 裁剪失败），并把 `cropped=False`
    如实记进 evidence。理由：视觉角色的价值是「给颜色/材质一个第二意见」，
    为了一次裁剪失败把它整条停掉不划算；但「有没有裁」必须可查 ——
    否则两次实验的视觉输入不一样、结果不可比，而报告里看不出来。

    ⚠ 这里必须接受 **`Path`**，不能只认 `str`：调用方最自然的写法就是传
    `Path`，而 `isinstance(image, str)` 对 `Path` 是 False ——
    那样会**静默退化成整图**（`cropped=False`），看起来一切正常，
    实际是"目标物体在图里只占 3% 像素"的小物件属性全靠模型猜。
    """
    if region.bbox is None:
        return image, False
    pil = _pil_module()
    if pil is None:
        return image, False
    try:
        if hasattr(image, "crop"):
            img = image.convert("RGB")
        elif isinstance(image, str) and image.startswith("data:"):
            return image, False                  # 已经是 data URL，没原始像素可裁
        elif isinstance(image, str) or hasattr(image, "__fspath__"):
            with pil.open(os.fspath(image)) as fh:
                img = fh.convert("RGB")
        else:
            return image, False

        w, h = img.size
        x0, y0, x1, y1 = region.bbox
        bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
        left = max(0, int(round(x0 - bw * CROP_MARGIN)))
        top = max(0, int(round(y0 - bh * CROP_MARGIN)))
        right = min(w, int(round(x1 + bw * CROP_MARGIN)))
        bottom = min(h, int(round(y1 + bh * CROP_MARGIN)))
        if right - left < 2 or bottom - top < 2:
            return image, False
        return img.crop((left, top, right, bottom)), True
    except Exception:                          # noqa: BLE001  裁剪失败不该让 describe 失败
        return image, False


# ============================================================================
# 3. 解析模型的回复
# ============================================================================


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def _extract_json(text: str) -> Any:
    """从回复里抠出 JSON。三条路径，按可靠性排序。

    为什么容忍这么多形态：视觉模型（尤其是小尺寸多模态模型）经常在 JSON 外面
    包一层解释。为格式瑕疵丢掉一次已经花掉的调用不划算 —— 但也**不能不管**：
    `_parse_attributes` 会把「解析不出任何属性」如实变成空结果 + 违规记录，
    而不是伪造一条默认值。
    """
    raw = (text or "").strip()
    if not raw:
        raise VLMError("视觉后端返回了空内容")
    for candidate in [b for b in _JSON_FENCE_RE.findall(raw)] + [raw]:
        try:
            return json.loads(candidate.strip())
        except (ValueError, TypeError):
            pass
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except (ValueError, TypeError):
            pass
    start, end = raw.find("["), raw.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except (ValueError, TypeError):
            pass
    raise VLMError("视觉后端返回的不是 JSON：%r" % raw[:200])


def _normalize(value: Any) -> str:
    return str(value or "").strip().strip("。.,，;；\"'").lower()


def _parse_attributes(
    payload: Any,
    attrs: Sequence[str],
    candidates: Mapping[str, Sequence[str]] | None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> tuple[list[Attribute], tuple[str, ...], tuple[str, ...]]:
    """把 JSON 变成 `Attribute` 列表，返回 `(属性, 闭集违规, 低置信度)`。

    三个必须守住的点：
      · **顺序按 `attrs` 走**，不按模型给的顺序 —— 逐字节稳定才可 diff；
      · **缺 confidence 记 0.0**（见模块 docstring 第 3 条）；
      · **闭集违规不算错、但要记**，并把置信度压到 0 让上层必须上报。

    `threshold` 必须由调用方传（即 `VLM.threshold`），**不能用模块常量** ——
    否则「把阈值调高/调低」这件事只改了报告里印出来的那个数字，
    而 `low_confidence` 清单还是按 0.6 算的，两边对不上。
    """
    entries: dict[str, Mapping[str, Any]] = {}

    if isinstance(payload, Mapping):
        inner = payload.get("attributes", payload)
        if isinstance(inner, Mapping):
            # `{"color": "black", "confidence": {...}}` 这种省事写法也认
            for name in attrs:
                if name in inner:
                    v = inner[name]
                    entries[name] = v if isinstance(v, Mapping) else {"value": v}
        elif isinstance(inner, list):
            for item in inner:
                if isinstance(item, Mapping) and item.get("name"):
                    entries[str(item["name"]).strip().lower()] = item
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, Mapping) and item.get("name"):
                entries[str(item["name"]).strip().lower()] = item

    out: list[Attribute] = []
    closed_violations: list[str] = []
    low: list[str] = []

    for name in attrs:
        item = entries.get(name)
        if item is None:
            # 模型漏答了一条。**不补默认值** —— 记成空值 + 零置信度，让上层看得见。
            out.append(Attribute(name=name, value="", confidence=0.0,
                                 in_closed_set=False, raw=""))
            closed_violations.append(name)
            continue

        raw_value = item.get("value")
        value = "" if raw_value is None else str(raw_value).strip()

        try:
            conf = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = 0.0 if conf != conf else max(0.0, min(1.0, conf))   # NaN → 0

        allowed = (candidates or {}).get(name)
        in_set = True
        if allowed:
            lookup = {_normalize(a): a for a in allowed}
            hit = lookup.get(_normalize(value))
            if hit is None:
                in_set = False
                conf = 0.0
                closed_violations.append(name)
            else:
                value = hit                      # 归一到闭集里的**原写法**（大小写等）

        if conf < threshold:
            low.append(name)

        out.append(Attribute(name=name, value=value, confidence=conf,
                             in_closed_set=in_set, raw="" if raw_value is None else str(raw_value)))

    return out, tuple(closed_violations), tuple(dict.fromkeys(low))


# ============================================================================
# 4. 角色② 本体
# ============================================================================


_PROMPT = """\
下面这张图是从一张房间照片里裁出来的，里面有一个「{label}」。

请**只看这张图**，回答这个物体在视觉上的下列属性：
{attrs}

要求：
1. 只输出 JSON，不要任何解释文字。格式：
   {{"attributes": [{{"name": "<属性名>", "value": "<取值>", "confidence": <0~1 的小数>}}]}}
2. `confidence` 是你对这个判断的把握程度，必须给。看不清就给低分 —— **低分是可以接受的，猜错不是**。
3. 只回答上面列出的属性，不要多答别的，也不要描述它的位置、朝向、远近或大小。
{candidates}
"""


def _build_prompt(attrs: Sequence[str], candidates: Mapping[str, Sequence[str]] | None,
                  label: str) -> str:
    attr_lines = []
    for a in attrs:
        allowed = (candidates or {}).get(a)
        if allowed:
            attr_lines.append("  - %s ∈ {%s}（**必须从这个集合里选一个词**）"
                              % (a, ", ".join(str(x) for x in allowed)))
        else:
            attr_lines.append("  - %s（用最简短的词回答，例如颜色就一个词）" % a)
    if candidates:
        tail = ("4. 凡是给了取值集合的属性，`value` **必须原样落在集合内**，"
                "不要用同义词，不要拼新的词。")
    else:
        tail = "4. 用最简短的说法（颜色一个词、材质一个词），不要写句子。"
    return _PROMPT.format(
        label=label or "物体",
        attrs="\n".join(attr_lines),
        candidates=tail,
    )


class VLM:
    """`describe()` 的实现，持有一个**视觉端点**的客户端。

    `client` 可注入（单测零联网）；不注入时按 `SPATIAL_VISION_*` → `VADAR_VISION_*`
    → 文本端点的顺序解析配置。**模型名从不硬编码**。
    """

    #: ★ 契约：形参只许是这四个。`check_signature()` 按它核对。
    CONTRACT_PARAMS: tuple[str, ...] = ("image", "region", "attrs", "candidates")

    def __init__(
        self,
        client: Any | None = None,
        *,
        settings: LLMSettings | None = None,
        threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if client is None:
            text = LLMSettings.from_env("text", environ=environ)
            st = settings or LLMSettings.from_env("vision", environ=environ, vision_fallback=text)
            client = LLMClient(settings=st, environ=environ)
        self.client = client
        self.threshold = float(threshold)
        #: 最近一次调用的侧事实 —— 工具层拿它填 evidence（模型名、用量、耗时）。
        self.last_call: DescribeCall = DescribeCall(threshold=self.threshold)

    # -- 便利 ----------------------------------------------------------------

    @property
    def settings(self) -> LLMSettings:
        return self.client.settings

    @property
    def usage(self) -> Any:
        return self.client.usage

    def ready(self) -> bool:
        return bool(self.settings.ready())

    def describe_dict(self) -> dict[str, Any]:
        """进实验产物的形态（密钥只以掩码出现）。"""
        d = self.settings.describe()
        d["confidence_threshold"] = self.threshold
        d["attrs"] = list(ATTRS)
        return d

    # -- 契约本体 ------------------------------------------------------------

    def describe(
        self,
        image: Any,
        region: Region,
        attrs: Sequence[str],
        candidates: dict[str, list[str]] | None = None,
    ) -> list[Attribute]:
        """看图回答 `attrs` 里的属性。**返回值里没有位置信息，也不接受空间提问。**

        - `attrs` 超出 `ATTRS` → `VLMError`（参数写错是程序的问题，不是场景的问题）；
        - `candidates` 给了就强制闭集，落不进去的属性会被标 `in_closed_set=False` 且置信度压 0；
        - 置信度低于 `self.threshold` 的**原样返回**，由调用方上报 —— 本函数不替调用方做取舍。
        """
        import time

        want = [str(a).strip().lower() for a in (attrs or ())]
        if not want:
            raise VLMError("attrs 为空 —— 至少指定一个属性（可用：%s）" % ", ".join(ATTRS))
        bad = [a for a in want if a not in ATTRS]
        if bad:
            raise VLMError(
                "不支持的属性 %s。可用：%s。（空间类属性（位置/朝向/远近/大小）"
                "**刻意不在此列** —— 空间关系一律由几何层计算。）"
                % (bad, ", ".join(ATTRS))
            )

        t0 = time.perf_counter()
        before = self.client.usage.snapshot()
        prompt = _build_prompt(want, candidates, region.label)

        try:
            cropped_image, cropped = _crop_for_region(image, region)
            data_url = to_data_url(cropped_image)
        except VLMError as exc:
            self.last_call = DescribeCall(
                model=self.settings.model, base_url=self.settings.base_url,
                threshold=self.threshold, requested=tuple(want),
                prompt_chars=len(prompt), error=str(exc),
            )
            raise

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }]

        try:
            reply = self.client.chat(messages, purpose="describe")
        except LLMError as exc:
            self.last_call = DescribeCall(
                model=self.settings.model, base_url=self.settings.base_url,
                threshold=self.threshold, requested=tuple(want), cropped=cropped,
                image_chars=len(data_url), prompt_chars=len(prompt),
                elapsed_s=time.perf_counter() - t0,
                # 稳定前缀便于统计（与截断那条用同一套约定）：`error` 字段要能被 grep 分类，
                # 而不是一句每次都不同的自然语言。
                error="BACKEND_UNAVAILABLE：%s" % str(exc)[:360],
            )
            raise VLMError("视觉后端调用失败：%s" % exc) from exc

        try:
            payload = _extract_json(reply.text)
        except VLMError as exc:
            self.last_call = DescribeCall(
                model=self.settings.model, base_url=self.settings.base_url,
                threshold=self.threshold, requested=tuple(want), cropped=cropped,
                image_chars=len(data_url), prompt_chars=len(prompt),
                elapsed_s=time.perf_counter() - t0,
                usage=dict(reply.usage or {}), error=str(exc)[:400],
            )
            raise

        values, violations, low = _parse_attributes(payload, want, candidates, self.threshold)
        if reply.truncated:
            # 截断的 JSON 会解析成"少了几条属性"，看起来像模型漏答 —— 必须区分开。
            low = tuple(dict.fromkeys(low + tuple(v for v in want if v not in low)))

        after = self.client.usage.snapshot()
        self.last_call = DescribeCall(
            model=self.settings.model,
            base_url=self.settings.base_url,
            endpoint_label=str(self.settings.label),
            threshold=self.threshold,
            requested=tuple(want),
            returned=tuple(a.name for a in values if a.value),
            cropped=cropped,
            image_chars=len(data_url),
            prompt_chars=len(prompt),
            elapsed_s=time.perf_counter() - t0,
            usage=_delta_usage(before, after),
            violations=violations,
            low_confidence=low,
            error="FINISH_REASON_LENGTH（输出被截断，属性可能不完整）" if reply.truncated else "",
        )
        return values

    # -- 别名：让工具层读起来像契约 -------------------------------------------

    __call__ = describe


def _delta_usage(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """视觉角色的用量差 —— 直接复用 `llm.adapter.usage_delta` 的口径。"""
    from llm.adapter import usage_delta

    return usage_delta(before, after)


# ============================================================================
# 5. 架构自检：签名里不得出现任何空间参数
# ============================================================================


def check_signature(fn: Any | None = None) -> tuple[str, ...]:
    """返回 `describe()` 签名里**违反禁令**的形参名（空表示合规）。

    这是「空间幻觉在类型层面写不出来」这句话的可执行版本：
    只要有人给 `describe` 加上 `distance`、`left_of`、`above` 之类的形参，
    这个函数就会报出来 —— 而本模块**在导入时就调用它**（见文件末尾），
    于是违规的代码根本跑不起来。
    """
    target = fn if fn is not None else VLM.describe
    params = list(inspect.signature(target).parameters)
    bad: list[str] = []
    for p in params:
        low = p.lower()
        if any(term in low for term in SPATIAL_PARAM_TERMS):
            bad.append(p)
    return tuple(bad)


#: ★ 导入期硬断言。**故意不让它可配置** —— 这条约束的价值全在「不能悄悄失效」。
_violations = check_signature()
if _violations:
    raise RuntimeError(
        "角色② 的契约被破坏了：`describe()` 的形参 %s 里出现了空间词汇。\n"
        "  §13.3(3) 要求「空间幻觉在类型层面就写不出来」—— 加了这类参数，"
        "角色② 就能被问「谁在左边」，而它看图猜出来的空间判断会污染整个场景图。\n"
        "  空间关系必须由 scene_graph/relations.py 的几何计算回答。"
        % (list(_violations),)
    )
