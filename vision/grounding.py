"""GroundingDINO —— 开放词汇检测（L1 感知第 ① 步）。

用 transformers 原生实现（`AutoProcessor` + `AutoModelForZeroShotObjectDetection`），
**不用**编译版 `groundingdino` 包：那个要现场编译 CUDA 扩展，没有官方 Windows 支持，
而权重是同一套（`grounding-dino-tiny` = SwinT-OGC 的 transformers 移植版，
172.3 M 参数、657 MiB fp32，与原版 0.69 GB 的 `.pth` 同源）。

三个必须写死在这里的坑（都是 Phase 0 实测出来的，不是查文档抄的）：

① **prompt 必须小写、每个标签以句点结尾**：`"sofa. chair. table."`。
   写成 `"sofa . chair ."`（点号两侧带空格）会被切错 token。检测回来的标签
   也跟着是小写 —— 下游 `SceneGraph.by_label` 因此做了大小写不敏感匹配。

② **返回的框是绝对像素 xyxy**，不是归一化的 cxcywh —— 与编译版相反。
   消费方按像素直接用即可；点云分辨率不同时由 `builder.py` 显式换算。

③ **后处理阈值参数名在版本间变过**：`box_threshold` → `threshold`。
   所以这里用 `inspect.signature` 读真实签名，而不是钉死一个版本再猜 ——
   猜错的症状是 `TypeError` 或更糟：传进了但被忽略，阈值静默失效。
"""

from __future__ import annotations

import inspect
from typing import Any, Sequence

import numpy as np

from vision.types import Detection, box_xyxy_of

__all__ = ["DEFAULT_PROMPT", "GroundingDetector", "post_process_kwargs"]

#: 已验证的默认 prompt（客厅类室内图）。用 `--prompt` 覆盖。
#: 标签数量会直接影响延迟与误检率：标签越多，同一个区域越容易被两个词各命中一次。
DEFAULT_PROMPT = "sofa. chair. table. picture. mirror."

#: 默认阈值。0.30 / 0.25 与 Phase 0 探针一致，保证历史数字可比。
DEFAULT_BOX_THRESHOLD = 0.30
DEFAULT_TEXT_THRESHOLD = 0.25


def post_process_kwargs(
    processor: Any,
    box_threshold: float,
    text_threshold: float,
    target_sizes: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """按**真实签名**拼后处理参数（坑 ③）。

    独立成函数是为了能单测：这个逻辑一旦错，错误方式是「阈值被静默忽略」，
    不是抛异常 —— 只能靠回归测试锁住。
    """
    params = set(inspect.signature(processor.post_process_grounded_object_detection).parameters)
    thr_kw = "threshold" if "threshold" in params else "box_threshold"
    kw: dict[str, Any] = {thr_kw: box_threshold, "target_sizes": list(target_sizes)}
    if "text_threshold" in params:
        kw["text_threshold"] = text_threshold
    return kw


def _first_present(d: Any, *names: str) -> Any:
    for n in names:
        try:
            v = d[n]
        except (KeyError, IndexError, TypeError):
            continue
        if v is not None:
            return v
    return None


class GroundingDetector:
    """检测器句柄。**模型是这个对象的属性**，不是局部变量。

    为什么值得单写一段说明：Phase 0 的探针里 `run_gdino()` 把 model 放在局部变量，
    函数一返回权重就被回收 —— 于是「三个模型同时常驻」量出来只有 291 MB = 假的
    （§20 Step 0.5b）。做成类的属性，生命周期就是显式的。
    """

    def __init__(self, processor: Any, model: Any, model_id: str, device: Any) -> None:
        self.processor = processor
        self.model = model
        self.model_id = model_id
        self.device = device

    @classmethod
    def from_pretrained(cls, model_id: str, device: Any) -> "GroundingDetector":
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        processor = AutoProcessor.from_pretrained(model_id)
        model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            .to(device)
            .eval()
        )
        return cls(processor, model, model_id, device)

    @property
    def n_params_m(self) -> float:
        return sum(p.numel() for p in self.model.parameters()) / 1e6

    def __call__(
        self,
        image: Any,
        prompt: str = DEFAULT_PROMPT,
        *,
        box_threshold: float = DEFAULT_BOX_THRESHOLD,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    ) -> list[Detection]:
        """检测。返回列表可能为空 —— 「图里没有 prompt 中的任何东西」是合法结果。"""
        import torch

        if not prompt.strip():
            raise ValueError("prompt 不能为空")

        inputs = self.processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            **post_process_kwargs(
                self.processor, box_threshold, text_threshold, [image.size[::-1]]
            ),
        )
        res = results[0]

        boxes = _first_present(res, "boxes")
        scores = _first_present(res, "scores")
        labels = _first_present(res, "labels", "text_labels") or []

        if boxes is None or len(boxes) == 0:
            return []

        arr = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
        arr = arr.reshape(-1, 4)
        sc = (
            scores.detach().cpu().numpy().reshape(-1)
            if torch.is_tensor(scores)
            else np.asarray(scores, dtype=float).reshape(-1)
        )

        out: list[Detection] = []
        for i, b in enumerate(arr):
            label = str(labels[i]).strip().lower() if i < len(labels) else ""
            if not label:
                # 无标签的框对「按类别指代」毫无用处，且会让 object_id 变成 `_1`。
                continue
            out.append(
                Detection(
                    label=label,
                    # 负 logit 经 sigmoid 后可能极小；score 在 schema 里是 [0,1]，
                    # 这里夹一下，免得一个 1.0000001 让 Pydantic 校验炸掉。
                    score=float(min(max(float(sc[i]), 0.0), 1.0)) if i < sc.size else 0.0,
                    box_xyxy=box_xyxy_of(b),
                )
            )
        return out
