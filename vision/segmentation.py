"""SAM2.1 —— 提示式分割（L1 感知第 ② 步）。

为什么 Omni3D 主路径也必须分割（而 VADAR 只在 CLEVR/GQA 支线用）：
    因为三维质心**必须**取掩码内的点云中位数。用检测框会把背景算进去 ——
    实测 9 个物体上均值偏 83 mm、最大偏 208 mm，而关系判定容差是 50 mm
    （§20 Step 0.5b）。所以这是正确性要求，不是可选优化。

★ **必须批量一次调用**：9 个框一次调用 200 ms，逐个调用 1512 ms，差 7.57×。
  这是实测值，不是估计。`__call__` 因此只接受「一批框」，不给单框重载 ——
  接口形状本身就在阻止调用方写出逐个调用的版本（§18 Phase 1 硬约束）。

两个会**静默出错**的 API 陷阱（必须关键字传参 + 自省签名）：

① `Sam2Processor.__call__` 的第一个参数叫 `images=` 不是 `image=`。
   传错的报错是 `ValueError: Either images or original_sizes must be provided`
   —— 指向的是**另一个参数**，极难排查。

② `post_process_masks` **没有** `reshaped_input_sizes` 参数。真实签名是
   `(masks, original_sizes, mask_threshold=0.0, binarize=True, max_hole_area=0.0,
     max_sprinkle_area=0.0, apply_non_overlapping_constraints=False)`。
   按位置传第三个参数会被当成 **`mask_threshold`** —— 二值化阈值被静默改掉，
   不报错、不警告，只是掩码悄悄变形。
"""

from __future__ import annotations

import inspect
from typing import Any, Sequence

import numpy as np

__all__ = ["Sam2Segmenter"]


class Sam2Segmenter:
    """分割器句柄。模型与处理器都是实例属性（见 `grounding.py` 里的同一说明）。"""

    def __init__(self, processor: Any, model: Any, model_id: str, device: Any,
                 model_class: str) -> None:
        self.processor = processor
        self.model = model
        self.model_id = model_id
        self.device = device
        self.model_class = model_class
        self._call_params = set(inspect.signature(processor.__call__).parameters)
        self._post_params = set(inspect.signature(processor.post_process_masks).parameters)

    @classmethod
    def from_pretrained(cls, model_id: str, device: Any) -> "Sam2Segmenter":
        import transformers
        from transformers import Sam2Processor

        processor = Sam2Processor.from_pretrained(model_id)
        last_err: Exception | None = None
        # 权重自带的 config 声明 architectures: ["Sam2VideoModel"]，
        # 但图像提示这条路两者都能走；先试 Sam2Model，再退 Sam2VideoModel。
        for cls_name in ("Sam2Model", "Sam2VideoModel"):
            klass = getattr(transformers, cls_name, None)
            if klass is None:
                continue
            try:
                model = klass.from_pretrained(model_id).to(device).eval()
                return cls(processor, model, model_id, device, cls_name)
            except Exception as exc:  # noqa: BLE001 —— 换下一个候选类再试
                last_err = exc
        raise RuntimeError(f"无法用已知的任一类加载 SAM2：{last_err}")

    @property
    def n_params_m(self) -> float:
        return sum(p.numel() for p in self.model.parameters()) / 1e6

    # -- 内部 -----------------------------------------------------------------

    def _image_kwarg(self) -> str:
        # 坑 ① 的处理：不假设参数名，读真实签名。
        if "images" in self._call_params:
            return "images"
        if "image" in self._call_params:
            return "image"
        raise RuntimeError(
            f"该 processor 既无 images 也无 image 参数；可用：{sorted(self._call_params)}"
        )

    def _post_kwargs(self, out: Any, inputs: dict) -> dict:
        """坑 ② 的处理：**全部关键字传参**，且只传签名里确实存在的。"""
        if getattr(out, "pred_masks", None) is None:
            return {}
        kw: dict[str, Any] = {"masks": out.pred_masks.cpu()}
        for name in ("original_sizes", "reshaped_input_sizes"):
            if name in self._post_params and name in inputs:
                v = inputs[name]
                kw[name] = v.cpu() if hasattr(v, "cpu") else v
        return kw

    def _split(
        self, masks: Any, n_boxes: int, iou: np.ndarray | None
    ) -> np.ndarray:
        """把 `post_process_masks` 的返回值归成 `(N, H, W)` bool。

        形状在不同版本/不同 batch 下并不唯一：可能是
        `list[(N, M, H, W)]`、`(1, N, M, H, W)`，单框时是 `(M, H, W)`。
        这里逐个消歧，而不是假设一种。M 是「每框 3 个候选掩码」，
        用 `iou_scores` 选最好的那个 —— 不选而固定取 0 会明显变差。
        """
        t = masks[0] if isinstance(masks, (list, tuple)) and len(masks) else masks
        a = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        while a.ndim > 4:
            a = a[0]
        if a.ndim == 4:
            if a.shape[0] != n_boxes and a.shape[1] == n_boxes:
                a = a.transpose(1, 0, 2, 3)
            elif a.shape[0] != n_boxes:
                # 兜底：把前面几维压平后按 N 切开（宁可明确报错也不静默错位）
                flat = a.reshape(-1, a.shape[-2], a.shape[-1])
                if flat.shape[0] < n_boxes:
                    raise RuntimeError(
                        f"掩码数 {flat.shape[0]} 少于框数 {n_boxes}，无法一一对应"
                    )
                a = flat[:n_boxes][:, None]
        elif a.ndim == 3:
            a = a[None] if n_boxes == 1 else a[:, None]
        elif a.ndim == 2:
            a = a[None, None]
        else:
            raise RuntimeError(f"无法解释的掩码形状：{a.shape}")

        out = np.zeros((n_boxes, a.shape[-2], a.shape[-1]), dtype=bool)
        for i in range(n_boxes):
            stack = a[i]
            if stack.shape[0] == 1:
                best = stack[0]
            elif iou is not None and iou.shape[0] > i and iou.shape[1] == stack.shape[0]:
                best = stack[int(np.argmax(iou[i]))]
            else:
                best = stack[0]
            out[i] = best if best.dtype == bool else (best > 0)
        return out

    # -- 公开 -----------------------------------------------------------------

    def __call__(self, image: Any, boxes: Sequence[Sequence[float]]) -> np.ndarray:
        """`boxes`: N 个 xyxy → `(N, H, W)` bool 掩码（**输入图像**分辨率）。

        N 为 0 时返回空数组，不调用模型 —— 调用一个零框的 SAM2 只会浪费时间。
        """
        import torch

        n = len(boxes)
        if n == 0:
            h, w = (image.size[1], image.size[0]) if hasattr(image, "size") else (0, 0)
            return np.zeros((0, h, w), dtype=bool)

        inputs = self.processor(
            **{
                self._image_kwarg(): image,
                "input_boxes": [[list(map(float, b)) for b in boxes]],
                "return_tensors": "pt",
            }
        )
        inputs = {
            k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()
        }

        with torch.no_grad():
            out = self.model(**inputs)

        kw = self._post_kwargs(out, inputs)
        masks = self.processor.post_process_masks(**kw) if kw else None
        if masks is None:
            raise RuntimeError("SAM2 未返回掩码（post_process_masks 不可用或 pred_masks 缺失）")

        iou = self._iou_matrix(getattr(out, "iou_scores", None), n)
        return self._split(masks, n, iou)

    @staticmethod
    def _iou_matrix(raw: Any, n_boxes: int) -> np.ndarray | None:
        if raw is None:
            return None
        a = raw.detach().cpu().numpy() if hasattr(raw, "detach") else np.asarray(raw)
        while a.ndim > 2:
            a = a[0]
        a = np.atleast_2d(a)
        if a.shape[0] != n_boxes and a.shape[1] == n_boxes:
            a = a.T
        return a
