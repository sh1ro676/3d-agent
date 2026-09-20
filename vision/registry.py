"""`PerceptionStack` —— 三个模型的常驻/懒加载/显存记账策略。

对应方案文档 §17 目录树里的 `vision/registry.py`。

**它存在要解决的两个具体问题**（不是抽象的「架构整洁」）：

① **生命周期必须是显式的。** Phase 0 的探针把 GroundingDINO 的 model 放在
   局部变量里，函数一返回权重就被回收 —— 于是「三模型同时常驻」测出来是
   291 MB，而真实值是 1200 MB（§20 Step 0.5b）。做成实例属性后，
   「谁还持有引用」这个问题有了确定答案。

② **显存必须有账。** 实测三模型常驻合计 1200 MB = 8188 MiB 的 14.7%，
   这是「8GB 上还能塞下一个本地 LLM」这条判断的**唯一依据**。
   所以每次加载都要记账，而且记账结果要进 `build_meta` 落盘。

默认走本地权重目录（`.cache/models/...`，由 `phase0/01c`、`01d` 脚本固定版本），
目录不存在才退回 hub id —— 前者保证离线可用与版本可复现，后者保证换台机器能跑。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from vision.types import Detection, DepthField

__all__ = ["PerceptionStack", "ModelSpec", "PROJECT_ROOT", "DEFAULT_GDINO", "DEFAULT_SAM2", "DEFAULT_UNIDEPTH"]

#: 项目根目录。`vision/` 的上一级。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_GDINO_DIR = PROJECT_ROOT / ".cache" / "models" / "grounding-dino-tiny"
DEFAULT_SAM2_DIR = PROJECT_ROOT / ".cache" / "models" / "sam2.1-hiera-base-plus"

#: 本地权重缺失时退回的 hub id。
DEFAULT_GDINO = "IDEA-Research/grounding-dino-tiny"
DEFAULT_SAM2 = "facebook/sam2.1-hiera-base-plus"
DEFAULT_UNIDEPTH = "lpiccinelli/unidepth-v2-vits14"


@dataclass(slots=True)
class ModelSpec:
    """一个模型的「去哪儿取权重」描述，以及取到之后的记账。

    **刻意不是 frozen**：它是可变账本 —— 加载后要往里写来源、耗时、显存、参数量。
    做成 frozen 会让 `_load()` 只能重建整个对象，而重建意味着
    「谁拿着旧引用」这个问题重新出现一次，正是本模块要消灭的那类 bug。
    （同一个理由也解释了为什么它不做成 Pydantic 模型：它不是数据契约，
    是一块账本，没有需要校验的外部输入。）
    """

    role: str
    local_dir: Path | None
    hub_id: str
    loaded: bool = False
    source: str = ""
    load_s: float = 0.0
    resident_mb: float = 0.0
    n_params_m: float = 0.0
    note: str = ""

    def resolve(self) -> tuple[str, str]:
        """返回 `(可加载的 id, 来源标记)`。本地有权重就用本地。"""
        if self.local_dir is not None and (self.local_dir / "model.safetensors").is_file():
            return str(self.local_dir), "local"
        return self.hub_id, "hub"

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "hub_id": self.hub_id,
            "loaded": self.loaded,
            "source": self.source,
            "load_s": round(self.load_s, 2),
            "resident_mb": round(self.resident_mb, 1),
            "n_params_m": round(self.n_params_m, 1),
            "note": self.note,
        }


@dataclass
class PerceptionStack:
    """L1 感知栈。三个模型**按需加载**，第一次用到才拉起来。

    `device` 为 None 时自动选：有 CUDA 用 CUDA，否则 CPU（CPU 上这三个模型
    慢到不可用，但至少能让单元测试与流程联调跑完，并明确打上 `cpu=True` 标记）。
    """

    device: Any = None
    gdino_id: str | None = None
    sam2_id: str | None = None
    unidepth_repo: str = DEFAULT_UNIDEPTH
    specs: dict[str, ModelSpec] = field(default_factory=dict)

    _gdino: Any = None
    _sam2: Any = None
    _unidepth: Any = None

    def __post_init__(self) -> None:
        # 空值一律视为「用默认」。
        # 为什么要在类里兜这一层，而不是要求调用方别传 None：
        # CLI 的参数默认值天然就是 `None`（`--unidepth` 不传就是 None），
        # 于是 `PerceptionStack(unidepth_repo=args.unidepth)` 会把默认值覆盖成 None，
        # 症状是 `from_pretrained(None)` → 报错信息里出现 `hf-mirror.com/None/...`，
        # 指向一个根本不存在的仓库名，很难一眼看出是「没传」而不是「传错了」。
        if not self.unidepth_repo:
            self.unidepth_repo = DEFAULT_UNIDEPTH
        if self.device is None:
            self.device = self._auto_device()
        if not self.specs:
            self.specs = {
                "grounding": ModelSpec(
                    role="grounding",
                    local_dir=DEFAULT_GDINO_DIR,
                    hub_id=self.gdino_id or DEFAULT_GDINO,
                    note="开放词汇检测；prompt 须小写句点结尾",
                ),
                "segmentation": ModelSpec(
                    role="segmentation",
                    local_dir=DEFAULT_SAM2_DIR,
                    hub_id=self.sam2_id or DEFAULT_SAM2,
                    note="提示式分割；必须批量一次调用（快 7.57×）",
                ),
                "depth": ModelSpec(
                    role="depth",
                    local_dir=None,  # UniDepth 权重在 HF 缓存里，由 from_pretrained 直接取
                    hub_id=self.unidepth_repo,
                    note="点云来源；许可 CC BY-NC 4.0，不可商用",
                ),
            }
        if self.gdino_id:
            self.specs["grounding"] = ModelSpec(
                role="grounding", local_dir=None, hub_id=self.gdino_id,
                note=self.specs["grounding"].note,
            )
        if self.sam2_id:
            self.specs["segmentation"] = ModelSpec(
                role="segmentation", local_dir=None, hub_id=self.sam2_id,
                note=self.specs["segmentation"].note,
            )

    # -- 设备与显存 ------------------------------------------------------------

    @staticmethod
    def _auto_device() -> Any:
        import torch

        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @property
    def is_cuda(self) -> bool:
        return getattr(self.device, "type", "cpu") == "cuda"

    def resident_mb(self) -> float:
        """当前 torch 已分配的显存（MB）。CPU 上恒为 0。"""
        if not self.is_cuda:
            return 0.0
        import torch

        return float(torch.cuda.memory_allocated() / 1024 ** 2)

    def peak_mb(self) -> float:
        if not self.is_cuda:
            return 0.0
        import torch

        return float(torch.cuda.max_memory_allocated() / 1024 ** 2)

    # -- 懒加载 ----------------------------------------------------------------

    def _load(self, role: str) -> Any:
        """加载指定角色并把耗时/显存记进 `specs[role]`。已加载则直接返回。"""
        spec = self.specs[role]
        before = self.resident_mb()
        t0 = time.time()
        model_id, source = spec.resolve()

        if role == "grounding":
            from vision.grounding import GroundingDetector

            obj = GroundingDetector.from_pretrained(model_id, self.device)
        elif role == "segmentation":
            from vision.segmentation import Sam2Segmenter

            obj = Sam2Segmenter.from_pretrained(model_id, self.device)
        elif role == "depth":
            from vision.depth import UniDepthLifter

            obj = UniDepthLifter.from_pretrained(model_id, self.device)
        else:
            raise KeyError(f"未知角色 {role!r}")

        spec.loaded = True
        spec.source = source
        spec.load_s = time.time() - t0
        spec.n_params_m = float(getattr(obj, "n_params_m", 0.0))
        # 记账用增量而不是「加载后总量」——三模型共存时，后者无法归因到某一个。
        spec.resident_mb = max(0.0, self.resident_mb() - before)
        return obj

    @property
    def grounding(self) -> Any:
        if self._gdino is None:
            self._gdino = self._load("grounding")
        return self._gdino

    @property
    def segmentation(self) -> Any:
        if self._sam2 is None:
            self._sam2 = self._load("segmentation")
        return self._sam2

    @property
    def depth(self) -> Any:
        if self._unidepth is None:
            self._unidepth = self._load("depth")
        return self._unidepth

    def load_all(self) -> "PerceptionStack":
        """一次性把所有模型拉起来。

        批量评测时应该显式调用它：懒加载会让「中途首次用到某模型」的那一题
        多出几秒加载时间，污染延迟指标的尾部分布。预热之后再计时才干净。
        """
        _ = self.grounding, self.segmentation, self.depth
        return self

    def unload(self, role: str) -> None:
        """卸载一个模型并释放缓存。用于串行加载策略与显存紧张的对照实验。"""
        import torch

        attr = {"grounding": "_gdino", "segmentation": "_sam2", "depth": "_unidepth"}[role]
        setattr(self, attr, None)
        self.specs[role].loaded = False
        if self.is_cuda:
            torch.cuda.empty_cache()

    # -- PerceptionLike 接口 ---------------------------------------------------

    def detect(
        self,
        image: Any,
        prompt: str,
        *,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
    ) -> list[Detection]:
        return self.grounding(image, prompt,
                              box_threshold=box_threshold,
                              text_threshold=text_threshold)

    def segment(self, image: Any, boxes: Sequence[Sequence[float]]) -> Any:
        return self.segmentation(image, boxes)

    def lift(self, image: Any, camera_K: Any = None) -> DepthField:
        """`camera_K` 非空 = 已知内参，原样透传给 `UniDepthLifter`。

        放在这里（而不是 registry 自己存一份）是因为内参是**逐图的**：
        它的 cx/cy 与图像分辨率绑定，换一张图就失效。registry 只管模型生命周期，
        不该持有任何与具体图像绑定的状态。
        """
        return self.depth(image, camera_K=camera_K)

    # -- 统计 ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """进 `SceneGraph.build_meta` 的感知栈指纹（可复现性的一半）。"""
        total = 0.0
        if self.is_cuda:
            import torch

            total = float(torch.cuda.get_device_properties(0).total_memory / 1024 ** 2)
        return {
            "device": str(self.device),
            "device_total_mib": round(total, 0),
            "resident_mb": round(self.resident_mb(), 1),
            "resident_pct": round(self.resident_mb() / total * 100, 1) if total else None,
            "peak_mb": round(self.peak_mb(), 1),
            "models": {k: v.as_dict() for k, v in self.specs.items()},
        }
