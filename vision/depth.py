"""UniDepthV2 —— 单目度量三维（L1 感知第 ③ 步，也是整个项目的立足点）。

本文件是全项目**最重要的一段包装**，因为 Phase 0 的核心发现就发生在这里：

    早期基线只读 `depth` 这一路：
        preds = self.unidepth_model.infer(rgb)["depth"].squeeze().cpu().numpy()

    而 UniDepth 同一个 `infer()` 还返回 `points`（相机系 XYZ 点云，单位米）。
    源码里 `depth` 就是 `points` 的 z 列：
        unidepthv2.py:334  out["radius"] = points.norm(dim=1, keepdim=True)
        unidepthv2.py:335  out["depth"]  = points[:, -1:]
    实测 `points[2] - depth` 的最大差是 **0.000e+00**（逐位相同）。

    ⟹ 只读 `depth` 等于手里握着完整的三维点云，却只留 z 列、丢掉 x/y。
       升级到真三维**不需要任何新模型、不需要任何额外显存**。

三条写进代码的结论：

① 返回 `points`，不是 `depth`。`DepthField` 里两者都留，但下游一律用 `points`。

② **不要**用 `intrinsics` 反投影重建点云。`rays` 是 decoder **预测**的方向场
   （不是解析针孔网格），`intrinsics` 是**另一个独立预测头**；而且 `infer()` 内部
   会先 padding 到 ratio_bounds、再 resize 到 pixels_bounds、最后把点云裁回
   （`:282-336`），K 是解析换算出来的。实测两者差 3.69%（均值）/ 5.30%（最大）——
   是正常量级，不是 bug。用 `points` 就不必付这道二次误差。
   判定用的也是相对误差：480p、2–4 m 场景里要求 2 cm（0.5%）对单目模型不现实。

③ 输入约定沿用实测通过的那一套：`uint8`、`(3,H,W)`、**无 batch 维、无归一化**
   （`torch.from_numpy(np.array(image)).permute(2,0,1)`）。这是 Phase 0 实测通过的
   调用方式，换成「标准」的 float/归一化反而没有验证过。

④ **内参来源决定横向尺度，量级远超换模型。** Phase 0 Step 6 用仓库自带的
   GT 深度（`assets/demo/depth.png`，毫米）逐像素对比了两条路径：

       路径                        深度ARel   δ<1.25   三维误差中位   三维相对中位
       infer(rgb)                    19.8%    57.1%      1.943 m        59.0%
       infer(rgb, camera=GT K)       11.7%    93.2%      0.267 m         9.1%

   同一张图、同一份权重，只因为「让模型猜相机」还是「把相机告诉它」，
   三维误差中位数差了 **7.3 倍**。而 UniDepth 自己的 `scripts/demo.py:14`
   从来就是传 camera 的 —— README 把它写成 "as well"（锦上添花）是误导。

   所以 `camera_K` 是本模块的一等参数。**并且**：传入 camera 后模型回传的
   `intrinsics` 依旧是错的（实测仍写 fx=163.7），必须用传入值覆盖 ——
   这条规则集中在 `resolve_intrinsics()` 里，可单独单测。

实测代价（§5，RTX 4060 Laptop）：34.2 M 参数 / 130.4 MB 权重 / 加载 3.1 s /
单帧 789–1120 ms / 峰值 486 MB allocated、604 MB reserved。
带 `camera` 条件时单帧 +73 ms（47 → 120 ms，`torch.cuda.synchronize()` 夹住的同步计时）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vision.types import DepthField

__all__ = ["UniDepthLifter", "DEFAULT_REPO", "resolve_intrinsics"]

#: 权重 130.4 MB。许可 **CC BY-NC 4.0** —— 课程/学术可用，不可商用（报告须注明）。
DEFAULT_REPO = "lpiccinelli/unidepth-v2-vits14"

#: 键名优先级：先精确匹配，再退化成子串搜索。
#: 用「先精确后模糊」而不是纯模糊，是因为模糊搜索会在版本改名时**悄悄**命中
#: 一个含义不同的键（例如 `depth_features` 也含 "depth"）。
_POINTS_KEYS = ("points", "pts_3d", "point_cloud", "xyz")
_DEPTH_KEYS = ("depth", "depth_map")
_K_KEYS = ("intrinsics", "K", "camera_k")


def _pick(out: dict, keys: tuple[str, ...], fragments: tuple[str, ...] = ()) -> str | None:
    for k in keys:
        if k in out:
            return k
    for k in sorted(out.keys()):
        kl = str(k).lower()
        if any(f in kl for f in fragments):
            return k
    return None


def _to_numpy(t: Any) -> np.ndarray:
    a = t.detach().float().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    return np.asarray(a, dtype=np.float64)


def resolve_intrinsics(
    predicted: Any,
    provided: Any = None,
) -> tuple[np.ndarray, str]:
    """决定 `points` 究竟按**哪一份**内参生成，并把它回传给下游。

    规则只有两条，但第 ① 条是 Phase 0 Step 6 的一个真实陷阱：

    ① 给了 `provided` → **实际生效的是 provided**（`infer(camera=...)` 把 camera
       转成 rays 喂进 decoder，`unidepthv2.py:361-362`），但模型回传的
       `out["intrinsics"]` 仍是它**自己相机头**的输出 —— 实测传入 GT 的
       `fx=518.9`，回传依然写 `fx=163.7`。
       ⟹ 必须用 `provided` **覆盖**回传值。否则 `scene.camera_intrinsics`
       会留下一个「与点云不自洽」的 K：点云是按 519 生成的，K 却记着 163.7，
       下游拿它做可视化或反投影核对时，会得到一个自相矛盾的结果，
       而这正是最难查的一类 bug —— 每一步单独看都对。

    ② 没给 → 用回传值，来源标为 `predicted`。此时调用方**必须**配合
       `vision.geometry.check_fov` 判断可信度。见 `phase0/probe_depth_gt.py`：
       这条路径在同一张图上三维误差中位 1.943 m，是给了 GT 内参时的 7.3 倍。
    """
    P = np.asarray(predicted, dtype=np.float64).reshape(3, 3)
    if provided is None:
        return P, "predicted"
    K = np.asarray(provided, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(K).all():
        raise ValueError("传入的已知内参含非有限值")
    if K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
        raise ValueError(
            f"传入的已知内参 fx/fy 必须为正，收到 fx={K[0, 0]}, fy={K[1, 1]}"
        )
    return K, "provided"


class UniDepthLifter:
    """升维器句柄。模型是实例属性（见 `grounding.py` 的同一说明）。"""

    def __init__(self, model: Any, repo: str, device: Any) -> None:
        self.model = model
        self.repo = repo
        self.device = device

    @classmethod
    def from_pretrained(cls, repo: str, device: Any) -> "UniDepthLifter":
        from unidepth.models import UniDepthV2

        model = UniDepthV2.from_pretrained(repo).to(device).eval()
        return cls(model, repo, device)

    @property
    def n_params_m(self) -> float:
        return sum(p.numel() for p in self.model.parameters()) / 1e6

    def __call__(self, image: Any, camera_K: Any = None) -> DepthField:
        """`camera_K` 非空 = 已知内参，会被当作条件喂给模型（见 `resolve_intrinsics`）。"""
        import torch

        from vision.geometry import check_fov

        rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(self.device)
        camera = None
        if camera_K is not None:
            # 3×3 tensor 会被 `infer` 自动转成 Pinhole（README.md:147-152），
            # 但显式构造 Pinhole 更好：它把「这是一个针孔假设」写在了调用点上，
            # 而不是藏在一段 isinstance 判断里。
            from unidepth.utils.camera import Pinhole

            K_in = np.asarray(camera_K, dtype=np.float64).reshape(3, 3)
            camera = Pinhole(
                K=torch.tensor(K_in, dtype=torch.float32, device=self.device).unsqueeze(0)
            )

        with torch.no_grad():
            out = self.model.infer(rgb, camera) if camera is not None else self.model.infer(rgb)

        k_pts = _pick(out, _POINTS_KEYS, ("point", "xyz", "cloud"))
        if k_pts is None:
            raise RuntimeError(
                f"UniDepth 未返回点云，实际返回键：{sorted(out.keys())}。"
                "本项目依赖 `points`，不能用 depth × K 重建（见模块 docstring ②）。"
            )
        pts = _to_numpy(out[k_pts])
        if pts.ndim == 4:
            pts = pts[0]
        if pts.ndim != 3:
            raise RuntimeError(f"点云形状无法解释：{pts.shape}")
        if pts.shape[0] != 3 and pts.shape[-1] == 3:
            pts = pts.transpose(2, 0, 1)
        if pts.shape[0] != 3:
            raise RuntimeError(f"点云第一维不是 3：{pts.shape}")

        k_depth = _pick(out, _DEPTH_KEYS, ("depth",))
        if k_depth is not None:
            depth = _to_numpy(out[k_depth])
            while depth.ndim > 2:
                depth = depth[0]
        else:
            # `depth` 按定义就是 z 列，缺了也能精确重建 —— 这不是猜测，是恒等式。
            depth = pts[2].copy()

        k_K = _pick(out, _K_KEYS, ("intrin", "camera_k"))
        if k_K is None:
            if camera_K is None:
                raise RuntimeError(f"UniDepth 未返回内参，实际返回键：{sorted(out.keys())}")
            K_pred = np.asarray(camera_K, dtype=np.float64).reshape(3, 3)
        else:
            K_pred = _to_numpy(out[k_K])
            while K_pred.ndim > 2:
                K_pred = K_pred[0]
            if K_pred.shape != (3, 3):
                raise RuntimeError(f"内参不是 3×3：{K_pred.shape}")

        # ★ 实际生效的那一份 K（可能是覆盖后的 provided），见 `resolve_intrinsics`。
        K, source = resolve_intrinsics(K_pred, camera_K)

        hp, wp = int(pts.shape[1]), int(pts.shape[2])
        wi, hi = image.size
        fov = check_fov(K, (int(hi), int(wi)))
        if source == "predicted" and not fov.plausible:
            # 只在控制台提示一次。真正把它变成「会进报告的错误」是 builder 的事
            # （那里才有 warnings 通道），这一层只负责不让它静默。
            import warnings as _warnings

            _warnings.warn(
                f"UniDepth 预测的内参视场不可信（HFoV {fov.hfov_deg:.1f}°，"
                f"reason={fov.reason}）—— 所有横向米制尺寸可能被整体放大。"
                "有已知内参请通过 camera_K 传入。详见 phase0/probe_depth_gt.py。",
                RuntimeWarning,
                stacklevel=2,
            )

        raw = {
            str(k): (list(v.shape) if hasattr(v, "shape") else type(v).__name__)
            for k, v in out.items()
        }
        return DepthField(
            points_chw=pts,
            depth_hw=depth,
            intrinsics=K,
            grid_hw=(hp, wp),
            image_hw=(int(hi), int(wi)),
            raw=raw,
            intrinsics_source=source,  # type: ignore[arg-type]
            fov=fov,
        )
