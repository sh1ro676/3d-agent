#!/usr/bin/env python3
r"""
Phase 0 / Step 4 -- SAM2.1 segmentation probe (the last estimated numbers).

probe3d.py replaced every estimated figure in the plan document except SAM2's.
This script closes that gap, and while the mask is in hand it also measures
whether SAM2 is actually worth using for the 3D centre.

  A. Sam2Model cost on the 4060: parameters, load time, per-box latency and
     peak VRAM -- measured STANDALONE, so the number is not polluted by the
     other two models
  B. batched prompting: one call carrying N boxes vs N calls carrying one box.
     A living room has ~9 objects, so this decides the Phase 1 call shape
  C. all three models resident at once (GroundingDINO + SAM2 + UniDepth).
     This is the budget the project actually has to live inside
  D. box-median vs mask-median 3D centre. probe3d.py takes the median over
     every pixel inside the bounding box, which necessarily swallows
     background. If the mask moves the centre by more than the tolerances used
     in the plan document, the mask must replace the box in Phase 1 -- and
     that becomes a measurement rather than a preference.

Run (Windows native):

    D:\3D_Spatial_Agent\venvs\vision\Scripts\python.exe D:\3D_Spatial_Agent\phase0\probe_sam2.py

Weights live in .cache\models\sam2.1-hiera-base-plus (fetch with
01d_fetch_sam2.ps1). Falls back to the bare hub id when that directory is
missing, which would then download into the HF cache instead.

Optional:
    --image  C:\path\to\photo.jpg   a real photo instead of the synthetic room
    --prompt "sofa. chair. table."  detector labels, lowercase, period after each
    --boxes  "10,20,110,200;300,50,400,300"   skip detection, use these xyxy boxes
    --max-boxes 9                   cap how many boxes go into the batch test
    --out    D:\...\probe_sam2_result.json
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULT: dict = {}


# ----------------------------------------------------------------------
# tiny helpers, kept independent of probe3d so this script also works alone
# ----------------------------------------------------------------------
def hr(title: str) -> None:
    print()
    print("=" * 68)
    print("  " + title)
    print("=" * 68)


def reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def reserved_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_reserved() / 1024 ** 2


def loaded_mb() -> float:
    """Currently live allocation, without resetting anything."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024 ** 2


def as_chw(t) -> np.ndarray:
    a = t.detach().float().cpu().numpy()
    if a.ndim == 4:
        a = a[0]
    if a.ndim == 3 and a.shape[0] == 3:
        return a
    if a.ndim == 3 and a.shape[-1] == 3:
        return a.transpose(2, 0, 1)
    raise ValueError(f"cannot read as 3xHxW: {a.shape}")


def load_probe3d():
    """Import probe3d as a module so its verified detector/depth code is reused
    rather than reimplemented (and silently drifting from it)."""
    spec = importlib.util.spec_from_file_location("probe3d", HERE / "probe3d.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not locate probe3d.py next to this script")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["probe3d"] = mod
    spec.loader.exec_module(mod)
    return mod


def default_sam2_id() -> str:
    local = ROOT / ".cache" / "models" / "sam2.1-hiera-base-plus"
    return str(local) if (local / "model.safetensors").is_file() \
        else "facebook/sam2.1-hiera-base-plus"


# ----------------------------------------------------------------------
# SAM2 loading + inference
# ----------------------------------------------------------------------
def load_sam2(model_id: str, device: torch.device):
    """Load Sam2Processor + a model class.

    The published config declares architectures: ["Sam2VideoModel"], so
    Sam2VideoModel is tried after Sam2Model rather than instead of it. Both
    accept the same image-prompt weights; only the wrapper differs.
    """
    import transformers
    from transformers import Sam2Processor

    proc = Sam2Processor.from_pretrained(model_id)
    last_err = None
    for cls_name in ("Sam2Model", "Sam2VideoModel"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            print(f"  [skip] transformers has no {cls_name}")
            continue
        try:
            model = cls.from_pretrained(model_id).to(device).eval()
            print(f"  model class         : {cls_name}")
            return proc, model, cls_name
        except Exception as e:                                   # noqa: BLE001
            last_err = e
            print(f"  [try ] {cls_name} failed: {type(e).__name__}: {e}")
    raise RuntimeError(f"could not load SAM2 with any known class: {last_err}")


def sam2_call(proc, model, image, boxes, device):
    """One processor call carrying `boxes` (list of xyxy) for a single image.

    The kwarg names are introspected rather than assumed: several transformers
    processors renamed `image` to `images`, and the failure mode is a
    misleading "Either images or original_sizes must be provided" raised from
    inside the processor, which points at the wrong argument entirely.
    """
    sig = set(inspect.signature(proc.__call__).parameters)
    img_kw = "images" if "images" in sig else "image"
    if "input_boxes" not in sig:
        raise RuntimeError(
            f"this processor exposes no input_boxes kwarg; available: {sorted(sig)}")
    inputs = proc(**{img_kw: image, "input_boxes": [boxes], "return_tensors": "pt"})
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v)
              for k, v in inputs.items()}

    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        out = model(**inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.time() - t0) * 1000

    masks = None
    if hasattr(proc, "post_process_masks"):
        try:
            psig = set(inspect.signature(proc.post_process_masks).parameters)
            pkw: dict = {"masks": out.pred_masks.cpu()}
            for name in ("original_sizes", "reshaped_input_sizes"):
                if name in psig and name in inputs:
                    v = inputs[name]
                    pkw[name] = v.cpu() if torch.is_tensor(v) else v
            masks = proc.post_process_masks(**pkw)
        except Exception as e:                                   # noqa: BLE001
            print(f"  [warn] post_process_masks failed: {type(e).__name__}: {e}")
            print(f"         inputs carried: {sorted(inputs.keys())}")
    return masks, out, latency_ms


def masks_to_bool(masks) -> np.ndarray:
    """Normalise whatever post_process_masks returned into (N, H, W) bool.

    Handles: list of tensors, a single tensor of shape
    (1, N, num_masks, H, W) or (N, num_masks, H, W), and logits vs bools.
    When several masks come back per box the highest-IoU one is picked
    separately via pick_best().
    """
    if isinstance(masks, (list, tuple)):
        t = masks[0]
    else:
        t = masks
    a = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    while a.ndim > 4:
        a = a[0]
    if a.dtype != bool:
        a = a > 0
    return a


def pick_best(mask_stack: np.ndarray, iou_scores=None, idx: int = 0) -> np.ndarray:
    """mask_stack: (num_masks, H, W) for one box -> one (H, W) bool mask."""
    if mask_stack.ndim == 2:
        return mask_stack
    if iou_scores is not None and len(iou_scores) == mask_stack.shape[0]:
        idx = int(np.argmax(np.asarray(iou_scores).reshape(-1)))
    return mask_stack[min(idx, mask_stack.shape[0] - 1)]


# ----------------------------------------------------------------------
# 3D centre from a pixel selector
# ----------------------------------------------------------------------
def centre_of(points_chw: np.ndarray, selector: np.ndarray):
    """Median 3D point over the selected pixels (selector: bool, H*W).

    Median, not mean: a box or a soft mask almost always catches a few pixels
    of a far surface, and those drag a mean off by hundreds of millimetres.
    """
    patch = points_chw.reshape(3, -1)[:, selector]
    if patch.shape[1] == 0:
        return None, 0
    patch = patch[:, np.isfinite(patch).all(axis=0)]
    if patch.shape[1] == 0:
        return None, 0
    return np.median(patch, axis=1), int(patch.shape[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--sam2", type=str, default=None)
    ap.add_argument("--gdino", type=str, default=None)
    ap.add_argument("--prompt", type=str, default="sofa. chair. table. picture. mirror.")
    ap.add_argument("--boxes", type=str, default=None,
                    help="'x1,y1,x2,y2;x1,y1,x2,y2' -- skip detection entirely")
    ap.add_argument("--max-boxes", type=int, default=9)
    ap.add_argument("--unidepth-repo", type=str, default="lpiccinelli/unidepth-v2-vits14")
    ap.add_argument("--out", type=str, default=str(HERE / "probe_sam2_result.json"))
    args = ap.parse_args()
    if not args.sam2:
        args.sam2 = default_sam2_id()

    print()
    print("Phase 0 / Step 4 -- SAM2.1 segmentation probe")
    print(f"  torch {torch.__version__}   cuda {torch.version.cuda}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
        print(f"  device {torch.cuda.get_device_name(0)}  ({total:.0f} MiB)")
        print(f"  free before start   : {torch.cuda.mem_get_info()[0]/1024**2:.0f} MiB")
    else:
        device = torch.device("cpu")
        total = 0.0
        print("  [warn] CUDA unavailable -- numbers will not describe the 4060")

    p3 = load_probe3d()
    if args.image:
        image = p3.Image.open(args.image).convert("RGB")
        print(f"  image  {args.image}  {image.size}")
    else:
        image = p3.synthetic_room()
        print(f"  image  synthetic fallback  {image.size}")
        print("         (pass --image /path/to/room.jpg for meaningful labels)")

    # ---------- explicit boxes, or detect them ----------
    boxes: list[list[float]] = []
    detections = []
    if args.boxes:
        for chunk in args.boxes.split(";"):
            vals = [float(v) for v in chunk.replace(" ", "").split(",")]
            if len(vals) == 4:
                boxes.append(vals)
                detections.append({"label": f"box{len(boxes)}", "score": None,
                                   "box_xyxy": vals})
        print(f"  boxes  {len(boxes)} supplied on the command line")
    else:
        if not args.gdino:
            args.gdino = p3.default_gdino_id()
        detections = p3.run_gdino(image, args.gdino, device, args.prompt) or []
        boxes = [list(map(float, d["box_xyxy"])) for d in detections]

    boxes = boxes[: args.max_boxes]
    detections = detections[: args.max_boxes]
    if not boxes:
        print("  [FAIL] no boxes -- nothing to segment. Widen --prompt.")

    def dump() -> None:
        """Flush results after every section.

        A full run loads three models and takes tens of seconds; if the host
        kills the process partway through, whatever was already measured must
        survive on disk rather than dying in a stdout buffer.
        """
        try:
            Path(args.out).write_text(
                json.dumps(RESULT, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    # ==================================================================
    # A. SAM2 standalone
    # ==================================================================
    hr("A.  SAM2.1 -- cost on this machine (standalone)")
    if not boxes:
        print("  [skip] no boxes available")
    reset_peak()
    t0 = time.time()
    proc, sam2, cls_name = load_sam2(args.sam2, device)
    load_s = time.time() - t0
    n_param = sum(p.numel() for p in sam2.parameters()) / 1e6
    wbytes = sum(p.numel() * p.element_size() for p in sam2.parameters())

    print(f"  repo id             : {args.sam2}")
    print(f"  parameters          : {n_param:.1f} M   ({wbytes/1024**2:.1f} MiB of weights)")
    print(f"  load time           : {load_s:.1f} s  (includes download if uncached)")
    print(f"  after load, live    : {loaded_mb():.0f} MB allocated")
    print()
    print(f"  processor.__call__ signature:")
    try:
        print(f"    {inspect.signature(proc.__call__)}")
    except Exception:                                            # noqa: BLE001
        print("    <not introspectable>")
    print(f"  processor.post_process_masks signature:")
    try:
        print(f"    {inspect.signature(proc.post_process_masks)}")
    except Exception:                                            # noqa: BLE001
        print("    <not introspectable>")

    # ---------- single box ----------
    print()
    print("  single-box prompt:")
    if boxes:
        m1 = None
        try:
            m1, out1, lat1 = sam2_call(proc, sam2, image, [boxes[0]], device)
            arr1 = masks_to_bool(m1) if m1 is not None else None
            if arr1 is not None:
                while arr1.ndim > 3:
                    arr1 = arr1[0]
                if arr1.ndim == 3:
                    best1 = pick_best(arr1)
                else:
                    best1 = arr1
                frac = float(best1.mean())
                x1, y1, x2, y2 = boxes[0]
                box_area = max(1.0, (x2 - x1) * (y2 - y1))
                cov = float(best1.sum()) / box_area
                print(f"    latency           : {lat1:.0f} ms")
                print(f"    peak VRAM allocated: {peak_mb():.0f} MB")
                print(f"    peak VRAM reserved : {reserved_mb():.0f} MB")
                print(f"    mask covers        : {frac*100:.1f}% of the image, "
                      f"{cov*100:.1f}% of its own box")
                RESULT["sam2_single"] = {
                    "latency_ms": round(lat1, 1),
                    "peak_vram_alloc_mb": round(peak_mb(), 1),
                    "peak_vram_reserved_mb": round(reserved_mb(), 1),
                    "mask_image_fraction": round(frac, 4),
                    "mask_box_coverage": round(cov, 4),
                }
            else:
                print("    [warn] no masks returned")
        except Exception as e:                                   # noqa: BLE001
            print(f"    [FAIL] {type(e).__name__}: {e}")
            RESULT["sam2_single"] = {"error": f"{type(e).__name__}: {e}"}

    RESULT["sam2_model"] = {
        "repo": args.sam2,
        "class": cls_name,
        "params_m": round(n_param, 1),
        "weights_mib": round(wbytes / 1024 ** 2, 1),
        "load_s": round(load_s, 2),
        "live_after_load_mb": round(loaded_mb(), 1),
    }
    dump()

    # ==================================================================
    # B. batched vs per-box
    # ==================================================================
    hr("B.  Batched prompting -- one call with N boxes vs N calls with one")
    if len(boxes) < 2:
        print("  [skip] needs at least two boxes")
    else:
        n = len(boxes)
        reset_peak()
        try:
            t0 = time.time()
            mb, outb, latb = sam2_call(proc, sam2, image, boxes, device)
            wall_batch = (time.time() - t0) * 1000
            arrb = masks_to_bool(mb) if mb is not None else None
            print(f"  one call, {n} boxes  : model {latb:.0f} ms   end-to-end {wall_batch:.0f} ms"
                  f"   peak {peak_mb():.0f} MB")
            batch_rec = {"boxes": n, "model_ms": round(latb, 1),
                         "end_to_end_ms": round(wall_batch, 1),
                         "peak_vram_alloc_mb": round(peak_mb(), 1),
                         "peak_vram_reserved_mb": round(reserved_mb(), 1)}
        except Exception as e:                                   # noqa: BLE001
            print(f"  one call, {n} boxes  : [FAIL] {type(e).__name__}: {e}")
            batch_rec = {"boxes": n, "error": f"{type(e).__name__}: {e}"}
            arrb = None

        reset_peak()
        per, wall_per = [], 0.0
        try:
            for i, b in enumerate(boxes):
                t0 = time.time()
                sam2_call(proc, sam2, image, [b], device)
                wall_per += (time.time() - t0) * 1000
            print(f"  {n} calls, 1 box each: total {wall_per:.0f} ms"
                  f"   peak {peak_mb():.0f} MB")
            per_rec = {"calls": n, "end_to_end_ms": round(wall_per, 1),
                       "peak_vram_alloc_mb": round(peak_mb(), 1),
                       "peak_vram_reserved_mb": round(reserved_mb(), 1)}
        except Exception as e:                                   # noqa: BLE001
            print(f"  {n} calls, 1 box each: [FAIL] {type(e).__name__}: {e}")
            per_rec = {"calls": n, "error": f"{type(e).__name__}: {e}"}

        RESULT["sam2_batch"] = {"batched": batch_rec, "per_box": per_rec,
                                "boxes_used": n}
        if "end_to_end_ms" in batch_rec and "end_to_end_ms" in per_rec:
            sp = per_rec["end_to_end_ms"] / max(batch_rec["end_to_end_ms"], 1e-9)
            print(f"  -> batching is {sp:.2f}x faster in wall time")
        dump()

    # ==================================================================
    # C. all three models resident
    # ==================================================================
    hr("C.  GroundingDINO + SAM2 + UniDepth all resident")
    print("  This is the budget the running system has to fit inside, not the")
    print("  sum of three standalone peaks: the allocator reuses freed blocks.")
    print()
    print(f"  SAM2 resident                : {loaded_mb():7.0f} MB")

    ud_out = None
    ud_model = None
    try:
        out, ud_model = p3.run_unidepth(image, device, args.unidepth_repo)
        ud_out = out
        print()
        print(f"  + UniDepth                   : {loaded_mb():7.0f} MB live")
    except Exception as e:                                       # noqa: BLE001
        print(f"  [warn] UniDepth failed: {type(e).__name__}: {e}")

    # Hold explicit references to all three models. run_gdino() keeps its model
    # in a local, so it is collected the moment the call returns and its VRAM
    # disappears -- measuring "resident" that way reports a number no real run
    # would ever see.
    reset_peak()
    gd_model = None
    try:
        if args.gdino is None:
            args.gdino = p3.default_gdino_id()
        from transformers import (AutoModelForZeroShotObjectDetection,
                                  AutoProcessor)
        gd_proc = AutoProcessor.from_pretrained(args.gdino)
        gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            args.gdino).to(device).eval()
        live = loaded_mb()
        print()
        print(f"  + GroundingDINO              : {live:7.0f} MB live")
        print(f"    -> SAM2 + UniDepth + GroundingDINO are all referenced here")
        if total:
            print(f"  share of {total:.0f} MiB           : {live/total*100:.1f}%")
        RESULT["three_models_resident"] = {
            "live_mb": round(live, 1),
            "peak_mb_while_hot": round(peak_mb(), 1),
            "reserved_mb_while_hot": round(reserved_mb(), 1),
            "device_total_mib": round(total, 0),
            "share_pct": round(live / total * 100, 1) if total else None,
            "unidepth_loaded": ud_model is not None,
        }
    except Exception as e:                                       # noqa: BLE001
        print(f"  [warn] GroundingDINO failed: {type(e).__name__}: {e}")
    dump()

    # ==================================================================
    # D. box-median vs mask-median
    # ==================================================================
    hr("D.  Does the mask improve the 3D centre?  (box-median vs mask-median)")

    if ud_out is None:
        print("  [skip] needs UniDepth `points`.")
    elif not boxes:
        print("  [skip] needs at least one box.")
    else:
        k_pts = p3.pick_key(ud_out, "point", "xyz", "cloud")
        if k_pts is None:
            print("  [skip] UniDepth returned no point-cloud key.")
        else:
            pts = as_chw(ud_out[k_pts])
            _, HP, WP = pts.shape
            WI, HI = image.size
            sx, sy = WP / WI, HP / HI
            print(f"  geometry grid {WP}x{HP}  image {WI}x{HI}  scale ({sx:.4f}, {sy:.4f})")
            print()
            print("   label        box px        n_box   n_mask  mask/box"
                  "   centre(box) m            centre(mask) m            |delta| m")
            rows = []
            for i, (b, d) in enumerate(zip(boxes, detections)):
                x1f, y1f, x2f, y2f = [float(v) for v in b]
                bx1 = int(max(0, min(WP - 1, round(x1f * sx))))
                bx2 = int(max(0, min(WP, round(x2f * sx))))
                by1 = int(max(0, min(HP - 1, round(y1f * sy))))
                by2 = int(max(0, min(HP, round(y2f * sy))))
                if bx2 <= bx1 or by2 <= by1:
                    continue

                sel_box = np.zeros((HP, WP), dtype=bool)
                sel_box[by1:by2, bx1:bx2] = True
                c_box, n_box = centre_of(pts, sel_box.reshape(-1))

                mrec = None
                try:
                    md, _out, _lat = sam2_call(proc, sam2, image, [b], device)
                    am = masks_to_bool(md) if md is not None else None
                    if am is not None:
                        while am.ndim > 3:
                            am = am[0]
                        if am.ndim == 3:
                            iou = None
                            try:
                                sc = getattr(_out, "iou_scores", None)
                                if sc is not None:
                                    iou = sc.detach().cpu().numpy().reshape(-1)[: am.shape[0]]
                            except Exception:                        # noqa: BLE001
                                iou = None
                            best = pick_best(am, iou)
                        else:
                            best = am
                        if best.shape != (HP, WP):
                            mrec = {"shape_mismatch": list(best.shape)}
                            best = None
                        if best is not None:
                            c_mask, n_mask = centre_of(pts, best.reshape(-1))
                            if c_box is not None and c_mask is not None:
                                delta = float(np.linalg.norm(c_box - c_mask))
                                lb = str(d.get("label", i))
                                frac_box = n_mask / max(n_box, 1)
                                print(f"   {lb:<12} [{bx1:4d},{by1:4d},{bx2:4d},{by2:4d}]"
                                      f" {n_box:7d} {n_mask:7d}  {frac_box*100:6.1f}%"
                                      f"   ({c_box[0]:6.3f},{c_box[1]:6.3f},{c_box[2]:6.3f})"
                                      f"   ({c_mask[0]:6.3f},{c_mask[1]:6.3f},{c_mask[2]:6.3f})"
                                      f"   {delta:7.4f}")
                                rows.append({"label": lb, "box_xyxy": [x1f, y1f, x2f, y2f],
                                             "n_box_px": n_box, "n_mask_px": n_mask,
                                             "mask_over_box": round(frac_box, 4),
                                             "centre_box_m": [round(float(v), 4) for v in c_box],
                                             "centre_mask_m": [round(float(v), 4) for v in c_mask],
                                             "delta_m": round(delta, 4)})
                except Exception as e:                           # noqa: BLE001
                    print(f"   {str(d.get('label', i)):<12} [warn] {type(e).__name__}: {e}")

            if rows:
                deltas = [r["delta_m"] for r in rows]
                covs = [1.0 - r["mask_over_box"] for r in rows]
                print()
                print(f"  |delta| over {len(rows)} objects: "
                      f"mean {np.mean(deltas)*1000:.0f} mm   max {max(deltas)*1000:.0f} mm")
                print(f"  background inside the boxes      : "
                      f"mean {np.mean(covs)*100:.1f}%   max {max(covs)*100:.1f}%")
                print()
                print("  How to read this: the plan document treats 50 mm as its")
                print("  relation tolerance (left_of/above use tol=0.05 m). If the")
                print("  centre moves by more than that when the background inside")
                print("  the box is removed, then the box-only centre is what")
                print("  destabilises the relation, and SAM2 is not optional.")
                RESULT["centre_comparison"] = {
                    "rows": rows,
                    "delta_mean_mm": round(float(np.mean(deltas) * 1000), 1),
                    "delta_max_mm": round(float(max(deltas) * 1000), 1),
                    "bg_in_box_mean_pct": round(float(np.mean(covs) * 100), 1),
                    "bg_in_box_max_pct": round(float(max(covs) * 100), 1),
                    "tol_reference_m": 0.05,
                }
                if float(max(deltas)) > 0.05:
                    print(f"  [PASS] max shift {max(deltas)*1000:.0f} mm exceeds the 50 mm")
                    print("         tolerance -> Phase 1 must use the mask, not the box.")
                else:
                    print(f"  [INFO] max shift {max(deltas)*1000:.0f} mm stays under the 50 mm")
                    print("         tolerance -> the box is survivable, but the mask still")
                    print("         removes the background pixels from any size estimate.")
                dump()

    # ==================================================================
    hr("VERDICT")
    sm = RESULT.get("sam2_model", {})
    if sm.get("params_m"):
        print(f"  [PASS] SAM2.1 loaded as {sm.get('class')}: {sm['params_m']} M params, "
              f"{sm.get('weights_mib')} MiB of weights")
    s1 = RESULT.get("sam2_single", {})
    if s1.get("latency_ms"):
        print(f"  [PASS] single-box prompt: {s1['latency_ms']:.0f} ms, "
              f"peak {s1['peak_vram_alloc_mb']:.0f} MB allocated")
    tm = RESULT.get("three_models_resident", {})
    if tm.get("share_pct") is not None:
        print(f"  [{'PASS' if tm['share_pct'] < 90 else 'WARN'}] all three models resident: "
              f"{tm['live_mb']:.0f} MB = {tm['share_pct']:.1f}% of the 4060")
    cc = RESULT.get("centre_comparison", {})
    if cc:
        print(f"  [{'PASS' if cc['delta_max_mm'] > 50 else 'INFO'}] mask vs box centre: "
              f"mean {cc['delta_mean_mm']:.0f} mm, max {cc['delta_max_mm']:.0f} mm "
              f"(background {cc['bg_in_box_mean_pct']:.1f}% of box pixels)")
    print()

    try:
        Path(args.out).write_text(json.dumps(RESULT, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"  measured numbers written to {args.out}")
    except OSError as e:
        print(f"  [warn] could not write {args.out}: {e}")

    print("  Replace the SAM2 estimates in 3D_Spatial_Agent_技术调研与实施方案.md.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
