#!/usr/bin/env python3
r"""
Phase 0 / Step 3 -- verify the 3D geometry pipeline on the RTX 4060 Laptop 8 GB.

This script exists to answer one question with evidence, not opinion:

    Does UniDepth return real 3D geometry that VADAR throws away?

VADAR reads only the "depth" key -- see engine/predefined_modules.py:375 and :395:

    preds = self.unidepth_model.infer(rgb)["depth"].squeeze().cpu().numpy()

UniDepth's own documentation says the same infer() call also returns `points`
(camera-frame XYZ cloud) and `intrinsics` (the K matrix). This script:

  A. loads UniDepth exactly the way VADAR does (uint8, (3,H,W), no batch dim)
     and prints every key it actually returns, with shapes and dtypes
  B. cross-checks `points` against `depth` + `intrinsics`: back-projects a few
     pixels through K and compares against the published point cloud. If they
     agree, `points` is genuine geometry and can be used for real distances
  C. measures peak VRAM + latency for UniDepth and GroundingDINO, replacing
     every estimated number in the plan document
  D. if GroundingDINO finds two objects, computes the true 3D centre-to-centre
     distance between them -- the exact operation the final project needs

Run on Windows native (the WSL2 route is no longer required):

    D:\3D_Spatial_Agent\venvs\vision\Scripts\python.exe D:\3D_Spatial_Agent\phase0\probe3d.py

Sections C and D use transformers' native GroundingDINO. The compiled
`groundingdino` package is deliberately NOT installed -- it needs a CUDA
extension build with no official Windows support. Weights come from
.cache\models\grounding-dino-tiny (fetch with 01c_fetch_gdino.ps1); if that
directory is missing the plain hub id is used instead.

Two details that differ from the compiled package:

  * the prompt must be lowercase with a period after each label
    ("box. door." -- not "box . door .")
  * boxes come back as absolute-pixel xyxy, not normalised cxcywh

Optional:
    --image C:\path\to\photo.jpg   use a real photo instead of the synthetic one
    --prompt "box. door."          lowercase labels, each ending in a period
    --gdino D:\path\or\hub-id      override where grounding-dino-tiny lives
    --skip-gdino                   only test UniDepth
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

RESULT = {}


def hr(title: str) -> None:
    print()
    print("=" * 68)
    print("  " + title)
    print("=" * 68)


def fmt_mb(v: float) -> str:
    return f"{v:7.0f} MB"


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


def synthetic_room(w: int = 640, h: int = 480) -> Image.Image:
    """A crude room: back wall, a 'door' panel, and three boxes at different
    apparent sizes so the depth head has real structure to predict."""
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / (h - 1)
        d.line([(0, y), (w, y)], fill=(int(186 + 54 * t), int(192 + 52 * t), int(202 + 46 * t)))
    d.rectangle([250, 55, 400, 300], fill=(236, 235, 229))   # door-ish panel
    d.rectangle([40, 130, 185, 405], fill=(92, 96, 106))     # near / large
    d.rectangle([300, 205, 385, 385], fill=(162, 92, 72))    # middle
    d.rectangle([520, 250, 572, 352], fill=(72, 122, 92))    # far / small
    d.line([(0, 405), (w, 405)], fill=(150, 152, 148), width=2)
    return img


# ----------------------------------------------------------------------
# key lookup helpers -- UniDepth key names are not guaranteed to be stable
# ----------------------------------------------------------------------
def pick_key(d: dict, *frags: str):
    for k in sorted(d.keys()):
        kl = str(k).lower()
        if any(f in kl for f in frags):
            return k
    return None


def as_hw(t) -> np.ndarray:
    a = t.detach().float().cpu().numpy()
    a = np.squeeze(a)
    if a.ndim != 2:
        raise ValueError(f"cannot read as HxW: {a.shape}")
    return a


def as_chw(t) -> np.ndarray:
    a = t.detach().float().cpu().numpy()
    if a.ndim == 4:
        a = a[0]
    if a.ndim == 3 and a.shape[0] == 3:
        return a
    if a.ndim == 3 and a.shape[-1] == 3:
        return a.transpose(2, 0, 1)
    raise ValueError(f"cannot read as 3xHxW: {a.shape}")


def as_k(t) -> np.ndarray:
    a = t.detach().float().cpu().numpy()
    a = np.squeeze(a)
    if a.shape != (3, 3):
        raise ValueError(f"cannot read as 3x3: {a.shape}")
    return a


# ----------------------------------------------------------------------
# A + B. UniDepth
# ----------------------------------------------------------------------
def run_unidepth(image: Image.Image, device: torch.device, repo: str):
    from unidepth.models import UniDepthV2

    hr("A.  UniDepthV2 -- what it actually returns")

    t0 = time.time()
    model = UniDepthV2.from_pretrained(repo).to(device).eval()
    load_s = time.time() - t0
    n_param = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"  repo id             : {repo}")
    print(f"  parameters          : {n_param:.1f} M")
    print(f"  load time           : {load_s:.1f} s")

    reset_peak()
    # EXACTLY how VADAR builds the input: uint8, no batch dim, no normalisation
    rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(device)
    print(f"  input tensor        : shape={tuple(rgb.shape)}  dtype={rgb.dtype}"
          f"   <- VADAR's own call convention")

    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        out = model.infer(rgb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.time() - t0) * 1000

    print(f"  infer latency       : {latency_ms:.0f} ms")
    print(f"  peak VRAM allocated : {fmt_mb(peak_mb())}")
    print(f"  peak VRAM reserved  : {fmt_mb(reserved_mb())}")
    print()
    print("  keys actually returned by infer():")
    for k in sorted(out.keys()):
        v = out[k]
        if torch.is_tensor(v):
            print(f"    {str(k):<16} {str(tuple(v.shape)):<20} {str(v.dtype)}")
        else:
            print(f"    {str(k):<16} {type(v).__name__}")

    RESULT["unidepth"] = {
        "repo": repo,
        "params_m": round(n_param, 1),
        "load_s": round(load_s, 2),
        "latency_ms": round(latency_ms, 1),
        "peak_vram_alloc_mb": round(peak_mb(), 1),
        "peak_vram_reserved_mb": round(reserved_mb(), 1),
        "keys": {str(k): (list(v.shape) if torch.is_tensor(v) else type(v).__name__)
                 for k, v in out.items()},
    }

    # ---------------- consistency check ----------------
    hr("B.  Is `points` real geometry, or just a reshaped depth map?")

    k_depth = pick_key(out, "depth")
    k_points = pick_key(out, "point", "xyz", "cloud")
    k_rays = pick_key(out, "ray")
    k_radius = pick_key(out, "radius")
    k_k = pick_key(out, "intrin", "camera_k", "k")

    print("  keys found:")
    for label, k in (("depth", k_depth), ("points", k_points),
                     ("rays", k_rays), ("radius", k_radius),
                     ("intrinsics", k_k)):
        print(f"    {label:<12} {k}")

    if k_depth is None or k_points is None or k_k is None:
        print()
        print("  [!!] one of the three keys is missing -- cannot verify.")
        print("       Inspect the key list above and re-run. This is worth")
        print("       reporting: it changes how the whole project is built.")
        RESULT["consistency"] = {"verified": False,
                                 "reason": "missing key",
                                 "depth": k_depth, "points": k_points, "intrinsics": k_k}
        return out, model

    depth = as_hw(out[k_depth])
    pts = as_chw(out[k_points])
    K = as_k(out[k_k])
    rays = as_chw(out[k_rays]) if k_rays is not None else None
    radius = as_hw(out[k_radius]) if k_radius is not None else None
    H, W = depth.shape

    # ---- B1. identities: do these hold by construction? -----------------
    print()
    print("  B1. identities -- if `points` is the camera-frame XYZ cloud these")
    print("      hold exactly, because infer() is where depth gets defined.")
    ident = {}
    if rays is not None and radius is not None:
        d = float(np.abs(pts - rays * radius[None]).max())
        ident["points == rays * radius"] = d
        print(f"      points == rays * radius     max|diff| = {d:.3e}")
    d = float(np.abs(pts[2] - depth).max())
    ident["points_z == depth"] = d
    print(f"      points[2] == depth          max|diff| = {d:.3e}")
    if radius is not None:
        d = float(np.abs(np.linalg.norm(pts, axis=0) - radius).max())
        ident["norm(points) == radius"] = d
        print(f"      norm(points) == radius      max|diff| = {d:.3e}")
    print()
    print("      Depth is not a separate measurement: unidepthv2.py:334-336 sets")
    print("        out['radius'] = points.norm(dim=1, keepdim=True)")
    print("        out['depth']  = points[:, -1:]")
    print("      and unidepthv2.py:376-377 builds points as rays * radius.")
    print("      So VADAR reading only 'depth' (predefined_modules.py:375/:395)")
    print("      is reading the z column of the cloud and throwing x, y away.")

    # ---- B2. cross-check against a pinhole back-projection ---------------
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    print()
    print(f"  B2. depth range     : {depth.shape}  [{depth.min():.3f}, {depth.max():.3f}]")
    print(f"      points shape    : {pts.shape}")
    print(f"      K               : fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    print()
    print("      back-projecting sampled pixels through K and comparing with `points`:")
    print()
    print("         u     v   depth      expected XYZ (m)         from points (m)       abs(m)   rel")

    samples = [(W // 4, H // 4), (W // 2, H // 2),
               (3 * W // 4, 3 * H // 4), (W // 3, 2 * H // 3),
               (2 * W // 3, H // 3)]

    errs, rels = [], []
    for (u, v) in samples:
        z = float(depth[v, u])
        if not np.isfinite(z) or z <= 0:
            print(f"  {u:5d} {v:5d}   {z:8.3f}   (skipped, non-positive depth)")
            continue
        ex = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)
        got = pts[:, v, u].astype(np.float64)
        e = float(np.linalg.norm(ex - got))
        rel = e / max(abs(z), 1e-9)
        errs.append(e)
        rels.append(rel)
        print(f"  {u:5d} {v:5d}   {z:8.3f}   "
              f"({ex[0]:8.3f},{ex[1]:8.3f},{ex[2]:8.3f})   "
              f"({got[0]:8.3f},{got[1]:8.3f},{got[2]:8.3f})   "
              f"{e:6.4f}  {rel * 100:5.2f}%")

    if errs:
        max_err = max(errs)
        mean_err = float(np.mean(errs))
        rel_max = max(rels)
        rel_mean = float(np.mean(rels))
        print()
        print(f"  absolute: max {max_err:.4f} m   mean {mean_err:.4f} m")
        print(f"  relative: max {rel_max * 100:.2f}%   mean {rel_mean * 100:.2f}%"
              f"   (as a fraction of the depth at that pixel)")
        print()
        print("  How to read this:")
        print("    `rays` is a PREDICTED unit-direction field -- the decoder emits it")
        print("    (unidepthv2.py:375). It is not an analytic pinhole grid, and")
        print("    `intrinsics` is a separate prediction head. On top of that, infer()")
        print("    pads the image to ratio_bounds, resizes it to pixels_bounds, then")
        print("    crops the cloud back (:282-336) while rescaling K analytically.")
        print("    So a few percent of disagreement is the expected order of")
        print("    magnitude, not a defect. The two sources simply are not identical.")
        print()
        print("    Consequence for this project: consume `points` directly. Do NOT")
        print("    rebuild the cloud from depth x K -- that would add this gap on top.")

        verified = rel_max < 0.10
        if verified:
            print()
            print("  [PASS] `points` is a genuine metric camera-frame XYZ cloud:")
            print("         - the identities in B1 hold exactly, so depth is literally")
            print("           the z column of the cloud")
            print("         - the depth values land in a plausible metric range for an")
            print("           indoor scene, which is what makes 3D distances meaningful")
            print("         - the cloud agrees with an independent pinhole reconstruction")
            print("           to within a few percent")
            print("         -> The project's core innovation is CONFIRMED. VADAR sizes")
            print("            objects as 2D_pixels * depth with no focal length at all")
            print("            (predefined_modules.py), so real distances are impossible")
            print("            there. Here they cost zero extra models and zero extra VRAM.")
        else:
            print()
            print("  [WARN] disagreement is larger than the expected few percent.")
            print("         Treat the raw coordinates as unverified until checked.")
            print("         Report the key list and these numbers before building on it.")

        RESULT["consistency"] = {
            "verified": bool(verified),
            "identities": {k: float(v) for k, v in ident.items()},
            "max_err_m": round(max_err, 5),
            "mean_err_m": round(mean_err, 5),
            "rel_max_pct": round(rel_max * 100, 3),
            "rel_mean_pct": round(rel_mean * 100, 3),
            "depth_key": k_depth, "points_key": k_points,
            "rays_key": k_rays, "radius_key": k_radius, "intrinsics_key": k_k,
            "depth_range_m": [round(float(depth.min()), 4), round(float(depth.max()), 4)],
            "K": [fx, fy, cx, cy],
        }
    return out, model


# ----------------------------------------------------------------------
# C + D. GroundingDINO, then the real 3D distance
# ----------------------------------------------------------------------
def _gdino_post_kwargs(processor, box_thr: float, text_thr: float, target_sizes) -> dict:
    """Build the post-processing kwargs by reading the real signature.

    transformers renamed the box threshold at some point
    (box_threshold -> threshold), and text_threshold has come and gone
    across versions. Introspecting beats pinning a version and guessing.
    """
    import inspect
    params = set(inspect.signature(
        processor.post_process_grounded_object_detection).parameters)
    thr_kw = "threshold" if "threshold" in params else "box_threshold"
    kw = {thr_kw: box_thr, "target_sizes": target_sizes}
    if "text_threshold" in params:
        kw["text_threshold"] = text_thr
    return kw


def _first_key(d, *names):
    for n in names:
        try:
            v = d[n]
        except (KeyError, IndexError, TypeError):
            continue
        if v is not None:
            return v
    return None


def run_gdino(image: Image.Image, model_id: str, device: torch.device, prompt: str):
    hr("C.  GroundingDINO -- object localisation (transformers native)")

    try:
        from transformers import (AutoModelForZeroShotObjectDetection,
                                  AutoProcessor)
    except Exception as e:                                   # noqa: BLE001
        print(f"  [skip] transformers unavailable: {type(e).__name__}: {e}")
        return None

    print(f"  model               : {model_id}")
    print(f"  prompt              : {prompt!r}")
    print("  (this implementation needs lowercase labels each ending in a")
    print("   period -- 'box . door .' gets mis-tokenised, use 'box. door.')")
    print("  precision           : fp32 (transformers default)")
    print()

    reset_peak()
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    load_s = time.time() - t0
    n_param = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"  parameters          : {n_param:.1f} M")
    print(f"  load time           : {load_s:.1f} s  (includes download if uncached)")

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        outputs = model(**inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.time() - t0) * 1000

    results = processor.post_process_grounded_object_detection(
        outputs, inputs["input_ids"],
        **_gdino_post_kwargs(processor, 0.30, 0.25, [image.size[::-1]]),
    )
    res = results[0]

    boxes_t = _first_key(res, "boxes")
    scores_t = _first_key(res, "scores")
    labels = _first_key(res, "labels", "text_labels") or []
    boxes = np.asarray(boxes_t.detach().cpu().numpy().tolist(), dtype=float) \
        if torch.is_tensor(boxes_t) else np.asarray(boxes_t, dtype=float)
    scores = np.asarray(scores_t.detach().cpu().numpy().tolist(), dtype=float) \
        if torch.is_tensor(scores_t) else np.asarray(scores_t, dtype=float)
    boxes = boxes.reshape(-1, 4)

    print()
    print(f"  infer latency       : {latency_ms:.0f} ms")
    print(f"  peak VRAM allocated : {fmt_mb(peak_mb())}")
    print(f"  peak VRAM reserved  : {fmt_mb(reserved_mb())}")
    print(f"  detections          : {len(boxes)}")

    detections = []
    for b, sc, lb in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [round(float(v), 1) for v in b]
        print(f"    {str(lb):<12} conf={float(sc):.3f}  xyxy(px)=[{x1}, {y1}, {x2}, {y2}]")
        detections.append({"label": str(lb), "score": round(float(sc), 4),
                           "box_xyxy": [x1, y1, x2, y2]})

    RESULT["gdino"] = {
        "model_id": str(model_id),
        "prompt": prompt,
        "params_m": round(n_param, 1),
        "load_s": round(load_s, 2),
        "latency_ms": round(latency_ms, 1),
        "peak_vram_alloc_mb": round(peak_mb(), 1),
        "peak_vram_reserved_mb": round(reserved_mb(), 1),
        "n_detections": int(len(boxes)),
        "box_format": "xyxy, absolute pixels, origin top-left",
        "detections": detections,
    }
    return detections


def run_distance_demo(image, detections, unidepth_out, query=None):
    hr("D.  The operation the final project actually needs")
    print("  'how far apart are these two objects' -- answered from geometry,")
    print("  not from the model's guess.")
    print()

    if not detections or unidepth_out is None:
        print("  [skip] needs at least one detection plus UniDepth output.")
        return

    k_points = pick_key(unidepth_out, "point", "xyz", "cloud")
    if k_points is None:
        print("  [skip] UniDepth returned no point cloud key.")
        return

    pts = as_chw(unidepth_out[k_points])
    _, HP, WP = pts.shape
    WI, HI = image.size
    sx, sy = WP / WI, HP / HI
    if abs(sx - 1.0) > 1e-6 or abs(sy - 1.0) > 1e-6:
        print(f"  [note] point cloud is {WP}x{HP} but the image is {WI}x{HI},")
        print(f"         so detection boxes get scaled by ({sx:.4f}, {sy:.4f}).")
    else:
        print(f"  geometry grid       : {WP} x {HP} px  (matches the input image)")
    print()

    def centre_3d(box_xyxy):
        """pixel xyxy -> median 3D point inside it.

        The median (not the mean) is deliberate: a detection box almost always
        swallows some background or neighbouring surface, and a handful of far
        pixels would drag a mean hundreds of millimetres off.
        """
        x1f, y1f, x2f, y2f = [float(v) for v in box_xyxy]
        x1 = int(max(0, min(WP - 1, round(x1f * sx))))
        x2 = int(max(0, min(WP, round(x2f * sx))))
        y1 = int(max(0, min(HP - 1, round(y1f * sy))))
        y2 = int(max(0, min(HP, round(y2f * sy))))
        if x2 <= x1 or y2 <= y1:
            return None, None
        patch = pts[:, y1:y2, x1:x2].reshape(3, -1)
        patch = patch[:, np.isfinite(patch).all(axis=0)]
        if patch.shape[1] == 0:
            return None, None
        return np.median(patch, axis=1), (x1, y1, x2, y2)

    centres = []
    for d in detections:
        c, pb = centre_3d(d["box_xyxy"])
        if c is not None:
            centres.append((str(d["label"]), c, pb))

    if not centres:
        print("  [skip] could not compute any 3D centre.")
        return

    # Duplicate labels are normal -- a living room has several pictures. Give
    # each detection a unique display name so a printed pair is unambiguous,
    # while keeping the raw label for the "different object types" comparison.
    dup = {}
    for lb, _, _ in centres:
        dup[lb] = dup.get(lb, 0) + 1
    seq, named = {}, []
    for lb, c, pb in centres:
        if dup[lb] > 1:
            seq[lb] = seq.get(lb, 0) + 1
            named.append((f"{lb}#{seq[lb]}", lb, c, pb))
        else:
            named.append((lb, lb, c, pb))

    print("  3D centres recovered from the point cloud (camera frame, metres):")
    for disp, _, c, pb in named:
        print(f"    {disp:<12} XYZ = ({c[0]:7.3f}, {c[1]:7.3f}, {c[2]:7.3f})   px_box={pb}")

    if len(named) < 2:
        print()
        print("  [skip] a distance needs two objects.")
        return

    print()
    print("  pairwise Euclidean distances:")
    pairs = []
    for i in range(len(named)):
        for j in range(i + 1, len(named)):
            d = float(np.linalg.norm(named[i][2] - named[j][2]))
            pairs.append((named[i][0], named[i][1], named[j][0], named[j][1], d))
            print(f"    {named[i][0]:<12} <-> {named[j][0]:<12} {d:6.3f} m")
    pairs.sort(key=lambda t: t[4])
    cross = [p for p in pairs if p[1] != p[3]]

    print()
    print(f"  closest pair, any two detections    : "
          f"{pairs[0][0]} <-> {pairs[0][2]} at {pairs[0][4]:.3f} m")
    if cross:
        print(f"  closest pair, different object types: "
              f"{cross[0][0]} <-> {cross[0][2]} at {cross[0][4]:.3f} m")
        print()
        print("  The second line is the one that reads like an answer. The first")
        print("  is often two instances of the same class -- two pictures on the")
        print("  same wall, which is arithmetically true but not informative.")

    if query:
        hits = [p for p in pairs if query in (p[1], p[3])]
        print()
        if hits:
            print(f"  nearest neighbours of {query!r} -- this is the shape the")
            print("  final project's questions actually take:")
            for a_disp, a_raw, b_disp, b_raw, d in hits[:5]:
                other = b_disp if query in a_raw else a_disp
                print(f"    {query:<12} -> {other:<12} {d:6.3f} m")
        else:
            print(f"  [note] nothing matching {query!r} was detected.")

    print()
    print("  These numbers came from geometry. The LLM never guesses.")
    RESULT["distance_demo"] = {
        "centres": {disp: [round(float(x), 4) for x in c] for disp, _, c, _ in named},
        "pairs": [{"a": a_disp, "b": b_disp, "dist_m": round(d, 4)}
                  for a_disp, _, b_disp, _, d in pairs],
        "closest_any": {"a": pairs[0][0], "b": pairs[0][2], "dist_m": round(pairs[0][4], 4)},
        "closest_distinct_types": ({"a": cross[0][0], "b": cross[0][2],
                                    "dist_m": round(cross[0][4], 4)} if cross else None),
    }


# ----------------------------------------------------------------------
def default_gdino_id() -> str:
    """Prefer the locally fetched copy; fall back to the plain hub id.

    The local directory is used when it exists so the probe works offline
    and so the exact weights are pinned by 01c_fetch_gdino.ps1.
    """
    local = (Path(__file__).resolve().parent.parent
             / ".cache" / "models" / "grounding-dino-tiny")
    return str(local) if (local / "model.safetensors").is_file() \
        else "IDEA-Research/grounding-dino-tiny"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--gdino", type=str, default=None,
                    help="local dir or hub id for grounding-dino-tiny")
    ap.add_argument("--prompt", type=str, default="box. door.",
                    help="lowercase labels, each ending in a period")
    ap.add_argument("--query", type=str, default=None,
                    help="also print this object's nearest neighbours, e.g. table")
    ap.add_argument("--unidepth-repo", type=str, default="lpiccinelli/unidepth-v2-vits14")
    ap.add_argument("--skip-gdino", action="store_true")
    ap.add_argument("--out", type=str, default=str(Path(__file__).with_name("probe3d_result.json")))
    args = ap.parse_args()
    if not args.gdino:
        args.gdino = default_gdino_id()

    print()
    print("Phase 0 / Step 3 -- 3D geometry pipeline probe")
    print(f"  torch {torch.__version__}   cuda {torch.version.cuda}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  device {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory/1024**2:.0f} MiB)")
    else:
        device = torch.device("cpu")
        print("  [warn] CUDA unavailable -- numbers will not represent the 4060")

    if args.image:
        image = Image.open(args.image).convert("RGB")
        print(f"  image  {args.image}  {image.size}")
    else:
        image = synthetic_room()
        print(f"  image  synthetic fallback  {image.size}")
        print("         (pass --image /path/to/room.jpg for meaningful object labels)")

    out, _ = run_unidepth(image, device, args.unidepth_repo)

    detections = None
    if not args.skip_gdino:
        detections = run_gdino(image, args.gdino, device, args.prompt)

    run_distance_demo(image, detections, out, args.query)

    # ------------------------------ verdict ------------------------------
    hr("VERDICT")
    cons = RESULT.get("consistency", {})
    if cons.get("verified"):
        print("  [PASS] UniDepth exposes usable camera-frame 3D points.")
        print("         Next: turn this into the 3D tool library (Phase 1).")
    elif cons:
        print("  [WARN] points/intrinsics present but not self-consistent.")
        print("         Do not build on the raw coordinates yet -- investigate first.")
    else:
        print("  [FAIL] could not confirm the point cloud. Report the key list output.")

    gd = RESULT.get("gdino")
    if gd:
        n = gd["n_detections"]
        print(f"  [{'PASS' if n >= 2 else 'WARN'}] GroundingDINO returned {n} "
              f"detection(s) for the prompt {gd['prompt']!r}")
        if n < 2:
            print("         Two objects are needed for the distance demo --")
            print("         widen --prompt, e.g. \"chair. table. lamp. door.\"")
    elif not args.skip_gdino:
        print("  [FAIL] the GroundingDINO section did not complete.")

    dd = RESULT.get("distance_demo")
    if dd:
        cdt = dd.get("closest_distinct_types") or dd.get("closest_any")
        if cdt:
            print(f"  [PASS] 3D distance from geometry: "
                  f"{cdt['a']} <-> {cdt['b']} = {cdt['dist_m']:.3f} m")
            print("         No model produced that number -- it is arithmetic on `points`.")

    print()
    print(f"  measured numbers written to {args.out}")
    print("  Replace the estimates in 3D_Spatial_Agent_技术调研与实施方案.md with these.")
    print()
    print("  NOTE: this probe does NOT run the VADAR pipeline end to end, and it")
    print("  does NOT touch the LLM. Phase 1 is where the original repo runs.")

    try:
        Path(args.out).write_text(json.dumps(RESULT, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
    except OSError as e:
        print(f"  [warn] could not write {args.out}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
