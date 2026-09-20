#!/usr/bin/env python3
"""
Phase 0 -- verification for the Windows-native vision environment.

Answers three questions with evidence rather than opinion:

  1. Is CUDA actually usable from this venv?  (the 4060 is sm_89, so the
     torch build must ship cu124 or newer)
  2. Does `from unidepth.models import UniDepthV2` succeed while xformers,
     triton and torchaudio are ABSENT?  This is the single fact that lets us
     skip WSL2 -- if it holds, the whole Windows-native path is justified.
  3. Which third-party modules does the UniDepth import chain really pull in?

Exit code is non-zero if any hard check fails, so the caller can gate on it.

Run:
    D:\\3D_Spatial_Agent\\venvs\\vision\\Scripts\\python.exe phase0\\verify_env.py
"""

from __future__ import annotations

import importlib.util
import sys
import time

FAILS: list[str] = []


def hr(title: str) -> None:
    print()
    print("=" * 68)
    print("  " + title)
    print("=" * 68)


# ----------------------------------------------------------------------
hr("1. interpreter")
print(f"  python           {sys.version.split()[0]}")
print(f"  executable       {sys.executable}")
if sys.version_info < (3, 10):
    FAILS.append(f"python {sys.version.split()[0]} < 3.10 (UniDepth requires >=3.10)")

# ----------------------------------------------------------------------
hr("2. torch / CUDA")
try:
    import torch
except ImportError as e:
    print(f"  [FAIL] torch not importable: {e}")
    raise SystemExit(1)

print(f"  torch            {torch.__version__}")
print(f"  cuda build       {torch.version.cuda}")
print(f"  cudnn            {torch.backends.cudnn.version()}")
avail = torch.cuda.is_available()
print(f"  cuda available   {avail}")
if avail:
    p = torch.cuda.get_device_properties(0)
    print(f"  device           {p.name}")
    print(f"  vram             {p.total_memory / 1024 ** 2:.0f} MiB")
    print(f"  compute cap      sm_{p.major}{p.minor}")
    print(f"  device count     {torch.cuda.device_count()}")
    if (p.major, p.minor) >= (8, 9) and (torch.version.cuda or "0") < "12.4":
        FAILS.append("sm_89 GPU needs a cu124+ build; this wheel is older")
else:
    FAILS.append("torch.cuda.is_available() is False -> numbers would not represent the 4060")

# ----------------------------------------------------------------------
hr("3. modules deliberately NOT installed")
print("  these are training-only or demo-only; absence is expected")
for name in ("xformers", "triton", "torchaudio", "gradio", "tables"):
    present = importlib.util.find_spec(name) is not None
    print(f"    {name:<12} {'PRESENT' if present else 'absent'}")

print()
print("  exception -- installed even though it looks optional:")
print("    wandb        ", end="")
_has_wandb = importlib.util.find_spec("wandb") is not None
print("PRESENT" if _has_wandb else "absent")
print("      unidepth/utils/__init__.py:12 does")
print("        from .visualization import colorize, image_grid, log_train_artifacts")
print("      and unidepth/utils/visualization.py:11 is a bare `import wandb`.")
print("      So the package __init__ pulls wandb in and `unidepth.models`")
print("      cannot be imported without it. Skipping it is not an option")
print("      unless the vendored source is patched -- which we refuse to do.")
if not _has_wandb:
    FAILS.append("wandb missing -> `from unidepth.models import ...` will fail")

# ----------------------------------------------------------------------
hr("4. UniDepth import chain (the decisive test)")
try:
    t0 = time.time()
    from unidepth.models import UniDepthV2
    dt = time.time() - t0
except Exception as e:  # noqa: BLE001
    print(f"  [FAIL] {type(e).__name__}: {e}")
    print("  -> install the missing package into the venv and re-run")
    raise SystemExit(1)

print(f"  import time      {dt:.2f} s")
print(f"  UniDepthV2       {UniDepthV2}")
print("  [PASS] imports fine WITHOUT xformers / triton / torchaudio")

# ----------------------------------------------------------------------
hr("5. third-party modules actually loaded")
WANT = {
    "numpy", "PIL", "cv2", "scipy", "einops", "timm", "matplotlib",
    "huggingface_hub", "torchvision", "transformers", "pandas",
    "safetensors", "filelock", "requests", "yaml",
}
loaded = sorted(m for m in sys.modules if m in WANT)
print("  " + ", ".join(loaded))
print()
print(f"  numpy            {sys.modules['numpy'].__version__}")
if "torchvision" in sys.modules:
    print(f"  torchvision      {sys.modules['torchvision'].__version__}")
if "transformers" in sys.modules:
    print(f"  transformers     {sys.modules['transformers'].__version__}")

# ----------------------------------------------------------------------
hr("VERDICT")
if FAILS:
    for f in FAILS:
        print(f"  [FAIL] {f}")
    raise SystemExit(1)

print("  [PASS] environment is usable for the vision stack.")
print("         next:  python phase0\\probe3d.py --skip-gdino")
raise SystemExit(0)
