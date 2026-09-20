#!/usr/bin/env bash
# ============================================================
#  Phase 0 / Step 2 -- build the 3D-vision toolchain inside WSL2
#  Target: Ubuntu 22.04 (Python 3.10) on RTX 4060 Laptop 8 GB
#
#  Run from inside Ubuntu:
#    bash /mnt/d/3D_Spatial_Agent/phase0/01_setup_ubuntu.sh
#
#  Optional:
#    --with-cuda-toolkit   also install CUDA Toolkit 12.2 (nvcc, ~3 GB).
#                          Only needed if GroundingDINO's fused kernel fails
#                          to build and you want the compiled CUDA op.
#
#  Notes on real VADAR constraints (verified against the source):
#   * the source tree MUST be named "VADAR" -- engine/predefined_modules.py:17
#     hardcodes `from VADAR.prompts.vqa_prompt import ...`
#   * sam2 is imported at module top level (predefined_modules.py:12-13),
#     so the sam2 PACKAGE must be installed even though the Omni3D path
#     never uses it. Its 300 MB checkpoint is NOT needed.
#   * xformers is deliberately NOT installed. VADAR's setup.sh installs
#     xformers==0.0.24, which drags in Triton and breaks easily under WSL.
#     Nothing in the Omni3D path requires it.
# ============================================================

set -euo pipefail

WITHD_CUDA=0
for a in "$@"; do
  case "$a" in
    --with-cuda-toolkit) WITHD_CUDA=1 ;;
    *) echo "unknown option: $a"; exit 2 ;;
  esac
done

# Project root, derived from this script's own location so the whole
# tree can be moved (C: -> D:, or anywhere else) without editing paths.
#   script lives at  <ROOT>/phase0/01_setup_ubuntu.sh
#   => ROOT          /mnt/d/3D_Spatial_Agent
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROBE="$PROJECT_ROOT/phase0/probe3d.py"
SRC="$PROJECT_ROOT/vendor/VADAR"          # original clone, dir name is load-bearing

# NOTE on placement: the VADAR source and the venv stay in the WSL ext4
# filesystem (~/), NOT on /mnt/d. Two reasons:
#   1. pip -e builds and torch imports over the 9p mount are 3-10x slower
#   2. /mnt/d is drvfs: it cannot hold POSIX permissions, which breaks
#      some installs and git operations
# The D: drive holds the project's *sources* (this repo, scripts, docs).
# Phase 0/1 artifacts live in ext4. Keep it that way.
VADAR_DIR="$HOME/VADAR"
MODELS="$VADAR_DIR/models"
VENV="$HOME/venvs/vadar"

step() { echo ""; echo "=== $* ==="; }
ok()   { echo "  [ ok ] $*"; }
warn() { echo "  [warn] $*"; }
die()  { echo "  [fail] $*"; exit 1; }

# ---------------------------------------------------------------
step "1/8  GPU passthrough"
# ---------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  if [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
    export PATH="/usr/lib/wsl/lib:$PATH"
  fi
fi
command -v nvidia-smi >/dev/null 2>&1 \
  || die "nvidia-smi not found. WSL GPU passthrough is not working -> fix the Windows NVIDIA driver first."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
ok "GPU visible from WSL"

# ---------------------------------------------------------------
step "2/8  system packages"
# ---------------------------------------------------------------
sudo apt-get update -qq
sudo apt-get install -y -qq \
  python3 python3-venv python3-dev \
  build-essential git wget curl unzip ca-certificates \
  pkg-config cmake ninja-build \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
  >/dev/null
ok "apt packages installed"

# CUDA toolkit is opt-in: pip's torch wheel already ships the CUDA runtime,
# but no nvcc. Without nvcc GroundingDINO builds its C++-only fallback.
if [ "$WITHD_CUDA" = "1" ]; then
  warn "installing CUDA Toolkit 12.2 (this takes a while)"
  cd /tmp
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq cuda-toolkit-12-2 >/dev/null
  echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> "$HOME/.bashrc"
  echo 'export CUDA_HOME=/usr/local/cuda-12.2' >> "$HOME/.bashrc"
  export PATH=/usr/local/cuda-12.2/bin:$PATH
  export CUDA_HOME=/usr/local/cuda-12.2
  ok "nvcc: $(nvcc --version | tail -1)"
else
  warn "skipping CUDA Toolkit -- GroundingDINO will build its fallback extension"
fi

# ---------------------------------------------------------------
step "3/8  VADAR source tree (directory name is load-bearing)"
# ---------------------------------------------------------------
[ -d "$SRC" ] || die "expected the cloned VADAR source at $SRC -- did the workspace move?"
if [ ! -d "$VADAR_DIR/.git" ]; then
  cp -r "$SRC" "$VADAR_DIR"
  ok "copied source into $VADAR_DIR"
else
  ok "$VADAR_DIR already present"
fi
[ -f "$VADAR_DIR/engine/predefined_modules.py" ] || die "copy looks incomplete"
ok "VADAR at commit $(git -C "$VADAR_DIR" rev-parse --short HEAD)"

# ---------------------------------------------------------------
step "4/8  Python 3.10 virtualenv"
# ---------------------------------------------------------------
python3 --version | sed 's/^/  /'
python3 -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,10) else 1)' \
  || die "VADAR pins transformers==4.45.2 and expects Python 3.10. Use Ubuntu 22.04."
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade pip wheel setuptools
ok "venv ready at $VENV"

# ---------------------------------------------------------------
# MEASURED 2026-09-15: huggingface.co is unreachable from this network
# (times out at 12s+) while hf-mirror.com answers in ~2s.
# UniDepth pulls its weights through from_pretrained(), and Omni3D-Bench
# also lives on the Hub -- both must go through the mirror.
# ---------------------------------------------------------------
export HF_ENDPOINT="https://hf-mirror.com"
if ! grep -q "HF_ENDPOINT" "$HOME/.bashrc" 2>/dev/null; then
  echo 'export HF_ENDPOINT=https://hf-mirror.com' >> "$HOME/.bashrc"
fi
ok "HF_ENDPOINT -> $HF_ENDPOINT  (huggingface.co direct is blocked)"

# ---------------------------------------------------------------
step "5/8  PyTorch 2.2.0 (VADAR's pinned version)"
# ---------------------------------------------------------------
# MEASURED 2026-09-15 from this network: download.pytorch.org returns HTTP 403,
# so VADAR's own setup.sh line
#     pip install torch==2.2.0 ... --index-url https://download.pytorch.org/whl/cu122
# WILL FAIL HERE. Use a domestic mirror instead.
# The PyPI wheel of torch 2.2.0 bundles CUDA 12.1. sm_89 (the 4060) has been
# supported since CUDA 11.8, so cu121 is fully sufficient -- cu122 is not needed.
TORCH_OK=0
for IDX in "https://pypi.tuna.tsinghua.edu.cn/simple" "https://mirrors.aliyun.com/pypi/simple" ""; do
  if [ -n "$IDX" ]; then
    echo "  trying index: $IDX"
    if pip install -q torch==2.2.0 torchvision==0.17.0 --index-url "$IDX"; then TORCH_OK=1; break; fi
  else
    echo "  trying index: PyPI default"
    if pip install -q torch==2.2.0 torchvision==0.17.0; then TORCH_OK=1; break; fi
  fi
done
[ "$TORCH_OK" = "1" ] || die "could not install torch from any index"
pip config set global.index-url "https://pypi.tuna.tsinghua.edu.cn/simple" >/dev/null 2>&1 || true
python - <<'PY'
import sys, torch
print("  torch      =", torch.__version__)
print("  cuda build =", torch.version.cuda)
avail = torch.cuda.is_available()
print("  available  =", avail)
if avail:
    print("  device     =", torch.cuda.get_device_name(0))
    print("  cc         = sm_%d%d" % torch.cuda.get_device_capability(0))
sys.exit(0 if avail else 1)
PY
ok "torch sees the GPU"

# ---------------------------------------------------------------
step "6/8  vision models"
# ---------------------------------------------------------------
mkdir -p "$MODELS"
cd "$MODELS"

clone_once() {
  if [ -d "$1/.git" ]; then ok "$1 already cloned"; else git clone -q "$2" "$1"; ok "cloned $1"; fi
}

# UniDepthV2 -- the module VADAR under-uses. Model id used by VADAR itself:
#   UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vits14")   <- predefined_modules.py:635
clone_once UniDepth https://github.com/lpiccinelli-eth/UniDepth.git
pip install -q -e UniDepth --no-build-isolation

# SAM2 -- package mandatory (top-level import), checkpoint NOT needed for Omni3D
clone_once sam2 https://github.com/facebookresearch/sam2.git
SAM2_BUILD_CUDA=0 pip install -q -e sam2 --no-build-isolation

# GroundingDINO -- SwinT-OGC detector, the only real compilation risk
clone_once GroundingDINO https://github.com/IDEA-Research/GroundingDINO.git
if pip install -q -e GroundingDINO --no-build-isolation; then
  ok "GroundingDINO installed"
else
  warn "GroundingDINO build failed."
  warn "Retry with:  bash \$0 --with-cuda-toolkit"
fi
mkdir -p GroundingDINO/weights
CKPT="GroundingDINO/weights/groundingdino_swint_ogc.pth"
if [ ! -f "$CKPT" ]; then
  wget -q -O "$CKPT" \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
fi
ok "GroundingDINO SwinT-OGC weights present ($(du -h "$CKPT" | cut -f1))"

# ---------------------------------------------------------------
step "7/8  VADAR requirements (+ two missing deps)"
# ---------------------------------------------------------------
cd "$VADAR_DIR"
pip install -q -r requirements.txt
# requirements.txt is incomplete: engine/engine.py:6 does `import pandas`
# and the eval scripts use tqdm. Install them explicitly.
pip install -q pandas tqdm gdown
python -c "import pandas, tqdm, transformers, torch; print('  transformers =', transformers.__version__)"
ok "python deps resolved"

# ---------------------------------------------------------------
step "8/8  done"
# ---------------------------------------------------------------
cat <<EOF

============================================================
 Toolchain is up.
============================================================

  VADAR source : $VADAR_DIR
  models       : $MODELS
  venv         : $VENV

NEXT -- the real point of today:

    source $VENV/bin/activate
    python $PROBE

probe3d.py does two things:
  (a) measures real peak VRAM + latency for GroundingDINO / UniDepth
      -> replaces every estimated number in the plan document
  (b) prints every key UniDepth returns and checks whether 'points'
      and 'intrinsics' are real geometry

(b) matters most. VADAR reads only ["depth"] (predefined_modules.py:375 and :395)
and throws the other keys away. If 'points' is genuine, the project's
core innovation is confirmed on day one -- with zero extra VRAM.

EOF
