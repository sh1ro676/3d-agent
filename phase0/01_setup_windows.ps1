<#
Phase 0 / Step 1 -- Windows-native vision-stack environment
==========================================================

Creates <project>\venvs\vision and installs everything the 3D vision stack
needs. This replaces the WSL-based route (00_install_wsl.ps1 /
01_setup_ubuntu.sh), which is no longer required.

Why Windows native is sufficient (verified against the actual source)
  - UniDepth builds with setuptools.build_meta and declares no ext_modules,
    so `pip install -e` needs no compiler.
  - Its requirements.txt pins triton>=2.4.0 and xformers>=0.0.26, but every
    import of them sits inside try/except ImportError:
        unidepth/models/backbones/metadinov2/attention.py:21
        unidepth/models/backbones/metadinov2/block.py:26
        unidepth/models/backbones/metadinov2/swiglu_ffn.py:37
        unidepth/layers/nystrom_attention.py:9
    attention.py falls back to F.scaled_dot_product_attention and even says
    so in a comment: "new pytorch have good attn efficient, no need for
    xformers". triton has no Windows wheels at all.
  - GroundingDINO and SAM2 have native pure-PyTorch implementations inside
    transformers, so no CUDA kernel compilation is needed either.

Deliberately NOT installed
  triton, xformers        training-only (see above)
  torchaudio              unused at inference
  gradio, wandb, tables   demo / training / dataset tooling
  旧的 numpy==1.25.0      conflicts with UniDepth's numpy>=2.0.0

Everything is written to D: -- C: has under 25 GB free, and the torch cu124
wheel alone is roughly 2.4 GB.

Usage
  powershell -ExecutionPolicy Bypass -File phase0\01_setup_windows.ps1
#>

# Deliberately NOT "Stop".
#
# Under $ErrorActionPreference = "Stop", any write to stderr by a native
# command (pip, git, curl) is promoted to a *terminating* NativeCommandError.
# That is what killed the 2026-09-16 run: pip was mid-download of scipy, wrote
# something to stderr, and the whole script aborted with exit 1 while the log
# simply stopped mid-line. Correct pattern for native tools is to let them run
# and inspect $LASTEXITCODE -- which every call below already does.
$ErrorActionPreference = "Continue"

# ---- derive the project root from this script's own location ----------
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$LogDir      = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir -ErrorAction SilentlyContinue | Out-Null

# Keep a transcript so a failed run is diagnosable without re-running it.
$Transcript = Join-Path $LogDir "setup_vision.log"
try { Start-Transcript -Path $Transcript -Force -ErrorAction Stop | Out-Null }
catch { Write-Host "  (transcript unavailable: $($_.Exception.Message))" }

$VenvDir     = Join-Path $ProjectRoot "venvs\vision"
$CacheDir    = Join-Path $ProjectRoot ".cache\pip"
$HfHome      = Join-Path $ProjectRoot ".cache\huggingface"
$UniDepthSrc = Join-Path $ProjectRoot "vendor\UniDepth"
$LockFile    = Join-Path $ScriptDir   "requirements-vision.lock.txt"
$PipLog      = Join-Path $LogDir     "setup_vision_pip.log"

$PypiMirror = "https://pypi.tuna.tsinghua.edu.cn/simple"
$TorchIndex = "https://download.pytorch.org/whl/cu124"
$TorchVer   = "2.6.0"
$TvVer      = "0.21.0"

Write-Host ""
Write-Host "=== 3D Spatial Agent / Phase 0 Step 1 (Windows native) ==="
Write-Host "  project root : $ProjectRoot"
Write-Host "  venv         : $VenvDir"
Write-Host ""

# ---- keep every downloaded byte on D: --------------------------------
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
New-Item -ItemType Directory -Force -Path $HfHome   | Out-Null
$env:PIP_CACHE_DIR = $CacheDir
$env:HF_HOME       = $HfHome
$env:HF_ENDPOINT   = "https://hf-mirror.com"
$env:PYTHONUTF8    = "1"
Write-Host "  pip cache    : $env:PIP_CACHE_DIR"
Write-Host "  hf home      : $env:HF_HOME"
Write-Host "  hf endpoint  : $env:HF_ENDPOINT"
Write-Host ""

# ---- 1. pick an interpreter ------------------------------------------
$Tag = $null
foreach ($v in @("3.12", "3.11", "3.13")) {
    & py "-$v" -c "1" *> $null
    if ($LASTEXITCODE -eq 0) { $Tag = $v; break }
}
if (-not $Tag) {
    throw "No CPython 3.11-3.13 found via the py launcher. Install Python 3.12 from python.org and re-run."
}
Write-Host "--- 1. interpreter: py -$Tag"

# ---- 2. create the venv ----------------------------------------------
# Reuse the venv if it is already runnable.
#
# Do NOT use `venv --clear`: it deletes the whole tree, which trips the
# environment's bulk-delete guard (observed failing on 1540 files under
# venvs\vision\Lib). Reusing is also the safer default -- an existing venv
# with a working interpreter should not be wiped just to re-run the install.
$Py = Join-Path $VenvDir "Scripts\python.exe"
if (Test-Path $Py) {
    Write-Host "    reusing existing venv (delete the folder by hand to rebuild)"
} else {
    & py "-$Tag" -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}
if (-not (Test-Path $Py)) { throw "venv interpreter missing: $Py" }
& $Py -c "import sys; print('    created with python', sys.version.split()[0])"
if ($LASTEXITCODE -ne 0) { throw "venv interpreter is not runnable" }

# ---- 3. build tooling -------------------------------------------------
Write-Host ""
Write-Host "--- 2. pip / setuptools / wheel"
& $Py -m pip install --upgrade pip setuptools wheel `
    -i $PypiMirror --timeout 120 --retries 5 --log $PipLog
if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed" }

# ---- 4. torch + torchvision ------------------------------------------
# Prefer wheels that 01b_fetch_torch.ps1 already pulled with curl.
#
# Why: `pip install torch --index-url <pytorch>` was measured hanging -- the
# 2.5 GB transfer sat at 0 bytes for 18+ minutes while the same URL served
# curl at 4.2 MB/s. pip's downloader has no resume and no throughput output,
# so a stall is silent. Fetching with curl first makes it visible + resumable.
Write-Host ""
Write-Host "--- 3. torch $TorchVer + torchvision $TvVer  (cu124)"
$WheelDir = Join-Path $ProjectRoot ".cache\wheels"
$LocalTorch = Get-ChildItem $WheelDir -Filter "torch-$TorchVer+cu124-*.whl" -ErrorAction SilentlyContinue |
    Where-Object { $_.Length -gt 2000MB }
if ($LocalTorch) {
    Write-Host "    using local wheels in $WheelDir  (offline for the big files)"
    # No --no-index: the small pure-python deps (sympy, filelock, networkx,
    # jinja2, fsspec, typing-extensions, numpy, pillow) still come from PyPI,
    # which was fast (3.4 MB/s). --find-links just wins for the huge ones.
    & $Py -m pip install "torch==$TorchVer+cu124" "torchvision==$TvVer+cu124" `
        --find-links $WheelDir -i $PypiMirror --timeout 180 --retries 6 --log $PipLog
} else {
    Write-Host "    no local wheels found -- falling back to the pytorch index"
    Write-Host "    (if this hangs, run phase0\01b_fetch_torch.ps1 first)"
    & $Py -m pip install "torch==$TorchVer" "torchvision==$TvVer" `
        --index-url $TorchIndex --timeout 300 --retries 6 --log $PipLog
}
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }

# ---- 5. the rest of the runtime deps ---------------------------------
#
# Split into small batches ON PURPOSE.
#
# Observed 2026-09-16: a single `pip install` of all 18 packages was killed
# mid-download around the two-minute mark -- twice, with pip's own --log
# showing no ERROR and no exit line, i.e. the process was terminated from
# outside rather than failing. The same scipy wheel then downloaded fine in
# 52 s when installed on its own. So: keep every pip invocation short.
#
# Installed individually because the wheel is large (~37 MB at ~0.7 MB/s).
$depBatches = @(
    @("timm", "einops", "huggingface-hub", "safetensors", "transformers"),
    @("scipy"),
    @("opencv-python"),
    @("matplotlib", "pandas", "imageio"),
    @("tabulate", "termcolor", "protobuf", "trimesh", "h5py"),
    # wandb is nominally training-only, but it cannot be skipped:
    # unidepth/utils/__init__.py:12 imports log_train_artifacts from
    # .visualization, and that module does a bare `import wandb` at line 11.
    # So `from unidepth.models import UniDepthV2` fails without it.
    @("wandb")
)
Write-Host ""
Write-Host "--- 4. runtime dependencies (in $($depBatches.Count) short batches)"
$i = 0
foreach ($batch in $depBatches) {
    $i++
    Write-Host "    [$i/$($depBatches.Count)] $($batch -join ', ')"
    & $Py -m pip install @batch -i $PypiMirror --timeout 60 --retries 3 --log $PipLog
    if ($LASTEXITCODE -ne 0) { throw "dependency install failed on batch $i ($($batch -join ', '))" }
}

# ---- 6. UniDepth from the vendored source ----------------------------
Write-Host ""
Write-Host "--- 5. UniDepth (source install, --no-deps on purpose)"
if (-not (Test-Path (Join-Path $UniDepthSrc "pyproject.toml"))) {
    Write-Host "    source missing -- cloning"
    & git clone --depth 1 https://github.com/lpiccinelli-eth/UniDepth.git $UniDepthSrc
    if ($LASTEXITCODE -ne 0) { throw "UniDepth clone failed" }
}
& $Py -m pip install --no-deps --no-build-isolation -e $UniDepthSrc `
    -i $PypiMirror --timeout 180 --retries 6 --log $PipLog
if ($LASTEXITCODE -ne 0) { throw "unidepth install failed" }

# ---- 7. verification --------------------------------------------------
Write-Host ""
Write-Host "--- 6. verification"
$Verify = Join-Path $ScriptDir "verify_env.py"
& $Py $Verify
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "verification failed. Most likely a missing package in the"
    Write-Host "unidepth import chain -- install it and re-run verify_env.py:"
    Write-Host "  $Py -m pip install <package> -i $PypiMirror"
    throw "verification failed"
}

# ---- 8. lock the resolved versions -----------------------------------
Write-Host ""
Write-Host "--- 7. writing $LockFile"
& $Py -m pip freeze | Out-File -Encoding utf8 $LockFile

Write-Host ""
Write-Host "=== done ==="
Write-Host "  activate : & '$VenvDir\Scripts\Activate.ps1'"
Write-Host "  next     : & '$Py' '$ScriptDir\probe3d.py' --skip-gdino"
Write-Host ""
try { Stop-Transcript | Out-Null } catch { }
