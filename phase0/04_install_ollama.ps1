# ============================================================
#  Phase 0 / Step 4 -- local LLM backend on Windows (no API key)
#
#  Why on Windows and not in WSL:
#    WSL2 is not installed yet and needs a reboot. This script
#    gives us a working OpenAI-compatible endpoint NOW, so the
#    VADAR prompt-protocol test (05_probe_vadar_prompt.py) can
#    run today. Ollama uses the 4060 directly through CUDA.
#
#  What it does:
#    1. detect / install Ollama
#    2. redirect the model store to D: (C: has only ~22 GB free)
#    3. start the server
#    4. pull qwen3.5:4b  (3.4 GB, 256K ctx, text+image)
#    5. VERIFY the endpoint with a real generate call
#
#  Does NOT need administrator rights.
#  Run:  powershell -ExecutionPolicy Bypass -File D:\3D_Spatial_Agent\phase0\04_install_ollama.ps1
#
#  Optional:
#    -Model qwen3.5:9b                     the 6.6 GB variant (tight on 8 GB)
#    -Model qwen3.5:2b                     the 2.7 GB variant, for pipeline debugging
#    -SkipInstall                          assume Ollama is present
#
#  MODEL NAMING (verified against ollama.com/library on 2026-09-15):
#    The library family is "qwen3.5", NOT the older "qwen3-vl".
#    Local tags and on-disk sizes:
#      qwen3.5:0.8b   1.0 GB      qwen3.5:27b   17 GB
#      qwen3.5:2b     2.7 GB      qwen3.5:35b   24 GB
#      qwen3.5:4b     3.4 GB      qwen3.5:122b  81 GB
#      qwen3.5:9b     6.6 GB (also tagged :latest)
#    All tags are 256K context, Text + Image, and carry the
#    vision / tools / thinking capability labels.
#    "qwen3.5:cloud" and "qwen3.5:397b-cloud" are CLOUD-ONLY -- they do
#    not run locally and must not be used for the offline experiments.
#    If a pull 404s, the family was renamed again: check
#    https://ollama.com/library/qwen3.5/tags and pass -Model.
# ============================================================

[CmdletBinding()]
param(
    [string]$Model = "qwen3.5:4b",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$ProgressPreference    = "SilentlyContinue"

function Ok($m)   { Write-Host ("  [ ok ] " + $m) -ForegroundColor Green }
function Info($m) { Write-Host ("  [ .. ] " + $m) -ForegroundColor Cyan }
function Warn($m) { Write-Host ("  [warn] " + $m) -ForegroundColor Yellow }
function Fail($m) { Write-Host ("  [fail] " + $m) -ForegroundColor Red }

# ---------- configuration ----------
$ModelStore = "D:\ollama\models"          # C: only has ~22 GB free
$DlDir      = "D:\3D_Spatial_Agent\downloads"
$Endpoint   = "http://127.0.0.1:11434"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Phase 0 / Step 4 -- local LLM backend (Ollama on Windows)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# ---------- 0. GPU sanity ----------
Write-Host ""
Info "checking the GPU"
$smi = $null
try { $smi = (nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | Out-String).Trim() } catch {}
if ($smi) { Ok ("GPU: " + $smi) } else { Warn "nvidia-smi not found -- Ollama will fall back to CPU and be unusably slow" }

# ---------- 1. locate or install Ollama ----------
Write-Host ""
Info "looking for Ollama"
$exe = $null
foreach ($c in @(
    (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"),
    (Join-Path $env:ProgramFiles "Ollama\ollama.exe")
)) { if (Test-Path $c) { $exe = $c; break } }
if (-not $exe) {
    $g = Get-Command ollama -ErrorAction SilentlyContinue
    if ($g) { $exe = $g.Source }
}

if ($exe) {
    Ok ("found: " + $exe)
} elseif ($SkipInstall) {
    Fail "-SkipInstall was given but Ollama was not found"
    exit 1
} else {
    Warn "Ollama not installed -- installing now"
    $installed = $false

    # prefer winget: silent, no GUI, no guessing at installer flags
    $wg = Get-Command winget -ErrorAction SilentlyContinue
    if ($wg) {
        Info "trying winget (this downloads ~700 MB)"
        try {
            & winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements --disable-interactivity
            if ($LASTEXITCODE -eq 0) { $installed = $true; Ok "winget install finished" }
            else { Warn ("winget exited with code " + $LASTEXITCODE) }
        } catch { Warn ("winget failed: " + $_.Exception.Message) }
    } else {
        Warn "winget not available"
    }

    if (-not $installed) {
        # fallback: download and run the official installer.
        # NOTE: the silent flags below are NOT verified against the current
        # OllamaSetup.exe. If it opens a GUI window, just click through it --
        # that is expected, not an error.
        New-Item -ItemType Directory -Force -Path $DlDir | Out-Null
        $setup = Join-Path $DlDir "OllamaSetup.exe"
        Info ("downloading " + $setup)
        try {
            Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $setup -UseBasicParsing
            Ok ("downloaded " + [math]::Round((Get-Item $setup).Length/1MB,1) + " MB")
            Info "running the installer (a window may appear -- click through it)"
            Start-Process -FilePath $setup -Wait
            $installed = $true
        } catch {
            Fail ("could not download the installer: " + $_.Exception.Message)
            Write-Host ""
            Write-Host "  Do it by hand instead:"
            Write-Host "    1. open https://ollama.com/download/windows"
            Write-Host "    2. run the installer"
            Write-Host "    3. re-run this script with -SkipInstall"
            exit 1
        }
    }

    # re-locate
    foreach ($c in @(
        (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"),
        (Join-Path $env:ProgramFiles "Ollama\ollama.exe")
    )) { if (Test-Path $c) { $exe = $c; break } }
    if (-not $exe) {
        $g = Get-Command ollama -ErrorAction SilentlyContinue
        if ($g) { $exe = $g.Source }
    }
    if (-not $exe) {
        Fail "Ollama still not found after install. Close this shell, open a NEW one, and re-run with -SkipInstall."
        exit 1
    }
    Ok ("installed: " + $exe)
}

# ---------- 2. redirect the model store to D: ----------
# Ollama reads OLLAMA_MODELS at server start. Setting it at User scope makes
# it stick across reboots without touching the machine-wide environment.
Write-Host ""
Info "pointing the model store at D:"
New-Item -ItemType Directory -Force -Path $ModelStore | Out-Null
$cur = [Environment]::GetEnvironmentVariable("OLLAMA_MODELS", "User")
if ($cur -eq $ModelStore) {
    Ok ("OLLAMA_MODELS already = " + $ModelStore)
} else {
    [Environment]::SetEnvironmentVariable("OLLAMA_MODELS", $ModelStore, "User")
    $env:OLLAMA_MODELS = $ModelStore
    Ok ("OLLAMA_MODELS set to " + $ModelStore + "  (was: " + $(if ($cur) { $cur } else { "<unset>" }) + ")")
}

# Reduce KV-cache VRAM so a 4B model + 16K context still fits in 8 GB.
# These are User-scope too; a restart of the server picks them up.
$tuning = @{
    "OLLAMA_KV_CACHE_TYPE"   = "q8_0"    # 8-bit KV cache: ~half the VRAM of f16
    "OLLAMA_FLASH_ATTENTION" = "1"
    "OLLAMA_CONTEXT_LENGTH"  = "16384"   # VADAR's prompts are 3-5K tokens; 16K leaves headroom
    "OLLAMA_KEEP_ALIVE"      = "10m"     # keep the model resident while we iterate
}
foreach ($k in $tuning.Keys) {
    $v = $tuning[$k]
    $old = [Environment]::GetEnvironmentVariable($k, "User")
    if ($old -ne $v) {
        [Environment]::SetEnvironmentVariable($k, $v, "User")
        Set-Item -Path ("Env:" + $k) -Value $v
        Ok ("  " + $k.PadRight(24) + "= " + $v)
    } else {
        Ok ("  " + $k.PadRight(24) + "= " + $v + " (unchanged)")
    }
}
Warn "env vars are read at server start -- if the server was already running, restart it below"

# ---------- 3. (re)start the server ----------
Write-Host ""
Info "restarting the Ollama server so the new env vars take effect"
Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Start-Process -FilePath $exe -ArgumentList "serve" -WindowStyle Hidden
$alive = $false
for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Milliseconds 800
    try {
        $null = Invoke-RestMethod -Uri "$Endpoint/api/tags" -TimeoutSec 3 -ErrorAction Stop
        $alive = $true; break
    } catch { }
}
if ($alive) { Ok ("server responding at " + $Endpoint) }
else { Fail "server did not come up within ~16s. Start it manually: `"$exe`" serve"; exit 1 }

# ---------- 4. pull the model ----------
# Model names in the Ollama library do get renamed between releases
# ("qwen3-vl" became "qwen3.5"). A bare 404 here would be a confusing
# dead end, so try the requested name first and fall back to the
# documented alternates in order of preference.
$Candidates = @($Model, "qwen3.5:4b", "qwen3.5:2b", "qwen3.5:9b") | Select-Object -Unique
$Pulled = $null
foreach ($cand in $Candidates) {
    Write-Host ""
    Info ("pulling " + $cand + "  (one-time download; the first may be ~3-4 GB)")
    & $exe pull $cand
    if ($LASTEXITCODE -eq 0) { $Pulled = $cand; Ok ("model present: " + $cand); break }
    Warn ("  pull failed for " + $cand + "  (exit " + $LASTEXITCODE + ")")
}
if (-not $Pulled) {
    Fail "no candidate model could be pulled."
    Write-Host "  Check the live tag list and re-run with an explicit name:"
    Write-Host "    https://ollama.com/library/qwen3.5/tags"
    Write-Host "    -Model <name>"
    exit 1
}
if ($Pulled -ne $Model) { Warn ("  falling back from " + $Model + " to " + $Pulled) ; $Model = $Pulled }

# ---------- 5. VERIFY with a real generate call ----------
# Not optional. A model that lists in /api/tags but cannot generate is a
# failure mode we have to rule out before trusting it in the pipeline.
Write-Host ""
Info "verifying with a real generate call"
$tags = Invoke-RestMethod -Uri "$Endpoint/api/tags" -TimeoutSec 10
$found = $tags.models | Where-Object { $_.name -eq $Model -or $_.model -eq $Model }
if ($found) {
    Ok ("  " + $Model + "  on-disk size = " + [math]::Round($found.size/1GB,2) + " GB")
} else {
    Warn "  $Model did not appear in /api/tags"
    Write-Host "  tags seen:"
    foreach ($m in $tags.models) { Write-Host ("    " + $m.name) }
}

$probe = @{
    model   = $Model
    prompt  = "Reply with exactly this and nothing else: <docstring>test</docstring><signature>def _t(image, bbox):</signature>"
    stream  = $false
    options = @{ temperature = 0; num_predict = 96 }
} | ConvertTo-Json -Depth 5

$t0 = Get-Date
try {
    $resp = Invoke-RestMethod -Uri "$Endpoint/api/generate" -Method Post -Body $probe -ContentType "application/json" -TimeoutSec 300
    $dt = [math]::Round(((Get-Date) - $t0).TotalSeconds, 2)
    Ok ("  generate OK in " + $dt + " s")
    $txt = $resp.response
    if ($txt.Length -gt 400) { $txt = $txt.Substring(0,400) + " ..." }
    Write-Host ""
    Write-Host "  --- model reply ---" -ForegroundColor DarkGray
    Write-Host ("  " + ($txt -replace "`n", "`n  ")) -ForegroundColor DarkGray
    Write-Host "  -------------------" -ForegroundColor DarkGray

    $hasTags = ($resp.response -match "<signature>")
    if ($hasTags) { Ok "  the model emitted <signature> tags -- the protocol is understood" }
    else { Warn "  no <signature> tag in the reply -- run 05_probe_vadar_prompt.py for the full test" }
} catch {
    Fail ("generate call failed: " + $_.Exception.Message)
    exit 1
}

# ---------- 6. hand off ----------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " Local LLM backend is up -- still zero API keys involved." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  endpoint : $Endpoint"
Write-Host "  OpenAI-compatible base URL : $Endpoint/v1"
Write-Host "  model    : $Model"
Write-Host "  weights  : $ModelStore"
Write-Host ""
Write-Host "NEXT -- the actual question this was set up to answer:"
Write-Host ""
Write-Host "    D:\Users\ROG\anaconda3\python.exe D:\3D_Spatial_Agent\phase0\05_probe_vadar_prompt.py"
Write-Host ""
Write-Host "That script feeds VADAR's real SIGNATURE_PROMPT (verbatim from"
Write-Host "vendor/VADAR/prompts/) to this endpoint and scores whether the model"
Write-Host "obeys VADAR's tag contract -- which VADAR parses with zero tolerance."
Write-Host ""
