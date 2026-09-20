<#
Phase 0 / Step 1c -- fetch the grounding-dino-tiny weights with curl
====================================================================

WHY THIS FILE EXISTS
--------------------
Same reason as 01b_fetch_torch.ps1: the HF transfer path is unreliable here
and the failure mode is silent.

Measured on this machine (2026-09-16):

    huggingface.co  direct            timeout
    hf-mirror.com via huggingface_hub  0.53 MB/s, killed mid-transfer
    hf-mirror.com via curl             2.78 MB/s, resumable

Two further traps this script avoids:

1. `snapshot_download` pulls BOTH `model.safetensors` (657 MB) and
   `pytorch_model.bin` (660 MB) -- the same weights twice. This script
   fetches only the safetensors, so the repo costs ~660 MB instead of 1.3 GB.

2. `hf-xet` is the slow transport. It is bypassed entirely by using curl,
   so no HF_HUB_DISABLE_XET juggling is needed.

The files land as a plain directory, which `from_pretrained()` accepts
directly -- no HF cache, no Windows symlink warnings:

    AutoProcessor.from_pretrained("D:\3D_Spatial_Agent\.cache\models\grounding-dino-tiny")

Usage
  powershell -ExecutionPolicy Bypass -File phase0\01c_fetch_gdino.ps1
#>

$ErrorActionPreference = "Stop"

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$DstDir      = Join-Path $ProjectRoot ".cache\models\grounding-dino-tiny"
New-Item -ItemType Directory -Force -Path $DstDir | Out-Null

$Repo = "IDEA-Research/grounding-dino-tiny"
$Base = "https://hf-mirror.com/$Repo/resolve/main"

# MinBytes guards against a truncated file being mistaken for a finished one.
# Plain byte counts for the small files: PowerShell has no "B" numeric suffix
# (only KB/MB/GB/TB), so `100B` is parsed as a command name -- the script died
# with CommandNotFoundException before downloading anything.
# pytorch_model.bin is deliberately absent -- it duplicates the safetensors.
$Files = @(
    @{ Name = "model.safetensors";         MinBytes = 650MB },
    @{ Name = "config.json";               MinBytes = 1KB   },
    @{ Name = "preprocessor_config.json";  MinBytes = 100   },
    @{ Name = "tokenizer.json";            MinBytes = 100KB },
    @{ Name = "tokenizer_config.json";     MinBytes = 500   },
    @{ Name = "special_tokens_map.json";   MinBytes = 50    },
    @{ Name = "added_tokens.json";         MinBytes = 20    },
    @{ Name = "vocab.txt";                 MinBytes = 100KB }
)

Write-Host ""
Write-Host "=== Phase 0 / Step 1c -- fetch $Repo (curl) ==="
Write-Host "  dest : $DstDir"
Write-Host ""

foreach ($f in $Files) {
    $path = Join-Path $DstDir $f.Name
    $url  = "$Base/$($f.Name)"

    $have = 0
    if (Test-Path $path) { $have = (Get-Item $path).Length }

    if ($have -ge $f.MinBytes) {
        Write-Host "--- $($f.Name)"
        Write-Host "    already complete ($([math]::Round($have/1MB,1)) MB), skipping"
        continue
    }

    Write-Host "--- $($f.Name)"
    if ($have -gt 0) {
        Write-Host "    resuming from $([math]::Round($have/1MB,1)) MB"
    } else {
        Write-Host "    starting fresh"
    }

    # Each pass is a fresh curl that resumes with -C -.
    #
    # The stall defence is --speed-limit/--speed-time: if throughput falls
    # under 50 KB/s for 60 s, curl aborts instead of hanging forever. The
    # outer loop then resumes from wherever the partial file ends.
    #
    # -sS --no-progress-meter keeps curl's progress bar off stderr; under
    # $ErrorActionPreference = "Stop" any stderr line from a native command
    # surfaces as a terminating NativeCommandError.
    $attempt = 0
    while ($attempt -lt 8) {
        $attempt++
        $ErrorActionPreference = "Continue"
        $curlOut = & curl.exe -sS --no-progress-meter -L -C - `
            --retry 4 --retry-delay 5 --retry-all-errors `
            --connect-timeout 30 `
            --speed-limit 51200 --speed-time 60 `
            -o $path `
            -w "http=%{http_code} got=%{size_download}B avg=%{speed_download}B/s secs=%{time_total}" `
            $url 2>&1
        $rc = $LASTEXITCODE
        $ErrorActionPreference = "Stop"

        $now = 0
        if (Test-Path $path) { $now = (Get-Item $path).Length }

        Write-Host ("    attempt {0}: {1} MB  {2}" -f $attempt, [math]::Round($now/1MB,2), ($curlOut -join " ").Trim())

        if ($now -ge $f.MinBytes) { break }

        # http=416 means the requested range is unsatisfiable, i.e. the local
        # file already reaches the remote size -- curl cannot resume past the
        # end. Without this branch the loop burns every attempt on a file that
        # is already complete.
        if (($curlOut -join " ") -match "http=416" -and $now -gt 0) {
            Write-Host "    server reports the file is already complete (http=416)"
            break
        }
        if ($attempt -lt 8) {
            Write-Host "    incomplete -- resuming in 5 s"
            Start-Sleep -Seconds 5
        }
    }

    $now = (Get-Item $path).Length
    if ($now -lt $f.MinBytes) {
        throw "incomplete: $($f.Name) is $([math]::Round($now/1MB,2)) MB, expected >= $([math]::Round($f.MinBytes/1MB,2)) MB. Re-run this script to resume."
    }
    Write-Host "    ok -- $([math]::Round($now/1MB,2)) MB"
}

Write-Host ""
Write-Host "=== done -- grounding-dino-tiny ready ==="
Get-ChildItem $DstDir | ForEach-Object {
    Write-Host ("  {0,-30} {1,10} bytes" -f $_.Name, $_.Length)
}
Write-Host ""
