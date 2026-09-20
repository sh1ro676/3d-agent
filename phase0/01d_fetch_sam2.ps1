<#
Phase 0 / Step 1d -- fetch the sam2.1-hiera-base-plus weights with curl
====================================================================

WHY THIS FILE EXISTS
--------------------
Same reason as 01b_fetch_torch.ps1 and 01c_fetch_gdino.ps1: the HF transfer
path on this machine is unreliable and fails silently. Measured 2026-09-16:

    hf-mirror.com via curl   ~2.5 MB/s, resumable with -C -

Two traps this script avoids:

1. The repo ships the SAME weights twice:

       model.safetensors         323,476,296 bytes   <- transformers reads this
       sam2.1_hiera_base_plus.pt 323,606,802 bytes   <- original PyTorch format

   We use `Sam2Model` from transformers, so the .pt file is dead weight.
   Fetching only the safetensors keeps the download at ~308 MiB instead of
   ~617 MiB.

2. A single curl invocation cannot finish 308 MiB here: the host kills long
   commands. Each pass is therefore capped with --max-time and the outer loop
   resumes. Two passes were enough in practice.

The files land as a plain directory, which `from_pretrained()` accepts
directly -- no HF cache, no Windows symlink warnings, no HF_HOME juggling:

    Sam2Processor.from_pretrained("D:\3D_Spatial_Agent\.cache\models\sam2.1-hiera-base-plus")
    Sam2Model.from_pretrained(  "D:\3D_Spatial_Agent\.cache\models\sam2.1-hiera-base-plus")

Usage
  powershell -ExecutionPolicy Bypass -File phase0\01d_fetch_sam2.ps1
#>

$ErrorActionPreference = "Stop"

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$DstDir      = Join-Path $ProjectRoot ".cache\models\sam2.1-hiera-base-plus"
New-Item -ItemType Directory -Force -Path $DstDir | Out-Null

$Repo = "facebook/sam2.1-hiera-base-plus"
$Base = "https://hf-mirror.com/$Repo/resolve/main"

# MinBytes guards against a truncated file being mistaken for a finished one.
# Plain byte counts for the small files: PowerShell has no "B" numeric suffix
# (only KB/MB/GB/TB), so `100B` is parsed as a command name and the script dies
# with CommandNotFoundException before downloading anything.
# sam2.1_hiera_base_plus.pt is deliberately absent -- it duplicates the
# safetensors in a format transformers does not read.
$Files = @(
    @{ Name = "model.safetensors";              MinBytes = 300MB },
    @{ Name = "config.json";                    MinBytes = 1KB   },
    @{ Name = "preprocessor_config.json";       MinBytes = 100   },
    @{ Name = "processor_config.json";          MinBytes = 50    },
    @{ Name = "video_preprocessor_config.json"; MinBytes = 100   }
)

Write-Host ""
Write-Host "=== Phase 0 / Step 1d -- fetch $Repo (curl) ==="
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

    # Each pass is a fresh curl that resumes with -C -. --max-time keeps a
    # single pass inside the host's command lifetime; the loop provides the
    # rest of the transfer. --speed-limit/--speed-time aborts a stalled
    # transfer instead of hanging. -sS --no-progress-meter keeps curl quiet on
    # stderr, because under $ErrorActionPreference = "Stop" any stderr line
    # from a native command surfaces as a terminating NativeCommandError.
    $attempt = 0
    while ($attempt -lt 10) {
        $attempt++
        $ErrorActionPreference = "Continue"
        $curlOut = & curl.exe -sS --no-progress-meter -L -C - `
            --max-time 100 `
            --retry 3 --retry-delay 5 --retry-all-errors `
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
        # end. Without this branch the loop burns all ten attempts on a file
        # that is already complete.
        if (($curlOut -join " ") -match "http=416" -and $now -gt 0) {
            Write-Host "    server reports the file is already complete (http=416)"
            break
        }
        if ($attempt -lt 10) {
            Write-Host "    incomplete -- resuming in 3 s"
            Start-Sleep -Seconds 3
        }
    }

    $now = (Get-Item $path).Length
    if ($now -lt $f.MinBytes) {
        throw "incomplete: $($f.Name) is $([math]::Round($now/1MB,2)) MB, expected >= $([math]::Round($f.MinBytes/1MB,2)) MB. Re-run this script to resume."
    }
    Write-Host "    ok -- $([math]::Round($now/1MB,2)) MB"
}

Write-Host ""
Write-Host "=== done -- sam2.1-hiera-base-plus ready ==="
Get-ChildItem $DstDir | ForEach-Object {
    Write-Host ("  {0,-34} {1,12} bytes" -f $_.Name, $_.Length)
}
Write-Host ""
