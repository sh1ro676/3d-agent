<#
Phase 0 / Step 1b -- fetch the torch + torchvision cu124 wheels with curl
=========================================================================

WHY THIS FILE EXISTS
--------------------
`pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124`
was observed to hang: the 2.5 GB wheel download sat at 0 bytes for 18+
minutes with the process alive and idle (measured: 0.00 MB/s over a 25 s
sample, no partial file anywhere on disk).

The source itself is healthy -- a plain 3 MB range request to the very same
URL returns HTTP 206 at ~4.2 MB/s:

    download.pytorch.org/whl/cu124/   4.19 MB/s   (official, fastest)
    mirror.sjtu.edu.cn/pytorch-whe  ~0.87 MB/s
    mirrors.aliyun.com/pytorch-whe  ~0.12 MB/s   (times out)
    mirror.nju.edu.cn/pytorch-whe    HTTP 404

So the fix is to take pip out of the download path for the big wheels:
curl them to a local directory first, then let pip install from there via
--find-links (see step 3 of 01_setup_windows.ps1).

curl is used because it resumes (`-C -`), retries cleanly, and reports real
throughput, so a stall is visible instead of silent.

Usage
  powershell -ExecutionPolicy Bypass -File phase0\01b_fetch_torch.ps1
  powershell -ExecutionPolicy Bypass -File phase0\01b_fetch_torch.ps1 -PythonTag 3.12
#>

param(
    [string]$PythonTag = "3.12"
)

$ErrorActionPreference = "Stop"

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$WheelDir    = Join-Path $ProjectRoot ".cache\wheels"
New-Item -ItemType Directory -Force -Path $WheelDir | Out-Null

# cp312 / cp311 / cp313 -- the wheel tag follows the interpreter ABI
$CpTag = "cp" + ($PythonTag -replace "\.", "")

$TorchVer = "2.6.0"
$TvVer    = "0.21.0"
$Base     = "https://download.pytorch.org/whl/cu124"

# MinBytes guards against a truncated file being mistaken for a finished one.
$Wheels = @(
    @{ Name = "torch-$TorchVer+cu124-$CpTag-$CpTag-win_amd64.whl";
       MinBytes = 2000MB },
    @{ Name = "torchvision-$TvVer+cu124-$CpTag-$CpTag-win_amd64.whl";
       MinBytes = 3MB }
)

Write-Host ""
Write-Host "=== Phase 0 / Step 1b -- fetch torch wheels (curl) ==="
Write-Host "  python tag : $CpTag"
Write-Host "  wheel dir  : $WheelDir"
Write-Host ""

foreach ($w in $Wheels) {
    $file = Join-Path $WheelDir $w.Name
    # %2B is the URL-encoded '+' that the pytorch index actually uses
    $url  = "$Base/" + ($w.Name -replace "\+", "%2B")

    $have = 0
    if (Test-Path $file) { $have = (Get-Item $file).Length }

    if ($have -ge $w.MinBytes) {
        Write-Host "--- $($w.Name)"
        Write-Host "    already complete ($([math]::Round($have/1MB,1)) MB), skipping"
        continue
    }

    Write-Host "--- $($w.Name)"
    if ($have -gt 0) {
        Write-Host "    resuming from $([math]::Round($have/1MB,1)) MB"
    } else {
        Write-Host "    starting fresh"
    }

    # Retry loop, each pass a fresh curl that resumes with -C -.
    #
    # The stall defence is --speed-limit/--speed-time: if throughput drops
    # under 50 KB/s for 60 s, curl aborts instead of hanging forever -- which
    # is exactly the failure pip showed us. The outer loop then resumes.
    #
    # -sS --no-progress-meter keeps curl from writing its progress bar to
    # stderr; under $ErrorActionPreference = "Stop" any stderr line from a
    # native command surfaces as a terminating NativeCommandError.
    $attempt = 0
    while ($attempt -lt 6) {
        $attempt++
        $ErrorActionPreference = "Continue"
        $curlOut = & curl.exe -sS --no-progress-meter -L -C - `
            --retry 4 --retry-delay 5 --retry-all-errors `
            --connect-timeout 30 `
            --speed-limit 51200 --speed-time 60 `
            -o $file `
            -w "http=%{http_code} got=%{size_download}B avg=%{speed_download}B/s secs=%{time_total}" `
            $url 2>&1
        $rc = $LASTEXITCODE
        $ErrorActionPreference = "Stop"

        $now = 0
        if (Test-Path $file) { $now = (Get-Item $file).Length }

        Write-Host ("    attempt {0}: {1} MB  {2}" -f $attempt, [math]::Round($now/1MB,1), ($curlOut -join " ").Trim())

        if ($now -ge $w.MinBytes) { break }
        if ($attempt -lt 6) {
            Write-Host "    incomplete -- resuming in 5 s"
            Start-Sleep -Seconds 5
        }
    }

    $now = (Get-Item $file).Length
    if ($rc -ne 0 -and $now -lt $w.MinBytes) {
        throw "curl kept failing for $($w.Name) (last exit $rc, $([math]::Round($now/1MB,1)) MB on disk). Re-run this script to resume."
    }
    if ($now -lt $w.MinBytes) {
        throw "incomplete: $($w.Name) is $([math]::Round($now/1MB,1)) MB, expected >= $([math]::Round($w.MinBytes/1MB,1)) MB. Re-run this script to resume."
    }
    Write-Host "    ok -- $([math]::Round($now/1MB,1)) MB"
}

Write-Host ""
Write-Host "=== done -- wheels ready for offline install ==="
Get-ChildItem $WheelDir -Filter *.whl | ForEach-Object {
    Write-Host ("  {0}  ({1} MB)" -f $_.Name, [math]::Round($_.Length/1MB,1))
}
Write-Host ""
