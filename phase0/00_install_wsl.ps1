# ============================================================
#  Phase 0 / Step 1 -- Windows host preparation
#  MUST run as Administrator.
#
#  What it does:
#    1. enables the two Windows features WSL2 needs
#    2. reboots gate (the hypervisor only starts after a restart)
#    3. installs Ubuntu-22.04 into D:\WSL
#
#  Why D: and not C:
#    C: has only ~22 GB free. WSL + CUDA libs + model weights +
#    datasets needs roughly 60-90 GB. D: has ~381 GB free.
#    Ubuntu 22.04 is chosen because it ships Python 3.10,
#    which is what VADAR's pinned deps expect.
#
#  Project root (Windows) : D:\3D_Spatial_Agent
#  Project root (WSL)     : /mnt/d/3D_Spatial_Agent
#  Keep every path below in sync with that root.
# ============================================================

$ErrorActionPreference = "Stop"

function Ok($m)   { Write-Host ("[ ok ] " + $m) -ForegroundColor Green }
function Info($m) { Write-Host ("[ .. ] " + $m) -ForegroundColor Cyan }
function Warn($m) { Write-Host ("[warn] " + $m) -ForegroundColor Yellow }
function Fail($m) { Write-Host ("[fail] " + $m) -ForegroundColor Red }

# ---------- 0. admin check ----------
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Fail "This script must run as Administrator."
    Write-Host ""
    Write-Host "  Right-click 'Windows PowerShell' -> 'Run as administrator', then re-run:"
    Write-Host "    powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit 1
}
Ok "running with administrator rights"

# ---------- 1. plan ----------
$Distro     = "Ubuntu-22.04"
$TargetRoot = "D:\WSL"
$TargetPath = Join-Path $TargetRoot $Distro

# ---------- 2. disk guard ----------
$dFree = [math]::Round((Get-PSDrive D).Free / 1GB, 1)
Write-Host ""
Write-Host ("  D: free space = " + $dFree + " GB")
if ($dFree -lt 90) {
    Warn "Under 90 GB free on D:. CUDA wheels + weights + Omni3D-Bench may not fit."
} else {
    Ok "enough disk space on D:"
}

# ---------- 3. hardware guard ----------
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
Write-Host ("  CPU = " + $cpu.Name)
if ($cpu.VirtualizationFirmwareEnabled -ne $true) {
    Fail "CPU virtualization is DISABLED in BIOS/UEFI."
    Write-Host "  Reboot into BIOS and enable SVM Mode (AMD) / VT-x (Intel), then re-run."
    exit 1
}
Ok "CPU virtualization enabled in firmware"

# ---------- 4. enable the two Windows features ----------
Write-Host ""
$features = @("Microsoft-Windows-Subsystem-Linux", "VirtualMachinePlatform")
foreach ($f in $features) {
    $state = (Get-WindowsOptionalFeature -Online -FeatureName $f).State
    if ($state -eq "Enabled") {
        Ok ($f + " already enabled")
    } else {
        Info ("enabling " + $f + " (was " + $state + ") ...")
        Enable-WindowsOptionalFeature -Online -FeatureName $f -All -NoRestart | Out-Null
        Ok ($f + " enabled -- a reboot is now required")
    }
}

# ---------- 5. reboot gate ----------
Write-Host ""
$hv = (Get-CimInstance Win32_ComputerSystem).HypervisorPresent
Write-Host ("  HypervisorPresent = " + $hv)
if ($hv -ne $true) {
    Warn "The hypervisor is not running yet, so WSL2 cannot start."
    Write-Host ""
    Write-Host "  DO THIS NOW:"
    Write-Host "    1. reboot Windows"
    Write-Host "    2. open PowerShell as Administrator again"
    Write-Host "    3. re-run this same script"
    Write-Host ""
    exit 0
}
Ok "hypervisor is running"

# ---------- 6. WSL CLI ----------
Write-Host ""
Info "updating the WSL CLI and kernel (needed for --location support) ..."
& wsl.exe --update
& wsl.exe --set-default-version 2
Ok "WSL default version set to 2"

# ---------- 7. install the distro onto D: ----------
Write-Host ""
$raw = (& wsl.exe --list --quiet 2>$null | Out-String)
$installed = ($raw -replace "`0", "").Trim()
if ($installed -match [regex]::Escape($Distro)) {
    Ok ($Distro + " is already installed")
} else {
    New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null
    Info ("installing " + $Distro + " into " + $TargetPath + " ...")
    & wsl.exe --install -d $Distro --location $TargetPath --no-launch
    if ($LASTEXITCODE -ne 0) {
        Warn "--location is unsupported on older WSL builds. Falling back to the default location."
        & wsl.exe --install -d $Distro --no-launch
        if ($LASTEXITCODE -ne 0) {
            Fail "distro installation failed. Run 'wsl --install -d $Distro' manually and read the output."
            exit 1
        }
        Warn ("installed to the default location. Move it to D: later with:")
        Write-Host "         wsl --shutdown"
        Write-Host "         wsl --export $Distro D:\WSL\$Distro.tar"
        Write-Host "         wsl --unregister $Distro"
        Write-Host "         wsl --import $Distro D:\WSL\$Distro D:\WSL\$Distro.tar --version 2"
    } else {
        Ok ("installed to " + $TargetPath)
    }
}

# ---------- 8. hand off ----------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " Windows side is ready." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "NEXT -- run these three commands:"
Write-Host ""
Write-Host ("  1) enter Ubuntu        :  wsl -d " + $Distro)
Write-Host  "  2) verify GPU passthru :  nvidia-smi"
Write-Host  "                           (must list: RTX 4060 Laptop GPU, 8188 MiB)"
Write-Host  "  3) build the toolchain :  bash /mnt/d/3D_Spatial_Agent/phase0/01_setup_ubuntu.sh"
Write-Host ""
Write-Host "If step 2 fails, the Windows NVIDIA driver is the problem, not WSL."
Write-Host "Driver 560.76 is fine; just make sure it is the Game Ready / Studio driver"
Write-Host "installed on Windows (never install a Linux driver inside WSL)."
Write-Host ""
