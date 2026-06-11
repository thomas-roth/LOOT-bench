$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Helper to catch native command failures
function Watch-Command {
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Host "Installing packages ..." -ForegroundColor Blue

$INSTALL_EXTRA = "cpu"

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi *>$null
    if ($LASTEXITCODE -eq 0) {
        $smiOutput = nvidia-smi 2>&1 | Out-String
        $match = [regex]::Match($smiOutput, "CUDA Version:\s*(\d+)\.(\d+)")

        if ($match.Success) {
            $major = [int]$match.Groups[1].Value
            $minor = [int]$match.Groups[2].Value
            $cudaVersion = "$major.$minor"

            if ($major -gt 12 -or ($major -eq 12 -and $minor -ge 8)) {
                $INSTALL_EXTRA = "cu128"
                Write-Host "NVIDIA GPU detected! CUDA $cudaVersion >= 12.8, installing cu128 build ..." -ForegroundColor Blue
            } else {
                Write-Host "NVIDIA GPU detected but CUDA $cudaVersion < 12.8. Installing CPU build ..." -ForegroundColor Blue
            }
        } else {
            Write-Host "NVIDIA GPU detected but could not parse CUDA version. Installing CPU build ..." -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "No NVIDIA GPU detected. Installing CPU build ..." -ForegroundColor Blue
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Error: 'uv' is not installed or not on PATH." -ForegroundColor Red
    Write-Host "Install it first, for example: irm https://astral.sh/uv/install.ps1 | iex" -ForegroundColor Yellow
    exit 1
}

uv sync --extra $INSTALL_EXTRA
Watch-Command

uv run pre-commit install