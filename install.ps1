$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Helper to catch native command failures
function Watch-Command {
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Host "(1/2) Initializing genesis-world submodule ..." -ForegroundColor Blue
git submodule update --init
cd ext/genesis-world
git sparse-checkout init --no-cone
git sparse-checkout set examples
cd ../..
Watch-Command

Write-Host "(2/2) Installing packages ..." -ForegroundColor Blue

$gpuDetected = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    # Run nvidia-smi and suppress output to see if it succeeds
    nvidia-smi *>$null
    if ($LASTEXITCODE -eq 0) {
        $gpuDetected = $true
    }
}

if ($gpuDetected) {
    Write-Host "NVIDIA GPU detected! Installing PyTorch with CUDA support ..." -ForegroundColor Blue
    $INSTALL_EXTRA = "cuda"
} else {
    Write-Host "No NVIDIA GPU detected! Installing PyTorch without CUDA support ..." -ForegroundColor Blue
    $INSTALL_EXTRA = "cpu"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Error: 'uv' is not installed or not on PATH." -ForegroundColor Red
    Write-Host "Install it first, for example: irm https://astral.sh/uv/install.ps1 | iex" -ForegroundColor Yellow
    exit 1
}

uv sync --extra $INSTALL_EXTRA
Watch-Command
