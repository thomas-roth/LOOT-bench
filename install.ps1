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

$cudaVersion = $null
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi *>$null
    if ($LASTEXITCODE -eq 0) {
        $cudaVersion = (nvidia-smi --query-gpu=cuda_version --format=csv,noheader | Select-Object -First 1).Trim()
        if (-not $cudaVersion) {
            $cudaVersion = (nvidia-smi --version | Select-String -Pattern "CUDA Version\s*:\s*(\S+)" | ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -First 1)
        }
    }
}

if ($cudaVersion) {
    if ($cudaVersion.StartsWith("12.9")) {
        $INSTALL_EXTRA = "cu129"
    } elseif ($cudaVersion.StartsWith("13.0")) {
        $INSTALL_EXTRA = "cu130"
    } else {
        $INSTALL_EXTRA = "cpu"
    }

    if ($INSTALL_EXTRA -eq "cpu") {
        Write-Host "NVIDIA GPU detected but CUDA $cudaVersion is not supported (expected 12.9 or 13.0). Installing CPU build ..." -ForegroundColor Blue
    } else {
        Write-Host "NVIDIA GPU detected! Installing PyTorch for CUDA $cudaVersion ..." -ForegroundColor Blue
    }
} else {
    $INSTALL_EXTRA = "cpu"
    Write-Host "No NVIDIA GPU detected! Installing PyTorch without CUDA support ..." -ForegroundColor Blue
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Error: 'uv' is not installed or not on PATH." -ForegroundColor Red
    Write-Host "Install it first, for example: irm https://astral.sh/uv/install.ps1 | iex" -ForegroundColor Yellow
    exit 1
}

uv sync --extra $INSTALL_EXTRA
Watch-Command

uv run pre-commit install
