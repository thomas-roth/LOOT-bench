$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Helper to catch native command failures
function Watch-Command {
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Host "Installing packages ..." -ForegroundColor Blue

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Error: 'uv' is not installed or not on PATH." -ForegroundColor Red
    Write-Host "Install it first, for example: irm https://astral.sh/uv/install.ps1 | iex" -ForegroundColor Yellow
    exit 1
}

uv sync
Watch-Command

uv run pre-commit install
