# Set strict error handling
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

uv sync

uv run pre-commit install