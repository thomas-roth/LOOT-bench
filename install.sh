#!/bin/bash

COLOR='\033[0;34m' # blue
YELLOW='\033[0;33m'
NC='\033[0m'

set -euo pipefail # Exit script on error, undefined variable, or error in pipeline

echo -e "${COLOR}Installing packages ...${NC}"

INSTALL_EXTRA="cpu"

if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    SMI_OUTPUT=$(nvidia-smi 2>&1)
    if [[ "$SMI_OUTPUT" =~ CUDA\ Version:[[:space:]]*([0-9]+)\.([0-9]+) ]]; then
        MAJOR="${BASH_REMATCH[1]}"
        MINOR="${BASH_REMATCH[2]}"
        CUDA_VERSION="$MAJOR.$MINOR"

        if [ "$MAJOR" -gt 12 ] || { [ "$MAJOR" -eq 12 ] && [ "$MINOR" -ge 8 ]; }; then
            INSTALL_EXTRA="cu128"
            echo -e "${COLOR}NVIDIA GPU detected! CUDA ${CUDA_VERSION} >= 12.8, installing cu128 build ...${NC}"
        else
            echo -e "${COLOR}NVIDIA GPU detected but CUDA ${CUDA_VERSION} < 12.8. Installing CPU build ...${NC}"
        fi
    else
        echo -e "${YELLOW}NVIDIA GPU detected but could not parse CUDA version. Installing CPU build ...${NC}"
    fi
else
    echo -e "${COLOR}No NVIDIA GPU detected. Installing CPU build ...${NC}"
fi

if ! command -v uv &> /dev/null; then
    echo -e "${COLOR}Error: 'uv' is not installed or not on PATH.${NC}"
    echo -e "${COLOR}Install it first, for example: curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
    exit 1
fi

uv sync --extra "$INSTALL_EXTRA"

uv run pre-commit install