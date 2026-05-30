#!/bin/bash

COLOR='\033[0;34m' # blue
NC='\033[0m'

set -euo pipefail # Exit script on error, undefined variable, or error in pipeline


echo -e "${COLOR}(1/2) Initializing genesis-world submodule ...${NC}"
git submodule update --init
cd ext/genesis-world
git sparse-checkout init --no-cone
git sparse-checkout set examples
cd ../..

echo -e "${COLOR}(2/2) Installing packages ...${NC}"
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' \r')
    if [ -z "$CUDA_VERSION" ]; then
        CUDA_VERSION=$(nvidia-smi --version 2>/dev/null | tr -d '\r' | sed -n 's/.*CUDA Version[[:space:]]*:[[:space:]]*//p' | head -n 1)
    fi
    case "$CUDA_VERSION" in
        12.9*)
            INSTALL_EXTRA="cu129"
            ;;
        13.0*)
            INSTALL_EXTRA="cu130"
            ;;
        *)
            INSTALL_EXTRA="cpu"
            ;;
    esac

    if [ "$INSTALL_EXTRA" = "cpu" ]; then
        echo -e "${COLOR}NVIDIA GPU detected but CUDA ${CUDA_VERSION:-unknown} is not supported (expected 12.9 or 13.0). Installing CPU build ...${NC}"
    else
        echo -e "${COLOR}NVIDIA GPU detected! Installing PyTorch for CUDA ${CUDA_VERSION} ...${NC}"
    fi
else
    INSTALL_EXTRA="cpu"
    echo -e "${COLOR}No NVIDIA GPU detected! Installing PyTorch without CUDA support ...${NC}"
fi

if ! command -v uv &> /dev/null; then
    echo -e "${COLOR}Error: 'uv' is not installed or not on PATH.${NC}"
    echo -e "${COLOR}Install it first, for example: curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
    exit 1
fi

uv sync --extra "$INSTALL_EXTRA"

uv run pre-commit install
