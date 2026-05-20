#!/bin/bash

COLOR='\033[0;34m' # blue
NC='\033[0m'

set -euo pipefail # Exit script on error, undefined variable, or error in pipeline


echo -e "${COLOR}(1/2) Initializing genesis-world submodule ...${NC}"
git submodule update --init

echo -e "${COLOR}(2/2) Installing packages ...${NC}"
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo -e "${COLOR}NVIDIA GPU detected! Installing PyTorch with CUDA support ...${NC}"
    INSTALL_EXTRA="cuda"
else
    echo -e "${COLOR}No NVIDIA GPU detected! Installing PyTorch without CUDA support ...${NC}"
    INSTALL_EXTRA="cpu"
fi

if ! command -v uv &> /dev/null; then
    echo -e "${COLOR}Error: 'uv' is not installed or not on PATH.${NC}"
    echo -e "${COLOR}Install it first, for example: curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
    exit 1
fi

uv sync --extra "$INSTALL_EXTRA"
