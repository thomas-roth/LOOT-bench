#!/bin/bash

COLOR='\033[0;34m' # blue
YELLOW='\033[0;33m'
NC='\033[0m'

set -euo pipefail # Exit script on error, undefined variable, or error in pipeline

echo -e "${COLOR}Installing packages ...${NC}"

if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}Error: 'uv' is not installed or not on PATH.${NC}"
    echo -e "${YELLOW}Install it first, for example: curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
    exit 1
fi

uv sync

uv run pre-commit install
