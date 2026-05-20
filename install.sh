#!/usr/bin/env bash

set -euo pipefail

uv sync

pre-commit install
