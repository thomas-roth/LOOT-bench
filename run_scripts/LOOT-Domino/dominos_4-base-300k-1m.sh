#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export G3A_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$HERE/../sweep.sh" \
  dominos_4-base-300k-1m \
  LOOT-Domino-4Pieces-Dk1 \
  dataset/qc/dominos_4-base_2026-09-11.npz \
  "$@" \
  -- \
  --offline_steps 300000 \
  --online_steps 1000000 \
  --eval_interval 200000 \
