#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOOT_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$HERE/../sweep.sh" \
  dominos_3-base-300k-1m \
  LOOT-Domino-3Pieces-Dk1 \
  dataset/qc/dominos_3-base_101-demos_split_2026-09-06.npz \
  "$@" \
  -- \
  --offline_steps 300000 \
  --online_steps 1000000 \
  --eval_interval 200000
