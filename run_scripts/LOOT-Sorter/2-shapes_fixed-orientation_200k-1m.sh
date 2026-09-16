#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOOT_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$HERE/../sweep.sh" \
  Sorter_2-Shapes_Fixed-Orientation_200k-1m \
  LOOT-Sorter-2Shapes-FixedOrientation-Dk1 \
  dataset/qc/sorter_2-shapes_fixed-orientation_100+4-demos_2026-09-16.npz \
  --discount 0.995 \
  "$@" \
  -- \
  --offline_steps 200000 \
  --online_steps 1000000 \
  --eval_interval 300000 \
  --agent.discount 0.995
