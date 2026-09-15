#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOOT_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$HERE/../sweep.sh" \
  Sorter_5-Shapes_Fixed-Orientation_700k-1m \
  LOOT-Sorter-5Shapes-FixedOrientation-Dk1 \
  dataset/qc/sorter_5-shapes_fixed-orientation_57-demos_2026-09-14.npz \
  --alphas 1,10,100 \
  --num-seeds 3 \
  --discount 0.995 \
  "$@" \
  -- \
  --offline_steps 700000 \
  --online_steps 1000000 \
  --eval_interval 300000 \
  --agent.discount 0.995
