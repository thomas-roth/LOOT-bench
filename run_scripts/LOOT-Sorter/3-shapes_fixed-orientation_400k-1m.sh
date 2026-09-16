#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOOT_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$HERE/../sweep.sh" \
  Sorter_3-Shapes_Fixed-Orientation_400k-1m \
  LOOT-Sorter-3Shapes-FixedOrientation-Dk1 \
  dataset/qc/sorter_3-shapes_fixed-orientation_111-demos_2026-09-15.npz \
  --discount 0.995 \
  "$@" \
  -- \
  --offline_steps 400000 \
  --online_steps 1000000 \
  --eval_interval 300000 \
  --agent.discount 0.995
