#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOOT_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$HERE/../../sweep.sh" \
  place_rand-s_fixed-t_50k-600k \
  LOOT-Place-RandS-FixedT-Dk1_uni \
  dataset/qc/place_rand-s_fixed-t_99-demos_2026-08-30_11-08-25.npz \
  "$@" \
  -- \
  --offline_steps 50000 \
  --online_steps 600000 \
  --eval_interval 30000 \
  --eval_episodes 100
