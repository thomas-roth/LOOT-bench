#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOOT_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$HERE/../../sweep.sh" \
  place_rand-s_fixed-t_400k-1m \
  LOOT-Place-RandS-FixedT-Dk1 \
  dataset/qc/place_rand-s_fixed-t_100-demos_2026-09-24.npz \
  "$@" \
  -- \
  --offline_steps 400000 \
  --online_steps 1000000 \
  --eval_interval 300000
