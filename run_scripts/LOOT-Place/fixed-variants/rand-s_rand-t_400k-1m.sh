#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOOT_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$HERE/../../sweep.sh" \
  place_rand-s_rand-t_400k-1m \
  LOOT-Place-RandS-RandT-Dk1 \
  dataset/qc/place_rand-s_rand-t_99-demos_2026-09-11.npz \
  --alphas 100 \
  "$@" \
  -- \
  --offline_steps 400000 \
  --online_steps 1000000 \
  --eval_interval 300000 \
  --eval_episodes 50
