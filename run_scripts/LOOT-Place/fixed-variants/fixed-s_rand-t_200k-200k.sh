#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOOT_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$HERE/../../sweep.sh" \
  place_fixed-s_rand-t_200k-200k \
  LOOT-Place-FixedS-RandT-Dk1_uni \
  dataset/qc/place_fixed-s_rand-t_100-demos_2026-08-30_10-51-01.npz \
  "$@" \
  -- \
  --offline_steps 200000 \
  --online_steps 200000 \
  --eval_interval 30000
