#!/usr/bin/env bash
# Alpha sweep of LOOT-Place-FixedS-FixedT-Dk1 on the stock QC-FQL config.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOOT_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$HERE/../../sweep.sh" \
  place_fixed-s_fixed-t_200k-0 \
  LOOT-Place-FixedS-FixedT-Dk1 \
  dataset/qc/place_fixed-s_fixed-t_99-demos_2026-08-30_10-46-36.npz \
  "$@" \
  -- \
  --offline_steps 200000 \
  --online_steps 0 \
  --eval_interval 30000
