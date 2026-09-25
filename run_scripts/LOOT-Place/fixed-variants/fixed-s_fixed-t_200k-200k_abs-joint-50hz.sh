#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOOT_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$HERE/../../sweep.sh" \
  place_fixed-s_fixed-t_200k-200k_joint-abs-50hz \
  LOOT-Place-FixedS-FixedT-Dk1 \
  dataset/qc/place_fixed-s_fixed-t_100-demos_regen-50hz.npz \
  "$@" \
  -- \
  --offline_steps 200000 \
  --online_steps 200000 \
  --eval_interval 30000
