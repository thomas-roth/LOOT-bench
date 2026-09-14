#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOOT_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$HERE/../sweep.sh" \
  1Henge-AlphaSweep-Discount0_995 \
  LOOT-Stack-1Henge-Dk1 \
  dataset/qc/stack_1henge_103-demos_2026-09-13_21-50-00.npz \
  "$@" \
  -- \
  --offline_steps 400000 \
  --online_steps 1000000 \
  --eval_interval 300000 \
  --discount 0.995
