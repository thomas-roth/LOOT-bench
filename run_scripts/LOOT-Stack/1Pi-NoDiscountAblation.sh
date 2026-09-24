#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOOT_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$HERE/../sweep.sh" \
  1Pi-NoDiscountAblation \
  LOOT-Stack-1Pi-Dk1 \
  dataset/qc/stack_1pi_101+2-demos_2026-09-17.npz \
  "$@" \
  --alphas 100 \
  -- \
  --offline_steps 400000 \
  --online_steps 1000000 \
  --eval_interval 300000
