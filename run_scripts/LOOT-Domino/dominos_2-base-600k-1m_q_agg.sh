#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOOT_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$HERE/../sweep.sh" \
  dominos_2-base-600k-1m_q_agg \
  LOOT-Domino-2Pieces-Dk1 \
  dataset/qc/dominos_2-base_101-demos_split_2026-09-06.npz \
  "$@" \
  -- \
  --agent.q_agg=min \
  --agent.num_qs=10 \
  --offline_steps 600000 \
  --online_steps 1000000 \
  --eval_interval 100000
