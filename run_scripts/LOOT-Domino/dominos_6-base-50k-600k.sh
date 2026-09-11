#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export G3A_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$HERE/../sweep.sh" \
  dominos_6-base_50k-600k \
  LOOT-Domino-6Pieces-Dk1_uni \
  dataset/qc/dominos_6-base_101-demos_2026-09-06.npz \
  "$@" \
  -- \
  --agent.normalize_q_loss=True \
  --offline_steps 50000 \
  --online_steps 600000 \
  --eval_interval 50000 \
  --eval_episodes 10
