#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOOT_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$HERE/../sweep.sh" \
  dominos_2-base-600k-0_critic_off \
  LOOT-Domino-2Pieces-Dk1 \
  dataset/qc/dominos_2-base_101-demos_split_2026-09-06.npz \
  "$@" \
  -- \
  --agent.actor_type=best-of-n \
  --agent.actor_num_samples=1 \
  --agent.value_hidden_dims="(32,)" \
  --offline_steps 600000 \
  --online_steps 0 \
  --eval_interval 100000
