#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export G3A_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$HERE/../sweep.sh" \
  place_fixed-s_fixed-t_200k-200k \
  G3A-Place-FixedS-FixedT-Dk1_uni \
  dataset/qc/place_eef_fixed-s_fixed-t_99-demos_2026-08-30_10-46-36.npz \
  "$@" \
  -- \
  --agent=agents/acfql.py \
  --offline_steps 200000 \
  --online_steps 200000 \
  --eval_interval 30000 \
  --eval_episodes 50 \
  --video_episodes 5
