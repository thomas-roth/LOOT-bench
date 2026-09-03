#!/usr/bin/env bash
# Alpha sweep of G3A-Place-FixedS-FixedT-Dk1_uni on the stock QC-FQL config.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export G3A_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$HERE/../../sweep.sh" \
  place_fixed-s_fixed-t_1m-1m \
  G3A-Place-FixedS-FixedT-Dk1_uni \
  dataset/qc/place_eef_fixed-s_fixed-t_99-demos_2026-08-30_10-46-36.npz \
  "$@" \
  -- \
  --agent=agents/acfql.py \
  --video_episodes 5
