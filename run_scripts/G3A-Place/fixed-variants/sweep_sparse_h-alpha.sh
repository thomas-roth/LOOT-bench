#!/usr/bin/env bash
# Chunk-length x BC-weight grid on sparse-reward G3A-Place-FixedS-FixedT-Dk1_uni.
# Every cell is one call of fixed-s_fixed-t_100k-0.sh
# The cells run one after another.
# Each cell is its own wandb group, named <start time>_sparse_h-<h>_alpha-<a>.
#
# Environment overrides:
#   NUM_SEEDS   seeds per cell           (default: 3)
#   DATASET     the .npz to train on     (default: the inner script's)
#   RUN_DIR     where runs are written   (default: /var/tmp/$USER/qc-runs)
#
# Any extra arguments are passed through to every cell.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$HERE/fixed-s_fixed-t_100k-0.sh"

NUM_SEEDS="${NUM_SEEDS:-3}"
export NUM_SEEDS

# A cell that fails does not stop the grid: the cells are independent, and a
# partial grid still answers part of the question. The failed cells are named
# at the end and the script exits non-zero.
failed=()
for H in 5 7 10; do
  for A in 100 500 1000 5000 10000; do
    cell="sparse_h-${H}_alpha-${A}"
    echo "[grid] cell $cell"
    status=0
    RUN_GROUP="$cell" "$RUN" \
      --sparse=True \
      --horizon_length "$H" \
      --agent.alpha="$A" \
      --offline_steps 200000 \
      "$@" || status=$?
    # Ctrl-C reaches the inner script first; it stops its own seeds and exits
    # 130. Passing that on stops the grid rather than starting the next cell.
    if [ "$status" -eq 130 ]; then
      echo "[grid] interrupted at cell $cell" >&2
      exit 130
    fi
    if [ "$status" -ne 0 ]; then
      echo "[grid] cell $cell failed with status $status" >&2
      failed+=("$cell")
    fi
  done
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "[grid] failed cells: ${failed[*]}" >&2
  exit 1
fi
echo "[grid] all cells done"
