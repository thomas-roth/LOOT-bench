#!/usr/bin/env bash
# Seed sweep of G3A-Place-FixedS-FixedT-Dk1_uni on the QC-FQL config.
#
# The config is stock agents/acfql.py: the actor is trained with distill-ddpg
# (bc_flow + alpha * distill + q loss), so the critic steers the actor through
# its gradient. It is the same arm as the critic-steering variants, on the
# fixed-spawn, fixed-target variant and on 100k offline steps only.
#
# The script starts one run per seed, all at the same time, and puts all of
# them in the same wandb group. The seeds are drawn at random, so a sweep is
# an independent sample and not a repeat of the last one. The runs of the sweep
# are grouped together.
#
# A seed that fails does not stop the others: the runs are independent and a
# group of two is still worth having. The script names the failed seeds at
# the end and exits non-zero. Ctrl-C stops the whole sweep.
#
# Each run writes its own stdout to a log file located in the run's own folder:
#
#   $RUN_DIR/<task>/<run group>/seed-<seed>/train.log
#
# Environment overrides:
#   NUM_SEEDS   how many random seeds  (default: 3)
#   SEEDS       exact seeds to run     (default: NUM_SEEDS random ones)
#   RUN_GROUP   the wandb group        (default: <script name>_<start time>)
#   RUN_DIR     where runs are written (default: /var/tmp/$USER/qc-runs)
#
# Any extra arguments are passed through to train_qc.sh, for every seed.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export G3A_RUN_SCRIPT="$NAME"
TASK=G3A-Place-FixedS-FixedT-Dk1_uni

NUM_SEEDS="${NUM_SEEDS:-3}"
RUN_GROUP="${RUN_GROUP:-$(date +%Y-%m-%d_%H-%M-%S)_place_${NAME}}"
RUN_DIR="${RUN_DIR:-/var/tmp/${USER:-$(id -un)}/qc-runs}"
export RUN_DIR

if [ -z "${SEEDS:-}" ]; then
  seeds=()
  while [ "${#seeds[@]}" -lt "$NUM_SEEDS" ]; do
    s=$((RANDOM * 32768 + RANDOM))
    case " ${seeds[*]-} " in *" $s "*) continue ;; esac # Resample seed if already in sweep.
    seeds+=("$s")
  done
  SEEDS="${seeds[*]}"
fi

echo "[sweep] group  $RUN_GROUP"
echo "[sweep] seeds  $SEEDS"
echo "[sweep] runs   $RUN_DIR/$TASK/$RUN_GROUP"

pids=()
declare -A seed_of=()
for seed in $SEEDS; do
  log_dir="$RUN_DIR/$TASK/$RUN_GROUP/seed-$seed"
  mkdir -p "$log_dir"

  "$REPO/train_qc.sh" "$TASK" "$REPO/out/qc/place_fixed-s_fixed-t_149-demos_2026-08-27_01-11-30.npz" \
    --seed="$seed" \
    --agent=agents/acfql.py \
    --offline_steps 100000 \
    --online_steps 0 \
    --eval_interval 20000 \
    --eval_episodes 20 \
    --video_episodes 5 \
    --run_group="$RUN_GROUP" \
    "$@" >"$log_dir/train.log" 2>&1 &

  pids+=("$!")
  seed_of["$!"]="$seed"
  echo "[sweep] seed $seed started, pid $!, log $log_dir/train.log"
done

# Ctrl-C stops every run of the sweep.
stop_rest() {
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap 'echo "[sweep] interrupted, stopping the runs" >&2; stop_rest; exit 130' INT TERM

# Wait list for the runs still going.
waiting=("${pids[@]}")
failed=()
while [ "${#waiting[@]}" -gt 0 ]; do
  status=0
  wait -n -p done_pid "${waiting[@]}" || status=$?
  seed="${seed_of[${done_pid:-}]:-?}"
  rest=()
  for pid in "${waiting[@]}"; do
    [ "$pid" = "${done_pid:-}" ] || rest+=("$pid")
  done
  waiting=("${rest[@]}")
  left=${#waiting[@]}
  if [ "$status" -ne 0 ]; then
    echo "[sweep] seed $seed failed with status $status, see $RUN_DIR/$TASK/$RUN_GROUP/seed-$seed/train.log" >&2
    failed+=("$seed")
  else
    echo "[sweep] seed $seed done"
  fi
  echo "[sweep] $left run(s) still going"
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "[sweep] group $RUN_GROUP: failed seeds: ${failed[*]}" >&2
  exit 1
fi
echo "[sweep] group $RUN_GROUP: all seeds done"
