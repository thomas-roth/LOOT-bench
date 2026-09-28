#!/usr/bin/env bash
# NQL x alpha grid on LOOT-Place-FixedS-FixedT-Dk1, offline only.
#
# Question: with NQL on (the sweep.sh default) and alpha=100, the 200k-200k
# arms stall at 0.0 success through the whole offline phase and then jump to
# ~1.0 online -- under dense rewards AND, per the new controlled runs
# (ecyoqpgf/apok0k24/pt5w9nyo), under sparse rewards on the same dataset.
# The old sparse-learns-offline signal was an old-dataset confound, so reward
# type is out as the cause. Remaining suspect: alpha=100 with NQL is 10-1000x
# above the FQL-recommended NQL range (0.03-10), over-constraining the policy
# to the BC flow (offline q_loss pinned at 1.0, flat critic).
#
# Grid (same dataset, same steps, same eval cadence everywhere):
#   arm 1: NQL on  (stock default), low alphas on the normalized scale
#   arm 2: NQL off, high alphas on the raw-Q scale -- deliberately NOT the
#          same list: an alpha number is incomparable across the flag (NQL
#          divides Q by per-batch mean|Q|), so each arm sweeps its own scale.
#          What IS shared across arms is the seed list (see below).
#   arm 3: critic-off (best-of-n, n=1) pure-BC-flow baseline at one alpha;
#          alpha is a label here, not a knob (it does not enter that arm's
#          actor loss -- see fixed-s_fixed-t_critic-off_600k-0.sh)
#
# Offline-only 200k-0: extending only the winners online is the follow-up.
# Sparse reward (the default -- Place is sparse unless --dense is passed):
# the rest of the benchmark trains sparse, the stall reproduces under sparse
# on this dataset (ecyoqpgf/apok0k24/pt5w9nyo), so dense would add a
# non-standard arm for no diagnostic gain. Same .npz trains under either
# reward, so no new demos are needed.
#
# Seeds: all three arms run the SAME seeds (cross-arm comparability -- paired
# design, arms differ only in config, not in seed draw). Fixed in the file so
# re-running joins the same groups and --resume keeps working; override with
# --seeds A,B,C on the command line (yours wins, it reaches every arm).
# --num-seeds will NOT work here (the fixed --seeds below takes precedence).
# --dry-run plans all three sweeps; --resume continues them.
#
# Reading the result: if low-alpha NQL arms (0.1/1/10) learn offline while the
# old NQL/alpha-100 runs stall, the stall is over-regularization, and NQL
# stays the default with retuned (lower) alphas. If NQL-off learns where
# NQL-on stalls, compare the magnitudes (q/distill/bc_flow/|Q|), not the
# alpha numbers.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOOT_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

# One list per NQL scale -- do not merge them. Seeds are shared across arms
# (see above); alphas are not (an alpha number is incomparable across NQL).
ALPHAS_NQL_TRUE="0.1,1,10,100"
ALPHAS_NQL_FALSE="100,1000,10000"
SEEDS="101,202,303"
DATASET="dataset/qc/place_fixed-s_fixed-t_100-demos_2026-09-24.npz"
TASK="LOOT-Place-FixedS-FixedT-Dk1"

"$HERE/../../sweep.sh" \
  place_fixed-s_fixed-t_200k-0_nql-grid_nql-True \
  "$TASK" \
  "$DATASET" \
  --alphas "$ALPHAS_NQL_TRUE" \
  --seeds "$SEEDS" \
  "$@" \
  -- \
  --offline_steps 200000 \
  --online_steps 0 \
  --eval_interval 30000

"$HERE/../../sweep.sh" \
  place_fixed-s_fixed-t_200k-0_nql-grid_nql-False \
  "$TASK" \
  "$DATASET" \
  --alphas "$ALPHAS_NQL_FALSE" \
  --agent.normalize_q_loss=False \
  --seeds "$SEEDS" \
  "$@" \
  -- \
  --offline_steps 200000 \
  --online_steps 0 \
  --eval_interval 30000

"$HERE/../../sweep.sh" \
  place_fixed-s_fixed-t_200k-0_nql-grid_critic-off \
  "$TASK" \
  "$DATASET" \
  --alphas 100 \
  --seeds "$SEEDS" \
  "$@" \
  -- \
  --agent.actor_type=best-of-n \
  --agent.actor_num_samples=1 \
  --offline_steps 200000 \
  --online_steps 0 \
  --eval_interval 30000
