#!/usr/bin/env bash
# Pure-BC-flow run of G3A-Place-Easy-Dk1_uni on the QC-FQL A/B config.
#
# A/B goal: isolate the effect of the critic/distill losses on the actor.
# The config agents/acfql_g3a_place_bc.py (in the qc checkout) trains a
# pure-BC actor (best-of-n, 1 sample); the smoke run showed this no longer
# drifts the cube away from its start pose, unlike the stock distill-ddpg.
# The critic is still trained but is never used to steer the actor.
#
# Any extra arguments are passed through to train_qc.sh, e.g.:
#   ./run_g3a_place_bc.sh --run_group=G3A-Place-bcf1 --seed=1

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$REPO/train_qc.sh" G3A-Place-Easy-Dk1_uni "$REPO/out/qc/place-easy.npz" \
  --agent=agents/acfql_g3a_place_bc.py \
  --offline_steps 1000000 \
  --online_steps 1000000 \
  --eval_episodes 20 \
  --video_episodes 20 \
  "$@"
