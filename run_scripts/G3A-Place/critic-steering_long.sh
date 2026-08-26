#!/usr/bin/env bash
# Real (non-BC) run of G3A-Place-Easy-Dk1_uni on the QC-FQL A/B config.
#
# A/B goal: baseline against the pure-BC runs (critic-off*.sh).
# The stock agents/acfql.py config trains the actor with distill-ddpg
# (bc_flow + alpha * distill + q loss); the critic steers the actor.
#
# Any extra arguments are passed through to train_qc.sh.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

exec "$REPO/train_qc.sh" G3A-Place-Easy-Dk1_uni "$REPO/out/qc/place-easy_2026-08-23_17-08-12.npz" \
  --seed=42 \
  --agent=agents/acfql.py \
  --offline_steps 1000000 \
  --online_steps 1000000 \
  --eval_interval 50000 \
  --eval_episodes 20 \
  --video_episodes 5 \
  --run_group=place_critic-steering \
  "$@"
