#!/usr/bin/env bash
# Full-critic (best-of-32 + Q-steering) run of G3A-Place-RandS-RandT-Dk1_uni on QC-FQL.
#
# A/B goal: critic used for BOTH selection and steering.
# agents/acfql_full_critic.py sets actor_type=distill-ddpg-select: the actor
# loss is bc_flow + alpha*distill + q_loss (so the critic gradient-steers the
# actor toward high-Q chunks), and at every decision point the agent samples
# 32 action chunks and the critic (TD-trained each step, argmax over its Q
# ensemble) selects which chunk to execute.
#
# This contrasts the two single-role arms:
#   critic-steering_long.sh        (distill-ddpg:   steering only)
#   critic-selection_long.sh  (best-of-n, n=32: selection only)
#
# Any extra arguments are passed through to train_qc.sh.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export G3A_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$REPO/train_qc.sh" G3A-Place-RandS-RandT-Dk1_uni "$REPO/out/qc/place-rands-randt_2026-08-23_17-08-12.npz" \
  --seed=42 \
  --agent=agents/acfql_full_critic.py \
  --offline_steps 1000000 \
  --online_steps 1000000 \
  --eval_interval 50000 \
  --eval_episodes 20 \
  --video_episodes 6 \
  --run_group=place_critic-selection-steering \
  "$@"
