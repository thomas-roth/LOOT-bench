#!/usr/bin/env bash
# Best-of-32 run of G3A-Place-Easy-Dk1_uni on QC-FQL.
#
# A/B goal: the closed-loop critic arm — the critic picks the action.
# Stock agents/acfql.py, with actor_type=best-of-n and actor_num_samples=32:
# the actor is trained as a pure BC flow (best-of-n zeroes the distill and Q
# terms in acfql.actor_loss), and at every decision point the agent samples
# 32 action chunks and the critic (TD-trained each step, argmax over its Q
# ensemble) selects which chunk to execute. The critic steers behaviour only
# through selection, never through the actor's gradient.
#
# This contrasts the two existing arms:
#   critic-steering_long.sh  (distill-ddpg: critic steers the actor's gradient)
#   critic-off*.sh     (best-of-n, n=1: critic never selects anything)
#
# Any extra arguments are passed through to train_qc.sh.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

exec "$REPO/train_qc.sh" G3A-Place-Easy-Dk1_uni "$REPO/out/qc/place-easy_2026-08-23_17-08-12.npz" \
  --seed=42 \
  --agent=agents/acfql.py \
  --agent.actor_type=best-of-n \
  --agent.actor_num_samples=32 \
  --offline_steps 1000000 \
  --online_steps 1000000 \
  --eval_episodes 20 \
  --video_episodes 20 \
  --run_group=place_critic-selection \
  "$@"
