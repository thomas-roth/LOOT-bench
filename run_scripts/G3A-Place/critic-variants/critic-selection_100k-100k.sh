#!/usr/bin/env bash
# Best-of-32 run of G3A-Place-RandS-RandT-Dk1_uni on QC-FQL.
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
#   critic-steering_*.sh  (distill-ddpg: critic steers the actor's gradient)
#   critic-off_*.sh       (best-of-n, n=1: critic never selects anything)
#
# Any extra arguments are passed through to train_qc.sh.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export G3A_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$REPO/train_qc.sh" G3A-Place-RandS-RandT-Dk1_uni \
  "$REPO/dataset/qc/place_joint_rand-s_rand-t_149-demos_2026-08-15_16-18-15+2026-08-23_17-08-12.npz" \
  --agent=agents/acfql.py \
  --agent.actor_type=best-of-n \
  --agent.actor_num_samples=32 \
  --agent.normalize_q_loss=True \
  --start_pose_noise=0.05 \
  --seed=42 \
  --offline_steps=100000 \
  --online_steps=100000 \
  --eval_interval=20000 \
  --eval_episodes=20 \
  --video_episodes=6 \
  --run_group=place_critic-selection \
  "$@"
