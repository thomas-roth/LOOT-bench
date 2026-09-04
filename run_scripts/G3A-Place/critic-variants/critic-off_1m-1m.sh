#!/usr/bin/env bash
# Pure BC flow on G3A-Place-RandS-RandT-Dk1_uni: the critic trains but never
# acts. best-of-n zeroes the distill and Q terms in acfql.actor_loss, and with
# actor_num_samples=1 the argmax has nothing to pick between.
#
# Extra arguments pass through to train_qc.sh.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export G3A_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$REPO/train_qc.sh" G3A-Place-RandS-RandT-Dk1_uni \
  "$REPO/dataset/qc/place_joint_rand-s_rand-t_149-demos_2026-08-15_16-18-15+2026-08-23_17-08-12.npz" \
  --agent=agents/acfql.py \
  --agent.actor_type=best-of-n \
  --agent.actor_num_samples=1 \
  --agent.normalize_q_loss=True \
  --start_pose_noise=0.05 \
  --seed=42 \
  --offline_steps=1000000 \
  --online_steps=1000000 \
  --eval_interval=50000 \
  --eval_episodes=20 \
  --video_episodes=6 \
  --run_group=place_critic-off \
  "$@"
