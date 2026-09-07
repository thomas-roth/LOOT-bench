#!/usr/bin/env bash
# Pure BC flow on LOOT-Place-RandS-RandT-Dk1: the critic trains but never
# acts. best-of-n zeroes the distill and Q terms in acfql.actor_loss, and with
# actor_num_samples=1 the argmax has nothing to pick between.
#
# Nothing reads the critic here, so it is shrunk to one hidden layer of 32 units
# (value_hidden_dims) to cut the offline step cost.
#
# ADR-016 note: this now trains a 20 Hz end-effector-delta dataset at the task's
# own start_pose_noise (0.0). Earlier runs in this run_group used absolute joint
# actions at 0.05, so wandb pools two incomparable sets under one name.
#
# Extra arguments pass through to train_qc.sh.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export LOOT_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$REPO/train_qc.sh" LOOT-Place-RandS-RandT-Dk1 \
  "$REPO/dataset/qc/place_rand-s_rand-t_100-demos_2026-09-04_14-09-57.npz" \
  --agent=agents/acfql.py \
  --agent.actor_type=best-of-n \
  --agent.actor_num_samples=1 \
  --agent.value_hidden_dims="(32,)" \
  --agent.normalize_q_loss=True \
  --seed=42 \
  --offline_steps=1000000 \
  --online_steps=1000000 \
  --eval_interval=50000 \
  --eval_episodes=20 \
  --video_episodes=6 \
  --run_group=place_critic-off \
  "$@"
