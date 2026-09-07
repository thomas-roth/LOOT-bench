#!/usr/bin/env bash
# Best-of-32 plus Q-steering on LOOT-Place-RandS-RandT-Dk1: the critic does
# both jobs. distill-ddpg-select keeps the full actor loss (bc_flow +
# alpha * distill + q_loss) and still lets the critic argmax over 32 sampled
# chunks at every decision.
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
  --agent.actor_type=distill-ddpg-select \
  --agent.actor_num_samples=32 \
  --agent.normalize_q_loss=True \
  --seed=42 \
  --offline_steps=100000 \
  --online_steps=100000 \
  --eval_interval=20000 \
  --eval_episodes=20 \
  --video_episodes=6 \
  --run_group=place_critic-selection-steering \
  "$@"
