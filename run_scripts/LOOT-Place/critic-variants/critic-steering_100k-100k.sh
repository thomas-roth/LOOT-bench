#!/usr/bin/env bash
# Stock QC-FQL on LOOT-Place-RandS-RandT-Dk1: distill-ddpg trains the actor
# with bc_flow + alpha * distill + q_loss, so the critic steers the actor's
# gradient but does not pick the action.
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
  "$REPO/dataset/qc/place_rand-s_rand-t_99-demos_2026-09-11.npz" \
  --agent.normalize_q_loss=True \
  --seed=42 \
  --offline_steps=100000 \
  --online_steps=100000 \
  --eval_interval=20000 \
  --eval_episodes=20 \
  --video_episodes=6 \
  --run_group=place_critic-steering \
  "$@"
