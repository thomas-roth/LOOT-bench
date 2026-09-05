#!/usr/bin/env bash
# Best-of-32 plus Q-steering on G3A-Place-RandS-RandT-Dk1_uni: the critic does
# both jobs. distill-ddpg-select keeps the full actor loss (bc_flow +
# alpha * distill + q_loss) and still lets the critic argmax over 32 sampled
# chunks at every decision.
#
# Extra arguments pass through to train_qc.sh.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export G3A_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$REPO/train_qc.sh" G3A-Place-RandS-RandT-Dk1_uni \
  "$REPO/dataset/qc/place_joint_rand-s_rand-t_149-demos_2026-08-15_16-18-15+2026-08-23_17-08-12.npz" \
  --agent=agents/acfql.py \
  --agent.actor_type=distill-ddpg-select \
  --agent.actor_num_samples=32 \
  --agent.normalize_q_loss=True \
  --start_pose_noise=0.05 \
  --seed=42 \
  --offline_steps=100000 \
  --online_steps=100000 \
  --eval_interval=20000 \
  --eval_episodes=20 \
  --video_episodes=6 \
  --run_group=place_critic-selection-steering \
  "$@"
