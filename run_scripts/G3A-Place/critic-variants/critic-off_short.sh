#!/usr/bin/env bash
# Pure-BC-flow run of G3A-Place-RandS-RandT-Dk1_uni on the QC-FQL A/B config.
#
# A/B goal: isolate the effect of the critic/distill losses on the actor.
# Stock agents/acfql.py with actor_type=best-of-n and actor_num_samples=1:
# the actor loss is pure bc_flow (best-of-n zeroes the distill and Q terms
# in acfql.actor_loss), and the single-sample selection makes the critic
# argmax a no-op, so the critic is trained but never used. The smoke run
# showed this no longer drifts the cube away from its start pose, unlike
# the stock distill-ddpg.
#
# Any extra arguments are passed through to train_qc.sh.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

exec "$REPO/train_qc.sh" G3A-Place-RandS-RandT-Dk1_uni "$REPO/out/qc/place-rands-randt_2026-08-23_17-08-12.npz" \
  --seed=42 \
  --agent=agents/acfql.py \
  --agent.actor_type=best-of-n \
  --agent.actor_num_samples=1 \
  --offline_steps 100000 \
  --online_steps 100000 \
  --eval_interval 20000 \
  --eval_episodes 20 \
  --video_episodes 5 \
  --run_group=place_critic-off \
  "$@"
