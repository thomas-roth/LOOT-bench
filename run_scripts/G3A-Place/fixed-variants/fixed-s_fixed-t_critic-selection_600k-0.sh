#!/usr/bin/env bash
# Selection-only arm of G3A-Place-FixedS-FixedT-Dk1_uni, offline only.
#
# Diagnostic, not a candidate configuration. It is the middle point of the
# critic axis, between two arms that are already measured on this dataset:
#
#   critic off (best-of-n, n=1)  the critic is trained and never read
#   steering   (distill-ddpg)    the critic moves the actor by its gradient
#
# The critic-off arm keeps 0.65 success at 600k. The stock steering arm keeps
# 0.32 at the same step, and every seed of it crosses down between 300k and
# 400k. So the critic costs about 0.30 of success, and it supplies the sharp
# crossing. This arm says *how* the critic costs it.
#
# actor_type=best-of-n takes the `else` branch of acfql.actor_loss for every n,
# so distill_loss and q_loss are zero and the actor is the same pure BC flow as
# the critic-off arm. n=32 is the only difference: at each decision the agent
# integrates 32 action chunks from the flow, and the critic picks one by the
# argmax of its Q ensemble. The critic therefore changes behaviour by selection
# alone, never by the actor's gradient.
#
# Reading the result. Compare against critic-off (0.65 at 600k) and the stock
# arm (0.32 at 600k), on these same three seeds.
#
#   Holds near 0.65 -> the critic ranks actions correctly. Only the distill/Q
#     gradient path hurts the actor. That is a configuration fault, and the
#     next arms are ones that weaken that path: normalize_q_loss, a smaller
#     Q weight, or selection in place of steering.
#
#   Falls toward 0.32 -> the critic's values are wrong, and steering is only
#     the fastest way to show it. No change to how the critic is *used* will
#     help. The next question is the critic's target: q_agg, the discount, or
#     the data it bootstraps from.
#
# q_agg stays at its default `mean`. It aggregates the ensemble both in the TD
# target and in this arm's selection argmax, so changing it here would move two
# things at once and break the A/B.
#
# `alpha` does not enter the actor loss in this arm. It is a label that keeps
# the tags comparable with the other two arms, not a knob.
#
# The seeds are the critic-off arm's seeds. The arms must differ by the critic
# and not by the seed draw.
#
# Cost. acfql.critic_loss calls sample_actions on next_observations every
# training step, so n=32 integrates the flow 32 times per step during training,
# not only at eval. The critic-off arm took about 2.1 h per run for 600k steps,
# three at once. This arm is slower by an amount that only a measurement gives:
# the flow sampling scales by 32, the critic and BC-flow parts do not. Start it
# and read the step rate in the log before you plan around a finish time.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export G3A_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$HERE/../../sweep.sh" \
  place_fixed-s_fixed-t_600k-0_critic-selection \
  G3A-Place-FixedS-FixedT-Dk1_uni \
  dataset/qc/place_eef_fixed-s_fixed-t_99-demos_2026-08-30_10-46-36.npz \
  --alphas 10000 \
  --seeds 864194177,195836927,628594327 \
  "$@" \
  -- \
  --agent=agents/acfql.py \
  --agent.actor_type=best-of-n \
  --agent.actor_num_samples=32 \
  --offline_steps 600000 \
  --online_steps 0 \
  --eval_interval 50000 \
  --eval_episodes 50 \
  --video_episodes 5
