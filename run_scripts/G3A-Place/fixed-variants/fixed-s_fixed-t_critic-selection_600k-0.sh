#!/usr/bin/env bash
# Selection-only arm of G3A-Place-FixedS-FixedT-Dk1_uni, offline only.
#
# Diagnostic: the middle point of the critic axis, between two arms already
# measured on this dataset -- critic off (best-of-n, n=1; critic trained but
# never read) keeps 0.65 success at 600k, stock steering (distill-ddpg) keeps
# 0.32 and crosses down between 300k-400k. So the critic costs ~0.30 success
# and supplies the crossing; this arm asks *how*.
#
# actor_type=best-of-n takes the `else` branch of acfql.actor_loss for every
# n, so the actor is the same pure BC flow as critic-off. n=32 is the only
# difference: the agent integrates 32 action chunks and the critic picks one
# by argmax of its Q ensemble -- selection only, never the actor's gradient.
#
# Reading the result (vs. critic-off 0.65 and stock 0.32 at 600k, same seeds):
#   Holds near 0.65 -> critic ranks correctly; only the distill/Q gradient
#     path hurts the actor. Next: normalize_q_loss, a smaller Q weight, or
#     selection in place of steering.
#   Falls toward 0.32 -> the critic's values are wrong; steering just showed
#     it fastest. Next: the critic's target (q_agg, discount, bootstrap data).
#
# q_agg stays `mean` (default) -- it aggregates both the TD target and this
# arm's selection argmax, so changing it here would move two things at once.
# `alpha` is a label here, not a knob (unused in this arm's actor loss).
# Seeds match the critic-off arm on purpose -- the A/B must differ only by
# the critic.
#
# Cost: n=32 integrates the flow 32x per training step (acfql.critic_loss
# samples next_observations every step, not just at eval). critic-off took
# ~2.1h per run at 600k, three at once; this scales up by an amount only a
# measurement gives -- read the step rate in the log before planning a finish
# time.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export G3A_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$HERE/../../sweep.sh" \
  place_fixed-s_fixed-t_600k-0_critic-selection \
  G3A-Place-FixedS-FixedT-Dk1_uni \
  dataset/qc/place_fixed-s_fixed-t_99-demos_2026-08-30_10-46-36.npz \
  --alphas 10000 \
  --seeds 864194177,195836927,628594327 \
  "$@" \
  -- \
  --agent.actor_type=best-of-n \
  --agent.actor_num_samples=32 \
  --offline_steps 600000 \
  --online_steps 0 \
  --eval_interval 50000
