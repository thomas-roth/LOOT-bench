#!/usr/bin/env bash
# Pure-BC-flow arm of G3A-Place-FixedS-FixedT-Dk1_uni, offline only.
#
# Diagnostic, not a candidate configuration. The stock arm
# (fixed-s_fixed-t_1m-1m.sh) loses success during the offline phase on all
# three seeds -- 0.83 mean over steps 100k-300k, 0.29 over 700k-1M -- while
# every training metric improves. Two mechanisms fit that: the fit to the demo
# states sharpens and the closed-loop policy stops recovering from its own
# drift, or the critic extrapolates on a static dataset. This arm separates
# them.
#
# actor_type=best-of-n with actor_num_samples=1 takes the `else` branch of
# acfql.actor_loss: distill_loss and q_loss are zero, so the actor is a pure BC
# flow that no critic gradient can move, and n=1 makes the critic's argmax a
# no-op. The critic is still TD-trained, and never used.
#
# Reading the result. The stock arm on these same three seeds averages 0.83
# over steps 100k-300k and 0.29 over 700k-1M, and every seed crosses down
# between 300k and 400k. Compare the shape of this curve against that, not the
# level: a pure BC flow can sit lower from the start and still be the control.
#
#   Falls the same way -> no critic gradient reaches this actor, so the critic
#     did not cause it. What is left is the BC fit itself: it sharpens on the
#     demo states, and the closed-loop policy stops recovering from its own
#     drift. Next is data over the states the policy reaches -- recovery demos
#     (ADR-013), or the online phase. No critic change can help.
#
#   Stays flat -> the critic caused the collapse in the stock arm. Next are
#     --agent.q_agg min (it is `mean` over 2 heads today, so nothing in the
#     critic is pessimistic) and the selection-only arm, best-of-n with n=32.
#
# `alpha` does not enter the actor loss in this arm. Pass one value; it is a
# label that keeps the tags comparable with the stock arm, not a knob.
#
# 600k offline steps is well past the crossing, and the online phase is not
# part of the question, so it is turned off. best-of-n samples actions by
# integrating the flow, which the critic's TD target does every step, so this
# runs about 2.4x the stock arm per step: roughly 11 h per run, three seeds at
# once.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export G3A_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$HERE/../../sweep.sh" \
  place_fixed-s_fixed-t_600k-0_critic-off \
  G3A-Place-FixedS-FixedT-Dk1_uni \
  dataset/qc/place_eef_fixed-s_fixed-t_99-demos_2026-08-30_10-46-36.npz \
  "$@" \
  -- \
  --agent=agents/acfql.py \
  --agent.actor_type=best-of-n \
  --agent.actor_num_samples=1 \
  --offline_steps 600000 \
  --online_steps 0 \
  --eval_interval 50000 \
  --eval_episodes 50 \
  --video_episodes 6
