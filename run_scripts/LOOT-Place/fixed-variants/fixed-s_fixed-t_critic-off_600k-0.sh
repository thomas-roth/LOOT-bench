#!/usr/bin/env bash
# Pure-BC-flow arm of LOOT-Place-FixedS-FixedT-Dk1_uni, offline only.
#
# Diagnostic: isolates whether the stock arm's offline success collapse
# (0.83 mean over steps 100k-300k -> 0.29 over 700k-1M, despite every
# training metric improving) comes from the critic, or from the BC fit
# sharpening on demo states while the closed-loop policy drifts.
#
# actor_type=best-of-n with actor_num_samples=1 takes the `else` branch of
# acfql.actor_loss: distill_loss and q_loss are zero, so the actor is a pure
# BC flow no critic gradient can move, and n=1 makes the critic's argmax a
# no-op. The critic is still TD-trained, just unused.
#
# Reading the result: compare this curve's shape against the stock arm's
# (crosses down between 300k-400k), not its level -- a pure BC flow can sit
# lower throughout and still be the control.
#   Falls the same way -> the BC fit itself, not the critic. Next: recovery
#     demos (ADR-013), or the online phase.
#   Stays flat -> the critic caused the stock arm's collapse. Next:
#     --agent.q_agg min, or the selection-only arm (best-of-n, n=32).
#
# `alpha` is a label here, not a knob -- it does not enter this arm's actor
# loss. 600k offline / 0 online: well past the crossing, online phase is out
# of scope. best-of-n integrates the flow on every TD target, so this runs
# ~2.4x the stock arm per step (~11h for three seeds at once).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOOT_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$HERE/../../sweep.sh" \
  place_fixed-s_fixed-t_600k-0_critic-off \
  LOOT-Place-FixedS-FixedT-Dk1_uni \
  dataset/qc/place_fixed-s_fixed-t_99-demos_2026-08-30_10-46-36.npz \
  "$@" \
  -- \
  --agent.actor_type=best-of-n \
  --agent.actor_num_samples=1 \
  --offline_steps 600000 \
  --online_steps 0 \
  --eval_interval 50000
