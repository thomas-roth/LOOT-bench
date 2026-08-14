# Closed experiments: do not run these again

The diagnosis behind ADR-001 spent 15 tickets and many hours of GPU time to
remove suspects. Each row below is closed by a measurement. Running one again
measures nothing new.

| Do not run | Why it is closed |
|---|---|
| Raising the discount | Two arms at γ = 0.995 and 0.999 left success at zero and moved `q_mean` from -20.32 to -18.33, against a predicted -40 and -200. The horizon is not the binding constraint. |
| A dense (shaped) reward arm | The dense dataset pays 31 % more per step, so the critic level owed a move to -26.7. It moved 11 % of that, and success stayed at the noise floor. |
| `horizon_length=1` | Action chunking is cleared. The deviation grows by the same amount at every position inside a chunk, with no step at the boundary. |
| An inference-time `flow_steps` sweep | 10, 50 and 200 steps give 0.397, 0.407 and 0.425. The residual spread is the learned velocity field, not the integration. |
| Reading `q_mean` as a value estimate | The level is set by `alpha`, the weight that holds the actor near the data, not by the MDP. The critic returns positive values, which are impossible when every reward is at or below zero, so it cannot rank actions. |
| More work on eval-reset coverage | Demonstration and evaluation resets draw from one distribution, and dataset actions pushed through the adapter score 9 successes in 10. |

## Traps worth knowing, in upstream `qc` at `d8a428f`

- `--agent.discount` is the TD discount. The top-level `--discount` flag only
  weights rewards inside a chunk and never reaches the bootstrap, so tuning it
  measures nothing. Raising `--horizon_length` does not extend the discount
  horizon either.
- `--sparse` maps `(rewards != 0.0) * -1.0`, which is safe only under ADR-006.
- Observation standardization is worth leaving on and is not a lead. It cuts
  the start-state error by 29 % and the sampling noise by 37 %, moves the bias
  barely, and leaves success at the noise floor.

## What counts as a rising curve

The noise floor is 1 success in 750 evaluation episodes (0.13 %), and that one
came from a diverged policy that swiped the cube by accident. A single 0.02
reading is noise. A result counts only when a non-zero success repeats over
consecutive evaluation points. Every number above comes from one seed
(seed 42), so no arm can yet separate "the change worked" from "this seed was
lucky".
