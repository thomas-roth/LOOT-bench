# One task reward in `[-1, 0]`, exactly zero only at success

Each task has one reward term. It stays inside `[-1, 0]` and reads exactly `0`
only on the success step. This replaces the earlier pair of `progress` and
`success` terms, each of weight 1.0, which summed into `[0, 2]`.

## Why the sign

Every task ends on success. With a positive reward for each step, ending the
episode costs the agent its income, so the best policy is to stay next to the
goal and never touch it. With a reward in `[-1, 0]` every step costs
something, and scoring is the only way to stop paying.

## Why exactly zero

QC's `--sparse` flag maps `(rewards != 0.0) * -1.0`. Any non-success step that
reads exactly `0.0` is then relabelled a success. A task can report full
progress before its success gate latches, because the objects are in place but
still moving. The shaped reward is therefore floored at `-MIN_SHORTFALL`
(1e-3), and only the success branch returns a true zero.

## The one-step lag

The reward lags the observation by one control step, on purpose. mjlab's
`step()` runs the reward and termination managers before the command manager,
so both read the success and progress buffers as they stood at the end of the
previous step. Success reached during step `k` is paid at step `k + 1`, which
is also the step that terminates. The last transition of a successful episode
therefore carries `r == 0` together with `terminated`, which is the absorbing
state offline RL wants.

The lag is identical when recording and when evaluating, so it cancels. Do not
correct it on one side alone, and make any tool that recomputes a reward along
a recording read the pre-step row.

## Consequences

- Rewards are not scaled by the timestep. `make_manipulation_env_cfg` pins
  mjlab's `scale_rewards_by_dt` to `False`. The default multiplies each reward
  by `step_dt`, which would tie the numbers in a recorded dataset to the
  decimation.
- `ball` keeps a sparse form. It has no progress signal, and the distance to
  the target says little about whether a throw will land, so it sits at `-1`
  until it scores.
- `tests/tasks/check_success_reward.py` asserts the convention on the live
  environment. It snapshots success and then steps, because a same-step
  comparison fails on scenes that are still settling.
