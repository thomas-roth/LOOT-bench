# The gym adapter is a seam and nothing else

`loot_bench/qc/env.py` presents one mjlab task to `qc` as a plain single
gymnasium environment. It does no reward shaping, no observation rewriting and
no action remapping. Anything that changes the MDP belongs in the task
definition, where the recorder sees it too.

## The load-bearing choices

Auto-reset is off. `qc`'s evaluation and online loops reset themselves and read
the final transition of each episode. With mjlab's auto-reset, the observation
returned next to `done` is the *next* episode's first observation. That
corrupts both the last transition in the replay buffer and the info dict the
success metric is read from.

Observation corruption is off, which matches how the demonstrations are
recorded. The offline and the online observations must come from one
distribution. No task declares observation noise any more, so the adapter's
flag is redundant rather than load-bearing; it stays as the seam's own
guarantee, because a cfg that arrives with corruption on must not reach a
training run.

`terminated` and `truncated` pass through as mjlab reports them. Its
termination manager routes each term by its own `time_out` flag, so the buffers
are already disjoint. Both can be true on one step, for a task that succeeds on
its last step, and that survives. Forcing truncation there would tell the
learner to bootstrap past a real absorbing state.

Success is latched over the episode, not sampled at the end. A success gate can
unlatch, for example when a block settles a little and the at-rest test stops
holding, while the episode ends a step later on the termination that already
fired. Reading the final step alone would score that episode a failure. The
latch keys off `reward == 0.0`, which ADR-006 already defines as success, so the
adapter needs no second opinion.

Observations are copied out of simulator memory. mjlab returns a view that the
next step overwrites, and `qc` holds the previous observation across the step to
build its transition. Without the copy, every transition would record its own
next state as its current state. `tests/qc/test_env.py` pins this.

`info` publishes contact telemetry in the **flat** form only. `qc`'s
`evaluation.py` reads the flat key and falls back to the nested
`info["proprio"]` form, so publishing only the nested form would leave the
counter blind. The first version of this adapter published neither, and the
contact counter read exactly 0.0 for every policy in every run. It now emits
the flat keys, which are the form the trainer handles safely in both branches.
They are telemetry: no success rule reads them. Do not compare the contact
columns of a run recorded before `dda5b2a` with those of a run recorded after
it.

The key names are a **cross-repo contract**: the adapter produces them in
`loot_bench/qc/episode.py` and `qc/evaluation.py` consumes them by name. The two
live in separate repositories with no version pin between them, and a drift is
silent: the consumer falls through to `None` and the counters read 0.0
rather than raising. After changing either side, run a short eval and check
`eval/pad_contact_runs` is non-zero; the DK1 pads always self-contact, so a
zero there means the contract broke.

**Amended.** The keys were renamed to say what they measure: `gripper_contact`
became `pad_contact_force`, `object_contact` became `object_contact_force`,
and on the `qc` side `num_gripper_contacts`/`avg_gripper_contact_length` became
`pad_contact_runs`/`pad_contact_run_steps`. The sensor names `grip_contact` and
`object_contact` are unchanged, so a key name and its sensor name now differ on
purpose. The same amendment fixed the flat-key read described above, which
previously ignored the nested branch it had just taken. Contact columns are
therefore not comparable across this change either.

## Consequence

`gymnasium` is now a direct dependency. It also arrives through mjlab, but the
adapter does not depend on that staying true.
