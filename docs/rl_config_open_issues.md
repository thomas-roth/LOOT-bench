# RL / simulation config — open issues (QC-FQL)

Supersedes `rl_mjlab_config_review.md` (2026-07-06) and
`rl_decisions_for_max.md`, both written against PPO configs and an older
category layout (`cube`/`insertion`/`ball`, a `--reward-mode` CLI flag)
before the `stack`/`domino`/`shape_sorter`/`peg_insertion` rename, the
uniform-reward-weight cleanup, and the scatter-sampler rewrite. Most of
those findings were either PPO-exploration-specific — moot now that
**QC-FQL, not PPO, is the trainer** (`g3a_bench/scripts/qc_fql/`; the PPO
runner config is kept only so `g3a-play` and the mjlab registry tooling
work, per `tasks/common/registration.py`) — or have since been resolved by
unrelated refactors. This doc keeps only what's still true against the
current codebase and still open, regardless of training algorithm.

## Resolved or no-longer-applicable (dropped, for traceability)

- **Success-termination/dt-scaling, failure-terminations-as-escape-hatches,
  hinge-curriculum timing, γ=0.99 bootstrap horizon, unclipped Gaussian
  wrist actions** (old F1/F2/F3/F5/F9): all describe a PPO *exploration*
  problem — a random policy discovering sparse success by trial and error.
  QC-FQL's actor imitates demonstrations (no reward in its BC loss) and its
  critic uses short n-step returns (H=5, no long bootstrap chain); none of
  these apply.
- **Success-reward weight inconsistency** (old F4, was 2.0/5.0 across
  domains): every currently-registered category (`stack`, `domino`,
  `shape_sorter`, `ball`) now uses a uniform `success` weight of 1.0.
  (`peg_insertion` still has old 2.0/5.0 weights, but it's archived and
  unregistered — not part of any active training run.)
- **Scatter placement not quite collision-free for rotated boxes** (old
  F10): the sampler was rewritten (`tasks/common/scatter.py:ScatterGrid`)
  to size cell pitch/jitter from each object's bounding-circle radius
  specifically so this can't happen, and it's now covered by
  `tests/tasks/test_scatter.py::test_tiling_pitch_and_jitter_are_collision_free`.
- **`max_iterations` split / `save_interval` W&B checkpoint volume** (old
  F11 notes): both are PPO-runner-only settings, kept solely for tooling
  compatibility (see above) — irrelevant to the QC-FQL training path.

## Remaining open issues

### 1. No registered category guards against EE-ground contact

None of `stack`, `domino`, `shape_sorter`, or `ball` wires an
`ee_contact_sensors` termination (checked every `config/__init__.py` and
`*_env_cfg.py` — none pass it to `register_category`/
`wire_manipulator_into_env`). `peg_insertion` had one (`ee_werkbox_collision`)
but is archived and unregistered. So in every currently-trainable task, the
wrist/fingers can grind against the terrain plane with no penalty or
episode-ending guard. Algorithm-independent — affects demo collection and
online fine-tuning rollouts alike.

### 2. Domain randomization is thin

Only finger-pad *sliding* friction is randomized
(`tasks/common/env_cfg.py`, `fingertip_friction_slide`, uniform 0.6-1.5).
No object mass, actuator-gain, or object-friction randomization anywhere.
Fine for a pure-sim benchmark; matters if sim-to-real transfer (the
GELLO/live-bridge direction) is a goal.

### 3. Solver iteration counts differ across categories with no stated reason

`stack`/`domino`: 10 solver iterations / 20 line-search; `ball`: 12/20;
`shape_sorter`: 15/25 (`peg_insertion`, archived: 15/25). Ball — a single
sphere on a plane, the least contact-rich task — running more iterations
than stack/domino (multi-body stacking with tight 1.5 cm tolerances) reads
backwards. Harmless if all are sufficient, but stacking stability is
exactly where too-few solver iterations shows up as jitter against the
success gate's position/velocity tolerances.

### 4. Ball's episode length is flat across every variant

`ball_env_cfg.py` sets `episode_length_s=12.0` regardless of variant —
Minigolf (Direct/UsingWall/Obstacle/Ramp) and Throw (U/MarkedArea/
WallVariation) all get the same budget. `stack`/`domino`/`shape_sorter`
instead scale episode length with task size (blocks/dominoes/pieces).
Worth checking whether the harder ball variants (e.g. `Ramp`, `WallVariation`)
need more time than `Direct`.
