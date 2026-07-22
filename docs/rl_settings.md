# RL settings

Overview of reward, observation, termination and randomization design across
task categories. 

## Shared design (all tasks)

**Observations**
- Actor and critic see the same terms; actor's copy has small noise added
  (e.g. ±0.01 on positions), critic's is clean — standard asymmetric
  actor-critic.
- Proprioception (all tasks): joint positions, joint velocities,
  end-effector pose.
- Task observations: category-specific, see below.
- Command: current goal (e.g. target stack height), where applicable.
- Last action.

**Actions**
- Single joint-position action term over all actuators, scaled to keep
  policy output near ±1.

**Rewards**
- Every task uses exactly two terms, both weight 1.0, no shaping/penalties:
  - `progress` — dense, fraction of task complete in `[0, 1]`.
  - `success` — sparse bonus once the success gate (below) holds; latches
    for the rest of the episode.
- Exception: **Ball has no `progress` term**, success only — deliberately
  reward-sparse to test long credit-assignment gaps.

**Success gate** — a task counts as done only once, simultaneously:
1. Task condition (category-specific placement/insertion/arrival test).
2. At rest — object speed below threshold (0.01 m/s default), so mid-motion
   objects aren't credited.
3. Released — gripper open *and* no measured grip contact force.
- Exception: **Shape Sorter skips the at-rest check** — insertion credited
  immediately, since the socket physically arrests the piece.

**Terminations (all tasks)**
- `time_out` — episode length exceeded (scales with difficulty).
- `success` — gate above fully satisfied.
- Categories may add extra failure/end conditions, see below.

**Randomization**
- Objects tiled into non-overlapping grid cells across the workspace, each
  getting random jitter + random starting yaw, every episode.
- Fixed props (boards, fields, walls) don't move mid-episode but their
  placement is randomized at reset too, within task-specific ranges.
- Deliberate anti-overfitting: nothing spawns in a fixed pose.

## Per-category specifics

### Stack
- **Progress**: tallest reached stack height ÷ target height.
- **Observations**: pose of every block.
- **Success**: top of a valid stack within a small tolerance of target
  height. `TwoStacks` variant also requires the two piles ≥6 cm apart.
- **Randomization**: blocks scattered once at episode start, ~23×42 cm
  workspace.

### Domino
- **Progress**: fraction of dominoes correctly placed; `KnockOver` variant
  switches to half placement + half toppled-fraction after the build phase.
- **Observations**: pose of every domino, per-slot placed flag, phase flag
  (`KnockOver` only: build vs. topple).
- **Success**: all dominoes upright and row-aligned (~15° tolerance).
  `KnockOver` additionally requires every domino fallen along the row, same
  direction, in row order.
- **Randomization**: row spacing (5 cm) intentionally < domino height so a
  fall can cascade; `DifferentDominoes` shrinks mass/thickness down the row.

### Shape Sorter
- **Progress**: mean fraction of pieces currently latched as inserted.
- **Observations**: cube pose, pose of every piece, per-socket
  inserted/not-inserted flag.
- **Success**: one-way "crossed into the socket" detector per piece — once
  a piece passes deep enough and aligned with the hole axis it latches
  inserted and can't be un-marked by pulling back out. No at-rest check.
- **Randomization**: cube face/yaw randomized (mode varies: fixed
  orientation, no forced hole-down, or fully free); pieces scattered in
  front of the sorter, or on a fixed ring for `randomPositions`.

### Ball (Minigolf / Throw)
- **Progress**: none — success only.
- **Observations**: ball position + velocity, field pose, wall pose (if
  present).
- **Success**: horizontal distance to goal under a target radius; hole
  goals also need the ball below the surface, flat goals just need it near
  the surface. At-rest threshold is per-variant, not the shared default.
- **Extra terminations**: `out_of_bounds` (>1.2 m from robot),
  `off_platform` (minigolf, fell off edge outside any pit),
  `wall_collision` (throw-with-wall, hit wall instead of clearing it).
- **Randomization**: ball spawn jitter, plus reset-time jitter on
  field/wall pose within small task-specific ranges.
