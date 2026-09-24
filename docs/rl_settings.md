# RL settings

Overview of the reward, observation, termination and randomization design
across the task categories.

This document is descriptive. The records in `docs/adr/` are the authority.
If this file and a record disagree, the record is correct.

## Shared design (all tasks)

**Observations**

- `actor` and `critic` read the same terms, and no term carries noise
  (`docs/adr/015-observations-carry-no-noise.md`). Training is offline from
  recorded demonstrations, and the demonstrations came through the
  uncorrupted observation.
- Proprioception (all tasks): joint positions, joint velocities, the
  end-effector pose, and — except Ball, which declares no grip sensor —
  the finger-pad contact force (1 wide: the same reduction the release
  gate reads, so the policy sees the quantity its episode terminates on,
  `docs/adr/018-contact-only-release-gate.md`).
- The end-effector pose is 9 wide: the position, then the first two columns
  of the rotation matrix. It is never a quaternion
  (`docs/adr/007-ee-pose-publishes-a-6d-rotation.md`).
- Task observations: category-specific, see below.
- Command: the current goal (for example Place's target point), where the
  category publishes one. Stack publishes none: a pi's level state is a
  function of the block poses the policy already sees.
- No last-action term. It made "repeat the last action" a trivially strong
  regressor, so a policy could score on the demos without reading the scene
  (`docs/adr/016-control-at-20hz-through-a-5cm-end-effector-delta.md`).

**Actions**

- Two terms, 7 wide, at a 0.05 s control step
  (`docs/adr/016-control-at-20hz-through-a-5cm-end-effector-delta.md`).
- The arm: `EndEffectorDeltaAction`, mjlab's differential IK in relative mode
  on the grasp site. `±1` is 0.05 m per position axis and 0.5 rad per rotation
  axis, per channel, and a zero action holds still.
- The gripper: `NormalizedJointPositionAction` on its one actuated joint. `-1`
  is the closed limit, `+1` the open one, read off the compiled model.
- Both clamp the raw action to `[-1, 1]`. The clamp is the convention, not a
  guard: differential IK scales whatever it is handed.

**Rewards**

- Every task has exactly one reward term, named `task`, at weight 1.0. There
  are no shaping terms and no penalties
  (`docs/adr/006-one-task-reward-in-minus-one-to-zero.md`).
- The term stays in `[-1, 0]` and reads exactly `0` only at success. Each
  task ends on success, so a positive per-step reward would pay a policy to
  stay at the goal and not touch it.
- A category with a progress signal shapes the interval as `progress - 1`. A
  category without one (Ball) stays at `-1` until it scores.
- A non-success step is floored at `-MIN_SHORTFALL` (1e-3). A task can report
  full progress before the success gate latches, and without the floor that
  step would read a true zero, which QC's `--sparse` relabels a success.
- The reward lags the observation by one control step. mjlab runs the reward
  and termination managers before the command manager, so success in step `k`
  is paid at step `k + 1`, which is also the step that terminates. The lag is
  the same when recording and when evaluating, so it cancels. Do not correct
  it on one side alone.
- Rewards are not scaled by the timestep (`scale_rewards_by_dt=False`), or
  the numbers in a recorded dataset would follow the decimation.

**Success gate**: a task is complete when all of these are true at the same
time:

1. Task condition (the category-specific placement, insertion or arrival
   test).
2. At rest: the object speed is at or below the threshold (0.01 m/s by
   default), so objects still in motion get no credit.
3. Released: the measured finger-pad contact force is at or below `MIN_GRIP_FORCE`
  (0.1 N) — contact-only, no gripper-joint check
  (`docs/adr/018-contact-only-release-gate.md`).

- Exception: **Sorter skips the at-rest check** (its gate threshold is
  infinite). The socket holds the piece, so the insertion is credited
  immediately.
- Exception: **Ball uses a distance test for the release**, not the gripper
  gate. The ball counts as released when it is more than `_RELEASE_DISTANCE`
  from the grasp site, because a thrown or putted ball leaves the gripper
  and the gripper can be in any state after that.

**Terminations (all tasks)**

- `time_out`: per-category budget, see table below. `success`: the gate
  above is fully satisfied. Extra failure/end conditions per category,
  see below.

| Category | Episode length |
| --- | --- |
| Stack | 75 / 150 / 225 s (25 s per block) |
| Domino | 20 s per domino, +30 s for `KnockOver` |
| Sorter | 20 s per piece |
| Ball | 10 / 12 / 16 / 20 s minigolf, 12 / 12 / 16 s throw |
| Place | 15 s (300 steps) |

**Randomization**

- Free objects stage from a fixed per-category spawn table: object `i` takes
  cell `i` with ±1 cm jitter and a random start yaw at every reset. The
  tables sit near the robot and keep the work area clear by construction, so
  no keep-out mask is needed
  (`docs/adr/019-fixed-spawn-points.md`). Place is exempt: its cube spawn and
  target draw uniformly over the shared envelope when random and pin exactly
  when fixed.
- Fixed props (boards, fields, walls) do not move during an episode. By
  default a reset event pins each one to `env_origins + init_state.pos`, so
  the prop is in the same place at every reset. A category can widen that
  range, and Ball is the only one that does: it samples the field pose, and
  the wall pose where the variant has a wall.
- Not every task randomizes everything. The domino row's centre x is pinned
  at every reset, because a teleoperator needs a layout that does not move
  between takes, and the `RandomRow` variants are the ones that opt back into
  a sampled centre y and yaw. Place switches its two axes independently, so `FixedS` pins
  the cube spawn and `FixedT` pins the target.
- The robot starts every episode at its home pose, plus a uniform sample of
  half-width `start_pose_noise`. The default is `0.0`
  (`docs/adr/008-episodes-start-at-the-home-pose.md`). A run can raise it
  with `train_qc.sh --start_pose_noise`, but the dataset stamp must then
  agree (`docs/adr/011-datasets-stamp-the-action-convention.md`).

## Per-category specifics

### Stack

- **Progress**: the fraction of pi levels complete, `levels / N`,
  instantaneous, so a collapse takes the credit back. A level is complete
  when some *resting* block lies horizontal (`|cos| <= TILT_COS_MAX`, 0.10)
  with its centre within `HEIGHT_TOL` (3 mm) of the level's target height
  (0.075 / 0.160 / 0.245 m). A block held in the gripper is excluded, so
  lifting one to the target height earns nothing until it is released
  (ADR-017). Under `--sparse` a 1Pi run is literally sparse; 3Pis
  keeps three coarse rungs that the flag flattens.
- **Observations**: the pose of every block. No command term.
- **Success**: every level complete, at rest and released. The criterion
  tests the lintel only, never the supports; a T and a trilithon both count
  (`docs/adr/020-a-pi-level-is-a-horizontal-block-at-a-fixed-height.md`).
  The two constants are coupled to the block geometry and checked at import,
  so neither can be tuned alone.
- **Randomization**: the blocks stage standing from the fixed table once at
  the start of the episode, each at a random yaw.
- **Episode length**: 75 / 150 / 225 s (1500 / 3000 / 4500 steps at 20 Hz),
  25 s per block, because a lintel costs a carry-and-place plus a knock-over
  and a re-grasp. At 4500 steps QC's 100-step horizon covers 2.2 % of an
  episode, against the 33-50 % band ADR-016 validates. That departure is
  accepted knowingly: a shorter budget yields silently truncated recordings,
  and the discount experiment is closed (ADR-014).

### Domino

- **Progress**: the placement fraction while building. The `KnockOver`
  variants use half the placement fraction during the build phase, then a
  latched 0.5 plus half the toppled fraction during the topple phase.
- **Observations**: the pose of every domino, a per-slot placed flag, the
  per-slot goal poses as command, and a `phase` flag (build or topple,
  `KnockOver` only).
- **Success**: every domino within 15 mm × margin of its slot, upright and
  row-aligned (15° × margin). `KnockOver` also needs every domino fallen
  along the row (>45° tilt, within a 45° cone of the row axis), all leaning
  the same way, and in row order.
- **Randomization**: the dominoes stage from the fixed table at every reset;
  spawns and row separate in x by construction. The row itself is pinned in
  x at 0.40, except in the `RandomRow` variants, which redraw the row centre
  y (±0.08 m) and yaw (±0.35 rad). The row spacing is always fixed (0.05 m),
  so the knock-over stays possible. Short rows are the first n slots of the
  6-row.

### Sorter

- **Progress**: the fraction of the pieces that are latched as inserted.
- **Observations**: the cube pose, the pose of every piece, and a per-socket
  inserted flag. The command (mouth positions + axes) exists but is not
  observed.
- **Success**: a one-way "crossed into the socket" detector per piece. Once a
  piece goes deep enough and aligned with the hole axis, it latches as
  inserted and a pull-back cannot un-mark it. Hole slack is 5 mm × margin.
  No at-rest check.
- **Randomization**: the cube sits at (0.375, 0) ±1 cm in every mode; only
  the face-down face and yaw vary (fixed: dead star face; no-rotation: an
  allowed face; full: any of the six). The pieces stage from a fixed table
  in front of the sorter with the insertion corridor kept clear.

### Ball (Minigolf / Throw)

- **Progress**: none. The reward is the sparse member of the convention: `-1`
  every step until the ball scores. A ball is either in the target zone or
  not, and the distance says little about where the throw will land.
- **Observations**: the ball position (no orientation) and linear velocity,
  the field pose (which holds the fixed positions of the obstacles, the
  ramp and so on), the target as command, and the wall pose for
  `Throw/WallVariation` only (other walls are baked into the field).
- **Success**: the horizontal distance to the goal is under the target
  radius (pit: physical hole radius 0.045, unscaled; throw: 0.04 × margin).
  A hole goal also needs the ball below the surface, and a flat goal only
  needs it near the surface. At-rest is 0.06 m/s, release is >0.08 m from
  the grasp site.
- **Extra terminations**: `out_of_bounds` (ball xy more than 1.2 m from the
  env origin), `off_platform` (minigolf, the ball fell off an edge outside any pit), and
  `wall_collision` (any variant with a wall, the ball hit the wall instead of
  clearing it).
- **Randomization**: the ball spawn (x 0.24–0.30, y ±0.08) and the field
  pose (x ±0.02, y ±0.03); the wall pose only where the variant has a
  movable wall (x ±0.06, sink 0..0.08). The command therefore tracks the
  target relative to the field's live pose, not as a constant offset.

### Place

- **Progress**: the fraction of the spawn-to-target travel completed,
  normalized by the constant `MAX_TRAVEL_DIST` and not by anything
  per-episode.
- **Observations**: the cube pose, and the command, which is the env-local
  target point.
- **Success**: the cube is within 0.02 m × margin (3D) of the target, at
  rest, and released.
- **Randomization**: the cube spawn and the target are two independent axes,
  and each variant pins or samples them separately.
- Place is the one-cube sanity check that the QC pipeline is brought up on.
  It has 300 steps (15 s at 20 Hz) and no difficulty axis.
