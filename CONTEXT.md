# G3A Benchmark

Long-horizon robot-manipulation benchmark tasks (block stacking, domino
placement, shape insertion, ball-to-target) registered into mjlab. Training
is QC-FQL (offline-to-online RL), which runs from a separate upstream `qc`
checkout; this repo holds only the seam to it, in `g3a_bench/qc/`. mjlab's
registry also requires a PPO cfg per task, but that RSL-RL PPO cfg is only a
registration requirement, not the training path used. See
`docs/TaskCategories.md` and each category's `README.md` for the per-category
deep dives.

## Language

### Tasks and registration

**Category**:
One of the four registered task families (stack, domino, shape sorter,
ball) plus the archived, unregistered peg insertion; a package under
`tasks/` with variants, assets and MDP terms.
_Avoid_: domain (overloaded with domain randomization)

**Variant**:
A frozen configuration enumerating one concrete task of a category (piece
set + arrangement, socket set, target kind). Registered as
`G3A-<Category>-<Task>[-<Variation>]-<Robot>`.

**Workbench**:
The shared tabletop-manipulation environment skeleton every task is built
on: manipulator at a table, proprioceptive + task observations, the one task
reward, success/time-out episode.

**Wiring**:
Filling the robot placeholders of a robot-agnostic task cfg from a
`ManipulatorInterface`. The single seam between tasks and robots.

**Fixed prop**:
A fixed-base scene entity (table, board, maze, decor) pinned to its
env-origin pose on every reset; mjlab wraps it as a mocap body. Contrast:
free bodies (pieces, pegs, ball) are placed by the command's resample.

**Task reward**:
The one reward term a task has, weight 1.0. It stays in `[-1, 0]` and reads
exactly `0` only at success. A category with a progress signal shapes the
interval; ball has none and sits at `-1` until it scores. See ADR-006.
_Avoid_: reward mode (the old name for a tier of `progress` + `success` terms)

### Control

**Control step**:
0.05 s — one action, 10 physics steps of 0.005 s (`decimation`). Every task
runs at this 20 Hz rate; it is `tasks/common/env_cfg.CONTROL_STEP_S` and
mjlab's `step_dt`. Both QC-validated reference domains use it, and the
recorded demos set a floor: at 10 Hz 3 % of demo steps stop fitting in one
action. See ADR-016.
_Avoid_: frame rate (the viewer's, unrelated), fps (the recorder's file
metadata, which follows this)

**Action box**:
The `[-1, 1]` cube every action lives in, per channel. `±1` is
`DELTA_POS_SCALE` (0.05 m) per position axis, `DELTA_ORI_SCALE` (0.5 rad) per
rotation axis, and the gripper's own stroke. Per-channel, not a norm: all
three position channels at `±1` is 0.087 m of travel. Quote it as "±5 cm per
axis", never "≤5 cm per step". The clamp to it is the convention itself, not
a safety net — without it the scale would be a unit rather than a bound.

**End-effector delta**:
What an action's first six channels mean: a pose delta on the robot's grasp
site, relative to where it currently is, solved into joint targets by
differential IK. A zero action holds still. The seventh channel is the
gripper, which is absolute.
_Avoid_: joint action, action scale (ADR-005's superseded rule, where an
action named an absolute joint position)

**Convention record**:
The stamp a dataset carries describing the MDP it was recorded under —
convention name, delta scales, gripper mapping, control step, task id
(`qc/dataset.STAMP_KEYS`). Compared against the environment a training run
builds, and raises on a mismatch. The failure it exists for is silent: the
loss falls and evaluation scores zero. See ADR-011.

### Success semantics

**Success criterion**:
A pure tensor function deciding a task-specific placement test (domino
upright/row-aligned, peg inserted, ball in target zone). No simulator
access; lives in the category's `mdp/success.py`.
_Avoid_: verdict function/kernel (old name), success check (ambiguous with
the gating)

**SuccessGate**:
The universal success gating: success criterion met AND scored objects at
rest AND released, latched per episode.

**Released**:
The robot has let go of the scored object. Gripper-joint opening for
cube/insertion; ball-to-grasp-site distance for ball.

**Episode success**:
Success latched since the last reset — once achieved it stays set for the
episode's metrics even if the state later degrades.

**SorterLatch**:
The shape sorter's scoring rule: a piece scores the moment it crosses the
mouth plane through its own hole into the cube, and stays scored while it
tumbles inside.

**Crossing plane**:
The depth (along the hole's outward axis) at which a sorter piece's tail has
cleared the cube's inner wall — fully inside.

**Insertion channel**:
The region in front of and through a bore/hole in which a peg or sorter
piece is considered "in transit": laterally within the hole outline, axially
between just outside the mouth and just below the crossing plane.

### Spawning

**ScatterGrid**:
Collision-free-by-construction spawn sampling: distinct cells of a grid,
each jittered by less than the half-cell size.

**Distractor**:
An extra free body with no goal slot/socket, present only to make the task
harder.
