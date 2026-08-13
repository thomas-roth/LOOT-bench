# G3A Benchmark

Long-horizon robot-manipulation benchmark tasks (block stacking, domino
placement, shape insertion, ball-to-target) registered into mjlab. Training
is QC-FQL (offline-to-online RL), run by the upstream `qc` repo against the
`g3a_bench.qc` OGBench-style factory (see the README's "Training" section).
mjlab's registry also requires a PPO cfg per task, but that RSL-RL PPO cfg is
only a registration requirement, not the training path used. See
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
on: manipulator at a table, proprioceptive + task observations,
regularization rewards with curriculum, success/time-out episode.

**Wiring**:
Filling the robot placeholders of a robot-agnostic task cfg from a
`ManipulatorInterface`. The single seam between tasks and robots.

**Fixed prop**:
A fixed-base scene entity (table, board, maze, decor) pinned to its
env-origin pose on every reset; mjlab wraps it as a mocap body. Contrast:
free bodies (pieces, pegs, ball) are placed by the command's resample.

**Reward mode**:
A named tier of active task reward terms (`progress` keeps the task's
completion-fraction shaping on top of the outcome signal -- discrete for
domino and the sorter, continuous distance for place; `success` keeps only
the sparse outcome signal). Regularization terms stay active in every mode.
A category may register one variant per mode (place does: `Easy` vs
`Easy-Dense`); the modes differ in `reward_max`, so each needs its own
recorded dataset.

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
