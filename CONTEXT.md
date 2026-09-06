# G3A Benchmark

Long-horizon robot-manipulation benchmark tasks (block stacking, domino
placement, shape insertion, ball-to-target, cube placing) registered into
mjlab. Training is QC-FQL (offline-to-online RL), which runs from a separate
upstream `qc` checkout; this repo holds only the seam to it, in
`g3a_bench/qc/`. mjlab's registry also requires a PPO cfg per task, but that
RSL-RL PPO cfg is only a registration requirement, not the training path used.
See `docs/task_categories.md` and each category's `README.md` for the
per-category deep dives.

## Language

### Tasks and registration

**Category**:
One of the five registered task families (stack, domino, shape sorter, ball,
place) plus the archived, unregistered peg insertion; a package under
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

**Shortfall floor**:
`MIN_SHORTFALL` (1e-3), the distance below zero every non-success reward must
keep. It is what makes QC's `--sparse` mapping safe.

**Reward lag**:
The one control step between a success and the reward that pays it, caused by
mjlab's manager order. It is identical when recording and when evaluating, so
it cancels. See ADR-006.

**Home pose**:
The absolute per-joint start pose (`START_JOINT_POS`) every episode resets to
and the teleop operator aligns the leader arm to. Contrast: the *rest* pose,
where the leader arm hangs, and the MJCF *keyframe*, which mjlab's offset reset
would use. See ADR-008.

**Start-pose noise**:
Half-width of the uniform sample added around the home pose at reset
(`start_pose_noise`, default `0.0`). A task fact first; `train_qc.sh
--start_pose_noise` (or `G3A_START_POSE_NOISE`) can override it for a run,
and the dataset stamp then has to agree or the load aborts.

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
convention name, delta scales, gripper mapping, control step, start-pose
noise, task id
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
rest AND released, latched per episode. The at-rest threshold is per
category: the shape sorter sets it to infinity, because a piece that has
crossed its hole is scored while it still tumbles inside the cube.

**Released**:
The robot has let go of the scored object. Gripper-joint opening for
cube/insertion; ball-to-grasp-site distance for ball.

**Episode success**:
Success latched since the last reset — once achieved it stays set for the
episode's metrics even if the state later degrades.

**Row slot**:
One of the fixed target poses a domino must be placed at, spaced closer than
a domino's height so a topple runs down the line. The row itself is at the
same workspace-centred pose every reset; only the `RandomRow` variants draw
its start position and yaw. The domino at index *i* carries its own identity
color, so a teleoperator can read which piece belongs to which slot. The
`KnockOver` variants add a `phase` observation, `0` while the line is being
built and `1` once the topple is required.

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

**Keep-out**:
Grid cells a `ScatterGrid` may not spawn into because they sit within a
clearance of a goal position — an object dropped on its own target is
awkward to align from there. Used by the domino row (`blocked_near`, whose
`keep_free` argument holds back enough cells to still place every object).

**Distractor**:
An extra free body with no goal slot/socket, present only to make the task
harder.

### Training seam

**Seam**:
The whole of what this repo gives the trainer: the gym adapter, the demo
converter and the dataset contract in `g3a_bench/qc/`. It changes no part of
the MDP, because the recorder must see the same MDP as the trainer. See
ADR-002 and ADR-009.

**Adapter**:
`g3a_bench/qc/env.py`, one mjlab task presented to `qc` as a plain single
gymnasium environment. Auto-reset off, corruption off, `terminated` and
`truncated` kept apart, observations copied out of simulator memory.

**Action convention**:
The `scale` and `offset` that map an action to joint targets. The converter
stamps the pair in force at record time into the dataset, and `load_dataset`
raises when it does not match the environment a run builds. See ADR-011.

**Boundary** (`terminals`):
The end of an episode in a converted dataset. Every episode end is one.

**Bootstrap mask** (`masks`):
Whether to bootstrap past a transition. Only an absorbing state — a success —
gets `0`. Not the same column as the boundary. See ADR-010.

**BC gate**:
The behaviour-cloning fit (`tests/qc/bc_gate.py`) read on a demonstration set
before a training run is started. A measurement instrument, not pipeline code:
observation in, the action chunk out, held-out error read back. See ADR-012.

**Recovery demonstration**:
A recorded attempt that starts off the ideal path, near the states a rollout
really visits. Contrast: a *nominal* demonstration, another clean run of the
same kind. Proposed as the answer to demonstrations that are too far apart to
interpolate between, and **rejected for now** in favour of pinning the spawn
and target instead; none has been recorded. See ADR-013.

**Closed experiment**:
A measurement that already removed a suspect, listed in ADR-014. Running one
again measures nothing new.

**Contact telemetry**:
The flat `pad_contact_force` / `object_contact_force` keys the adapter
publishes in `info`. A cross-repo contract with `qc/evaluation.py`, read by
name and with no version pin. No success rule reads them.

### Demonstrations

**Session**:
One teleop recording run, written to `out/demos/success|failure/<task>/<stamp>/`
and kept, when it is worth keeping, as `dataset/demos/<task>/<stamp>/` plus the
sibling `<stamp>_failure/`. The `.npz` converted from one or more sessions
lands in `out/qc/`, and is kept in `dataset/qc/` once a run script points at
it.
