# Spec: 20 Hz control through a 5 cm end-effector delta

Implements ADR-016. Read that record first — it holds the measurements behind
every number here, and this spec does not repeat them.

## Problem Statement

QC-FQL scores ≈0 success on `place` across 29 runs while grasping the cube in
83–100 % of episodes. The benchmark's MDP sits outside the control envelope QC
was validated in on two axes at once: it runs at 50 Hz where both reference
domains run at 20 Hz, and it commands absolute joint positions where both
command a ±5 cm end-effector delta. Under absolute joint control a fixed action
error becomes a pose-dependent Cartesian error — 0.0766 in action units moves
the end-effector 0.0719 m, against a 0.02 m success tolerance — so the policy
must invert the arm's kinematics itself. A researcher cannot tell whether the
benchmark is hard or the MDP is misconfigured.

## Solution

Every task moves to 20 Hz and a relative end-effector delta action, matching
robomimic `can` and OGBench `cube-*`. The action stays 7 wide: a 6D delta
(±0.05 m, ±0.5 rad) solved by mjlab's differential IK on the robot's grasp
site, plus the gripper as an absolute normalized stroke. Teleoperation records
in the same space, the dataset stamps the whole convention, and loading a
dataset recorded under any other convention raises.

## User Stories

1. As a researcher, I want the control rate to be 20 Hz, so that an action
   chunk of 5 spans 0.25 s and sits in the middle of QC's validated band.
2. As a researcher, I want actions to be end-effector deltas, so that the
   policy commands task-space motion instead of solving inverse kinematics.
3. As a researcher, I want the delta box to be ±0.05 m, so that the action
   space matches both reference domains exactly.
4. As a researcher, I want the rotation box to be ±0.5 rad, so that no free
   variable is introduced that nobody has swept.
5. As a researcher, I want `place` episodes to last 15 s, so that 98 % of
   demonstrations keep their successful ending.
6. As a researcher, I want the previous action out of the observation, so that
   "repeat the last action" stops being a trivially strong regressor.
7. As a researcher, I want a dataset recorded under an older convention to
   raise at load, so that I never again spend a sweep training against the
   wrong MDP.
8. As a researcher, I want the dataset to stamp its task id, so that a
   `RandS-RandT` recording cannot silently train against a `FixedS-FixedT` env.
9. As a researcher, I want the control rate stamped, so that changing the rate
   invalidates datasets loudly rather than silently.
10. As an operator, I want teleoperation to drive the arm through the same
    delta action the policy emits, so that a demonstration is reproducible by
    a bounded policy.
11. As an operator, I want the recorded action to be exactly what the simulator
    stepped with, so that recording and evaluation describe one MDP.
12. As an operator, I want an episode to start only once the leader is aligned
    with the reset pose, so that the opening transition is not a clamped jump.
13. As an operator, I want the recorder to use the training episode length, so
    that an over-long episode is visibly truncated while I fly it.
14. As an operator, I want gravity compensation gone, so that the commanded
    target is where I actually am.
15. As a task author, I want the rate and action term defined once in the
    shared workbench, so that no category can drift into its own MDP.
16. As a task author, I want to add a robot by declaring its actuator split and
    grasp site, so that no task file learns a robot-specific name.
17. As a task author, I want a robot without a declared actuator split to fail
    at build time, so that a misconfiguration cannot reach a training run.
18. As a maintainer, I want the differential IK solver to respect the arm's
    real velocity limit, so that the commanded target stays physically
    trackable.
19. As a maintainer, I want the solver to steer away from joint limits without
    shrinking the teleoperable range, so that ADR-005's guarantee survives.
20. As a maintainer, I want a test that fails when the recording and evaluation
    mappings disagree, so that the seam cannot break silently.

## Implementation Decisions

**Action space.** The shared workbench declares two action terms in place of
one. The first is mjlab's `DifferentialIKActionCfg` in relative mode, frame
type `site`, frame name the robot's grasp site, orientation weight positive so
the term is 6D, `delta_pos_scale` 0.05, `delta_ori_scale` 0.5. The second is
the existing normalized joint-position term restricted to the gripper
actuators, keeping "-1 is the closed limit, +1 the open limit". Total width 7.
The raw action is clamped to `[-1, 1]` before either term applies its scale.

**Solver parameters.** `joint_limit_weight` positive; `posture_weight` zero (a
6-DOF arm under a 6D objective has no null space); `damping` 0.05; `max_dq`
derived at build time from the controlled actuators' velocity limits times the
physics timestep, not written as a literal. The DK1's
`soft_joint_pos_limit_factor` becomes 1.0 so differential IK's soft limits
equal the hard ones — no other path in this repo reads the soft limits.

**Robot interface.** `ManipulatorInterface` gains an arm-actuator tuple and a
gripper-actuator tuple. The wiring seam fills both action terms from them and
sets the IK frame from the existing grasp site. Deriving the arm as "every
actuator that is not the gripper" is rejected. A robot that declares neither
must fail when the term is built.

**Rate and budget.** `decimation` 10 with the physics timestep unchanged at
0.005. `episode_length_s` stays expressed in seconds; `place`'s value becomes
15.0. Other categories keep their existing formulas.

**Observation.** The `actions` term is removed from the shared observation
group. `place` goes 42 → 35 wide. No other term changes.

**Dataset convention record.** The stamp becomes a record carrying a
convention name, the position and orientation delta scales, the gripper scale
and offset, the control step duration, and the task id. `load_dataset` asserts
every field against the environment being built and raises naming the field
that differs. The record travels as metadata and is filtered out before the
training keys are assembled, as the current stamp is.

**Teleop mapping.** A new pure function maps a leader joint pose to the 7-wide
delta action, given the current end-effector pose: forward-kinematics the
leader pose to a target grasp-site pose, take the position difference and the
orientation difference as an axis-angle, divide each by its scale, clamp, and
append the gripper channel under the existing normalized mapping. Gravity
compensation is deleted from the record path. A second pure predicate answers
whether a leader pose is within one delta box of the reset pose; the recorder
gates episode start on it.

**Recorder.** Builds its environment under the training episode length rather
than the endless `play` length.

**Deletions.** The absolute joint-position action term, its configuration
class, the gravity-compensation helper, and the record-time call to it.

## Testing Decisions

A good test here asserts external behaviour at the highest available seam: what
the environment does when stepped, and what the loader accepts or rejects. It
never reaches into a term's private buffers to assert an intermediate value.
The suite's existing habit of naming a test after the failure it prevents
should be kept.

**Seam 1 — the built environment** (`tests/tasks/test_actions.py`, replacing
its scale/offset tests, reusing its module-scoped built-env fixture). Cover:
the action is 7 wide; a zero action holds the end-effector still; a unit
position action moves it by the delta scale within tolerance; an action outside
the box is clamped and the environment agrees; the control step is 0.05 s; a
`place` episode times out at its budget; the observation is 35 wide and carries
no action term; commanding a pose near a joint limit does not violate it.

**Seam 2 — the dataset convention** (`tests/qc/test_convert.py`, extending
`TestStamp` and `TestStampCheck`). Cover: every convention field is published;
a matching dataset loads; each field mismatching individually is rejected with
a message naming it; a dataset carrying the old scale/offset stamp is rejected;
the training keys are unchanged. The existing `_FakeEnv`/`_FakeTerm` doubles
need reshaping for two action terms.

**Seam 3 — the pure mappings** (`tests/live_bridge/test_mapper.py`, alongside
`TestRawAction`). Cover: a leader pose at the current end-effector pose maps to
a zero delta; a leader pose one box away maps to a unit delta; a pose beyond
the box clamps; the gripper channel round-trips; the alignment predicate is
true at the reset pose and false one box beyond it.

**The seam test that matters most.** One test must fail if the recording and
evaluation mappings ever disagree: drive the built environment with a delta
produced by the leader mapping and assert the resulting end-effector pose
reaches the leader's requested pose within the 0.02 m success tolerance. This
replaces `test_the_teleop_mapper_agrees_with_the_action_term`, whose scale and
offset comparison no longer has a subject.

Prior art for the built-env style is `TestCompiledTask`; for the stamp doubles,
`TestStampCheck`; for simulator-free mapping tests, `TestRawAction` and
`tests/tasks/test_success.py`.

## Out of Scope

Re-recording demonstrations (hardware, and a separate session). Any training
run — `train_qc.sh` is Linux/cluster-only and cannot run on the development
box. The `alpha` sweep and the sparse-reward A/B, which are trainer-side flags.
Loosening the four-way success conjunction. Adding object-relative-to-end-
effector observations. Bimanual support: no bimanual task is registered, so one
IK frame suffices. YAM's actuator split beyond what is needed to keep it
building.

## Further Notes

`horizon_length` needs no change: at 20 Hz an action chunk of 5 spans 0.25 s,
which is exactly both reference domains, so the rate change closes the
chunk-length gap on its own.

Every existing `.npz` becomes unloadable by design. That is the convention
record working, not a regression — the one dataset in `out/qc/` should raise at
load with a message naming the convention name as the differing field.

`CONTEXT.md` gains "Control step" and loses the action-unit language once this
lands; `CLAUDE.md`'s action-convention bullet is rewritten and ADR-016 flips
from `proposed` to accepted, with ADR-005 marked superseded.
