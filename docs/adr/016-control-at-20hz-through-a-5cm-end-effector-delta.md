# Control at 20 Hz through a 5 cm end-effector delta

status: accepted — supersedes ADR-005. Implemented on `eef-delta-control`.

> **Amended after implementation, in two places.** Both are marked
> **Amended** below and both are mechanical, not directional: the design
> stands, two claims about *how mjlab's differential IK works* were wrong.
> `max_dq` turns out to be a torque cap rather than a rate limit, and the term
> does not correct gravity sag — it re-anchors to it. Every measurement in
> this record still holds; the numbers 0.04 rad and "compensation goes" do
> not.

Every task drops from 50 Hz to 20 Hz (`decimation` 4 → 10, `timestep`
unchanged at 0.005) and replaces absolute normalized joint-position actions
with a relative end-effector delta: mjlab's `DifferentialIKActionCfg` in
relative mode on the robot's grasp site, ±0.05 m and ±0.5 rad per control
step, plus the gripper as an absolute normalized stroke. The action stays 7
wide. This is the control envelope both QC-validated reference domains use,
and we were the outlier on both axes at once.

## Why

QC-FQL learns nothing on `place` (success ≈ 0 across 29 runs) while grasping
in 83–100 % of episodes once `alpha` is raised. The two reference domains the
QC paper validates on agree exactly with each other and disagree with us:

| | LOOT `place` (old) | robomimic `can` | OGBench `cube-single` |
|---|---|---|---|
| action | 7-d absolute joint pos, `[-1,1]` = full travel | 7-d `OSC_POSE` delta, `output_max=[.05,.05,.05,.5,.5,.5]` | 5-d DiffIK delta, `action_range=[.05,.05,.05,.3,1.0]` |
| rate | 50 Hz | 20 Hz | 20 Hz |
| `horizon_length=5` spans | 0.10 s | 0.25 s | 0.25 s |
| γ-horizon vs. episode | 100 steps = 20 % | 33 % | 50 % |

Two independent groups, two codebases, the same ±5 cm at 20 Hz. The chunk
length is the sharper of the two: QC's own Fig. 6 sweeps `h ∈ {1,5,10,25,50}`
at 20 Hz, where `h=1` (0.05 s) barely learns and `h=5–10` (0.25–0.5 s) is
best. Our `h=5` spanned 0.10 s — nearer the failing end than the working one.

The action space compounded it. Under absolute joint control a fixed action
error is a pose-dependent Cartesian error: measured in sim, an action error of
0.0766 (the BC gate's passing bar at the time, withdrawn since — ADR-012)
moves the end-effector 0.0719 m on the
worst channel, against a 0.02 m success tolerance. The policy had to invert
the arm's kinematics itself, which neither reference asks of it.

## The 5 cm box fits the demos, and 20 Hz is near its floor

Per-step end-effector displacement across the 99 recorded `place` demos, read
from the stored `ee_pose` observation. 20 Hz is bracketed by strides 2 and 3:

| rate | pos p90 | pos p99 | rot p99 | steps > 5 cm |
|---|---|---|---|---|
| 50 Hz (old) | 0.7 cm | 1.4 cm | 0.066 rad | 0.19 % |
| 25 Hz | 1.4 cm | 2.7 cm | 0.129 rad | 0.19 % |
| 16.7 Hz | 2.0 cm | 4.0 cm | 0.191 rad | 0.28 % |
| 10 Hz | 3.3 cm | 6.3 cm | 0.301 rad | 2.96 % |

At 20 Hz the demos sit at roughly half to three-quarters of the box at p99, so
±5 cm is fitted rather than lucky. The 10 Hz row is why the box also sets a
*floor* on the control rate: slow the loop further and human demonstrations
stop fitting inside it.

Rotation uses only ~26–38 % of a ±0.5 rad box. A fitted ±0.25 would balance
the two channels, and is rejected: robomimic's 0.5 is the value inside the
validated envelope, and the point of this record is to stop introducing free
variables nobody has swept.

## The tail is an artifact, not motion

Every step above 5 cm is the frame 0→1 transition — 66 of 99 episodes, exactly
one each — where the sim arm snaps from the reset pose toward the already-parked
leader. Under delta control that action clamps, so the opening transition would
be unreproducible by construction. The recorder must gate episode start on the
leader being within one delta box of the reset pose, rather than clamping the
action or dropping the row: only that keeps "the stored action is what
`env.step()` took" true for every row (ADR-011).

## Consequences

- **Every recorded `.npz` is invalid.** Both axes changed. `place` is
  re-recorded; the other categories have no demos to lose.
- **The stamp must grow.** `check_dataset_matches_env` reads `term.scale` /
  `term.offset`, which `DifferentialIKAction` does not have, and the control
  rate has never been stamped at all. It becomes a convention record — name,
  delta scales, gripper mapping, `step_dt`, **and the task id** — asserted at
  load. Without this, changing the rate reproduces the silent-mismatch failure
  ADR-011 exists for. The task id closes a second, already-realised hole: every
  `place` variant shares observation and action widths, so a `RandS-RandT`
  dataset trained silently against a `FixedS-FixedT` env for all fourteen runs
  of the α ladder. It is a different bug from this record's, but the schema is
  only open once.
- **`place`'s episode budget goes 10 s → 15 s.** `episode_length_s` stays in
  seconds; the *number* changes because the old 10 s truncated 20 of 99 demos,
  severing each one's successful ending. 15 s keeps 98 % of episodes whole and
  is 300 steps at 20 Hz — exactly robomimic `can`, whose γ-horizon fraction
  (100/300 = 33 %) it also matches. See the durations table below. Other
  categories keep their formulas; each gets this review when it is first
  recorded.
- **The `last_action` observation is dropped**, taking `place` from 42 to 35
  wide. It named a term that no longer exists, and it made "repeat the last
  action" a trivially strong regressor: 0.0242 held-out against the BC gate bar
  of 0.0766 then in force (withdrawn since — ADR-012).
- **Teleop records deltas.** The leader's FK pose minus the sim's current
  end-effector pose, over the scale. Both terms exist at record time, so the
  stored action is still exactly what the simulator stepped with.
- **Gravity compensation moves into the simulator.** *(Amended — this bullet
  originally read "goes", on the grounds that DiffIK closes the loop on the
  measured pose and corrects sag itself.)* The teleop-side compensation is
  indeed deleted: it inverted an open-loop PD sag, and there is no longer an
  open-loop joint target to correct. But the premise about DiffIK is false for
  mjlab's term. `apply_actions` commands `q_target = q_current + dq`, re-read
  from the *measured* position every substep, so there is no fixed setpoint to
  converge to — whatever sag the PD actuators carry at the end of a control
  step becomes the baseline the next one measures from. The sag does not
  persist, it compounds. Measured on `place`, grasp-site drift over 100
  zero-action steps:

  | bodies compensated | drift |
  |---|---|
  | none | 0.168 m — the arm walks into the table |
  | arm links only | 0.100 m |
  | links + fingers (shipped) | 0.000 m |

  Per-body `gravcomp = 1.0` on everything the arm actuators carry
  (`dk1_constants.GRAVITY_COMPENSATED_BODIES`). The fingers belong there even
  though their own joints see 0.0 N — the racks slide horizontally — because
  their 0.05 kg each still hangs off joint2 and joint3. Grip force is
  horizontal, so the 20 N cap is untouched. Accepted cost: the real DK1 has no
  gravity feedforward, so the sim arm holds station better than hardware; the
  alternative is an MDP in which standing still is impossible. Both reference
  domains arrive at the same place by another route — robomimic's OSC carries
  a gravity feedforward in its torque law.
- **All categories move together.** The rate and the action term live in
  `tasks/common/env_cfg.py`. A per-category fork would put two MDP conventions
  in one workbench.
- The old term is deleted rather than kept selectable. Git holds the old
  behaviour; a live switch would mean two conventions to stamp and test.

## Mechanics

- **The robot declares its own actuator split.** `ManipulatorInterface` gains
  `arm_actuators` and `gripper_actuators`; DiffIK takes the first plus
  `frame_name=grasp_site`, the normalized position term takes the second.
  Deriving the arm as "every actuator that is not the gripper" was rejected:
  it makes the residue load-bearing, so a later robot's torso or wrist-camera
  joint silently joins the IK chain.
- **`joint_limit_weight > 0`, and the DK1's `soft_joint_pos_limit_factor` goes
  0.9 → 1.0.** DiffIK only knows the soft limits, and ADR-005 chose the hard
  ones so the outer slice of each joint stayed teleoperable. Setting the factor
  to 1.0 makes the two identical and keeps both properties. The change is
  contained: no other path in this repo reads the soft limits — the DK1's reset
  is `tasks/common/events.reset_joints_to_pose`, which clamps to the hard limits
  deliberately, and mjlab's own soft-limit consumers (`reset_joints_by_offset`,
  the `joint_pos_limits` reward, the tracking commands) are unused here.
- **`max_dq = 0.28 rad`, derived from torque.** *(Amended — this originally
  read `0.04 rad`, derived from the DM4340's 8 rad/s velocity limit over one
  0.005 s physics step, on the reading that `max_dq` is a rate limit.)* It is
  not a rate limit. `apply_actions` commands `q_target = q_current + dq`
  afresh every physics substep, so `dq` never integrates into a trajectory: it
  *is* the instantaneous PD position error, and clamping it clamps torque at
  `kp · max_dq`. Measured: holding the arm at the start pose needs 0.0584 rad
  on joint2 (5.84 Nm over kp = 100). At 0.04 the joint gets 4 Nm against a
  28 Nm effort limit and the arm cannot stand up. It is now
  `min(effort_limit / kp)` over the arm actuators = 0.28 rad, the error past
  which the motor is already saturated and the clamp inert
  (`DK1_IK_MAX_DQ`, surfaced as `ManipulatorInterface.ik_max_dq`). mjlab's
  default of 0.5 is above that for this arm, so it would be inert too — the
  value matters for a robot whose motors saturate later, and deriving it means
  no robot has to be measured by hand.
- **`damping` stays at 0.05; `posture_weight` is 0.** A 6-DOF arm under a 6D
  task objective is exactly determined, so there is no null space for a posture
  term to act in. Singularity behaviour is `damping`'s job alone.

## What replaces the old tests

`tests/tasks/test_actions.py` is built entirely on `term.scale`/`term.offset`
and does not survive. Its one irreplaceable test is
`test_the_teleop_mapper_agrees_with_the_action_term` — the recording/evaluation
seam, the place where a demo and an evaluation silently stop describing the
same task. It is replaced by
`TestCompiledTask::test_the_teleop_mapper_reaches_what_the_leader_asked_for`:
leader FK → delta → DiffIK, driven on a built `place` env, asserted to land
inside the 0.02 m success tolerance. Without it, a broken mapping is only
discoverable by spending a cluster run.

The second replacement this record proposed — replaying a whole recorded
end-effector trajectory and bounding its tracking error — was **not built**,
and should not be. An open-loop replay of a *delta* sequence integrates its
own error rather than converging back, so its tracking number is a property of
the replay, not of the mapping. `record_replay.py` and `verify_replay.py` are
pass-throughs for the same reason; `state_replay.py` is the tool for "what did
this episode look like".

The rest of the suite: `tests/live_bridge/test_mapper.py` covers the teleop
side alone (FK, frame, clamp, gripper channel, and the alignment gate),
`tests/qc/test_convert.py` the convention record, and
`tests/live_bridge/test_recording_mixin.py::TestResetPhaseAlignmentGate` the
gate's effect on the recorder's state machine.

## What becomes of ADR-005's teleop guardrails

Gravity compensation and the `[-1, 1]` clamp were ADR-005's two teleop
guardrails, and both change meaning here. Compensation leaves the *teleop*
path: there is no longer an open-loop joint target for it to lead, and the
out-of-box case it caused near a joint limit (measured 1.014 on joint5) goes
with it. It reappears in the simulator instead, for every caller rather than
only the operator's trajectory — see the amended consequence above. The clamp
stays, and is now the convention itself rather than a guardrail: mjlab's
`DifferentialIKAction` scales whatever it is handed, so without the clamp
`delta_pos_scale` would be a unit rather than a bound and an action of 3.0
would honestly ask for 15 cm.

## The episode budget, and the recorder that ignored it

`live_record.py` builds its env with `play=True`, which sets
`episode_length_s = int(1e9)` — so the recorder has never had a timeout while
training and eval always capped. That is why the demo durations below run past
the budget at all. The recorder is changed to record under the *training*
episode length, so an over-long episode is visibly truncated as it is flown
rather than silently at conversion.

Durations of the 99 recorded `place` episodes, and what each candidate budget
would keep:

| budget | = steps @ 20 Hz | episodes surviving whole | frames inside |
|---|---|---|---|
| 10 s (old) | 200 | 79.8 % | 93.3 % |
| 12 s | 240 | 89.9 % | 97.4 % |
| **15 s** | **300** | **98.0 %** | **99.6 %** |
| 18 s | 360 | 100 % | 100 % |

Median episode 5.9 s, p90 11.8 s, max 17.5 s.

Optimise the episode column, not the frame column: truncation does not merely
drop frames, it severs a demo's successful ending and leaves a trajectory that
never scores. (An earlier analysis reported 35.5 % of frames as lost at the
10 s cap. That figure counts every frame of the 20 overrunning episodes; only
6.7 % sit past the cap. The episode-level number is the one that matters.)

18 s would catch the last two episodes, and is rejected for leaving the
reference envelope to buy 100 % over 98 %. Note the durations are 50 Hz
joint-teleop paces — if delta control slows the operator, 15 s has headroom
and 12 s does not.

## First recording

`FixedS-FixedT`, ~100 demos. One spawn, one target: the smallest set that can
answer whether the new control envelope moved anything, and it turns the α
ladder's fixed-eval-on-random-data result into a clean fixed-on-fixed one.
Random spawn and target are a difficulty axis to add once the envelope is known
to work — `can` fixes its target too.

`horizon_length` needs no change. At 20 Hz `h=5` spans 0.25 s, the middle of
QC's working band and exactly both references, so the rate change closes the
chunk-length gap as well.
