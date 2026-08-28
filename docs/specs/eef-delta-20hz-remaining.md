# Remaining work: `eef-delta-20hz`

Handoff for a follow-up agent. The implementation is complete through step 4 of
the spec and most of step 5; what is listed here is unfinished. Read
`docs/specs/eef-delta-20hz.md` and `docs/adr/016-*.md` first, then this file.

## State

**Done and verified.**

- `ManipulatorInterface` actuator split (`arm_actuators`, `gripper_actuators`,
  `ik_max_dq`), DK1 + YAM declarations, wiring seam.
- `tasks/common/actions.py`: `EndEffectorDeltaAction(Cfg)` with the `[-1, 1]`
  clamp, gripper-only `NormalizedJointPositionAction`, and the shared pure
  mappings `unclamped_delta_from_poses` / `delta_action_from_poses` /
  `pose_within_one_delta`.
- `tasks/common/env_cfg.py`: two action terms (7 wide), `decimation` 10
  (`step_dt` 0.05), `actions` observation removed (`place` 42 → 35).
- `place` episode budget 10 s → 15 s.
- Convention record: `qc/dataset.py` `STAMP_KEYS` + `check_dataset_matches_env`,
  `G3aEnv.action_convention()`, `convert.action_convention()`.
- Converter stores the recorded raw action through, stamps the task id, and
  rejects mixed-task sessions.
- Teleop: `live_bridge/mapper.py` rewritten (`leader_grasp_pose`,
  `leader_to_delta_action`, `leader_is_aligned`); gravity compensation deleted;
  recorder stores the raw action and builds under the training episode length.
- `tests/tasks/test_actions.py` rewritten (seam 1): **18 passed**.
- `tests/qc/test_env.py` updated: **20 passed**.
- `tests/trlc_dk1/`: **3 passed** (grasp physics unaffected by gravcomp).
- `uv run ruff check`, `ruff format`, `pyright` clean on `g3a_bench/`.

## Two deliberate deviations from ADR-016 — read before changing them

Both are measured, both are documented in code, and **both need the ADR
amended** (not yet done).

### 1. `max_dq` is derived from torque, not from the velocity limit

ADR-016 specifies `max_dq = velocity_limit × timestep` = 0.04 rad. That is
wrong for mjlab's term. `DifferentialIKAction.apply_actions` commands
`q_target = q_current + dq` afresh **every physics substep**, so `dq` never
integrates into a trajectory — it is the instantaneous PD position error, and
therefore a **torque cap** of `kp · max_dq`.

Measured: holding the arm at the start pose needs 0.0584 rad on joint2
(5.84 Nm / kp 100). At `max_dq = 0.04` the arm gets 4 Nm against a 28 Nm effort
limit and cannot stand up.

Now derived as `min(effort_limit / kp)` over the arm actuators = **0.28 rad** —
the error past which the motor is already saturated. Above ~0.28 the clamp is
inert. Lives in `DK1_IK_MAX_DQ`; see `ManipulatorInterface.ik_max_dq`.

### 2. Gravity compensation moved into the simulator, not deleted

ADR-016 deletes teleop gravity compensation because "DiffIK closes the loop on
the measured pose and corrects sag itself". **That premise is false for mjlab's
term**: it re-anchors its setpoint to the measured pose every control step, so
sag does not merely persist, it compounds.

Measured, 100 zero-action steps on `place`, grasp-site drift:

| bodies compensated | drift |
| --- | --- |
| none | 0.168 m (arm walks into the table) |
| arm links only | 0.100 m |
| links + fingers (**shipped**) | 0.000 m |

The fingers matter even though their own joints carry 0.0 N (racks slide along
±y, horizontal): their 0.05 kg each hangs off joint2/joint3. Grip force is
horizontal, so the 20 N cap is untouched — `tests/trlc_dk1/test_dk1_grasp.py`
passes. See `dk1_constants._compensate_arm_gravity`.

Accepted cost: the real DK1 has no gravity feedforward, so the sim arm holds
station better than hardware.

**Non-finding, for the record.** With both fixes the arm tracks a sustained
unit +x at 3.8 cm/step peak, ~2.1 cm/step sustained, against the demos' p99 of
2.7 cm/step. The earlier "the arm is too slow for a 5 cm box" reading was an
artifact of measuring with `max_dq = 0.04` and no gravcomp. No action needed.

## Outstanding

### A. Tests still referencing deleted APIs (blocking `pytest tests/`)

Each fails at import or on a stale assertion. None is conceptually hard.

1. **`tests/live_bridge/test_mapper.py`** — seam 3, the largest item. Still
   imports `action_scale_offset` and `lerobot_to_mjlab_raw_action`, both
   deleted. Rewrite `TestRawAction` against the new API. The spec names the
   cases: a leader pose at the current end-effector pose maps to a zero delta;
   a pose one box away maps to a unit delta; a pose beyond the box clamps; the
   gripper channel round-trips; `leader_is_aligned` is true at the reset pose
   and false one box beyond. Keep `TestGripperConversion` and
   `TestAbsoluteCtrl` as they are — `mjlab_gripper_meters` and
   `lerobot_to_mjlab_ctrl` both survive. Use `gripper_scale_offset()` (scalars)
   where the old tests used `action_scale_offset()` (7-vectors).

2. **`tests/qc/test_convert.py`** — seam 2. Needs:
   - `_columns` to emit an `actions` column of 7-wide deltas in `[-1, 1]`
     instead of `leader_actions` of leader poses, and `_build` to pass
     `task_id=`.
   - `_FakeTerm`/`_FakeEnv` replaced by a double exposing
     `action_convention()` returning the 7-field dict (see
     `g3a_bench/qc/dataset.py` `STAMP_KEYS`).
   - `TestStampCheck` extended: each field mismatching individually is rejected
     with a message naming it; a dataset carrying the old
     `action_scale`/`action_offset` stamp is rejected naming `convention`
     (`check_dataset_matches_env` raises "carries no 'convention'").
   - `TestReadingARecording._write` hardcodes `fps=50` → `20`, and must write a
     `task` matching what the test asserts is stamped.
   - Delete `TestActionConversion::test_actions_invert_back_to_the_recorded_pose`
     and `test_a_pose_at_a_joint_limit_stays_in_the_box` — both are about the
     deleted gravity compensation. Add one asserting `build_dataset` raises
     when a recorded action lies outside the box.

3. **`tests/live_bridge/test_recorder.py:309`** — asserts
   `extra_info["gravity_compensate"] is True`; the key and the `Recorder`
   kwarg are gone. Drop the assertion (the surrounding test also covers
   `start_pose_noise`, which survives).

4. **`tests/live_bridge/test_recording_mixin.py:97`** — passes
   `live_start_pose_rel=` to `_init_recording`, a parameter that no longer
   exists. Remove the argument. The alignment gate now calls
   `mapper.leader_is_aligned`, so a test that drives the reset phase needs its
   policy double to expose `latest_leader_action` as a 7-wide leader pose (and
   `latest_raw_action`, which the recording path now reads).

5. **`tests/live_bridge/test_debug_panel.py`** — `PLACE_NAMES` and one other
   list include the removed `"actions"` observation term. Drop it from both.

6. **`tests/qc/faults-visual/fault01_action_reach.py`** and `_visual.py` — read
   `data["action_scale"]`/`action_offset` and
   `action_manager.get_term("joint_pos")`. This directory reconstructs three
   *historical* faults for before/after pictures; it is not part of the pytest
   suite. Decide whether to port it or mark it as pinned to the pre-ADR-016
   convention in its README. Porting is probably not worth it — fault 01 is
   specifically about the action space this record replaces.

### B. Not started

7. **Full-suite run.** `uv run pytest tests/ -v` has not been run green
   end-to-end; only `tests/tasks/test_actions.py`, `tests/qc/test_env.py` and
   `tests/trlc_dk1/` have. Expect only the items above.

8. **Docs.** None of the documentation pass is done:
   - `CLAUDE.md` — the "One action unit is half a joint's travel" bullet must be
     rewritten for the delta convention (drop the "Being replaced" note), and
     the robot-agnostic section must mention the actuator split.
     `tasks/common/actions.NormalizedJointPositionAction` is now gripper-only.
   - `CONTEXT.md` — add a **Control step** term (0.05 s, 20 Hz); remove the
     action-unit language. Consider adding **Action box** and **Convention
     record**.
   - `docs/adr/016-*.md` — flip `status: proposed` to accepted, and amend the
     two claims contradicted above (the `max_dq` derivation, and "DiffIK
     corrects sag itself"). Keep the measurements; correct the mechanics.
   - `docs/adr/005-*.md` — flip to `status: superseded by ADR-016`.
   - `docs/adr/011-*.md` — its "converter repeats the record-time conversion"
     paragraph is now wrong; the converter copies the stored action through.
   - `g3a_bench/tasks/place/README.md` and `docs/TaskCategories.md` — check for
     50 Hz / 10 s / 42-wide observation references.
   - `tests/tasks/README.md` — `check_success_reward.py:40` still documents
     "decimation(4)".

9. **`tests/tasks/check_success_reward.py`** — its docstring hardcodes the old
   control dt; the step budget it uses may need rescaling for 20 Hz.

## Things worth not re-deriving

- `compute_pose_error` and `apply_delta_pose` are exact inverses, which is what
  makes the recorder and the action term unable to disagree. `delta_action_from_poses`
  is the *only* place a delta action is built; keep it that way.
- `axis_angle_from_quat` canonicalises `w >= 0`, so the quaternion sign
  instability `observations.py` warns about does not reach the delta.
- Position deltas are translation-invariant, so world and env-local frames give
  the same answer — but they must not be *mixed*. `leader_grasp_pose` returns
  base-frame; `EndEffectorDeltaAction.frame_pose` returns world-frame. The
  recorder subtracts `env_origins`.
- `lerobot` stores the task per frame as `task_index` into `meta.tasks`, a frame
  indexed *by the task string* — reverse-map it, do not index positionally.
- Open-loop replay of a delta sequence integrates its own error instead of
  converging back, unlike the old absolute joint targets. `record_replay.py` and
  `verify_replay.py` are now pass-throughs and their tracking numbers will drift
  along an episode. Use `state_replay.py` when the question is what the episode
  looked like.
