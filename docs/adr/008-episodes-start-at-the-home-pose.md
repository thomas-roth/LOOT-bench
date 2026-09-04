# Episodes start at the robot's home pose, with a settable sample around it

Each episode resets the arm to an absolute per-joint pose
(`ManipulatorInterface.reset_joint_pos`, the DK1's `START_JOINT_POS`, which is
the home pose) plus a uniform sample. `make_manipulation_env_cfg` exposes the
half-width as `start_pose_noise`, and its default is `0.0`.

## Why an absolute pose

mjlab's `reset_joints_by_offset` resets relative to the entity's default pose,
which for the DK1 is the MJCF keyframe. The recorder makes the operator align
the leader arm to a ghost pose before each recording, so every demonstration
began at that pose while every evaluated episode began at the keyframe,
1.7 rad away on joint2 and joint3. A policy was trained on demonstrations and
then evaluated from a state it had never seen. `tasks/common/events.reset_joints_to_pose`
closes that gap: the pose is a robot fact, the sample range is a task fact.

## Why the home pose

The first version of this change reset to the rest pose, which is where the
leader arm hangs when the operator lets go. Two costs retired it: joints 2 and
3 rest at 2.7 % and 1.8 % of their travel, so a policy had to saturate its
output only to hold still (see ADR-005), and the gripper pointed forward
instead of down at the table. The operator now aligns the leader arm to the
home pose. The recorder already required an alignment step, so this changes
which pose it asks for, not whether it asks.

## Consequences

- Sampled positions are clamped to the hard joint limits, not the soft ones. A
  soft clamp can move the start pose without saying so, and the recording and
  the evaluation must begin at the same place.
- `START_JOINT_POS` names the arm joints only, so the gripper is never
  sampled. Its whole stroke is 0.046 m, and any joint-space noise would start
  episodes half closed.
- `start_pose_noise = 0.0` gives a bit-identical start, which every demo replay
  check needs. It also offers a policy exactly one start state, which is the
  data-shape problem ADR-013 attacks. The constant start is inherited from
  `09e5dfa`, not introduced by the QC work; variety is new work, not a revert.
- `place` carries the seam for that work but is back at `0.0`
  (`place_env_cfg.START_POSE_NOISE`): it ran at `0.05` for the first demo
  sessions and now starts every episode at the home pose alone. The value
  lives on the task because the demo recorder and the QC run must use one
  number and neither passes an override. `live_record.py --start-pose-noise`
  overrides it per session and writes the value it used into the dataset
  sidecar, so a dataset says which start distribution it came from. Keep any
  value well under the recorder's 0.2 rad alignment gate, which is measured
  against the nominal pose.

## Measuring start-state variety

Any test of start-state variety must read peak-to-peak range, never standard
deviation. The observations are float32, and `np.std` over bit-identical
samples returns about 1e-6, so a non-zero standard deviation does not prove
variation and an implementation that randomizes nothing passes. A test must
also assert that a scene term varies, which proves the measurement would have
seen robot variation, and must read with observation corruption off, which no
longer needs arranging, because no task declares observation noise.
