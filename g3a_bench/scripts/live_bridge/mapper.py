"""The teleop half of the shared action convention.

A leader-arm pose is joint angles; an action is an end-effector delta
(``tasks/common/actions.py``). :func:`leader_to_delta_action` bridges them:
forward-kinematics the leader pose to the grasp-site pose it asks for, then
ask :func:`~g3a_bench.tasks.common.actions.delta_action_from_poses` for the
action that reaches it from where the simulated arm currently is.

The mapping and the action term cannot disagree, because neither owns the
arithmetic: both call into ``tasks/common/actions.py`` (the term applies
mjlab's ``apply_delta_pose``, this module inverts it through
``compute_pose_error``), and the two scales come from one pair of constants.

Gravity compensation is gone from here: an action now names a pose delta, so
there is no open-loop joint target left for the sag to push past (it used to
cause an out-of-box case near a joint limit, measured 1.014 on joint5). It did
not stop being necessary -- differential IK re-anchors to the measured
position every substep rather than integrating a target, so the sag compounds
instead (0.168 m of drift over 100 zero actions). The correction lives in the
simulator now, as per-body ``gravcomp``, once for every caller. See
``robots/trlc_dk1/dk1_constants._compensate_arm_gravity`` and ADR-016's
amended consequence.
"""

import functools

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_from_matrix

from g3a_bench.robots.trlc_dk1.dk1_actuators import ARM_JOINTS, GRIPPER_COMMAND_RANGE
from g3a_bench.robots.trlc_dk1.dk1_constants import (
    DK1_GRASP_SITE,
    START_JOINT_POS,
    get_unimanual_dk1_robot_cfg,
)
from g3a_bench.tasks.common.actions import (
    delta_action_from_poses,
    normalized_scale_offset,
    pose_within_one_delta,
)

_ARM_JOINT_NAMES = list(ARM_JOINTS.keys())

ACTUATED_JOINT_NAMES: tuple[str, ...] = tuple(_ARM_JOINT_NAMES) + ("gripper_left",)
"""The joints a leader pose names, in the order it arrives in.

The gripper is a coupled pair, so only ``gripper_left`` is actuated;
``gripper_right`` follows it through an equality constraint. Note this is no
longer the shape of an action: an action is six delta channels plus the
gripper, not seven joints.
"""

GRIPPER_ACTION_INDEX = 6
"""Where the gripper sits in the 7-wide action, after the 6-D delta."""

# Note, carried from an earlier revision: this leader's joint1 was
# hand-calibrated (DM_CAN.set_zero_position — see
# trlc-dk1/examples/calibration_follower.py) to a "zero" that doesn't match the
# MJCF asset's own joint1=0 reference (arm reaching forward, see CLAUDE.md's
# home-pose note). Any correction for it would be sim-visualization-only —
# never apply one to the real leader_action, which is sent to the real follower
# (whose calibration already agrees with the leader) and recorded to disk.
# No such correction is applied today.


def joint_pos_limits() -> tuple[np.ndarray, np.ndarray]:
    """The hard ``(lower, upper)`` ctrl limit of each actuated joint.

    Read from the compiled model rather than copied out of ``dk1.xml``, so
    editing the MJCF cannot leave a stale number behind here. Radians for the
    arm joints, metres for the gripper.
    """
    model, _, _ = _dk1_scratch()
    limits = np.array(
        [
            model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in ACTUATED_JOINT_NAMES
        ],
        dtype=np.float32,
    )
    return limits[:, 0], limits[:, 1]


def gripper_scale_offset() -> tuple[float, float]:
    """The ``(scale, offset)`` the gripper action term applies.

    The same numbers ``tasks.common.actions.NormalizedJointPositionAction``
    derives on the task side, from the same model and the same helper -- which
    is what lets a teleop gripper command and a policy's mean the same thing.
    Scalars: one gripper joint is actuated.
    """
    lower, upper = joint_pos_limits()
    scale, offset = normalized_scale_offset(lower, upper)
    return float(scale[GRIPPER_ACTION_INDEX]), float(offset[GRIPPER_ACTION_INDEX])


def mjlab_gripper_meters(norm_val: float) -> float:
    """Map real-robot gripper [0,1] → prismatic meters [0.0, ≈-0.0411].

    Inverts the controller's normalization through the pinion radius.
    """
    return float(
        np.interp(
            norm_val,
            [0.0, 1.0],
            [GRIPPER_COMMAND_RANGE["open"], GRIPPER_COMMAND_RANGE["closed"]],
        )
    )


def lerobot_to_mjlab_ctrl(joint_pos: np.ndarray) -> np.ndarray:
    """Convert real-robot joint_pos (7,) → MJLab data.ctrl (7,).

    Joints 0-5 (radians) pass through unchanged.
    Gripper (index 6) goes from normalised [0,1] to prismatic meters
    (see mjlab_gripper_meters).
    """
    ctrl = np.empty(7, dtype=np.float32)
    ctrl[:6] = joint_pos[:6]
    ctrl[6] = mjlab_gripper_meters(float(joint_pos[6]))
    return ctrl


@functools.cache
def _dk1_scratch() -> tuple[mujoco.MjModel, mujoco.MjData, int]:
    """The compiled DK1, built once and shared.

    Standalone and non-rendering. It answers two questions that both need the
    model rather than the MJCF text: what each joint's hard limit is, and
    where the grasp site ends up for a candidate joint pose. Built on first
    use, so importing this module stays cheap for callers that need neither.
    """
    from mjlab.entity import Entity

    model = Entity(get_unimanual_dk1_robot_cfg()).compile()
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, DK1_GRASP_SITE)
    assert site_id >= 0, f"no site named {DK1_GRASP_SITE!r} in the compiled DK1"
    return model, data, site_id


def leader_grasp_pose(joint_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The grasp-site pose (position, ``wxyz`` quaternion) a leader pose asks for.

    Forward kinematics on a scratch DK1 whose base sits at the origin, so the
    result is in the **robot base frame** -- the frame ``ee_pose_w`` publishes
    and the frame :func:`leader_to_delta_action` expects its ``current`` pose
    in. The orientation goes through mjlab's ``quat_from_matrix``, the same
    conversion differential IK applies to the same site.

    Args:
        joint_pos: (7,) leader pose -- radians for the arm, normalized
            ``[0, 1]`` for the gripper, as the hardware reports it.
    """
    model, data, site_id = _dk1_scratch()
    ctrl = lerobot_to_mjlab_ctrl(joint_pos)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    for name, value in zip(ACTUATED_JOINT_NAMES, ctrl, strict=True):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(model, data)

    pos = np.array(data.site_xpos[site_id], dtype=np.float32)
    xmat = torch.as_tensor(
        np.array(data.site_xmat[site_id], dtype=np.float32).reshape(1, 3, 3)
    )
    quat = quat_from_matrix(xmat)[0].numpy().astype(np.float32)
    return pos, quat


def leader_to_delta_action(
    joint_pos: np.ndarray, current_pos: np.ndarray, current_quat: np.ndarray
) -> np.ndarray:
    """The 7-wide action that drives the sim arm toward a leader pose.

    Six delta channels for the arm, then the gripper as an absolute normalized
    stroke. The result is inside ``[-1, 1]`` by construction, so it is an
    action a bounded policy could have emitted.

    Args:
        joint_pos: (7,) leader pose, as the hardware reports it.
        current_pos: (3,) the simulated grasp site's current position, **in
            the robot base frame** — subtract ``env_origins`` from the sim's
            world-frame site position. Mixing frames is wrong by one env
            origin, and silently so.
        current_quat: (4,) the simulated grasp site's current orientation,
            ``wxyz``.

    Returns:
        (7,) float32 raw action, exactly what ``env.step()`` takes.
    """
    target_pos, target_quat = leader_grasp_pose(joint_pos)
    delta = delta_action_from_poses(
        torch.as_tensor(np.asarray(current_pos, dtype=np.float32)).reshape(1, 3),
        torch.as_tensor(np.asarray(current_quat, dtype=np.float32)).reshape(1, 4),
        torch.as_tensor(target_pos).reshape(1, 3),
        torch.as_tensor(target_quat).reshape(1, 4),
    )

    action = np.empty(7, dtype=np.float32)
    action[:6] = delta[0].numpy()
    scale, offset = gripper_scale_offset()
    gripper_m = mjlab_gripper_meters(float(joint_pos[GRIPPER_ACTION_INDEX]))
    action[GRIPPER_ACTION_INDEX] = np.clip((gripper_m - offset) / scale, -1.0, 1.0)
    return action


@functools.cache
def reset_grasp_pose() -> tuple[np.ndarray, np.ndarray]:
    """The grasp-site pose every episode starts from, in the base frame.

    ``START_JOINT_POS`` seen through the same forward kinematics the leader is
    measured with, so :func:`leader_is_aligned` compares like with like.
    """
    pose = np.zeros(7, dtype=np.float32)
    pose[:6] = [START_JOINT_POS[name] for name in _ARM_JOINT_NAMES]
    return leader_grasp_pose(pose)


def leader_is_aligned(joint_pos: np.ndarray) -> bool:
    """Whether an episode may start with the leader held at this pose.

    True when one unclamped action carries the arm from the reset pose to
    where the leader is asking. ADR-016 measured the alternative: 66 of 99
    recorded demos opened with the arm snapping toward an already-parked
    leader, a jump no bounded policy can reproduce. Gating here rather than
    clamping the opening action keeps every stored action equal to the one
    ``env.step()`` took.

    The gate reads the *leader*, never the sim arm — the env reset yanks the
    sim arm to the start pose, which would satisfy a sim-based gate before the
    operator has aligned anything.
    """
    target_pos, target_quat = leader_grasp_pose(joint_pos)
    reset_pos, reset_quat = reset_grasp_pose()
    return bool(
        pose_within_one_delta(
            torch.as_tensor(reset_pos).reshape(1, 3),
            torch.as_tensor(reset_quat).reshape(1, 4),
            torch.as_tensor(target_pos).reshape(1, 3),
            torch.as_tensor(target_quat).reshape(1, 4),
        )[0].item()
    )
