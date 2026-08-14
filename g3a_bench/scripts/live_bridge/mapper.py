import functools

import mujoco
import numpy as np

from g3a_bench.robots.trlc_dk1.dk1_actuators import ARM_JOINTS, GRIPPER_COMMAND_RANGE
from g3a_bench.robots.trlc_dk1.dk1_constants import get_unimanual_dk1_robot_cfg
from g3a_bench.tasks.common.actions import normalized_scale_offset

_ARM_JOINT_NAMES = list(ARM_JOINTS.keys())

ACTUATED_JOINT_NAMES: tuple[str, ...] = tuple(_ARM_JOINT_NAMES) + ("gripper_left",)
"""The joints the action term drives, in the order it expects them.

The gripper is a coupled pair, so only ``gripper_left`` is actuated;
``gripper_right`` follows it through an equality constraint.
"""

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


def action_scale_offset() -> tuple[np.ndarray, np.ndarray]:
    """The ``(scale, offset)`` the joint action term applies.

    The same numbers ``tasks.common.actions.NormalizedJointPositionAction``
    derives on the task side, from the same model and the same helper. That is
    what lets a teleop action and a policy action mean the same thing.
    """
    lower, upper = joint_pos_limits()
    return normalized_scale_offset(lower, upper)


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


_GRAVITY_STIFFNESS = np.array(
    [ARM_JOINTS[n].kp for n in _ARM_JOINT_NAMES], dtype=np.float32
)


@functools.cache
def _dk1_scratch() -> tuple[mujoco.MjModel, mujoco.MjData, np.ndarray]:
    """The compiled DK1, built once and shared.

    Standalone and non-rendering. It answers two questions that both need the
    model rather than the MJCF text: what each joint's hard limit is, and what
    gravity torque holds a candidate pose. Built on first use, so importing
    this module stays cheap for callers that need neither.
    """
    from mjlab.entity import Entity

    model = Entity(get_unimanual_dk1_robot_cfg()).compile()
    data = mujoco.MjData(model)
    dofadr = np.array(
        [
            model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in _ARM_JOINT_NAMES
        ]
    )
    return model, data, dofadr


def gravity_compensate_ctrl(ctrl: np.ndarray) -> np.ndarray:
    """Push the commanded arm target past gravity sag so the PD actuator
    actually settles at ``ctrl`` instead of short of it.

    The actuators are soft PD position controllers (~2 Hz, tuned for RL
    residual steps) with no gravity feedforward, so a large one-shot teleop
    target — e.g. lifting the arm well above home — sags under gravity and
    never reaches the commanded angle. Standard fix: command
    ``target + g(target) / kp`` instead of ``target``, where ``g(target)`` is
    the gravity torque needed to hold that pose statically (kp is the
    actuator's own PD stiffness, so this exactly cancels the steady-state
    error). Gripper (index 6) passes through unchanged.
    """
    model, data, dofadr = _dk1_scratch()
    data.qvel[:] = 0.0
    for name, value in zip(_ARM_JOINT_NAMES, ctrl[:6], strict=True):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(model, data)
    gravity_torque = data.qfrc_bias[dofadr]

    compensated = ctrl.copy()
    compensated[:6] = ctrl[:6] + gravity_torque / _GRAVITY_STIFFNESS
    return compensated


def lerobot_to_mjlab_raw_action(
    joint_pos: np.ndarray,
    gravity_compensate: bool = False,
) -> np.ndarray:
    """Convert a leader-arm pose to the action that commands it in the sim.

    "Raw" is mjlab's word for the action before the term applies its scale and
    offset -- that is, exactly what ``env.step()`` takes. Under the shared
    convention (``tasks.common.actions``) the term computes
    ``target = action * scale + offset``, so this inverts it. The result is
    inside ``[-1, 1]`` for any pose the joints can hold.

    Args:
        gravity_compensate: If True, push the target past gravity sag (see
            gravity_compensate_ctrl) before computing the action. Opt-in and
            off by default — record_replay.py and tests rely on the exact
            uncompensated mapping.
    """
    ctrl = lerobot_to_mjlab_ctrl(joint_pos)
    if gravity_compensate:
        ctrl = gravity_compensate_ctrl(ctrl)
    scale, offset = action_scale_offset()
    return (ctrl - offset) / scale
