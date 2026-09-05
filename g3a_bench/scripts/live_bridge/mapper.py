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

# Leader joint1 zero differs from MJCF joint1=0; no correction applied.
# leader_action is sent to the follower and recorded unchanged.


def joint_pos_limits() -> tuple[np.ndarray, np.ndarray]:
    """Hard (lower, upper) ctrl limits read from the compiled model."""
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
    """(scale, offset) matching the task-side normalized action term."""
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
    """Compiled DK1 built once for limits and gravity torque."""
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
    """Add gravity feedforward (g/kp) so soft PD settles at ctrl."""
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
    scale: np.ndarray | None = None,
    offset: np.ndarray | None = None,
) -> np.ndarray:
    """Leader pose to sim raw action (inverts term scale and offset).

    Args:
        gravity_compensate: Apply gravity feedforward before inverting.
        scale: Action scale; defaults to compiled DK1 values.
        offset: Action offset; required together with scale.
    """
    ctrl = lerobot_to_mjlab_ctrl(joint_pos)
    if gravity_compensate:
        ctrl = gravity_compensate_ctrl(ctrl)
    if (scale is None) != (offset is None):
        raise ValueError("scale and offset must be provided together or not at all")
    if scale is None or offset is None:
        scale, offset = action_scale_offset()
    return (ctrl - offset) / scale
