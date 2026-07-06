import numpy as np

from g3a_bench.robots.trlc_dk1.dk1_actuators import DK1_UNIMANUAL_ACTION_SCALE
from g3a_bench.robots.trlc_dk1.dk1_constants import _HOME_JOINT_POS

_GRIPPER_OPEN = 0.0
_GRIPPER_CLOSED = -0.045

_ACTUATOR_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
]

HOME_JOINT_POS = np.array(
    [_HOME_JOINT_POS[n] for n in _ACTUATOR_NAMES] + [_HOME_JOINT_POS["gripper_left"]],
    dtype=np.float32,
)
ACTION_SCALE = np.array(
    [DK1_UNIMANUAL_ACTION_SCALE[n] for n in _ACTUATOR_NAMES]
    + [DK1_UNIMANUAL_ACTION_SCALE["gripper_left"]],
    dtype=np.float32,
)


def mjlab_gripper_meters(norm_val: float) -> float:
    """Map real-robot gripper [0,1] → prismatic meters [0.0, -0.045]."""
    return float(np.interp(norm_val, [0.0, 1.0], [_GRIPPER_OPEN, _GRIPPER_CLOSED]))


def lerobot_to_mjlab_ctrl(joint_pos: np.ndarray) -> np.ndarray:
    """Convert real-robot joint_pos (7,) → MJLab data.ctrl (7,).

    Joints 0-5 (radians) pass through unchanged.
    Gripper (index 6) goes from normalised [0,1] to prismatic meters.
    """
    ctrl = np.empty(7, dtype=np.float32)
    ctrl[:6] = joint_pos[:6]
    ctrl[6] = mjlab_gripper_meters(float(joint_pos[6]))
    return ctrl


def lerobot_to_mjlab_raw_action(joint_pos: np.ndarray) -> np.ndarray:
    """Convert to raw action for JointPositionActionCfg(use_default_offset=True).

    MJLab computes:  target = HOME_JOINT_POS + raw_action * ACTION_SCALE
    where the result recovers the original joint positions from lerobot_to_mjlab_ctrl.
    """
    ctrl = lerobot_to_mjlab_ctrl(joint_pos)
    return (ctrl - HOME_JOINT_POS) / ACTION_SCALE
