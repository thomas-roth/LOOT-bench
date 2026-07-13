import mujoco
import numpy as np

from g3a_bench.robots.trlc_dk1.dk1_actuators import (
    DK1_UNIMANUAL_ACTION_SCALE,
    get_arm_stiffness,
)
from g3a_bench.robots.trlc_dk1.dk1_constants import (
    _HOME_JOINT_POS,
    get_unimanual_dk1_robot_cfg,
)

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


# This leader's joint1 was hand-calibrated (DM_CAN.set_zero_position — see
# trlc-dk1/examples/calibration_follower.py) to a "zero" that doesn't match
# the MJCF asset's own joint1=0 reference (arm reaching forward, see
# CLAUDE.md's home-pose note). Sim-visualization-only correction — never
# apply this to the real leader_action, which is sent to the real follower
# (whose calibration already agrees with the leader) and recorded to disk.
# Re-tune (or remove) if the leader is re-calibrated.

# Physical joint limits from dk1.xml, in ctrl space (radians for joints, meters for gripper).
# Order: joint1..joint6, gripper_left — matches HOME_JOINT_POS / ACTION_SCALE.
JOINT_POS_LIMITS_LO = np.array(
    [-2.0943951, -0.0872665, -0.0872665, -1.95, -1.5707963, -2.0943951, -0.045],
    dtype=np.float32,
)
JOINT_POS_LIMITS_HI = np.array(
    [2.0943951, 3.1415927, 4.7123890, 1.5707963, 1.5707963, 2.0943951, 0.001],
    dtype=np.float32,
)

# Physical limits expressed in raw_action space: (ctrl_limit - HOME) / ACTION_SCALE.
# Use these to normalize teleop sim-actions to [-1, 1] with a fixed, session-independent
# transform that matches the full reachable joint range.
PHYSICAL_RAW_ACTION_MIN: np.ndarray = (
    JOINT_POS_LIMITS_LO - HOME_JOINT_POS
) / ACTION_SCALE
PHYSICAL_RAW_ACTION_MAX: np.ndarray = (
    JOINT_POS_LIMITS_HI - HOME_JOINT_POS
) / ACTION_SCALE


class ActionNormalizer:
    """Maps raw actions to the policy's [-1, 1] and back over a fixed range.

    One home for the convention previously duplicated across lerobot_to_qc
    (normalize), mjlab_env and play_qcfql (denormalize). Defaults to the
    physical DK1 joint range; construct from a dataset's stored action_min/max
    to honour a data-normalized dataset.
    """

    def __init__(self, action_min: np.ndarray, action_max: np.ndarray):
        self.action_min = np.asarray(action_min, dtype=np.float32)
        self.action_max = np.asarray(action_max, dtype=np.float32)

    @classmethod
    def physical(cls) -> "ActionNormalizer":
        return cls(PHYSICAL_RAW_ACTION_MIN, PHYSICAL_RAW_ACTION_MAX)

    def normalize(self, raw_action: np.ndarray) -> np.ndarray:
        """raw action → [-1, 1]. Guards zero-range (constant) dims against /0."""
        span = self.action_max - self.action_min
        rng = np.where(span > 1e-6, span, 1.0)
        return 2.0 * (raw_action - self.action_min) / rng - 1.0

    def denormalize(self, action: np.ndarray) -> np.ndarray:
        """[-1, 1] → raw action. Clips first: policy output must saturate at
        the physical limit, not extrapolate past it."""
        clipped = np.clip(action, -1.0, 1.0)
        span = self.action_max - self.action_min
        return (clipped + 1.0) / 2.0 * span + self.action_min


def lerobot_to_mjlab_ctrl(joint_pos: np.ndarray) -> np.ndarray:
    """Convert real-robot joint_pos (7,) → MJLab data.ctrl (7,).

    Joints 0-5 (radians) pass through unchanged
    Gripper (index 6) goes from normalised [0,1] to prismatic meters.
    """
    ctrl = np.empty(7, dtype=np.float32)
    ctrl[:6] = joint_pos[:6]
    ctrl[6] = mjlab_gripper_meters(float(joint_pos[6]))
    return ctrl


_GRAVITY_STIFFNESS = np.array(
    [get_arm_stiffness()[n] for n in _ACTUATOR_NAMES], dtype=np.float32
)
_gravity_model: mujoco.MjModel | None = None
_gravity_data: mujoco.MjData | None = None
_gravity_dofadr: np.ndarray | None = None


def _gravity_scratch():
    """Lazily build a standalone (non-rendering) DK1 model used only to
    evaluate gravity torque at a candidate target pose. Built on first use so
    importing this module stays cheap for callers that never enable
    gravity compensation (e.g. record_replay.py, tests).
    """
    global _gravity_model, _gravity_data, _gravity_dofadr
    if _gravity_model is None:
        from mjlab.entity import Entity

        entity = Entity(get_unimanual_dk1_robot_cfg())
        _gravity_model = entity.compile()
        _gravity_data = mujoco.MjData(_gravity_model)
        _gravity_dofadr = np.array(
            [
                _gravity_model.jnt_dofadr[
                    mujoco.mj_name2id(_gravity_model, mujoco.mjtObj.mjOBJ_JOINT, n)
                ]
                for n in _ACTUATOR_NAMES
            ]
        )
    return _gravity_model, _gravity_data, _gravity_dofadr


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
    model, data, dofadr = _gravity_scratch()
    data.qvel[:] = 0.0
    for name, value in zip(_ACTUATOR_NAMES, ctrl[:6], strict=True):
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
    """Convert to raw action for JointPositionActionCfg(use_default_offset=True).

    MJLab computes:  target = HOME_JOINT_POS + raw_action * ACTION_SCALE
    where the result recovers the original joint positions from lerobot_to_mjlab_ctrl.

    Args:
        gravity_compensate: If True, push the target past gravity sag (see
            gravity_compensate_ctrl) before computing raw_action. Opt-in and
            off by default — record_replay.py and tests rely on the exact
            uncompensated mapping.
    """
    ctrl = lerobot_to_mjlab_ctrl(joint_pos)
    if gravity_compensate:
        ctrl = gravity_compensate_ctrl(ctrl)
    return (ctrl - HOME_JOINT_POS) / ACTION_SCALE
