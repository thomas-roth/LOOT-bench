import mujoco
import numpy as np

from g3a_bench.robots.trlc_dk1.dk1_actuators import (
    ARM_JOINTS,
    DK1_ACTION_SCALE_UNIMANUAL,
    GRIPPER_COMMAND_RANGE,
)
from g3a_bench.robots.trlc_dk1.dk1_constants import (
    HOME_JOINT_POS,
    get_unimanual_dk1_robot_cfg,
)

_ARM_JOINT_NAMES = list(ARM_JOINTS.keys())

HOME_JOINT_POS_ACTUATED = np.array(
    [HOME_JOINT_POS[n] for n in _ARM_JOINT_NAMES] + [HOME_JOINT_POS["gripper_left"]],
    dtype=np.float32,
)
ACTION_SCALE_ACTUATED = np.array(
    [DK1_ACTION_SCALE_UNIMANUAL[n] for n in _ARM_JOINT_NAMES]
    + [DK1_ACTION_SCALE_UNIMANUAL["gripper_left"]],
    dtype=np.float32,
)


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
                for n in _ARM_JOINT_NAMES
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
    """Convert to raw action for JointPositionActionCfg(use_default_offset=True).

    MJLab computes:  target = HOME_ACTUATED_JOINT_POS + raw_action * ACTION_SCALE
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
    return (ctrl - HOME_JOINT_POS_ACTUATED) / ACTION_SCALE_ACTUATED
