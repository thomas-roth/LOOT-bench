"""TRLC-DK1 actuator configuration.

Motor specs (DAMIAO DM4340 / DM4310), effective inertias, arm/gripper
actuator builders, and action-scale computation.
"""

import math

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.utils.actuator import ElectricActuator, reflect_rotary_to_linear

# Motor armatures (rotor inertia reflected through the gearbox).
_ARMATURE_DM_4340 = 0.032
_ARMATURE_DM_4310 = 0.0018

# Same motors as the i2rt YAM (mjlab asset zoo reference values).
#
# velocity_limit is the motor's max/no-load speed (the zero-torque intercept of
# the torque-speed curve), taken from the official trlc_dk1_control config.py
# (DM4340_DQ_MAX / DM4310_DQ_MAX) and matching the YAM asset convention. NOTE:
# BuiltinPositionActuatorCfg does not currently consume velocity_limit, so these
# values are informational only (no torque-speed derating is modeled).
_DM_4340 = ElectricActuator(
    reflected_inertia=_ARMATURE_DM_4340,
    velocity_limit=8.0,  # rad/s, DM4340 max speed (config.py DM4340_DQ_MAX).
    effort_limit=28.0,  # Nm, per the official trlc_dk1_control config.py torque limit.
)
_DM_4310 = ElectricActuator(
    reflected_inertia=_ARMATURE_DM_4310,
    velocity_limit=30.0,  # rad/s, DM4310 max speed (config.py DM4310_DQ_MAX).
    effort_limit=10.0,
)

_DAMPING_RATIO = 2.0  # gripper PD damping ratio (arm uses explicit KD below).

_ARM_JOINTS: dict[str, ElectricActuator] = {
    "joint1": _DM_4340,
    "joint2": _DM_4340,
    "joint3": _DM_4340,
    "joint4": _DM_4310,
    "joint5": _DM_4310,
    "joint6": _DM_4310,
}

# Arm PD gains from the official trlc_dk1_control config.py MIT-mode gains
# (KP / KD). These are markedly stiffer than an inertia-derived critically-
# damped tuning, matching the real teleop controller and giving the end
# effector enough Cartesian stiffness (~700 N/m at the wrist) to push scene
# objects without large leader over-travel. Units: kp [Nm/rad], kd [Nm·s/rad].
_ARM_KP: dict[str, float] = {
    "joint1": 100.0,
    "joint2": 100.0,
    "joint3": 100.0,
    "joint4": 20.0,
    "joint5": 20.0,
    "joint6": 10.0,
}
_ARM_KD: dict[str, float] = {
    "joint1": 5.0,
    "joint2": 5.0,
    "joint3": 4.0,
    "joint4": 1.0,
    "joint5": 1.0,
    "joint6": 1.0,
}


def get_arm_stiffness() -> dict[str, float]:
    """Per-joint PD stiffness (kp), unprefixed — for callers that need the raw
    gain rather than a full actuator cfg (e.g. live-teleop gravity feedforward).
    """
    return dict(_ARM_KP)


def get_arm_actuators(prefix: str = "") -> tuple[BuiltinPositionActuatorCfg, ...]:
    return tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=(f"{prefix}{name}",),
            stiffness=_ARM_KP[name],
            damping=_ARM_KD[name],
            effort_limit=motor.effort_limit,
            armature=motor.reflected_inertia,
        )
        for name, motor in _ARM_JOINTS.items()
    )


##
# Gripper transmission parameters.
##

# DM4310 pinion with 8.75 mm radius drives both finger racks, so the
# transmission ratio is exact (unlike a crank): x = r * theta.
_GRIPPER_SPUR_RADIUS = 0.00875  # [m]

(
    _ARMATURE_DM_4310_LINEAR,
    _VELOCITY_LIMIT_DM_4310_LINEAR,
    _EFFORT_LIMIT_DM_4310_LINEAR,
) = reflect_rotary_to_linear(
    armature_rotary=_ARMATURE_DM_4310,
    velocity_limit_rotary=_DM_4310.velocity_limit,
    effort_limit_rotary=_DM_4310.effort_limit,
    transmission_ratio=_GRIPPER_SPUR_RADIUS,
)

_NATURAL_FREQ_GRIPPER = math.tau  # 1 Hz
_STIFFNESS_GRIPPER = _ARMATURE_DM_4310_LINEAR * _NATURAL_FREQ_GRIPPER**2
_DAMPING_GRIPPER = (
    2.0 * _DAMPING_RATIO * _ARMATURE_DM_4310_LINEAR * _NATURAL_FREQ_GRIPPER
)

# The DK1 datasheet rates the gripper at 114 N (the official trlc_dk1_control
# config.py caps the gripper motor at 1.0 Nm, which through the 8.75 mm pinion
# is exactly 114 N). Use the full rated grip force; no extra sim derating.
_EFFORT_LIMIT_GRIPPER_SAFE = 114.0 * 1.0


def get_gripper_actuator(prefix: str = "") -> BuiltinPositionActuatorCfg:
    # Only gripper_left is actuated; gripper_right is coupled via equality.
    return BuiltinPositionActuatorCfg(
        target_names_expr=(f"{prefix}gripper_left",),
        stiffness=_STIFFNESS_GRIPPER,
        damping=_DAMPING_GRIPPER,
        effort_limit=_EFFORT_LIMIT_GRIPPER_SAFE,
        armature=_ARMATURE_DM_4310_LINEAR,
    )


def _action_scale(
    actuators: tuple[BuiltinPositionActuatorCfg, ...],
) -> dict[str, float]:
    scale: dict[str, float] = {}
    for a in actuators:
        assert a.effort_limit is not None
        for n in a.target_names_expr:
            scale[n] = 0.25 * a.effort_limit / a.stiffness
    return scale


DK1_UNIMANUAL_ACTION_SCALE = _action_scale(
    (*get_arm_actuators(), get_gripper_actuator())
)
DK1_BIMANUAL_ACTION_SCALE = _action_scale(
    (
        *get_arm_actuators("left_"),
        get_gripper_actuator("left_"),
        *get_arm_actuators("right_"),
        get_gripper_actuator("right_"),
    )
)
