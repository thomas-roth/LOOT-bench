"""TRLC-DK1 actuator configuration.

Motor specs (DAMIAO DM4340 / DM4310), effective inertias, arm/gripper
actuator builders, and action-scale computation.
"""

import math

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.utils.actuator import ElectricActuator, reflect_rotary_to_linear

# Effective inertias: diagonal of the joint-space mass matrix at the home
# keyframe (see utils/convert_glb_meshes_to_obj.py provenance), plus motor armature.
_ARMATURE_DM_4340 = 0.032
_ARMATURE_DM_4310 = 0.0018

_EFFECTIVE_INERTIAS: dict[str, float] = {
    "joint1": 0.194 + _ARMATURE_DM_4340,
    "joint2": 0.347 + _ARMATURE_DM_4340,
    "joint3": 0.137 + _ARMATURE_DM_4340,
    "joint4": 0.017 + _ARMATURE_DM_4310,
    "joint5": 0.005 + _ARMATURE_DM_4310,
    "joint6": 0.0006 + _ARMATURE_DM_4310,
}

# Same motors as the i2rt YAM (mjlab asset zoo reference values).
_DM_4340 = ElectricActuator(
    reflected_inertia=_ARMATURE_DM_4340,
    velocity_limit=5.49,  # In rad/s. Corresponds to 52.5 rpm (DK1 datasheet).
    effort_limit=27.0,
)
_DM_4310 = ElectricActuator(
    reflected_inertia=_ARMATURE_DM_4310,
    velocity_limit=20.94,  # In rad/s. Corresponds to 200 rpm (DK1 datasheet) and 0.183 m/s (URDF with _GRIPPER_SPUR_RADIUS).
    effort_limit=10.0,
)

_NATURAL_FREQ = 2 * math.tau  # 2 Hz
_DAMPING_RATIO = 2.0

_ARM_JOINTS: dict[str, ElectricActuator] = {
    "joint1": _DM_4340,
    "joint2": _DM_4340,
    "joint3": _DM_4340,
    "joint4": _DM_4310,
    "joint5": _DM_4310,
    "joint6": _DM_4310,
}


def get_arm_actuators(prefix: str = "") -> tuple[BuiltinPositionActuatorCfg, ...]:
    return tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=(f"{prefix}{name}",),
            stiffness=_EFFECTIVE_INERTIAS[name] * _NATURAL_FREQ**2,
            damping=2.0 * _DAMPING_RATIO * _EFFECTIVE_INERTIAS[name] * _NATURAL_FREQ,
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

# The DK1 datasheet rates the gripper at 114 N; we additionally derate for sim
# stability (same practice as the mjlab YAM asset).
_EFFORT_LIMIT_GRIPPER_SAFE = 114.0 * 0.3


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
