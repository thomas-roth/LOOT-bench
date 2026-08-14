"""TRLC-DK1 actuator configuration.

Motor specs (DAMIAO DM4340 / DM4310), effective inertias, and the arm/gripper
actuator builders. There is no action scale here any more -- see the note at
the foot of the file. Motor specs and arm gains are taken from the official
trlc_dk1_control config.py to match the real-world controller. The gripper
uses a sim-specific soft tuning (see the gripper section below).
"""

import math
from dataclasses import dataclass

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.utils.actuator import ElectricActuator, reflect_rotary_to_linear

##
# Motors.
##

_ARMATURE_DM_4340 = 0.032
"[kg*m²], rotor inertia of DM4340 reflected through gearbox."

_ARMATURE_DM_4310 = 0.0018
"[kg*m²], rotor inertia of DM4310 reflected through gearbox."

_DM_4340 = ElectricActuator(
    reflected_inertia=_ARMATURE_DM_4340,
    velocity_limit=8.0,  # [rad/s], max speed in official config.py.
    effort_limit=28.0,  # [Nm], torque limit in official config.py.
)
_DM_4310 = ElectricActuator(
    reflected_inertia=_ARMATURE_DM_4310,
    velocity_limit=30.0,  # [rad/s], max speed in official config.py.
    effort_limit=10.0,  # [Nm], torque limit in official config.py.
)


##
# Arm joint actuators.
##


@dataclass(frozen=True)
class _ArmJointCfg:
    motor: ElectricActuator
    kp: float
    "[Nm/rad], joint stiffness."
    kd: float
    "[Nm*s/rad], joint damping."


ARM_JOINTS: dict[str, _ArmJointCfg] = {
    "joint1": _ArmJointCfg(_DM_4340, kp=100.0, kd=5.0),
    "joint2": _ArmJointCfg(_DM_4340, kp=100.0, kd=5.0),
    "joint3": _ArmJointCfg(_DM_4340, kp=100.0, kd=4.0),
    "joint4": _ArmJointCfg(_DM_4310, kp=20.0, kd=1.0),
    "joint5": _ArmJointCfg(_DM_4310, kp=20.0, kd=1.0),
    "joint6": _ArmJointCfg(_DM_4310, kp=10.0, kd=1.0),
}


def get_arm_actuators(prefix: str = "") -> tuple[BuiltinPositionActuatorCfg, ...]:
    return tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=(f"{prefix}{name}",),
            stiffness=joint.kp,
            damping=joint.kd,
            effort_limit=joint.motor.effort_limit,
            armature=joint.motor.reflected_inertia,
        )
        for name, joint in ARM_JOINTS.items()
    )


##
# Gripper transmission and control parameters.
#
# The real-life gripper runs the DM4310's internal (unpublished) position loop
# with grip forces coming from the torque cap: a servo that saturates its
# force cap after a small position error. The sim gripper keeps that shape
# (stiffness = cap / saturation error) but lowers the cap from the rated
# 114 N to 20 N: at the rated force, sim-stiffness contacts let the pads sink
# ~4.5 mm into task pieces and eject them from the grip, while 20 N holds
# them with sub-millimeter penetration (tests/trlc_dk1/test_dk1_grasp.py).
# Overdamped so the impact transient settles without bounce.
##

GRIPPER_MECHANICAL_RANGE = {"open": 0.001, "closed": -0.045}
"[m], mechanical gripper range mirroring `finger` joint range in xml/dk1.xml."
_GRIPPER_PINION_RADIUS = 0.00875
"[m], pinion radius of DM4310."
_ARMATURE_DM_4310_LINEAR, _, _ = reflect_rotary_to_linear(
    armature_rotary=_ARMATURE_DM_4310,
    velocity_limit_rotary=_DM_4310.velocity_limit,
    effort_limit_rotary=_DM_4310.effort_limit,
    transmission_ratio=_GRIPPER_PINION_RADIUS,
    # Pinion drives both finger racks => transmission ratio = pinion radius.
)
"[kg], rotor inertia of DM4310 reflected through pinion."

_GRIPPER_CLOSED_POS_RAD = -4.7
"[rad], gripper_closed_pos in official config.py."
GRIPPER_COMMAND_RANGE = {
    "open": 0.0,
    "closed": _GRIPPER_CLOSED_POS_RAD * _GRIPPER_PINION_RADIUS,
}
"""[m], real-robot controller's calibrated command range of gripper."""
_GRIPPER_MAX_TORQUE = 1.0
"[Nm], max_gripper_torque_n in official config.py."
_GRIPPER_RATED_FORCE = _GRIPPER_MAX_TORQUE / _GRIPPER_PINION_RADIUS
"[N], real gripper force. Derived from max torque and pinion radius."
_GRIPPER_EFFORT_LIMIT = 20.0
"[N], sim grip-force cap, lowered from the rated ~114 N (see section comment)."
_GRIPPER_SATURATION_ERROR = 0.0035
"[m], position error that yields full force."
_GRIPPER_STIFFNESS = _GRIPPER_EFFORT_LIMIT / _GRIPPER_SATURATION_ERROR
"[N/m], stiffness that yields the capped grip force at saturation error."
_GRIPPER_DAMPING_RATIO = 2.0
"Overdamped so the contact impact settles without bounce."
_GRIPPER_DAMPING = (
    _GRIPPER_DAMPING_RATIO
    * 2.0
    * math.sqrt(_GRIPPER_STIFFNESS * _ARMATURE_DM_4310_LINEAR)
)
"[kg/s], damping of the stiffness-spring-mass system."


##
# Gripper joint actuator.
##


def get_gripper_actuator(prefix: str = "") -> BuiltinPositionActuatorCfg:
    # Only gripper_left is actuated; gripper_right is coupled via equality in dk1.xml.
    return BuiltinPositionActuatorCfg(
        target_names_expr=(f"{prefix}gripper_left",),
        stiffness=_GRIPPER_STIFFNESS,
        damping=_GRIPPER_DAMPING,
        effort_limit=_GRIPPER_EFFORT_LIMIT,
        armature=_ARMATURE_DM_4310_LINEAR,
    )


##
# Action scale.
#
# There is none here any more. The DK1 used to publish a per-joint residual
# scale -- the position error producing a quarter of the motor's torque, about
# 0.07 rad on the big joints. That number only makes sense for an unbounded
# policy, which emits whatever magnitude it needs. Every task now uses
# tasks/common/actions.NormalizedJointPositionAction instead, where one action
# unit is half a joint's travel and the numbers come from the compiled model.
##
