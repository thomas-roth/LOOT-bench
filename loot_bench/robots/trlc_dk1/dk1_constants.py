"""TRLC-DK1 robot config assembly.

Loads the MJCF, builds keyframes, and composes the final ``EntityCfg`` for
single-arm and bimanual variants.  Motor/collision details live in
``dk1_actuators`` and ``dk1_collision``.
"""

import math
from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from loot_bench.robots.trlc_dk1.dk1_actuators import (
    ARM_JOINTS,
    get_arm_actuators,
    get_gripper_actuator,
)
from loot_bench.robots.trlc_dk1.dk1_collision import (
    FULL_COLLISION,
    GRIPPER_ONLY_COLLISION,
    GRIPPER_ONLY_WITH_SELF_COLLISION,
)

# Interface identifiers.

DK1_NAME_UNIMANUAL = "Dk1"
DK1_NAME_BIMANUAL = "Dk1_bi"
DK1_GRASP_SITE = "grasp_site"
DK1_EE_SUBTREE = "link6-7"
DK1_VIEWER_BODY = "base"

# The actuator split the two action terms are built from: the arm is the IK
# chain, the gripper is the normalized stroke. Listed from ARM_JOINTS rather
# than written out, so adding a joint to the arm cannot leave this stale.
DK1_ARM_ACTUATORS_UNIMANUAL = tuple(ARM_JOINTS)
DK1_GRIPPER_ACTUATORS_UNIMANUAL = ("gripper_left",)
DK1_ARM_ACTUATORS_BIMANUAL = tuple(
    f"{prefix}{name}" for prefix in ("left_", "right_") for name in ARM_JOINTS
)
DK1_GRIPPER_ACTUATORS_BIMANUAL = ("left_gripper_left", "right_gripper_left")
DK1_IK_MAX_DQ = min(
    joint.motor.effort_limit / joint.kp for joint in ARM_JOINTS.values()
)
"""[rad] 0.28, the smallest arm joint's PD saturation error (joint2/3: 28 Nm
over kp=100).

Differential IK's ``max_dq``, derived from the actuators and from *torque*, not
from a velocity limit -- see ``ManipulatorInterface.ik_max_dq`` for why."""

##
# MJCF.

_DK1_XML_PATH: Path = Path(__file__).resolve().parent / "xml" / "dk1.xml"
assert _DK1_XML_PATH.exists()


# Keyframe config.

# Ready pose: gripper pointing straight down (q2 - q3 - q4 = pi/2), grasp site
# hovering at roughly (0.30, 0.0, 0.20) in the base frame, fingers open.
HOME_JOINT_POS: dict[str, float] = {
    "joint1": 0.0,
    "joint2": 1.7,
    "joint3": 1.7,
    "joint4": -(math.pi / 2),
    "joint5": 0.0,
    "joint6": 0.0,
    "gripper_left": 0.0,
    "gripper_right": 0.0,
}


# Episode start pose (arm joints). Matches teleop align pose; gripper unsampled.
# Gripper kept open: 0.046 m stroke would start half-closed if sampled.
START_JOINT_POS: dict[str, float] = {
    f"joint{i}": HOME_JOINT_POS[f"joint{i}"] for i in range(1, 7)
}


# Rest pose measured on the physical DK1. Not the start pose; nothing reads it.
REST_JOINT_POS: dict[str, float] = {
    "joint1": +0.1565,
    "joint2": +0.0015,
    "joint3": +0.0015,
    "joint4": -0.0153,
    "joint5": +0.0153,
    "joint6": +0.0276,
}


def _home_keyframe(prefix: str = "") -> EntityCfg.InitialStateCfg:
    return EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={f"{prefix}{k}": v for k, v in HOME_JOINT_POS.items()},
        joint_vel={".*": 0.0},
    )


# Unimanual config: single DK1 arm.


GRAVITY_COMPENSATED_BODIES: tuple[str, ...] = (
    "link1-2",
    "link2-3",
    "link3-4",
    "link4-5",
    "link5-6",
    "link6-7",
    "camera_mount",
    "finger_left",
    "finger_right",
)
"""Every body the arm actuators hold up — the whole chain below ``base``.

The fingers belong here even though their *own* joints see no gravity load
(the racks slide along ±y, horizontal when the gripper points down, measured
at 0.0 N): their 0.05 kg each still hangs off joint2 and joint3, and leaving
them out leaves two thirds of the drift in place. Measured over 100 zero-action
steps: 0.168 m uncompensated, 0.100 m with the links alone, 0.000 m with the
fingers included. See :func:`_compensate_arm_gravity`."""


def _compensate_arm_gravity(spec: mujoco.MjSpec) -> None:
    """Cancel gravity on everything the arm actuators carry.

    **Why the arm needs this.** Differential IK commands
    ``q_target = q_current + dq``, re-anchored to the *measured* position every
    substep, so there is no fixed setpoint to converge to: whatever sag the PD
    actuators carry at the end of a control step becomes the baseline the next
    control step measures from. The sag does not merely persist, it compounds
    — measured on ``place``, a held zero action walked the grasp site 0.168 m
    down into the table over 100 steps. Compensated it is 0.000 m, which is
    what "a zero action holds the end-effector still" has to mean.

    ADR-016 deletes the teleop-side gravity compensation on the grounds that
    "DiffIK closes the loop on the measured pose and corrects sag itself".
    That premise is false for mjlab's term. This puts the same correction where
    it does hold: in the simulator, once, for every caller — rather than in the
    recorder, where it distorted only the operator's trajectory. Both reference
    domains arrive at the same place by another route (robomimic's OSC carries
    a gravity feedforward in its torque law).

    **Why this is safe for grasping.** ``gravcomp`` adds a vertical force at
    each body's COM. Grip force is horizontal — the racks' own axis — so the
    20 N cap and the sub-millimeter penetration that ``dk1_actuators`` tunes
    for are untouched; ``tests/trlc_dk1/test_dk1_grasp.py`` pins that.

    **Accepted sim-fidelity cost:** the real DK1's controller has no gravity
    feedforward, so the sim arm now holds station better than the hardware.
    Nothing in the benchmark reads the difference, and the alternative is an
    MDP where standing still is impossible.
    """
    # The bimanual spec attaches two copies of this one, so it inherits the
    # compensation with its own prefixes and needs no second call.
    for name in GRAVITY_COMPENSATED_BODIES:
        spec.body(name).gravcomp = 1.0


def get_unimanual_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(_DK1_XML_PATH))
    # Exclude base/link1-2: interpenetrating meshes lock joint1 otherwise.
    spec.add_exclude(
        bodyname1="base", bodyname2="link1-2"
    )  # TODO: check if actually needed
    _compensate_arm_gravity(spec)
    return spec


def get_unimanual_dk1_robot_cfg(full_collision: bool = False) -> EntityCfg:
    return EntityCfg(
        init_state=_home_keyframe(),
        collisions=(FULL_COLLISION if full_collision else GRIPPER_ONLY_COLLISION,),
        spec_fn=get_unimanual_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(*get_arm_actuators(), get_gripper_actuator()),
            # 1.0, not the usual 0.9: differential IK reads the *soft* limits for
            # its joint-limit residual, and shrinking them would leave the
            # outer slice of every joint uncommandable -- the slice the
            # operator teleoperates through (ADR-005, ADR-016). Nothing else
            # in this repo reads them: reset_joints_to_pose clamps to the hard
            # limits deliberately, and mjlab's own soft-limit consumers are
            # all unused here.
            soft_joint_pos_limit_factor=1.0,
        ),
    )


# Bimanual config: two DK1 arms on a shared mount, facing +x.

_BIMANUAL_ARM_SEPARATION = 0.4
"[m], distance between the two base centers."


def get_bimanual_spec(
    separation: float = _BIMANUAL_ARM_SEPARATION,
) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.modelname = "trlc_dk1_bimanual"
    half = separation / 2.0
    for prefix, y in (("left_", half), ("right_", -half)):
        arm = get_unimanual_spec()
        # Drop the per-arm worldbody light/camera; the scene provides its own.
        for cam in list(arm.cameras):
            if cam.name == "front_cam":
                arm.delete(cam)
        for light in list(arm.lights):
            arm.delete(light)
        frame = spec.worldbody.add_frame(pos=(0.0, y, 0.0))
        frame.attach_body(arm.body("base"), prefix, "")
    spec.worldbody.add_camera(
        name="head_camera",
        pos=(-0.1, 0.0, 0.55),
        xyaxes=(0, -1, 0, 0.5, 0, 0.5),
        resolution=(640, 480),
    )
    return spec


def get_bimanual_dk1_robot_cfg(full_collision: bool = False) -> EntityCfg:
    joint_pos: dict[str, float] = {}
    for prefix in ("left_", "right_"):
        joint_pos.update({f"{prefix}{k}": v for k, v in HOME_JOINT_POS.items()})
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos=joint_pos,
            joint_vel={".*": 0.0},
        ),
        collisions=(
            FULL_COLLISION if full_collision else GRIPPER_ONLY_WITH_SELF_COLLISION,
        ),
        spec_fn=get_bimanual_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                *get_arm_actuators("left_"),
                get_gripper_actuator("left_"),
                *get_arm_actuators("right_"),
                get_gripper_actuator("right_"),
            ),
            # 1.0 for the same reason as the unimanual config above.
            soft_joint_pos_limit_factor=1.0,
        ),
    )
