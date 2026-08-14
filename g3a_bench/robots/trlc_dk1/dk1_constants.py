"""TRLC-DK1 robot config assembly.

Loads the MJCF, builds keyframes, and composes the final ``EntityCfg`` for
single-arm and bimanual variants.  Motor/collision details live in
``dk1_actuators`` and ``dk1_collision``.
"""

import math
from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from g3a_bench.robots.interface import gripper_open_threshold
from g3a_bench.robots.trlc_dk1.dk1_actuators import (
    GRIPPER_MECHANICAL_RANGE,
    get_arm_actuators,
    get_gripper_actuator,
)
from g3a_bench.robots.trlc_dk1.dk1_collision import (
    FULL_COLLISION,
    GRIPPER_ONLY_COLLISION,
    GRIPPER_ONLY_WITH_SELF_COLLISION,
)

##
# Interface identifiers
#
# Consumed by get_unimanual_dk1_interface or get_bimanual_dk1_interface.
##

DK1_NAME_UNIMANUAL = "Dk1_uni"
DK1_NAME_BIMANUAL = "Dk1_bi"
DK1_GRASP_SITE = "grasp_site"
DK1_EE_SUBTREE = "link6-7"
DK1_VIEWER_BODY = "base"
DK1_GRIPPER_JOINTS_UNIMANUAL = ("gripper_left",)
DK1_GRIPPER_JOINTS_BIMANUAL = ("left_gripper_left", "right_gripper_left")
DK1_GRIPPER_OPEN_THRESHOLD = gripper_open_threshold(GRIPPER_MECHANICAL_RANGE)

##
# MJCF.
##

_DK1_XML_PATH: Path = Path(__file__).resolve().parent / "xml" / "dk1.xml"
assert _DK1_XML_PATH.exists()


##
# Keyframe config.
##

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


# Rest pose: where the leader arm hangs when the operator lets go, measured on
# the physical DK1. Teleop sessions start here, so episodes must start here too
# -- see ManipulatorInterface.reset_joint_pos. Far from HOME_JOINT_POS (q2 1.7
# vs 0.0015, q4 -1.5708 vs -0.0153), so the two are not interchangeable.
#
# Arm joints only, deliberately. The reset event samples its start-pose noise
# over exactly the joints listed here, and the gripper must not be sampled: its
# whole stroke is 0.046 m, so even a small joint-space range would start
# episodes with a half-closed hand. Omitted joints keep the keyframe value,
# which leaves the gripper open at every reset.
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


##
# Unimanual config: single DK1 arm.
##


def get_unimanual_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(_DK1_XML_PATH))
    # base/link1-2 mesh surfaces interpenetrate slightly at rest (joint1's own
    # axis). MuJoCo's default parent-child contact filter doesn't catch this
    # pair for some reason, so the resulting contact force fights the joint1
    # actuator and effectively locks the base rotation. Exclude explicitly.
    spec.add_exclude(
        bodyname1="base", bodyname2="link1-2"
    )  # TODO: check if actually needed
    return spec


def get_unimanual_dk1_robot_cfg(full_collision: bool = False) -> EntityCfg:
    return EntityCfg(
        init_state=_home_keyframe(),
        collisions=(FULL_COLLISION if full_collision else GRIPPER_ONLY_COLLISION,),
        spec_fn=get_unimanual_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(*get_arm_actuators(), get_gripper_actuator()),
            soft_joint_pos_limit_factor=0.9,
        ),
    )


##
# Bimanual config: two DK1 arms on a shared mount, facing +x.
##

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
            soft_joint_pos_limit_factor=0.9,
        ),
    )
