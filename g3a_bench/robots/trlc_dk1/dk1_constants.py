"""TRLC-DK1 robot config assembly.

Loads the MJCF, builds keyframes, and composes the final ``EntityCfg`` for
single-arm and bimanual variants.  Motor/collision details live in
``dk1_actuators`` and ``dk1_collision``.
"""

import math
from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from g3a_bench.robots.trlc_dk1.dk1_actuators import (
    get_arm_actuators,
    get_gripper_actuator,
)
from g3a_bench.robots.trlc_dk1.dk1_collision import (
    FULL_COLLISION,
    GRIPPER_ONLY_COLLISION,
)

##
# Interface identifiers (consumed by get_unimanual_dk1_interface or get_bimanual_dk1_interface).
##

DK1_UNIMANUAL_NAME = "Dk1_uni"
DK1_BIMANUAL_NAME = "Dk1_bi"
DK1_GRASP_SITE = "grasp_site"
DK1_EE_SUBTREE = "link6-7"
DK1_VIEWER_BODY = "base"

##
# MJCF and assets.
##

_DK1_XML: Path = Path(__file__).resolve().parent / "xmls" / "dk1.xml"
assert _DK1_XML.exists()


def get_unimanual_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_DK1_XML))


##
# Keyframe config.
##

# Ready pose: gripper pointing straight down (q2 - q3 - q4 = pi/2), grasp site
# hovering at roughly (0.30, 0.0, 0.20) in the base frame, fingers open.
_HOME_JOINT_POS: dict[str, float] = {
    "joint1": 0.0,
    "joint2": 1.7,
    "joint3": 1.7,
    "joint4": -(math.pi / 2),
    "joint5": 0.0,
    "joint6": 0.0,
    "gripper_left": 0.0,
    "gripper_right": 0.0,
}


def _home_keyframe(prefix: str = "") -> EntityCfg.InitialStateCfg:
    return EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={f"{prefix}{k}": v for k, v in _HOME_JOINT_POS.items()},
        joint_vel={".*": 0.0},
    )


##
# Single-arm config.
##


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
# Bimanual variant: two DK1 arms on a shared mount, facing +x.
##

_BIMANUAL_ARM_SEPARATION = 0.4  # [m] distance between the two base centers.


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
        joint_pos.update({f"{prefix}{k}": v for k, v in _HOME_JOINT_POS.items()})
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos=joint_pos,
            joint_vel={".*": 0.0},
        ),
        collisions=(FULL_COLLISION if full_collision else GRIPPER_ONLY_COLLISION,),
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
