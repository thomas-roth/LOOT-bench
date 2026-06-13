"""TRLC-DK1 constants.

Model converted from the official URDF (https://github.com/robot-learning-co/trlc-dk1,
Apache-2.0). Joints 1-3 use DAMIAO DM4340 motors, joints 4-6 DM4310, and the
gripper a DM4310 driving both fingers through an 8.75 mm spur gear
(rack-and-pinion, 114 N rated grip force, 0.183 m/s).
"""
# pyright: reportAttributeAccessIssue=false
# (mujoco ships no type stubs for the compiled MjSpec module)

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflect_rotary_to_linear
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

DK1_XML: Path = Path(__file__).resolve().parent / "xmls" / "dk1.xml"
assert DK1_XML.exists()


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(DK1_XML))


##
# Actuator config.
##

# Effective inertias: diagonal of the joint-space mass matrix at the home
# keyframe (see scripts/convert_meshes.py provenance), plus motor armature.
ARMATURE_DM_4340 = 0.032
ARMATURE_DM_4310 = 0.0018

EFFECTIVE_INERTIAS = {
    "joint1": 0.194 + ARMATURE_DM_4340,
    "joint2": 0.347 + ARMATURE_DM_4340,
    "joint3": 0.137 + ARMATURE_DM_4340,
    "joint4": 0.017 + ARMATURE_DM_4310,
    "joint5": 0.005 + ARMATURE_DM_4310,
    "joint6": 0.0006 + ARMATURE_DM_4310,
}

# Same motors as the i2rt YAM (mjlab asset zoo reference values).
DM_4340 = ElectricActuator(
    reflected_inertia=ARMATURE_DM_4340,
    velocity_limit=5.49,  # 52.5 rpm (DK1 datasheet).
    effort_limit=27.0,
)
DM_4310 = ElectricActuator(
    reflected_inertia=ARMATURE_DM_4310,
    velocity_limit=20.94,  # 200 rpm (DK1 datasheet).
    effort_limit=10.0,
)

NATURAL_FREQ = 2 * 2.0 * 3.1415926535  # 2 Hz
DAMPING_RATIO = 2.0

_ARM_JOINTS: dict[str, ElectricActuator] = {
    "joint1": DM_4340,
    "joint2": DM_4340,
    "joint3": DM_4340,
    "joint4": DM_4310,
    "joint5": DM_4310,
    "joint6": DM_4310,
}


def _arm_actuators(prefix: str = "") -> tuple[BuiltinPositionActuatorCfg, ...]:
    return tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=(f"{prefix}{name}",),
            stiffness=EFFECTIVE_INERTIAS[name] * NATURAL_FREQ**2,
            damping=2.0 * DAMPING_RATIO * EFFECTIVE_INERTIAS[name] * NATURAL_FREQ,
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
GRIPPER_SPUR_RADIUS = 0.00875  # [m]

(
    ARMATURE_DM_4310_LINEAR,
    VELOCITY_LIMIT_DM_4310_LINEAR,
    EFFORT_LIMIT_DM_4310_LINEAR,
) = reflect_rotary_to_linear(
    armature_rotary=ARMATURE_DM_4310,
    velocity_limit_rotary=DM_4310.velocity_limit,
    effort_limit_rotary=DM_4310.effort_limit,
    transmission_ratio=GRIPPER_SPUR_RADIUS,
)

NATURAL_FREQ_GRIPPER = 1.0 * 2.0 * 3.1415926535  # 1 Hz
STIFFNESS_GRIPPER = ARMATURE_DM_4310_LINEAR * NATURAL_FREQ_GRIPPER**2
DAMPING_GRIPPER = 2.0 * DAMPING_RATIO * ARMATURE_DM_4310_LINEAR * NATURAL_FREQ_GRIPPER

# The DK1 datasheet rates the gripper at 114 N; we additionally derate for sim
# stability (same practice as the mjlab YAM asset).
EFFORT_LIMIT_GRIPPER_SAFE = 114.0 * 0.3


def _gripper_actuator(prefix: str = "") -> BuiltinPositionActuatorCfg:
    # Only gripper_left is actuated; gripper_right is coupled via equality.
    return BuiltinPositionActuatorCfg(
        target_names_expr=(f"{prefix}gripper_left",),
        stiffness=STIFFNESS_GRIPPER,
        damping=DAMPING_GRIPPER,
        effort_limit=EFFORT_LIMIT_GRIPPER_SAFE,
        armature=ARMATURE_DM_4310_LINEAR,
    )


##
# Keyframe config.
##

# Ready pose: gripper pointing straight down (q2 - q3 - q4 = pi/2), grasp site
# hovering at roughly (0.30, 0.0, 0.20) in the base frame, fingers open.
HOME_JOINT_POS = {
    "joint1": 0.0,
    "joint2": 1.7,
    "joint3": 1.7,
    "joint4": -1.5708,
    "joint5": 0.0,
    "joint6": 0.0,
    "gripper_left": 0.0,
    "gripper_right": 0.0,
}


def _home_keyframe(prefix: str = "") -> EntityCfg.InitialStateCfg:
    return EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={f"{prefix}{k}": v for k, v in HOME_JOINT_POS.items()},
        joint_vel={".*": 0.0},
    )


##
# Collision config.
##

_PAD_GEOMS = r".*finger_(left|right)_pad_collision"

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={
        _PAD_GEOMS: 6,
        ".*_collision": 3,
    },
    friction={
        _PAD_GEOMS: (1, 5e-3, 5e-4),
        ".*_collision": (0.6,),
    },
    solref={
        _PAD_GEOMS: (0.01, 1),
    },
    priority={
        _PAD_GEOMS: 1,
    },
)

# Collisions limited to the wrist link and fingers: cheaper, and the arm body
# cannot get stuck on scene objects during training.
GRIPPER_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype={
        r".*(link6-7|finger_(left|right))(_pad)?_collision": 1,
        ".*_collision": 0,
    },
    conaffinity={
        r".*(link6-7|finger_(left|right))(_pad)?_collision": 1,
        ".*_collision": 0,
    },
    condim={
        _PAD_GEOMS: 6,
        ".*_collision": 3,
    },
    friction={
        _PAD_GEOMS: (1, 5e-3, 5e-4),
        ".*_collision": (0.6,),
    },
    solref={
        _PAD_GEOMS: (0.01, 1),
    },
    priority={
        _PAD_GEOMS: 1,
    },
)

##
# Final configs.
##


def get_dk1_robot_cfg(full_collision: bool = False) -> EntityCfg:
    return EntityCfg(
        init_state=_home_keyframe(),
        collisions=(FULL_COLLISION if full_collision else GRIPPER_ONLY_COLLISION,),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(*_arm_actuators(), _gripper_actuator()),
            soft_joint_pos_limit_factor=0.9,
        ),
    )


##
# Bimanual variant: two DK1 arms on a shared mount, facing +x.
##

BIMANUAL_ARM_SEPARATION = 0.4  # [m] distance between the two base centers.


def get_bimanual_spec(separation: float = BIMANUAL_ARM_SEPARATION) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.modelname = "trlc_dk1_bimanual"
    half = separation / 2.0
    for prefix, y in (("left_", half), ("right_", -half)):
        arm = get_spec()
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
        collisions=(FULL_COLLISION if full_collision else GRIPPER_ONLY_COLLISION,),
        spec_fn=get_bimanual_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                *_arm_actuators("left_"),
                _gripper_actuator("left_"),
                *_arm_actuators("right_"),
                _gripper_actuator("right_"),
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
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


DK1_ACTION_SCALE = _action_scale((*_arm_actuators(), _gripper_actuator()))
DK1_BIMANUAL_ACTION_SCALE = _action_scale(
    (
        *_arm_actuators("left_"),
        _gripper_actuator("left_"),
        *_arm_actuators("right_"),
        _gripper_actuator("right_"),
    )
)


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_dk1_robot_cfg())
    viewer.launch(robot.spec.compile())
