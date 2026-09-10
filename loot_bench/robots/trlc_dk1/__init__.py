from loot_bench.robots.interface import ManipulatorInterface
from loot_bench.robots.trlc_dk1.dk1_collision import DK1_PAD_GEOMS_PATTERN
from loot_bench.robots.trlc_dk1.dk1_constants import (
    DK1_ARM_ACTUATORS_BIMANUAL,
    DK1_ARM_ACTUATORS_UNIMANUAL,
    DK1_EE_SUBTREE,
    DK1_GRASP_SITE,
    DK1_GRIPPER_ACTUATORS_BIMANUAL,
    DK1_GRIPPER_ACTUATORS_UNIMANUAL,
    DK1_IK_MAX_DQ,
    DK1_NAME_BIMANUAL,
    DK1_NAME_UNIMANUAL,
    DK1_VIEWER_BODY,
    START_JOINT_POS,
    get_bimanual_dk1_robot_cfg,
    get_unimanual_dk1_robot_cfg,
)


def get_unimanual_dk1_interface() -> ManipulatorInterface:
    return ManipulatorInterface(
        name=DK1_NAME_UNIMANUAL,
        entity_cfg_fn=get_unimanual_dk1_robot_cfg,
        grasp_site=DK1_GRASP_SITE,
        arm_actuators=DK1_ARM_ACTUATORS_UNIMANUAL,
        gripper_actuators=DK1_GRIPPER_ACTUATORS_UNIMANUAL,
        ik_max_dq=DK1_IK_MAX_DQ,
        ee_subtree=DK1_EE_SUBTREE,
        pad_geoms=DK1_PAD_GEOMS_PATTERN,
        viewer_body=DK1_VIEWER_BODY,
        reset_joint_pos=START_JOINT_POS,
    )


def get_bimanual_dk1_interface() -> ManipulatorInterface:
    """The two-armed DK1. Declared, not registered against any task.

    Its actuator split names both arms, which is honest but not usable as it
    stands: the end-effector delta term drives one IK frame, and twelve arm
    joints under one 6-D objective are underdetermined. Registering a bimanual
    task needs a second frame first (out of scope, ADR-016).
    """
    return ManipulatorInterface(
        name=DK1_NAME_BIMANUAL,
        entity_cfg_fn=get_bimanual_dk1_robot_cfg,
        grasp_site=DK1_GRASP_SITE,
        arm_actuators=DK1_ARM_ACTUATORS_BIMANUAL,
        gripper_actuators=DK1_GRIPPER_ACTUATORS_BIMANUAL,
        ik_max_dq=DK1_IK_MAX_DQ,
        ee_subtree=DK1_EE_SUBTREE,
        pad_geoms=DK1_PAD_GEOMS_PATTERN,
        viewer_body=DK1_VIEWER_BODY,
    )
