from g3a_bench.robots.interface import ManipulatorInterface
from g3a_bench.robots.trlc_dk1.dk1_collision import DK1_PAD_GEOMS_PATTERN
from g3a_bench.robots.trlc_dk1.dk1_constants import (
    DK1_EE_SUBTREE,
    DK1_GRASP_SITE,
    DK1_GRIPPER_JOINTS_BIMANUAL,
    DK1_GRIPPER_JOINTS_UNIMANUAL,
    DK1_GRIPPER_OPEN_THRESHOLD,
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
        gripper_joints=DK1_GRIPPER_JOINTS_UNIMANUAL,
        gripper_open_threshold=DK1_GRIPPER_OPEN_THRESHOLD,
        ee_subtree=DK1_EE_SUBTREE,
        pad_geoms=DK1_PAD_GEOMS_PATTERN,
        viewer_body=DK1_VIEWER_BODY,
        reset_joint_pos=START_JOINT_POS,
    )


def get_bimanual_dk1_interface() -> ManipulatorInterface:
    return ManipulatorInterface(
        name=DK1_NAME_BIMANUAL,
        entity_cfg_fn=get_bimanual_dk1_robot_cfg,
        grasp_site=DK1_GRASP_SITE,
        gripper_joints=DK1_GRIPPER_JOINTS_BIMANUAL,
        gripper_open_threshold=DK1_GRIPPER_OPEN_THRESHOLD,
        ee_subtree=DK1_EE_SUBTREE,
        pad_geoms=DK1_PAD_GEOMS_PATTERN,
        viewer_body=DK1_VIEWER_BODY,
    )
