from g3a_bench.robots.interface import ManipulatorInterface
from g3a_bench.robots.trlc_dk1.dk1_actuators import (
    DK1_BIMANUAL_ACTION_SCALE,
    DK1_UNIMANUAL_ACTION_SCALE,
)
from g3a_bench.robots.trlc_dk1.dk1_collision import DK1_PAD_GEOM_PATTERN
from g3a_bench.robots.trlc_dk1.dk1_constants import (
    DK1_BIMANUAL_NAME,
    DK1_EE_SUBTREE,
    DK1_GRASP_SITE,
    DK1_UNIMANUAL_NAME,
    DK1_VIEWER_BODY,
    get_bimanual_dk1_robot_cfg,
    get_unimanual_dk1_robot_cfg,
)


def get_unimanual_dk1_interface() -> ManipulatorInterface:
    return ManipulatorInterface(
        name=DK1_UNIMANUAL_NAME,
        entity_cfg_fn=get_unimanual_dk1_robot_cfg,
        action_scale=DK1_UNIMANUAL_ACTION_SCALE,
        grasp_site=DK1_GRASP_SITE,
        ee_subtree=DK1_EE_SUBTREE,
        finger_pad_geoms=DK1_PAD_GEOM_PATTERN,
        viewer_body=DK1_VIEWER_BODY,
        # Arm joints driven by the EEF/IK controller; gripper kept on its own
        # joint action (only gripper_left is actuated, see dk1_actuators).
        arm_actuators=tuple(f"joint{i}" for i in range(1, 7)),
        gripper_actuators=("gripper_left",),
        # Wrist-mounted camera from the DK1 spec (prefixed with the "robot"
        # entity name in the scene); a consistent EEF-relative view for vision.
        camera_name="robot/wrist_camera",
    )


def get_bimanual_dk1_interface() -> ManipulatorInterface:
    return ManipulatorInterface(
        name=DK1_BIMANUAL_NAME,
        entity_cfg_fn=get_bimanual_dk1_robot_cfg,
        action_scale=DK1_BIMANUAL_ACTION_SCALE,
        grasp_site=DK1_GRASP_SITE,
        ee_subtree=DK1_EE_SUBTREE,
        finger_pad_geoms=DK1_PAD_GEOM_PATTERN,
        viewer_body=DK1_VIEWER_BODY,
    )
