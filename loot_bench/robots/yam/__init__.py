"""YAM manipulator interface.

Thin loot-bench adapter over the mjlab asset-zoo YAM robot, so every robot the
benchmark uses exposes the same ``get_<robot>_interface()`` entry point and owns
its own decisions (see ``loot_bench.robots.interface.ManipulatorInterface``).
"""

from mjlab.asset_zoo.robots import get_yam_robot_cfg

from loot_bench.robots.interface import ManipulatorInterface
from loot_bench.robots.yam.yam_constants import (
    YAM_ARM_ACTUATORS,
    YAM_EE_SUBTREE,
    YAM_GRASP_SITE,
    YAM_GRIPPER_ACTUATORS,
    YAM_GRIPPER_JOINTS,
    YAM_GRIPPER_OPEN_THRESHOLD,
    YAM_IK_MAX_DQ,
    YAM_NAME,
    YAM_PAD_GEOMS_PATTERN,
    YAM_VIEWER_BODY,
)


def get_yam_interface() -> ManipulatorInterface:
    return ManipulatorInterface(
        name=YAM_NAME,
        entity_cfg_fn=get_yam_robot_cfg,
        grasp_site=YAM_GRASP_SITE,
        arm_actuators=YAM_ARM_ACTUATORS,
        gripper_actuators=YAM_GRIPPER_ACTUATORS,
        ik_max_dq=YAM_IK_MAX_DQ,
        gripper_joints=YAM_GRIPPER_JOINTS,
        gripper_open_threshold=YAM_GRIPPER_OPEN_THRESHOLD,
        ee_subtree=YAM_EE_SUBTREE,
        pad_geoms=YAM_PAD_GEOMS_PATTERN,
        viewer_body=YAM_VIEWER_BODY,
    )
