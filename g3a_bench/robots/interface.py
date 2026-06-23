"""Robot-agnostic description of a manipulator for the benchmark tasks.

Tasks are written against this interface instead of hardcoding robot details
(site names, collision geom patterns, action scales). Adding a robot to the
benchmark means adding one `ManipulatorInterface` instance and registering the
task variants for it in each task's `config/` package.

The interface can be implemented with other robots for example YAM from mjlab asset zoo
"""

from collections.abc import Callable
from dataclasses import dataclass

from mjlab.entity import EntityCfg


@dataclass(frozen=True)
class ManipulatorInterface:
    name: str
    """Short identifier used in task ids (e.g. "Dk1")."""
    entity_cfg_fn: Callable[[], EntityCfg]
    action_scale: dict[str, float] | float
    grasp_site: str
    """Site between the fingertips; used for reach rewards/observations."""
    ee_subtree: str
    """Body name whose subtree is monitored for illegal ground contact."""
    finger_pad_geoms: str
    """Regex matching the fingertip pad geoms (friction domain randomization)."""
    viewer_body: str
    arm_joints_expr: str
    """Regex matching arm joint actuators (IK-controlled, excludes gripper)."""
    gripper_actuator_expr: str
    """Name of the actuated gripper joint (the other finger is coupled via equality)."""


def get_dk1_interface() -> ManipulatorInterface:
    from g3a_bench.robots.trlc_dk1 import DK1_ACTION_SCALE, get_dk1_robot_cfg

    return ManipulatorInterface(
        name="Dk1",
        entity_cfg_fn=get_dk1_robot_cfg,
        action_scale=DK1_ACTION_SCALE,
        grasp_site="grasp_site",
        ee_subtree="link6-7",
        finger_pad_geoms=r"finger_(left|right)_pad_collision",
        viewer_body="base",
        arm_joints_expr=r"joint\d",
        gripper_actuator_expr="gripper_left",
    )


def get_yam_interface() -> ManipulatorInterface:
    from mjlab.asset_zoo.robots import YAM_ACTION_SCALE, get_yam_robot_cfg

    return ManipulatorInterface(
        name="Yam",
        entity_cfg_fn=get_yam_robot_cfg,
        action_scale=YAM_ACTION_SCALE,
        grasp_site="grasp_site",
        ee_subtree="link_6",
        finger_pad_geoms=r"(lf|rf)_down.*_collision",
        viewer_body="arm",
        arm_joints_expr=r"joint\d",
        gripper_actuator_expr="left_finger",
    )
