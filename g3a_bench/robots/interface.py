"""Robot-agnostic description of a manipulator for the benchmark tasks.

Tasks are written against this interface instead of hardcoding robot details
(site names, collision geom patterns, action scales). Every concrete value
lives in the robot's own ``g3a_bench.robots.<robot>`` package, which exposes a
``get_<robot>_interface()`` factory -- nothing robot-specific belongs in this
module. Tasks consume an instance through the ``wire_manipulator_into_env``
adapter on the task side.
"""

from collections.abc import Callable
from dataclasses import dataclass

from mjlab.entity import EntityCfg


@dataclass(frozen=True)
class ManipulatorInterface:
    name: str
    """Short identifier used in task ids (e.g. "Dk1")."""
    entity_cfg_fn: Callable[[], EntityCfg]
    """Builds the robot's ``EntityCfg``; added to the scene as ``"robot"``."""
    action_scale: dict[str, float] | float
    """Per-actuator (or scalar) scale for the joint-position action."""
    grasp_site: str
    """Site between the fingertips; used for reach rewards/observations."""
    ee_subtree: str
    """Body name whose subtree is monitored for illegal ground contact."""
    finger_pad_geoms: str
    """Regex matching the fingertip pad geoms (friction domain randomization)."""
    viewer_body: str
    """Body the viewer camera tracks."""
    arm_actuators: tuple[str, ...] = (".*",)
    """Actuator name expressions for the arm joints, driven by the task-space
    (EEF / differential-IK) action. Excludes the gripper. Defaults to all
    actuators; set explicitly on grippered manipulators so the IK controller
    only moves the arm."""
    gripper_actuators: tuple[str, ...] = ()
    """Actuator name expressions for the gripper, driven by a direct joint
    action alongside the EEF controller. Empty means no separate gripper action
    (only used in the ``"eef"`` action mode)."""
    camera_name: str | None = None
    """Name of an existing MJCF camera to use for vision observations (prefixed,
    e.g. ``"robot/head_camera"``). ``None`` means the robot has no camera wired,
    so the ``"camera"`` observation mode is unavailable for it."""
