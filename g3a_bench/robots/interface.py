"""Robot-agnostic manipulator description for benchmark tasks.

Tasks use this interface plus wire_manipulator_into_env; concrete values live
in each g3a_bench.robots.<robot> package.
"""

from collections.abc import Callable
from dataclasses import dataclass

from mjlab.entity import EntityCfg

_GRIPPER_OPEN_FRACTION = 2 / 3
"Fraction of the gripper joint stroke beyond which the gripper counts as open."


def gripper_open_threshold(joint_range: dict[str, float]) -> float:
    """Open threshold from joint_range via _GRIPPER_OPEN_FRACTION."""
    return joint_range["closed"] + _GRIPPER_OPEN_FRACTION * (
        joint_range["open"] - joint_range["closed"]
    )


@dataclass(frozen=True)
class ManipulatorInterface:
    name: str
    """Short identifier used in task ids (e.g. "Dk1")."""
    entity_cfg_fn: Callable[[], EntityCfg]
    """Builds the robot's ``EntityCfg``; added to the scene as ``"robot"``."""
    grasp_site: str
    """Site between the fingertips; used for reach rewards/observations."""
    gripper_joints: tuple[str, ...]
    """Gripper joint(s) gating task success: all must exceed
    ``gripper_open_threshold`` for the object to count as released."""
    gripper_open_threshold: float
    """Joint position above which a gripper joint counts as open."""
    ee_subtree: str
    """Body name whose subtree is monitored for illegal ground contact."""
    pad_geoms: str
    """Fullmatch regex for finger-pad geoms; used as primary of contact-release."""
    viewer_body: str
    """Body the viewer camera tracks."""
    reset_joint_pos: dict[str, float] | None = None
    """Absolute start pose, or None to start from the entity's own keyframe."""
