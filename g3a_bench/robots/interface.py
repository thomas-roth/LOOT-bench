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

GRIPPER_OPEN_FRACTION = 2 / 3
"""Fraction of a gripper joint's stroke above which it counts as open.

One benchmark-wide convention so "released" means the same thing on every
robot; robots derive their ``gripper_open_threshold`` from their joint range
via :func:`gripper_open_threshold`.
"""


def gripper_open_threshold(joint_range: tuple[float, float]) -> float:
    """Threshold for a gripper joint that is open at its *upper* limit."""
    lo, hi = joint_range
    return lo + GRIPPER_OPEN_FRACTION * (hi - lo)


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
    gripper_joints: tuple[str, ...]
    """Gripper joint(s) gating task success: all must exceed
    ``gripper_open_threshold`` for the object to count as released."""
    gripper_open_threshold: float
    """Joint position above which a gripper joint counts as open. Derive it
    from the joint range with :func:`gripper_open_threshold` so a merely
    loosened grip does not count."""
    ee_subtree: str
    """Body name whose subtree is monitored for illegal ground contact."""
    finger_pad_geoms: str
    """Regex matching the fingertip pad geoms (friction domain randomization)."""
    viewer_body: str
    """Body the viewer camera tracks."""
