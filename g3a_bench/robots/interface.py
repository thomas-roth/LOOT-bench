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
