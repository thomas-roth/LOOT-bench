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

_GRIPPER_OPEN_FRACTION = 2 / 3
"Fraction of the gripper joint stroke beyond which the gripper counts as open."


def gripper_open_threshold(joint_range: dict[str, float]) -> float:
    """Joint position ``_GRIPPER_OPEN_FRACTION`` of the way from ``closed``
    to ``open`` in ``joint_range``."""
    return joint_range["closed"] + _GRIPPER_OPEN_FRACTION * (
        joint_range["open"] - joint_range["closed"]
    )


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
    pad_geoms: str
    """Entity-local, fullmatch regex for the finger-pad collision geoms;
    used as the primary of the gripper contact-release sensor (see
    ``tasks.common.commands.ContactReleaseGate``)."""
    viewer_body: str
    """Body the viewer camera tracks."""
    reset_joint_pos: dict[str, float] | None = None
    """Absolute joint pose to reset to, if the robot's episodes should not
    start from its keyframe/home pose. Set this to the pose teleop recordings
    actually start from, so the offline demonstrations and evaluation share a
    start state. ``None`` keeps mjlab's default (reset to the keyframe).

    This deliberately does *not* move the keyframe: ``default_joint_pos`` is
    also the reference for the ``joint_pos_rel`` observation and the zero
    point of the joint-position action, so moving it would silently
    reinterpret every recorded demonstration."""
