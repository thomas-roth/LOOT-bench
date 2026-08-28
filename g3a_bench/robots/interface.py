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
    grasp_site: str
    """Site between the fingertips. Used for reach rewards/observations, and
    as the frame differential IK drives (``tasks.common.actions``)."""
    arm_actuators: tuple[str, ...]
    """Actuator name expressions the end-effector delta term controls -- the
    IK chain, and nothing else.

    Declared rather than derived. "Every actuator that is not the gripper" was
    rejected because it makes the residue load-bearing: a later robot's torso
    or wrist-camera joint would silently join the IK chain (ADR-016). An empty
    tuple fails when the action term is built."""
    gripper_actuators: tuple[str, ...]
    """Actuator name expressions the normalized gripper term controls. The
    complement of :attr:`arm_actuators`, but stated rather than inferred."""
    ik_max_dq: float
    """[rad] how far differential IK's joint target may sit ahead of the
    measured joint position. Becomes the term's ``max_dq``.

    **This is a torque cap, not a rate limit**, and that is not obvious.
    mjlab's ``DifferentialIKAction.apply_actions`` commands
    ``q_target = q_current + dq`` afresh on *every physics substep*, so ``dq``
    never accumulates into a trajectory -- it is the instantaneous PD position
    error, and the actuator therefore delivers at most ``kp * max_dq``.

    ADR-016 specifies ``velocity_limit * timestep`` (0.04 rad on the DK1) on
    the reading that ``max_dq`` rate-limits the target. Measured, that value
    caps joint2 at ``100 * 0.04`` = 4 Nm against its 28 Nm effort limit --
    below the 5.84 Nm it takes to hold the arm at the start pose, so the arm
    cannot stand up. The velocity limit is simply the wrong quantity for a
    term that re-anchors to the measurement each substep; it would be the
    right one if the target were integrated.

    Derive it instead as the smallest ``effort_limit / stiffness`` over the
    arm actuators: the position error past which the motor is already
    saturated, so letting the target sit further ahead buys no more force.
    It still has to arrive through this interface -- mjlab's
    ``BuiltinPositionActuatorCfg`` values do not reach the compiled model in a
    form the term could read back."""
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
    """Absolute joint pose episodes start from, or ``None`` to start from the
    entity's own keyframe. Set it to the pose the robot rests in during
    teleoperation: the recorded demos and the evaluated episodes must start
    from the same state, otherwise a policy trained on the demos is evaluated
    from a state it never saw. Consumed by ``manipulator_wiring.
    wire_manipulator_into_env``, which swaps in
    ``tasks.common.events.reset_joints_to_pose``."""
