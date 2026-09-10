"""Robot-agnostic manipulator description for benchmark tasks.

Tasks use this interface plus wire_manipulator_into_env; concrete values live
in each loot_bench.robots.<robot> package.
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

    Torque cap, not a rate limit: ``DifferentialIKAction.apply_actions``
    commands ``q_target = q_current + dq`` fresh every physics substep, so
    ``dq`` is the instantaneous PD error and the actuator delivers at most
    ``kp * max_dq``. ADR-016's original ``velocity_limit * timestep`` reading
    (0.04 rad on the DK1) caps joint2 at 4 Nm against a 28 Nm effort limit --
    below the 5.84 Nm needed just to hold the start pose, so the arm sags.
    Velocity limit is the wrong quantity for a term that re-anchors every
    substep; it would be right if the target were integrated instead.

    Derive it as the smallest ``effort_limit / stiffness`` over the arm
    actuators: the position error past which the motor is already saturated,
    so letting the target sit further ahead buys no more force. Must arrive
    through this interface -- mjlab's ``BuiltinPositionActuatorCfg`` values
    don't reach the compiled model in a form the term can read back."""
    ee_subtree: str
    """Body name whose subtree is monitored for illegal ground contact."""
    pad_geoms: str
    """Fullmatch regex for finger-pad geoms; used as primary of contact-release."""
    viewer_body: str
    """Body the viewer camera tracks."""
    reset_joint_pos: dict[str, float] | None = None
    """Absolute start pose, or None to start from the entity's own keyframe."""
