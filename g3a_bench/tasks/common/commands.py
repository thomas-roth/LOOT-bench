"""Command-term helpers shared by the task categories.

The success metrics of the object-arrangement categories share a "gripper
released the object" gate on the robot's gripper joints.

The robot-specific values come from the :class:`~g3a_bench.robots.interface.ManipulatorInterface` via
``manipulator_wiring.wire_manipulator_into_env``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.sensor import ContactSensor


@dataclass(kw_only=True)
class GripperGateCommandCfg(CommandTermCfg):
    """Base for command cfgs whose success gate checks the gripper is open.

    Both fields are robot-specific and set by
    ``manipulator_wiring.wire_manipulator_into_env``; the placeholders here
    exist only so the robot-agnostic env cfg can be built first.
    """

    gripper_joints: tuple[str, ...] = ()
    """Set per-robot: all listed joints must be open to count as released."""
    gripper_open_threshold: float = 0.0
    """Set per-robot: joint position above which a joint counts as open."""


class GripperOpenGate:
    """Per-step "all gripper joints open" check for a command's success gate.

    Joint indices are resolved on first call: entity joint data is not
    available while command terms are constructed.
    """

    def __init__(self, cfg: GripperGateCommandCfg):
        assert cfg.gripper_joints, "manipulator wiring must set gripper_joints"
        self._joints = cfg.gripper_joints
        self._threshold = cfg.gripper_open_threshold
        self._joint_idx: list[int] | None = None

    def __call__(self, robot: Entity) -> torch.Tensor:
        if self._joint_idx is None:
            names = list(robot.joint_names)
            self._joint_idx = [names.index(j) for j in self._joints]
        q = robot.data.joint_pos[:, self._joint_idx]
        return (q > self._threshold).all(dim=-1)


GRIP_CONTACT_SENSOR_NAME = "grip_contact"
"""Sensor name every category's grip-release contact sensor is registered
under (see ``manipulator_wiring.wire_manipulator_into_env``'s
``grip_contact_sensor`` param).

**Unfiltered on purpose**: it has no secondary match, so it reports the pads
touching *anything*. That includes the other pad. On the DK1 the pad faces
meet at gripper joint -0.038 m and the joint travels to -0.045 m, so a gripper
closed on empty air reads 1.6 to 2.5 N here. The release gate is unharmed
(:class:`ContactReleaseGate` says why), but the number cannot answer "did it
touch the object". :data:`OBJECT_CONTACT_SENSOR_NAME` is the one that can.
"""

OBJECT_CONTACT_SENSOR_NAME = "object_contact"
"""Sensor name for the pads-against-the-manipulated-object contact sensor.

Same primary as ``grip_contact`` (the finger pads), plus a **secondary** match
on the object the task scores, so self-contact and table contact are filtered
out. Telemetry only: no success gate reads it. A category declares it by
adding the sensor and passing ``object_contact_sensor`` to
``registration.register_category``; a category with no single scored object
may leave it out, and the key then stays out of the info dict.
"""

# TODO: check with real robot
MIN_GRIP_FORCE = 0.1
"""[N] contact-force threshold above which a finger pad counts as still
gripping something."""


def pad_contact_force(sensor: ContactSensor) -> torch.Tensor:
    """(E,) strongest finger-pad contact force magnitude [N], per env.

    The one reduction of a pad contact sensor, used for both
    :data:`GRIP_CONTACT_SENSOR_NAME` and :data:`OBJECT_CONTACT_SENSOR_NAME`.
    :class:`ContactReleaseGate` thresholds it for the success gate, and
    ``qc/env.py`` publishes it as telemetry, so a run's reported contact force
    and its release decision read the same number. What differs between the two
    sensors is the *filter*, never the reduction.
    """
    force = sensor.data.force
    assert force is not None
    return force.norm(dim=-1).amax(dim=-1)


class ContactReleaseGate:
    """Per-step "pads not exerting grip force" check for a command's
    success gate.

    Complements :class:`GripperOpenGate`: the joint gate alone can be
    fooled by a wedged/diagonal grasp that crosses the open-joint threshold
    while still gripping. This reads actual contact force at the finger
    pads, independent of joint position. The sensor has no secondary
    (object) filter: it reports contact with anything, not just a tracked
    object, so this is coarser than "is it still holding the scored object".

    That coarseness includes **pad-against-pad** contact on a gripper closed
    on nothing. It costs nothing here, because this gate never runs alone:
    :class:`GripperOpenGate` requires the gripper joint above 2/3 of its
    stroke, and the DK1's pads cannot reach each other until 5/6 of the way
    closed. So every pose that can self-contact is already "not released" on
    the joint gate. Telemetry has no such second gate, which is why it reads
    :data:`OBJECT_CONTACT_SENSOR_NAME` instead.
    """

    def __init__(self, min_force: float = MIN_GRIP_FORCE):
        self._min_force = min_force

    def __call__(self, sensor: ContactSensor) -> torch.Tensor:
        return pad_contact_force(sensor) <= self._min_force
