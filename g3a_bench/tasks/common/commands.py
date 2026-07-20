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
``grip_contact_sensor`` param)."""

MIN_GRIP_FORCE = 0.5
"""[N] contact-force threshold above which a finger pad counts as still
gripping something. Not derived from actuator specs -- a conservative
starting value; verify empirically (log force during a real wedge grasp vs.
a genuine open release) and retune if wrong."""


class ContactReleaseGate:
    """Per-step "pads not exerting grip force" check for a command's
    success gate.

    Complements :class:`GripperOpenGate`: the joint gate alone can be
    fooled by a wedged/diagonal grasp that crosses the open-joint threshold
    while still gripping. This reads actual contact force at the finger
    pads, independent of joint position. The sensor has no secondary
    (object) filter -- it reports contact with anything, not just a
    tracked object -- so this is deliberately coarser than "is it still
    holding the scored object"; see the design doc for the trade-off.
    """

    def __init__(self, min_force: float = MIN_GRIP_FORCE):
        self._min_force = min_force

    def __call__(self, sensor: ContactSensor) -> torch.Tensor:
        force = sensor.data.force
        assert force is not None
        return force.norm(dim=-1).amax(dim=-1) <= self._min_force
