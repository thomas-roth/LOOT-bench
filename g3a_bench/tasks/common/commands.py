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
