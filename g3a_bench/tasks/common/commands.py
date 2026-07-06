"""Command-term helpers shared by the task categories.

The success metrics of the object-arrangement categories (cube, insertion)
share two mechanics: a "gripper released the object" gate on the robot's
gripper joints, and a jittered-grid scatter that places free bodies without
overlap on reset. Both live here so the categories cannot drift apart.
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


class JitteredCellGrid:
    """Scatter positions on a jittered grid, collision-free by construction.

    The region is split into ``nx * ny`` cells; ``sample`` picks distinct
    cells per row and jitters each pick by up to 40% of the cell half-extent,
    so two picks can never land on top of each other.
    """

    def __init__(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        nx: int,
        ny: int,
        device: torch.device | str,
    ):
        x0, x1 = x_range
        y0, y1 = y_range
        cx = torch.linspace(x0, x1, nx + 1, device=device)
        cy = torch.linspace(y0, y1, ny + 1, device=device)
        self._cells = torch.cartesian_prod(
            (cx[:-1] + cx[1:]) / 2, (cy[:-1] + cy[1:]) / 2
        )
        self._jitter = 0.4 * torch.tensor(
            [(x1 - x0) / nx / 2, (y1 - y0) / ny / 2], device=device
        )

    @property
    def num_cells(self) -> int:
        return self._cells.shape[0]

    def sample(self, n: int, count: int) -> torch.Tensor:
        """xy positions on ``count`` distinct jittered cells per row, (n, count, 2)."""
        assert count <= self.num_cells
        rank = torch.rand(n, self.num_cells, device=self._cells.device)
        pick = torch.argsort(rank, dim=1)[:, :count]
        xy = self._cells[pick]
        return xy + (torch.rand_like(xy) - 0.5) * 2.0 * self._jitter
