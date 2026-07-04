"""Insertion command: goal bores, peg scatter and insertion success.

One command term drives all three insertion kinds (single peg, two-sheet, shape
sorter). It owns:

- the world pose of every bore mouth (from :func:`socket_layout`, so geometry
  and success metric stay in sync),
- the per-reset scatter of pegs (and the offset of the movable top sheet), and
- the per-socket insertion test (axial depth + lateral centring + tilt, plus
  sheet alignment for the two-sheet variant).

Peg ``i`` is assigned to socket ``i``; extra pegs are distractors. Socket order
is the variant's socket order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from g3a_bench.tasks.insertion.assets import (
    BOARD_POS,
    SHEET_HALF_Z,
    TRAY_POS,
    TRAY_SPAWN_HALF,
    socket_layout,
)
from g3a_bench.tasks.insertion.variants import PEG_HALF_LEN, InsertionVariantCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass(kw_only=True)
class InsertionCommandCfg(CommandTermCfg):
    board_name: str
    peg_names: tuple[str, ...]
    """All peg entities; the first ``num_sockets`` are assigned to sockets."""
    variant: InsertionVariantCfg
    sheet_name: str | None = None
    """Movable top sheet entity (two-sheet variant only)."""
    sheet_align_tol: float = 0.010
    """Max top-sheet/bottom-hole offset to count as aligned."""
    peg_vel_threshold: float = 0.05
    """Max linear speed (m/s) of any assigned peg to count success."""
    gripper_open_threshold: float = -0.015
    """Min gripper_left joint position (m) to count success (> -0.015 = ≥2/3 open)."""

    def build(self, env: ManagerBasedRlEnv) -> InsertionCommand:
        return InsertionCommand(self, env)


class InsertionCommand(CommandTerm):
    cfg: InsertionCommandCfg

    def __init__(self, cfg: InsertionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        variant = cfg.variant
        self.variant = variant
        self.pegs: list[Entity] = [env.scene[n] for n in cfg.peg_names]
        self.sheet: Entity | None = (
            env.scene[cfg.sheet_name] if cfg.sheet_name is not None else None
        )
        self.num_pegs = len(self.pegs)
        self.num_sockets = variant.num_sockets
        assert self.num_pegs >= self.num_sockets

        E, S, N = self.num_envs, self.num_sockets, self.num_pegs
        dev = self.device

        frames = socket_layout(variant)
        mouth_local = torch.tensor(
            np.stack([f.mouth for f in frames]), device=dev, dtype=torch.float32
        )
        axis_local = torch.tensor(
            np.stack([f.axis_out for f in frames]), device=dev, dtype=torch.float32
        )
        board_pos = torch.tensor(BOARD_POS, device=dev, dtype=torch.float32)
        # Board is a fixed prop pinned to the env origin (identity orientation),
        # so world mouth poses are constant per env.
        origins = self._env.scene.env_origins  # (E, 3)
        self.socket_mouth_w = (
            origins.unsqueeze(1) + board_pos + mouth_local.unsqueeze(0)
        )  # (E, S, 3)
        self.socket_axis_w = axis_local.unsqueeze(0).expand(E, S, 3).contiguous()

        self.insert_depth = variant.insert_depth
        self.lateral_tol = variant.lateral_tol
        self.tilt_cos = math.cos(math.radians(variant.tilt_tol_deg))

        self.peg_pos_w = torch.zeros(E, N, 3, device=dev)
        self.inserted = torch.zeros(E, S, device=dev)
        self.success = torch.zeros(E, device=dev)
        self.episode_success = torch.zeros(E, device=dev)
        self._half_len = PEG_HALF_LEN

        # Resolved lazily: joint data isn't available at __init__ time.
        self._gripper_joint_idx: int = -1

        # Pegs are spawned upright into the parts tray ("Werkbox"): a jittered
        # grid over the tray's spawn region (env-local; env origins are added at
        # spawn time). Same for every variant.
        tx, ty, _ = TRAY_POS
        hx, hy = TRAY_SPAWN_HALF
        x0, x1 = tx - hx, tx + hx
        y0, y1 = ty - hy, ty + hy
        nx, ny = 2, 4
        cx = torch.linspace(x0, x1, nx + 1, device=dev)
        cy = torch.linspace(y0, y1, ny + 1, device=dev)
        self._cells = torch.cartesian_prod(
            (cx[:-1] + cx[1:]) / 2, (cy[:-1] + cy[1:]) / 2
        )
        self._cell_jitter = 0.4 * torch.tensor(
            [(x1 - x0) / nx / 2, (y1 - y0) / ny / 2], device=dev
        )
        assert self._cells.shape[0] >= N

        self.metrics["lateral_error"] = torch.zeros(E, device=dev)
        self.metrics["num_inserted"] = torch.zeros(E, device=dev)
        self.metrics["all_inserted"] = torch.zeros(E, device=dev)
        self.metrics["episode_success"] = torch.zeros(E, device=dev)
        if self.sheet is not None:
            self.metrics["sheet_align_error"] = torch.zeros(E, device=dev)

    # Helpers. --------------------------------------------------------------

    def assigned_peg_pos(self) -> torch.Tensor:
        """Position of the peg assigned to each socket, (E, S, 3)."""
        return self.peg_pos_w[:, : self.num_sockets]

    @property
    def command(self) -> torch.Tensor:
        mouth = (
            self.socket_mouth_w - self._env.scene.env_origins.unsqueeze(1)
        ).reshape(self.num_envs, -1)
        axis = self.socket_axis_w.reshape(self.num_envs, -1)
        return torch.cat([mouth, axis], dim=-1)

    # CommandTerm interface. ------------------------------------------------

    def _update_metrics(self) -> None:
        self.peg_pos_w = torch.stack([p.data.root_link_pos_w for p in self.pegs], dim=1)
        peg_quat = torch.stack([p.data.root_link_quat_w for p in self.pegs], dim=1)

        pos = self.assigned_peg_pos()  # (E, S, 3)
        quat = peg_quat[:, : self.num_sockets]
        axis = self.socket_axis_w
        rel = pos - self.socket_mouth_w
        axial = torch.sum(rel * axis, dim=-1)  # signed, + is outside
        center_depth = -axial
        lateral = torch.norm(rel - axial.unsqueeze(-1) * axis, dim=-1)

        w, x, y, z = quat.unbind(-1)
        peg_axis = torch.stack(
            [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=-1
        )
        cos_tilt = torch.sum(peg_axis * axis, dim=-1).abs()
        tip_depth = center_depth + self._half_len * cos_tilt

        inserted = (
            (tip_depth >= self.insert_depth)
            & (lateral < self.lateral_tol)
            & (cos_tilt > self.tilt_cos)
        )

        if self.sheet is not None:
            sheet_xy = self.sheet.data.root_link_pos_w[:, :2]
            align_err = torch.norm(sheet_xy - self.socket_mouth_w[:, 0, :2], dim=-1)
            inserted = inserted & (align_err < self.cfg.sheet_align_tol).unsqueeze(1)
            self.metrics["sheet_align_error"] = align_err

        self.inserted = inserted.float()
        all_in = inserted.all(dim=1)

        if self._gripper_joint_idx < 0:
            robot = self._env.scene["robot"]
            self._gripper_joint_idx = list(robot.joint_names).index("gripper_left")

        # Velocity gate: every assigned peg moving slowly.
        peg_lin_vel_w = torch.stack(
            [p.data.root_link_lin_vel_w for p in self.pegs], dim=1
        )  # (E, N, 3)
        assigned_vel = peg_lin_vel_w[:, : self.num_sockets]
        pegs_still = (
            assigned_vel.norm(dim=-1).max(dim=-1).values <= self.cfg.peg_vel_threshold
        )

        # Gripper-open gate.
        robot = self._env.scene["robot"]
        gripper_q = robot.data.joint_pos[:, self._gripper_joint_idx]
        gripper_open = gripper_q > self.cfg.gripper_open_threshold

        all_conditions = all_in & pegs_still & gripper_open
        self.success = all_conditions.float()
        self.episode_success = torch.maximum(self.episode_success, self.success)

        self.metrics["lateral_error"] = lateral.mean(dim=1)
        self.metrics["num_inserted"] = self.inserted.sum(dim=1)
        self.metrics["all_inserted"] = all_in.float()
        self.metrics["episode_success"] = self.episode_success

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        dev = self.device
        N = self.num_pegs
        origins = self._env.scene.env_origins[env_ids]

        self.success[env_ids] = 0.0
        self.episode_success[env_ids] = 0.0
        self.inserted[env_ids] = 0.0

        # Scatter pegs upright on distinct jittered cells inside the parts tray.
        pick = torch.argsort(torch.rand(n, self._cells.shape[0], device=dev), dim=1)[
            :, :N
        ]
        xy = self._cells[pick] + (torch.rand(n, N, 2, device=dev) - 0.5) * 2.0 * (
            self._cell_jitter
        )
        spawn_yaw = torch.rand(n, N, device=dev) * 2 * math.pi
        zeros = torch.zeros(n, device=dev)
        for i, peg in enumerate(self.pegs):
            z = (self._half_len + 0.002 + origins[:, 2]).unsqueeze(-1)
            p = torch.cat([xy[:, i, :] + origins[:, :2], z], dim=-1)
            quat = quat_from_euler_xyz(zeros, zeros, spawn_yaw[:, i])
            peg.write_root_link_pose_to_sim(
                torch.cat([p, quat], dim=-1), env_ids=env_ids
            )
            peg.write_root_link_velocity_to_sim(
                torch.zeros(n, 6, device=dev), env_ids=env_ids
            )

        # Place the movable top sheet, hole offset from the bottom hole.
        if self.sheet is not None:
            bottom = self.socket_mouth_w[env_ids, 0]  # (n, 3)
            ang = torch.rand(n, device=dev) * 2 * math.pi
            off = 0.03 + torch.rand(n, device=dev) * 0.02
            sx = bottom[:, 0] + off * torch.cos(ang)
            sy = bottom[:, 1] + off * torch.sin(ang)
            sz = origins[:, 2] + 2 * SHEET_HALF_Z + SHEET_HALF_Z + 0.001
            pos = torch.stack([sx, sy, sz], dim=-1)
            quat = quat_from_euler_xyz(zeros, zeros, torch.zeros(n, device=dev))
            self.sheet.write_root_link_pose_to_sim(
                torch.cat([pos, quat], dim=-1), env_ids=env_ids
            )
            self.sheet.write_root_link_velocity_to_sim(
                torch.zeros(n, 6, device=dev), env_ids=env_ids
            )

    def _update_command(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        for batch in env_indices:
            for s in range(self.num_sockets):
                mouth = self.socket_mouth_w[batch, s].cpu().numpy()
                done = bool(self.inserted[batch, s] > 0.5)
                color = (0.2, 0.8, 0.2, 0.5) if done else (0.9, 0.9, 0.2, 0.4)
                visualizer.add_sphere(
                    center=mouth, radius=0.015, color=color, label=f"bore_{batch}_{s}"
                )


def get_command(env: ManagerBasedRlEnv, name: str) -> InsertionCommand:
    return cast(InsertionCommand, env.command_manager.get_term(name))
