"""Peg-insertion command: goal bores, peg scatter and insertion success.

ARCHIVED (unregistered, untested): see ``README.md``. One command term drives
both insertion kinds (single peg, two-sheet). It owns:

- the world pose of every bore mouth (from :func:`socket_layout`, so geometry
  and success metric stay in sync). Boards are fixed props, so their mouths
  are constant,
- the per-reset scatter of pegs and the offset of the movable top sheet, and
- the insertion test (pure criteria in ``success.py``): instantaneous axial
  depth + lateral centring + tilt (+ sheet alignment).

Peg ``i`` is assigned to socket ``i``, in the variant's socket order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from g3a_bench.tasks.common.commands import GripperGateCommandCfg, GripperOpenGate
from g3a_bench.tasks.common.debug_markers import MARKER_INFLATE, marker_color
from g3a_bench.tasks.common.scatter import ScatterGrid
from g3a_bench.tasks.common.success import (
    OBJECT_VEL_THRESHOLD,
    SuccessGate,
)
from g3a_bench.tasks.peg_insertion.assets import (
    BOARD_POS,
    PALETTE,
    SHEET_TOP_ROOT_Z,
    frame_axes,
    socket_layout,
)
from g3a_bench.tasks.peg_insertion.mdp.success import (
    peg_axial_lateral,
    peg_axis_cos_tilt,
    peg_inserted,
)
from g3a_bench.tasks.peg_insertion.variants import (
    PEG_HALF_LEN,
    SHAPE_GEOMS,
    InsertionVariantCfg,
    Shape,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

# Peg spawn reach band (env-local xy) and the largest peg's bounding radius.
_WORKSPACE_X = (0.13, 0.27)
_WORKSPACE_Y = (-0.16, 0.16)
_OBJECT_RADIUS = 0.022


@dataclass(kw_only=True)
class InsertionCommandCfg(GripperGateCommandCfg):
    board_name: str
    peg_names: tuple[str, ...]
    """One peg entity per socket, in socket order."""
    variant: InsertionVariantCfg
    sheet_name: str | None = None
    """Movable top sheet entity (two-sheet variant only)."""
    sheet_align_tol: float = 0.010
    """Max top-sheet/bottom-hole offset to count as aligned."""
    peg_vel_threshold: float = OBJECT_VEL_THRESHOLD
    """Max linear speed (m/s) of any assigned peg to count success."""

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
        self.board: Entity = env.scene[cfg.board_name]
        self.num_pegs = len(self.pegs)
        self.num_sockets = variant.num_sockets
        assert self.num_pegs == self.num_sockets

        E, S = self.num_envs, self.num_sockets
        dev = self.device

        frames = socket_layout(variant)
        self._mouth_local = torch.tensor(
            np.stack([f.mouth for f in frames]), device=dev, dtype=torch.float32
        )
        self._axis_local = torch.tensor(
            np.stack([f.axis_out for f in frames]), device=dev, dtype=torch.float32
        )
        board_pos = torch.tensor(BOARD_POS, device=dev, dtype=torch.float32)
        origins = self._env.scene.env_origins  # (E, 3)
        # Fixed prop pinned to the env origin (identity orientation), so
        # world mouth poses are constant per env.
        self.socket_mouth_w = (
            origins.unsqueeze(1) + board_pos + self._mouth_local.unsqueeze(0)
        )  # (E, S, 3)
        self.socket_axis_w = self._axis_local.unsqueeze(0).expand(E, S, 3).contiguous()

        # Fixed-board insertion test tolerances and per-socket ghost-peg
        # marker data for debug vis: peg shape/color and the bore
        # cross-section basis (columns c0, c1, axis_out), matching the walls.
        self.insert_depth = variant.insert_depth
        self.lateral_tol = variant.lateral_tol
        self.tilt_cos = math.cos(math.radians(variant.tilt_tol_deg))
        self._socket_shapes: tuple[Shape, ...] = tuple(f.shape for f in frames)
        self._socket_colors = tuple(PALETTE[f.color] for f in frames)
        self._socket_mats = [np.stack(frame_axes(f.axis_out), axis=1) for f in frames]

        self.peg_pos_w = torch.zeros(E, self.num_pegs, 3, device=dev)
        self.peg_quat_w = torch.zeros(E, self.num_pegs, 4, device=dev)
        self.board_pos_w = torch.zeros(E, 3, device=dev)
        self.board_quat_w = torch.zeros(E, 4, device=dev)
        self.sheet_pos_w = torch.zeros(E, 3, device=dev)
        self.sheet_quat_w = torch.zeros(E, 4, device=dev)
        self.inserted = torch.zeros(E, S, device=dev)
        self._gate = SuccessGate(E, dev, vel_threshold=cfg.peg_vel_threshold)
        self._gripper_gate = GripperOpenGate(cfg)
        self._half_len = PEG_HALF_LEN

        # Pegs are spawned upright into the reach band, env-local (env origins
        # are added at spawn time), on a jittered grid sized for the largest
        # peg's bounding radius.
        self._scatter = ScatterGrid.tiling(
            _WORKSPACE_X, _WORKSPACE_Y, object_radius=_OBJECT_RADIUS, device=dev
        )
        assert self._scatter.num_cells >= self.num_pegs

        self.metrics["lateral_error"] = torch.zeros(E, device=dev)
        self.metrics["num_inserted"] = torch.zeros(E, device=dev)
        self.metrics["all_inserted"] = torch.zeros(E, device=dev)
        self.metrics["episode_success"] = torch.zeros(E, device=dev)
        if self.sheet is not None:
            self.metrics["sheet_align_error"] = torch.zeros(E, device=dev)

    # Helpers. --------------------------------------------------------------

    @property
    def success(self) -> torch.Tensor:
        """(E,) float: task done, scored pegs at rest, gripper open."""
        return self._gate.success

    @property
    def episode_success(self) -> torch.Tensor:
        """(E,) float: success latched since the last reset."""
        return self._gate.episode_success

    @property
    def axis_local(self) -> torch.Tensor:
        """(S, 3) outward bore axes in the board frame."""
        return self._axis_local

    def assigned_peg_pos(self) -> torch.Tensor:
        """Position of the peg assigned to each socket, (E, S, 3)."""
        return self.peg_pos_w[:, : self.num_sockets]

    def scored_fraction(self) -> torch.Tensor:
        """Task progress in [0, 1], (E,): fraction of sockets filled."""
        return self.inserted.sum(dim=1) / self.num_sockets

    @property
    def command(self) -> torch.Tensor:
        mouth = (
            self.socket_mouth_w - self._env.scene.env_origins.unsqueeze(1)
        ).reshape(self.num_envs, -1)
        axis = self.socket_axis_w.reshape(self.num_envs, -1)
        return torch.cat([mouth, axis], dim=-1)

    # CommandTerm interface. ------------------------------------------------

    def _update_metrics(self) -> None:
        self.board_pos_w = self.board.data.root_link_pos_w
        self.board_quat_w = self.board.data.root_link_quat_w
        self.peg_pos_w = torch.stack([p.data.root_link_pos_w for p in self.pegs], dim=1)
        self.peg_quat_w = torch.stack(
            [p.data.root_link_quat_w for p in self.pegs], dim=1
        )
        S = self.num_sockets

        pos = self.assigned_peg_pos()  # (E, S, 3)
        quat = self.peg_quat_w[:, :S]
        axis = self.socket_axis_w
        axial, lateral = peg_axial_lateral(pos, self.socket_mouth_w, axis)
        cos_tilt = peg_axis_cos_tilt(quat, axis)

        inserted = peg_inserted(
            axial,
            lateral,
            cos_tilt,
            half_len=self._half_len,
            insert_depth=self.insert_depth,
            lateral_tol=self.lateral_tol,
            tilt_cos=self.tilt_cos,
        )
        if self.sheet is not None:
            self.sheet_pos_w = self.sheet.data.root_link_pos_w
            self.sheet_quat_w = self.sheet.data.root_link_quat_w
            align_err = torch.norm(
                self.sheet_pos_w[:, :2] - self.socket_mouth_w[:, 0, :2], dim=-1
            )
            inserted = inserted & (align_err < self.cfg.sheet_align_tol).unsqueeze(1)
            self.metrics["sheet_align_error"] = align_err

        self.inserted = inserted.float()
        all_in = inserted.all(dim=1)

        peg_lin_vel_w = torch.stack(
            [p.data.root_link_lin_vel_w for p in self.pegs], dim=1
        )  # (E, N, 3)
        assigned_speed = peg_lin_vel_w[:, :S].norm(dim=-1)  # (E, S)

        released = self._gripper_gate(self._env.scene["robot"])
        self._gate.update(all_in, assigned_speed, released)

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

        self._gate.reset(env_ids)
        self.inserted[env_ids] = 0.0

        # Scatter pegs upright on distinct jittered cells (per-peg draws in
        # index order via sample_poses).
        xy, spawn_yaw = self._scatter.sample_poses(n, N)
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
            sz = origins[:, 2] + SHEET_TOP_ROOT_Z
            pos = torch.stack([sx, sy, sz], dim=-1)
            quat = quat_from_euler_xyz(zeros, zeros, zeros)
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
            success = bool(self.success[batch] > 0.5)
            for s in range(self.num_sockets):
                mouth = self.socket_mouth_w[batch, s].cpu().numpy()
                axis = self.socket_axis_w[batch, s].cpu().numpy()
                done = bool(self.inserted[batch, s] > 0.5)
                color = marker_color(self._socket_colors[s], done, success)
                # Ghost peg at the minimum successful insertion: tip at
                # insert_depth below the mouth, extending one peg length out.
                tip = mouth - axis * self.insert_depth
                top = tip + axis * (2.0 * self._half_len)
                shape = self._socket_shapes[s]
                geom = SHAPE_GEOMS[shape]
                label = f"bore_{batch}_{s}"
                if shape in ("square", "rect"):
                    half_xy = geom.rect_half or (geom.inradius, geom.inradius)
                    visualizer.add_box(
                        center=(tip + top) / 2.0,
                        size=np.array([half_xy[0], half_xy[1], self._half_len])
                        + MARKER_INFLATE,
                        mat=self._socket_mats[s],
                        color=color,
                        label=label,
                    )
                else:
                    # Triangle/hexagon have no polygon-prism primitive; a
                    # cylinder of the circumradius is the closest hull.
                    visualizer.add_cylinder(
                        start=tip,
                        end=top,
                        radius=geom.circumradius + MARKER_INFLATE,
                        color=color,
                        label=label,
                    )


def get_command(env: ManagerBasedRlEnv, name: str) -> InsertionCommand:
    return cast(InsertionCommand, env.command_manager.get_term(name))
