"""Goal-arrangement command for the cube task category.

At every resample the command:

1. samples goal poses for the arrangement (stack bases / domino row / push
   zones behind the maze wall),
2. scatters all pieces (goal pieces + distractors) on a jittered grid
   (collision-free by construction) over the workspace — restricted to the
   near side of the maze wall for the "push" arrangement,
3. randomly re-assigns goal slots among interchangeable pieces (same form and
   color), and
4. exposes the slot goal positions as the policy command.

Terminology: a *slot* is one position in the goal arrangement (slot order is
bottom-to-top within a stack, stacks concatenated; row order for dominos). A
*piece* is a free-body scene entity. `piece_for_slot` maps slots to pieces and
only differs from identity for interchangeable pieces.
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

from g3a_bench.tasks.common.commands import (
    GripperGateCommandCfg,
    GripperOpenGate,
    JitteredCellGrid,
)
from g3a_bench.tasks.common.debug_markers import MARKER_INFLATE, marker_color
from g3a_bench.tasks.cube.assets import (
    PALETTE,
    PUSH_BAND_HALF_Y,
    PUSH_GOAL_LANE_SPACING,
    PUSH_GOAL_X,
    PUSH_SPAWN_X,
)
from g3a_bench.tasks.cube.variants import (
    FORM_HALF_EXTENTS,
    CubeVariantCfg,
    form_height,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


def get_command(env: ManagerBasedRlEnv, name: str) -> PieceArrangementCommand:
    return cast(PieceArrangementCommand, env.command_manager.get_term(name))


@dataclass(kw_only=True)
class PieceArrangementCommandCfg(GripperGateCommandCfg):
    piece_names: tuple[str, ...]
    """All piece entities; the first `variant.num_goal_pieces` are the goal
    pieces in slot order, the rest are distractors."""
    variant: CubeVariantCfg
    workspace_x: tuple[float, float] = (0.22, 0.40)
    workspace_y: tuple[float, float] = (-0.18, 0.18)
    pos_threshold: float = 0.015
    """Max distance between piece and slot goal to count as placed."""
    upright_threshold_deg: float = 15.0
    """Max tilt (and yaw misalignment for dominos) to count as placed."""
    piece_vel_threshold: float = 0.05
    """Max linear speed (m/s) of any goal piece to count success."""

    def build(self, env: ManagerBasedRlEnv) -> PieceArrangementCommand:
        return PieceArrangementCommand(self, env)


class PieceArrangementCommand(CommandTerm):
    cfg: PieceArrangementCommandCfg

    def __init__(self, cfg: PieceArrangementCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        variant = cfg.variant
        self.variant = variant
        self.pieces: list[Entity] = [env.scene[n] for n in cfg.piece_names]
        self.num_pieces = len(self.pieces)
        self.num_slots = variant.num_goal_pieces
        assert self.num_pieces >= self.num_slots

        E, G, N = self.num_envs, self.num_slots, self.num_pieces
        dev = self.device

        self.slot_goal_pos = torch.zeros(E, G, 3, device=dev)  # world frame
        self.piece_for_slot = (
            torch.arange(G, device=dev, dtype=torch.long).unsqueeze(0).repeat(E, 1)
        )
        self.row_yaw = torch.zeros(E, device=dev)  # domino row direction
        self.piece_pos_w = torch.zeros(E, N, 3, device=dev)
        self.slot_at_goal = torch.zeros(E, G, device=dev)
        self.success = torch.zeros(E, device=dev)
        self.episode_success = torch.zeros(E, device=dev)

        # Slot heights -> goal z per slot (cumulative within each stack).
        goal_z = []
        for stack in variant.stacks:
            z = 0.0
            for p in stack:
                h = form_height(p.form)
                if variant.arrangement == "stack":
                    goal_z.append(z + h / 2)
                    z += h
                else:
                    goal_z.append(h / 2)
        self._slot_goal_z = torch.tensor(goal_z, device=dev)
        self._slot_to_stack = torch.tensor(
            variant.slot_to_stack, device=dev, dtype=torch.long
        )
        self._num_stacks = len(variant.stacks)

        # Interchangeable groups: slots holding identical pieces.
        self._groups: list[torch.Tensor] = []
        if variant.interchangeable:
            seen: dict[tuple, list[int]] = {}
            for i, p in enumerate(variant.goal_pieces):
                seen.setdefault((p.form, p.color), []).append(i)
            for idx in seen.values():
                if len(idx) > 1:
                    self._groups.append(torch.tensor(idx, device=dev, dtype=torch.long))

        # Piece half heights for spawn z.
        self._piece_half_h = torch.tensor(
            [form_height(p.form) / 2 for p in variant.pieces], device=dev
        )

        # Scatter grid (jittered cells) covering the spawn region: the whole
        # workspace, or only the near side of the maze wall for "push".
        if variant.arrangement == "push":
            self._spawn_grid = JitteredCellGrid(
                PUSH_SPAWN_X,
                (-PUSH_BAND_HALF_Y, PUSH_BAND_HALF_Y),
                nx=1,  # single column: the near band is too shallow for two
                ny=4,
                device=dev,
            )
        else:
            self._spawn_grid = JitteredCellGrid(
                cfg.workspace_x, cfg.workspace_y, nx=3, ny=6, device=dev
            )
        assert self._spawn_grid.num_cells >= N

        self._gripper_gate = GripperOpenGate(cfg)

        self.metrics["position_error"] = torch.zeros(E, device=dev)
        self.metrics["num_placed"] = torch.zeros(E, device=dev)
        self.metrics["all_placed"] = torch.zeros(E, device=dev)
        self.metrics["episode_success"] = torch.zeros(E, device=dev)

    @property
    def command(self) -> torch.Tensor:
        goals = self.slot_goal_pos - self._env.scene.env_origins.unsqueeze(1)
        out = goals.reshape(self.num_envs, -1)
        if self.variant.arrangement == "domino":
            out = torch.cat(
                [
                    out,
                    torch.cos(self.row_yaw[:, None]),
                    torch.sin(self.row_yaw[:, None]),
                ],
                dim=-1,
            )
        return out

    # Helpers used by observation/reward/termination terms. ----------------

    def slot_piece_pos(self) -> torch.Tensor:
        """Positions of the piece assigned to each slot, (E, G, 3)."""
        return self._gather_slot(self.piece_pos_w)

    def _gather_slot(self, per_piece: torch.Tensor) -> torch.Tensor:
        """Per-piece values (E, N, D) -> per-slot values via piece_for_slot."""
        idx = self.piece_for_slot.unsqueeze(-1).expand(-1, -1, per_piece.shape[-1])
        return torch.gather(per_piece, 1, idx)

    # CommandTerm interface. ------------------------------------------------

    def _update_metrics(self) -> None:
        self.piece_pos_w = torch.stack(
            [p.data.root_link_pos_w for p in self.pieces], dim=1
        )

        error = torch.norm(self.slot_goal_pos - self.slot_piece_pos(), dim=-1)
        at_goal = error < self.cfg.pos_threshold

        # Upright (and row alignment) checks for dominos.
        if self.variant.arrangement == "domino":
            quats = torch.stack([p.data.root_link_quat_w for p in self.pieces], dim=1)
            q = self._gather_slot(quats)  # (E, G, 4) wxyz
            w, x, y, z = q.unbind(-1)
            # Body z-axis dot world z.
            up_cos = 1.0 - 2.0 * (x * x + y * y)
            cos_tol = math.cos(math.radians(self.cfg.upright_threshold_deg))
            # Body x-axis (thin side) must align with the row direction
            # (mod pi) so the dominos can knock each other over.
            bx_x = 1.0 - 2.0 * (y * y + z * z)
            bx_y = 2.0 * (x * y + w * z)
            dir_x = torch.cos(self.row_yaw)[:, None]
            dir_y = torch.sin(self.row_yaw)[:, None]
            align = (bx_x * dir_x + bx_y * dir_y).abs()
            at_goal = at_goal & (up_cos > cos_tol) & (align > cos_tol)

        self.slot_at_goal = at_goal.float()
        all_placed = at_goal.all(dim=1)

        # Velocity gate: every goal piece moving slowly.
        piece_lin_vel_w = torch.stack(
            [p.data.root_link_lin_vel_w for p in self.pieces], dim=1
        )  # (E, N, 3)
        goal_vel = self._gather_slot(piece_lin_vel_w)  # (E, G, 3)
        pieces_still = (
            goal_vel.norm(dim=-1).max(dim=-1).values <= self.cfg.piece_vel_threshold
        )

        gripper_open = self._gripper_gate(self._env.scene["robot"])

        self.success = (all_placed & pieces_still & gripper_open).float()
        self.episode_success = torch.maximum(self.episode_success, self.success)

        self.metrics["position_error"] = error.mean(dim=1)
        self.metrics["num_placed"] = self.slot_at_goal.sum(dim=1)
        self.metrics["all_placed"] = all_placed.float()
        self.metrics["episode_success"] = self.episode_success

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        dev = self.device
        cfg = self.cfg
        G, N = self.num_slots, self.num_pieces
        origins = self._env.scene.env_origins[env_ids]

        self.success[env_ids] = 0.0
        self.episode_success[env_ids] = 0.0
        self.slot_at_goal[env_ids] = 0.0

        x0, x1 = cfg.workspace_x
        y0, y1 = cfg.workspace_y

        # --- Goal poses. ---
        if self.variant.arrangement == "stack":
            # Stack bases on a coarse jittered grid for separation.
            sx = torch.linspace(x0 + 0.03, x1 - 0.03, 2, device=dev)
            sy = torch.linspace(y0 + 0.04, y1 - 0.04, 3, device=dev)
            base_cells = torch.cartesian_prod(sx, sy)  # (6, 2)
            pick = torch.argsort(torch.rand(n, base_cells.shape[0], device=dev), dim=1)[
                :, : self._num_stacks
            ]
            bases = base_cells[pick]  # (n, num_stacks, 2)
            bases = bases + (torch.rand_like(bases) - 0.5) * 0.03
            slot_xy = bases[:, self._slot_to_stack, :]  # (n, G, 2)
        elif self.variant.arrangement == "push":
            # One goal zone per piece in the band behind the maze wall: fixed
            # y lanes are shuffled among the zones and jittered, x is sampled
            # freely within the band.
            gx0, gx1 = PUSH_GOAL_X
            lanes = (
                torch.arange(G, device=dev, dtype=torch.float32) - (G - 1) / 2
            ) * PUSH_GOAL_LANE_SPACING
            perm = torch.argsort(torch.rand(n, G, device=dev), dim=1)
            gy = lanes[perm] + (torch.rand(n, G, device=dev) - 0.5) * 0.03
            gx = gx0 + 0.02 + torch.rand(n, G, device=dev) * (gx1 - gx0 - 0.04)
            slot_xy = torch.stack([gx, gy], dim=-1)
        else:
            # Domino row: mostly lateral so the row always fits in x.
            spacing = self.variant.domino_spacing
            sign = torch.where(torch.rand(n, device=dev) < 0.5, 1.0, -1.0)
            yaw = sign * math.pi / 2 + (torch.rand(n, device=dev) - 0.5) * 1.2
            dir_xy = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
            extent = dir_xy.abs() * spacing * (G - 1)  # (n, 2)
            lo = torch.tensor([x0, y0], device=dev) + extent / 2 + 0.02
            hi = torch.tensor([x1, y1], device=dev) - extent / 2 - 0.02
            center = lo + torch.rand(n, 2, device=dev) * (hi - lo).clamp_min(0.0)
            steps = torch.arange(G, device=dev, dtype=torch.float32) - (G - 1) / 2
            slot_xy = center.unsqueeze(1) + steps.view(
                1, G, 1
            ) * spacing * dir_xy.unsqueeze(1)
            self.row_yaw[env_ids] = yaw

        goal = torch.cat(
            [slot_xy, self._slot_goal_z.expand(n, G).unsqueeze(-1)], dim=-1
        )
        self.slot_goal_pos[env_ids] = goal + origins.unsqueeze(1)

        # --- Slot assignment among interchangeable pieces. ---
        assign = torch.arange(G, device=dev).unsqueeze(0).repeat(n, 1)
        for group in self._groups:
            perm = torch.argsort(torch.rand(n, group.shape[0], device=dev), dim=1)
            assign[:, group] = group[perm]
        self.piece_for_slot[env_ids] = assign

        # --- Scatter all pieces on distinct jittered grid cells. ---
        xy = self._spawn_grid.sample(n, N)
        spawn_yaw = torch.rand(n, N, device=dev) * 2 * math.pi

        zeros = torch.zeros(n, device=dev)
        for i, piece in enumerate(self.pieces):
            z = (self._piece_half_h[i] + 0.002 + origins[:, 2]).unsqueeze(-1)
            pos = torch.cat([xy[:, i, :] + origins[:, :2], z], dim=-1)
            quat = quat_from_euler_xyz(zeros, zeros, spawn_yaw[:, i])
            pose = torch.cat([pos, quat], dim=-1)
            piece.write_root_link_pose_to_sim(pose, env_ids=env_ids)
            piece.write_root_link_velocity_to_sim(
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
            if self.variant.arrangement == "domino":
                # Dominos must align their thin side with the row direction.
                yaw = float(self.row_yaw[batch])
                c, s = math.cos(yaw), math.sin(yaw)
                mat = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            else:
                # Stack success is yaw-free.
                mat = np.eye(3)
            for i, piece in enumerate(self.variant.goal_pieces):
                goal = self.slot_goal_pos[batch, i].cpu().numpy()
                placed = bool(self.slot_at_goal[batch, i] > 0.5)
                color = marker_color(PALETTE[piece.color], placed, success)
                half = FORM_HALF_EXTENTS[piece.form]
                label = f"slot_goal_{batch}_{i}"
                if piece.form == "cylinder":
                    radius, hz = half
                    axis = np.array([0.0, 0.0, hz + MARKER_INFLATE])
                    visualizer.add_cylinder(
                        start=goal - axis,
                        end=goal + axis,
                        radius=radius + MARKER_INFLATE,
                        color=color,
                        label=label,
                    )
                else:
                    visualizer.add_box(
                        center=goal,
                        size=np.array(half) + MARKER_INFLATE,
                        mat=mat,
                        color=color,
                        label=label,
                    )
