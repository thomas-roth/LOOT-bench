"""Task command for the domino category.

At every resample the command:

1. draws a shared row start pose (domino-0's slot position) and row yaw —
   one draw, independent of the domino count — with the reach-band clamp
   sized for ``HARDEST_COUNT`` so the distribution never shifts with the
   variant's actual n (seed-prefix rule); slots extend from the start along
   the row direction,
2. scatters all dominoes upright elsewhere in the workspace on a jittered
   grid (collision-free by construction; per-domino draws in index order via
   ``ScatterGrid.sample_poses``).

Every step it re-checks each domino's placement (position + upright/row
alignment, ``success.py`` kernels) against its slot. ``KnockOver`` variants
additionally latch a build/topple ``phase`` once every slot is placed
simultaneously, then require the row to topple as a chain: every domino
fallen along the row axis, all leaning the same way, and their first-fall
steps monotonic along the row in that direction — both phases are gated by
the shared at-rest + released check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from g3a_bench.tasks.common.colors import PALETTE
from g3a_bench.tasks.common.commands import (
    GRIP_CONTACT_SENSOR_NAME,
    ContactReleaseGate,
    GripperGateCommandCfg,
    GripperOpenGate,
)
from g3a_bench.tasks.common.debug_markers import (
    MARKER_INFLATE,
    OUTLINE_RGBA,
    add_box_wireframe,
    marker_color,
)
from g3a_bench.tasks.common.scatter import ScatterGrid, write_upright_spawn
from g3a_bench.tasks.common.success import OBJECT_VEL_THRESHOLD, SuccessGate
from g3a_bench.tasks.domino.mdp.success import (
    chain_topple_success,
    domino_fallen_along_row,
    domino_upright_aligned,
)
from g3a_bench.tasks.domino.variants import (
    BASE_HALF_EXTENTS,
    DOMINO_COLORS,
    HARDEST_COUNT,
    SPACING,
    DominoVariantCfg,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

_EDGE_MARGIN = 0.02
_NEVER_FELL = 1.0e9
"""Sentinel fall-order value for a domino that has not chain-fallen yet."""
_PENDING_ALPHA = 0.75
"""Fill opacity for a slot's pending (not-yet-placed) marker. Higher than the
shared default (see debug_markers.PENDING_ALPHA) because the outline (below)
already signals "this is a ghost, not a real object" -- the fill is free to
sit close to the real domino color for a legible identity match."""


def get_command(env: ManagerBasedRlEnv, name: str) -> DominoCommand:
    return cast(DominoCommand, env.command_manager.get_term(name))


@dataclass(kw_only=True)
class DominoCommandCfg(GripperGateCommandCfg):
    domino_names: tuple[str, ...]
    """All domino entities, in row-slot index order."""
    variant: DominoVariantCfg
    workspace_x: tuple[float, float] = (0.22, 0.42)
    workspace_y: tuple[float, float] = (-0.18, 0.18)
    pos_threshold: float = 0.015
    """Max distance between a domino and its slot to count as placed (× margin)."""
    upright_threshold_deg: float = 15.0
    """Max tilt/row-misalignment to count as placed (× margin)."""
    fallen_tilt_deg: float = 45.0
    """Min tilt to count a domino as toppled (KnockOver)."""
    domino_vel_threshold: float = OBJECT_VEL_THRESHOLD
    """Max linear speed (m/s) of any domino to count success."""

    def build(self, env: ManagerBasedRlEnv) -> DominoCommand:
        return DominoCommand(self, env)


class DominoCommand(CommandTerm):
    cfg: DominoCommandCfg

    def __init__(self, cfg: DominoCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        variant = cfg.variant
        self.variant = variant
        self.dominoes: list[Entity] = [env.scene[n] for n in cfg.domino_names]
        self.num_dominoes = len(self.dominoes)
        G = self.num_dominoes

        E, dev = self.num_envs, self.device

        halves = variant.half_extents
        self._half_h = torch.tensor([h[2] for h in halves], device=dev)
        # Bounding-circle radius of the thickest (index-0) domino — the
        # widest footprint any spawn slot must clear.
        object_radius = math.hypot(BASE_HALF_EXTENTS[0], BASE_HALF_EXTENTS[1])
        self._scatter = ScatterGrid.tiling(
            cfg.workspace_x, cfg.workspace_y, object_radius=object_radius, device=dev
        )
        assert self._scatter.num_cells >= G

        self.slot_goal_pos = torch.zeros(E, G, 3, device=dev)  # world frame
        self.piece_for_slot = (
            torch.arange(G, device=dev, dtype=torch.long).unsqueeze(0).repeat(E, 1)
        )
        self.row_yaw = torch.zeros(E, device=dev)
        self.domino_pos_w = torch.zeros(E, G, 3, device=dev)
        self.domino_quat_w = torch.zeros(E, G, 4, device=dev)
        self.slot_at_goal = torch.zeros(E, G, device=dev)
        self.phase = torch.zeros(E, device=dev)
        self.progress = torch.zeros(E, device=dev)
        self.fall_order = torch.full((E, G), _NEVER_FELL, device=dev)
        self._gate = SuccessGate(E, dev, vel_threshold=cfg.domino_vel_threshold)
        self._gripper_gate = GripperOpenGate(cfg)
        self._contact_gate = ContactReleaseGate()
        self._grip_sensor = cast(
            ContactSensor, env.scene.sensors[GRIP_CONTACT_SENSOR_NAME]
        )

        self.metrics["placed_frac"] = torch.zeros(E, device=dev)
        self.metrics["phase"] = torch.zeros(E, device=dev)
        self.metrics["episode_success"] = torch.zeros(E, device=dev)

    @property
    def command(self) -> torch.Tensor:
        """(E, G*7): per-slot env-local target pose (pos xyz, quat wxyz).

        Every slot in a row shares the same upright/row-aligned target
        orientation. Object-major (slot index order == domino index order),
        so a shorter variant's command is a tail-truncation of a longer
        one's (demo-cutoff rule).
        """
        goals = self.slot_goal_pos - self._env.scene.env_origins.unsqueeze(1)
        zeros = torch.zeros_like(self.row_yaw)
        slot_quat = quat_from_euler_xyz(zeros, zeros, self.row_yaw)  # (E, 4)
        slot_quat = slot_quat.unsqueeze(1).expand(-1, self.num_dominoes, -1)
        return torch.cat([goals, slot_quat], dim=-1).reshape(self.num_envs, -1)

    # Helpers used by observation/reward/termination terms. ----------------

    @property
    def success(self) -> torch.Tensor:
        """(E,) float: task done, dominoes at rest, gripper open."""
        return self._gate.success

    @property
    def episode_success(self) -> torch.Tensor:
        """(E,) float: success latched since the last reset."""
        return self._gate.episode_success

    def slot_piece_pos(self) -> torch.Tensor:
        """Positions of the domino assigned to each slot, (E, G, 3)."""
        return self._gather_slot(self.domino_pos_w)

    def _gather_slot(self, per_piece: torch.Tensor) -> torch.Tensor:
        """Per-piece values (E, N, D) -> per-slot values via piece_for_slot."""
        idx = self.piece_for_slot.unsqueeze(-1).expand(-1, -1, per_piece.shape[-1])
        return torch.gather(per_piece, 1, idx)

    # CommandTerm interface. ------------------------------------------------

    def _update_metrics(self) -> None:
        self.domino_pos_w = torch.stack(
            [d.data.root_link_pos_w for d in self.dominoes], dim=1
        )
        self.domino_quat_w = torch.stack(
            [d.data.root_link_quat_w for d in self.dominoes], dim=1
        )  # (E, G, 4) wxyz

        if not self.variant.ordered:
            # Order-free assignment: match dominoes to slots by proximity so
            # any domino may fill any slot. Only valid when all dominoes are
            # identical (variants.py forbids different + not ordered). No
            # registered variant uses this yet; the knob exists for manual
            # experiments.
            dist = torch.cdist(self.slot_goal_pos, self.domino_pos_w)
            self.piece_for_slot = self._greedy_match(dist)

        pos = self.slot_piece_pos()
        q = self._gather_slot(self.domino_quat_w)
        margin = self.variant.margin
        error = torch.norm(self.slot_goal_pos - pos, dim=-1)
        at_goal = (error < self.cfg.pos_threshold * margin) & domino_upright_aligned(
            q, self.row_yaw, self.cfg.upright_threshold_deg * margin
        )
        self.slot_at_goal = at_goal.float()
        all_placed = at_goal.all(dim=1)
        placed_frac = self.slot_at_goal.mean(dim=1)

        speed = torch.stack(
            [d.data.root_link_lin_vel_w for d in self.dominoes], dim=1
        ).norm(dim=-1)  # (E, G)
        released = self._gripper_gate(self._env.scene["robot"]) & self._contact_gate(
            self._grip_sensor
        )

        if self.variant.knock_over:
            self.phase = torch.maximum(self.phase, all_placed.float())
            fallen, sign = domino_fallen_along_row(
                self.domino_quat_w, self.row_yaw, self.cfg.fallen_tilt_deg
            )
            # Latch first-fall steps, but only during the topple phase —
            # accidents while building must not count toward the chain.
            step = self._env.episode_length_buf.float().unsqueeze(-1)
            newly = fallen & (self.phase > 0.5).unsqueeze(-1)
            self.fall_order = torch.where(
                newly & (self.fall_order >= _NEVER_FELL), step, self.fall_order
            )
            task_done = (self.phase > 0.5) & chain_topple_success(
                fallen, sign, self.fall_order
            )
            fallen_frac = fallen.float().mean(dim=-1)
            self.progress = torch.where(
                self.phase > 0.5, 0.5 + 0.5 * fallen_frac, 0.5 * placed_frac
            )
        else:
            task_done = all_placed
            self.progress = placed_frac

        self._gate.update(task_done, speed, released)

        self.metrics["placed_frac"] = placed_frac
        self.metrics["phase"] = self.phase
        self.metrics["episode_success"] = self.episode_success

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        dev = self.device
        cfg = self.cfg
        G = self.num_dominoes
        origins = self._env.scene.env_origins[env_ids]

        self._gate.reset(env_ids)
        self.slot_at_goal[env_ids] = 0.0
        self.phase[env_ids] = 0.0
        self.fall_order[env_ids] = _NEVER_FELL
        self.piece_for_slot[env_ids] = (
            torch.arange(G, device=dev).unsqueeze(0).repeat(n, 1)
        )

        x0, x1 = cfg.workspace_x
        y0, y1 = cfg.workspace_y

        # --- Row start pose + yaw: one shared draw, independent of G, so
        # this stays a seed prefix across variants of different n. ---
        u = torch.rand(n, 4, device=dev)
        sign = torch.where(u[:, 0] < 0.5, 1.0, -1.0)
        yaw = sign * math.pi / 2 + (u[:, 1] - 0.5) * 1.2
        dir_xy = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)  # (n, 2)
        # Clamp for a HARDEST_COUNT row (anchored at the start), so a
        # shorter row's start-position range never depends on its actual n.
        extent = dir_xy * SPACING * (HARDEST_COUNT - 1)
        lo = torch.tensor([x0, y0], device=dev) - extent.clamp(max=0.0) + _EDGE_MARGIN
        hi = torch.tensor([x1, y1], device=dev) - extent.clamp(min=0.0) - _EDGE_MARGIN
        start = lo + u[:, 2:4] * (hi - lo).clamp_min(0.0)
        self.row_yaw[env_ids] = yaw

        steps = torch.arange(G, device=dev, dtype=torch.float32)
        slot_xy = start.unsqueeze(1) + steps.view(1, G, 1) * SPACING * dir_xy.unsqueeze(
            1
        )
        slot_z = self._half_h[0].expand(n, G).unsqueeze(-1)
        self.slot_goal_pos[env_ids] = torch.cat(
            [slot_xy, slot_z], dim=-1
        ) + origins.unsqueeze(1)

        # --- Scatter all dominoes upright on distinct jittered grid cells
        # (per-domino draws in index order). ---
        xy, spawn_yaw = self._scatter.sample_poses(n, G)
        write_upright_spawn(
            self.dominoes, xy, spawn_yaw, self._half_h, origins, env_ids
        )

    def _update_command(self) -> None:
        pass

    @staticmethod
    def _greedy_match(dist: torch.Tensor) -> torch.Tensor:
        """Greedy one-to-one matching for a batched square cost matrix.

        ``dist`` is (E, k, k) with rows = slots and columns = candidate
        dominoes; returns the (E, k) column index per row. Greedy on the
        global minimum recovers the perfect matching whenever one exists
        within ``pos_threshold`` — dominoes cannot overlap and neighboring
        slots are at least one domino width apart, so no domino can satisfy
        two slots — which is all the success metric needs.
        """
        num_envs, k, _ = dist.shape
        cost = dist.clone()
        assign = torch.zeros(num_envs, k, dtype=torch.long, device=dist.device)
        env_idx = torch.arange(num_envs, device=dist.device)
        for _ in range(k):
            flat = cost.flatten(1).argmin(dim=1)
            row, col = flat // k, flat % k
            assign[env_idx, row] = col
            cost[env_idx, row, :] = torch.inf
            cost[env_idx, :, col] = torch.inf
        return assign

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        for batch in env_indices:
            yaw = float(self.row_yaw[batch])
            c, s = math.cos(yaw), math.sin(yaw)
            mat = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            success = bool(self.success[batch] > 0.5)
            for i, half in enumerate(self.variant.half_extents):
                goal = self.slot_goal_pos[batch, i].cpu().numpy()
                placed = bool(self.slot_at_goal[batch, i] > 0.5)
                base_rgba = PALETTE[DOMINO_COLORS[i]]
                color = marker_color(
                    base_rgba, placed, success, pending_alpha=_PENDING_ALPHA
                )
                fill_size = np.array(half) + MARKER_INFLATE
                visualizer.add_box(
                    center=goal,
                    size=fill_size,
                    mat=mat,
                    color=color,
                    label=f"slot_goal_{batch}_{i}",
                )
                # Edge wireframe, not a second filled box: a bigger *filled*
                # box would enclose/hide the fill box entirely rather than
                # frame it (see debug_markers.OUTLINE_RADIUS).
                add_box_wireframe(
                    visualizer,
                    center=goal,
                    half_extents=fill_size,
                    mat=mat,
                    color=OUTLINE_RGBA,
                    label_prefix=f"slot_goal_outline_{batch}_{i}",
                )
