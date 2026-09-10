"""Domino command. Spawns dominoes from the fixed table, checks slot placement, latches topple.

The row's centre x is pinned at ``DOMINO_ROW_X``, so spawns and row separate
in x by construction: no keep-out mask, no rejection loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from loot_bench.tasks.common.colors import COLOR_NAMES, COLOR_VALUES
from loot_bench.tasks.common.commands import (
    GRIP_CONTACT_SENSOR_NAME,
    ContactReleaseGate,
)
from loot_bench.tasks.common.debug_markers import (
    MARKER_INFLATE,
    OUTLINE_RGBA,
    add_box_wireframe,
    marker_color,
)
from loot_bench.tasks.common.fixed_spawns import sample_fixed_spawns
from loot_bench.tasks.common.scatter import write_upright_spawn
from loot_bench.tasks.common.success import OBJECT_VEL_THRESHOLD, SuccessGate
from loot_bench.tasks.domino.mdp.success import (
    chain_topple_success,
    domino_fallen_along_row,
    domino_upright_aligned,
)
from loot_bench.tasks.domino.variants import (
    HARDEST_COUNT,
    SPACING,
    DominoVariantCfg,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

DOMINO_SPAWN_CELLS: tuple[tuple[float, float], ...] = (
    (0.210, 0.05),
    (0.210, -0.05),
    (0.305, 0.10),
    (0.305, -0.10),
    (0.210, 0.15),
    (0.210, -0.15),
)
"""Fixed spawn cells, object ``i`` takes cell ``i``.

Minimum pairwise separation 0.100, above the gripper-derived pitch. Every
cell's jittered poses stay left of ``DOMINO_LANE_EDGE`` while every row slot
stays right of it, so spawns and row separate in x by construction.
"""

DOMINO_ROW_X = 0.40
"""Pinned row-centre x (m): the HARDEST_COUNT row's centre sits here."""

DOMINO_ROW_TILT = 0.35
"""Max row yaw deviation (rad) from the row axis (``RandomRow``)."""

DOMINO_ROW_Y_RANGE = 0.08
"""Half-range (m) of the sampled row-centre y (``RandomRow``).

Sized so the HARDEST_COUNT row's endpoints stay on the table and inside the
radial reach bound at any tilt within ``DOMINO_ROW_TILT``.
"""

DOMINO_LANE_EDGE = 0.340
"""Row-lane near edge (m): spawns (plus jitter and footprint) stay left of it,
slots (minus footprint) stay right of it. The bound is ``DOMINO_ROW_X`` minus
``sin(DOMINO_ROW_TILT)`` times the row half-length, minus the domino
bounding-circle radius.
"""


def domino_spawn_is_clear(x: float, y: float, object_radius: float) -> bool:
    """Whether a jittered ``(x, y)`` spawn pose is clear of the row lane."""
    return x + object_radius <= DOMINO_LANE_EDGE


def domino_row_slots(
    centre: torch.Tensor, yaw: torch.Tensor, count: int
) -> torch.Tensor:
    """Row slot xy positions (local frame) for a pinned-centre row.

    Slots are anchored at the start of a ``HARDEST_COUNT`` row whose centre is
    ``centre``: slot ``k`` sits ``(k - (HARDEST_COUNT - 1) / 2) * SPACING``
    along ``yaw``, so a shorter variant's slots are the longer one's first
    ``count`` (seed-prefix rule) while the worst-case x excursion never
    depends on ``count``.

    Args:
        centre: (..., 2) row-centre xy (x is ``DOMINO_ROW_X``).
        yaw: (...,) row direction.
        count: number of slots.

    Returns:
        (..., count, 2) slot xy positions.
    """
    direction = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
    steps = torch.arange(count, device=centre.device, dtype=direction.dtype)
    steps = steps - (HARDEST_COUNT - 1) / 2.0
    shape = (1,) * (centre.dim() - 1) + (count, 1)
    return centre.unsqueeze(-2) + steps.reshape(shape) * SPACING * direction.unsqueeze(
        -2
    )


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
class DominoCommandCfg(CommandTermCfg):
    domino_names: tuple[str, ...]
    """All domino entities, in row-slot index order."""
    variant: DominoVariantCfg
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
        self._spawn_cells = torch.tensor(DOMINO_SPAWN_CELLS, device=dev)
        assert self._spawn_cells.shape[0] >= G, (
            f"spawn table holds {self._spawn_cells.shape[0]} cells, variant needs {G}"
        )

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
        self._contact_gate = ContactReleaseGate()
        self._grip_sensor = cast(
            ContactSensor, env.scene.sensors[GRIP_CONTACT_SENSOR_NAME]
        )

        self.metrics["placed_frac"] = torch.zeros(E, device=dev)
        self.metrics["phase"] = torch.zeros(E, device=dev)
        self.metrics["episode_success"] = torch.zeros(E, device=dev)

    @property
    def command(self) -> torch.Tensor:
        """(E, G*7): per-slot env-local target pose, in slot order."""
        goals = self.slot_goal_pos - self._env.scene.env_origins.unsqueeze(1)
        zeros = torch.zeros_like(self.row_yaw)
        slot_quat = quat_from_euler_xyz(zeros, zeros, self.row_yaw)  # (E, 4)
        slot_quat = slot_quat.unsqueeze(1).expand(-1, self.num_dominoes, -1)
        return torch.cat([goals, slot_quat], dim=-1).reshape(self.num_envs, -1)

    # Helpers used by observation/reward/termination terms. ----------------

    @property
    def success(self) -> torch.Tensor:
        """(E,) float: task done, dominoes at rest, released."""
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
            # Order-free matching by proximity. Needs identical dominoes.
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
        released = self._contact_gate(self._grip_sensor)

        if self.variant.knock_over:
            self.phase = torch.maximum(self.phase, all_placed.float())
            fallen, sign = domino_fallen_along_row(
                self.domino_quat_w, self.row_yaw, self.cfg.fallen_tilt_deg
            )
            # Latch first-fall steps only during the topple phase.
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
        G = self.num_dominoes
        origins = self._env.scene.env_origins[env_ids]

        self._gate.reset(env_ids)
        self.slot_at_goal[env_ids] = 0.0
        self.phase[env_ids] = 0.0
        self.fall_order[env_ids] = _NEVER_FELL
        self.piece_for_slot[env_ids] = (
            torch.arange(G, device=dev).unsqueeze(0).repeat(n, 1)
        )

        if not self.variant.random_row:
            # --- Fixed row: along +y, with the HARDEST_COUNT row centred at
            # (DOMINO_ROW_X, 0). Anchored at the start, so a shorter
            # variant's slots are still the longer one's first n. ---
            yaw = torch.full((n,), math.pi / 2, device=dev)
            centre_y = torch.zeros(n, device=dev)
        else:
            # --- Row centre y + yaw: one shared draw, independent of G, so
            # this stays a seed prefix across variants of different n. Only
            # x is pinned. ---
            u = torch.rand(n, 3, device=dev)
            sign = torch.where(u[:, 0] < 0.5, 1.0, -1.0)
            yaw = sign * math.pi / 2 + (u[:, 1] - 0.5) * 2.0 * DOMINO_ROW_TILT
            centre_y = (u[:, 2] - 0.5) * 2.0 * DOMINO_ROW_Y_RANGE
        self.row_yaw[env_ids] = yaw

        centre = torch.stack(
            [torch.full((n,), DOMINO_ROW_X, device=dev), centre_y], dim=-1
        )
        slot_xy = domino_row_slots(centre, yaw, G)
        slot_z = self._half_h[0].expand(n, G).unsqueeze(-1)
        self.slot_goal_pos[env_ids] = torch.cat(
            [slot_xy, slot_z], dim=-1
        ) + origins.unsqueeze(1)

        # --- Spawn domino i on cell i with +/-1 cm jitter (per-domino draws
        # in index order). Spawns and row separate in x by construction, so
        # there is no keep-out mask to consult. ---
        xy, spawn_yaw = sample_fixed_spawns(self._spawn_cells, n, G)
        write_upright_spawn(
            self.dominoes, xy, spawn_yaw, self._half_h, origins, env_ids
        )

    def _update_command(self) -> None:
        pass

    @staticmethod
    def _greedy_match(dist: torch.Tensor) -> torch.Tensor:
        """Greedy one-to-one slot to domino matching for (E, k, k) costs."""
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
                base_rgba = COLOR_VALUES[COLOR_NAMES[i]]
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
