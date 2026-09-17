"""Task command for the pi (stack) category.

There is no goal pose to sample: the level heights are per-task constants,
so at every resample the command only resets the success gate and spawns the
blocks from the fixed table (object ``i`` on cell ``i``, standing, with
+/-1 cm jitter via ``sample_fixed_spawns``; seed-prefix safe, so an easier
task's reset is a prefix of a harder task's).

Every step it recomputes which levels have a lintel (``success.py``
kernels, with the held block masked out per ADR-017) and feeds "all levels
standing" through the shared at-rest + released gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

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
from loot_bench.tasks.common.fixed_spawns import SPAWN_JITTER, sample_fixed_spawns
from loot_bench.tasks.common.scatter import write_upright_spawn
from loot_bench.tasks.common.success import OBJECT_VEL_THRESHOLD, SuccessGate
from loot_bench.tasks.stack.mdp.success import pi_levels_complete, pi_progress
from loot_bench.tasks.stack.variants import (
    BLOCK_HALF_EXTENTS,
    BLOCK_WIDTH,
    LEVEL_HEIGHT,
    PiVariantCfg,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

GRIP_REACH = 0.08
"""Max distance (m) from the grasp site for a block to count as held.

Shared with ``loot_bench/scripts/split_demos.py``, which reproduces the
held-block exclusion offline and would otherwise recompute a different
reward."""

GUIDE_UPRIGHT_SPACING = 0.036
"""Centre-to-centre spacing (m) of the ghost trilithon's uprights.

The guide is the only place a teleoperator sees the trilithon geometry, so it
draws the spacing the gripper actually needs: at least ``w/2 + 0.022``
= 32 mm to lower the second upright past the first with the 22 mm pad
half-width, at most ``L - w`` = 45 mm before the lintel stops overhanging
both, and 36 mm leaves ~4 mm of clearance on both that placement and the
end-to-end lintel grip (ADR-020)."""

_GOAL_PI_RGBA = (0.55, 0.75, 1.0, 1.0)  # base (pending) color, alpha unused

# Local-to-world rotation laying a block's long axis (local z) along world y.
_LINTEL_MAT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

PI_SPAWN_CELLS: tuple[tuple[float, float], ...] = (
    (0.22, 0.15),
    (0.22, -0.15),
    (0.22, 0.27),
    (0.22, -0.27),
    (0.35, 0.18),
    (0.35, -0.18),
    (0.35, 0.30),
    (0.35, -0.30),
    (0.45, 0.00),
)
"""Fixed spawn cells, object ``i`` takes cell ``i`` (3Pis uses all nine).

Clustered near the robot with the build zone left free, so a recording
session starts with the work area empty. 1Pi's reset is a prefix of
2Pis' and 2Pis' a prefix of 3Pis'. Minimum pairwise separation
0.12 m against the 0.088 m pitch an upright 20 mm-square prism needs
(``2 * 0.01 jitter + 2 * 0.01414 radius + 0.04 clearance``), so the table
carries real margin rather than sitting on the bound.
"""

BUILD_ZONE_X = (0.24, 0.32)
"""Build-zone x span (m), kept free of spawns.

Near the robot rather than far, which inverts the intuitive "work far, stage
near" arrangement: the reach dome tops out a three-level pi only near the
base (see ADR-019), so the pi stands where the arm can still place its
top lintel.
"""

BUILD_ZONE_HALF_WIDTH = 0.07
"""Build-zone |y| half-width (m), kept free of spawns."""

BUILD_HALO_MARGIN = 0.02
"""Extra margin (m) around the build zone, on top of object radius + jitter."""

GOAL_PI_CENTRE_XY = ((BUILD_ZONE_X[0] + BUILD_ZONE_X[1]) / 2, 0.0)
"""Base ghost position: build-zone middle, i.e. the middle of the spawn scatter."""

GOAL_PI_FREE_POS_XY = (0.0, -0.20)
"""FreePos ghost position: directly right of the robot, out of the build zone."""


def pi_spawn_is_clear(x: float, y: float, object_radius: float) -> bool:
    """Whether a jittered ``(x, y)`` spawn pose is clear of the build zone."""
    halo = object_radius + SPAWN_JITTER + BUILD_HALO_MARGIN
    x0, x1 = BUILD_ZONE_X
    return not (x0 - halo <= x <= x1 + halo and abs(y) <= BUILD_ZONE_HALF_WIDTH + halo)


def goal_pi_parts(
    levels: int, free_pos: bool = False
) -> list[tuple[int, str, np.ndarray, np.ndarray]]:
    """Ghost-trilithon parts for a ``levels``-level reference pi.

    Table-relative env-local poses. Base parks the ghost in the build-zone
    middle (the middle of the spawn scatter) as the build guide; ``free_pos``
    keeps it directly right of the robot, out of the way, so the operator
    builds anywhere. Lintel centres are the level targets, uprights
    ``GUIDE_UPRIGHT_SPACING`` apart: a legal build, copyable outright.
    Reach-probed for the L3 grasp band. Returns ``(level, part, center,
    mat)`` (``level`` 0-based).
    """
    half = np.array(BLOCK_HALF_EXTENTS) + MARKER_INFLATE
    cx, gy = GOAL_PI_FREE_POS_XY if free_pos else GOAL_PI_CENTRE_XY
    parts: list[tuple[int, str, np.ndarray, np.ndarray]] = []
    for level in range(levels):
        base = level * LEVEL_HEIGHT
        for sign in (-1.0, 1.0):
            parts.append(
                (
                    level,
                    f"upright_{sign}",
                    np.array(
                        [cx, gy + sign * GUIDE_UPRIGHT_SPACING / 2, base + half[2]]
                    ),
                    np.eye(3),
                )
            )
        parts.append(
            (
                level,
                "lintel",
                np.array([cx, gy, base + LEVEL_HEIGHT - BLOCK_WIDTH / 2]),
                _LINTEL_MAT,
            )
        )
    return parts


def get_command(env: ManagerBasedRlEnv, name: str) -> PiCommand:
    return cast(PiCommand, env.command_manager.get_term(name))


@dataclass(kw_only=True)
class PiCommandCfg(CommandTermCfg):
    block_names: tuple[str, ...]
    """All block entities, in the variant's (tail-extending) index order."""
    variant: PiVariantCfg
    grasp_cfg: SceneEntityCfg
    """Robot grasp site, used to find the held block; site set by wiring."""
    block_vel_threshold: float = OBJECT_VEL_THRESHOLD
    """Max linear speed (m/s) of any block to count success."""
    grip_reach: float = GRIP_REACH

    def build(self, env: ManagerBasedRlEnv) -> PiCommand:
        return PiCommand(self, env)


class PiCommand(CommandTerm):
    cfg: PiCommandCfg

    def __init__(self, cfg: PiCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        variant = cfg.variant
        self.variant = variant
        self.blocks: list[Entity] = [env.scene[n] for n in cfg.block_names]
        self.num_blocks = len(self.blocks)

        # The managers resolve a term's ``asset_cfg``; this command-owned one
        # is not, so resolve it here to get ``grasp_cfg.site_ids``.
        self.grasp_cfg = cfg.grasp_cfg
        self.grasp_cfg.resolve(env.scene)

        E, N, K = self.num_envs, self.num_blocks, variant.levels
        dev = self.device

        self._half_h = torch.full((N,), BLOCK_HALF_EXTENTS[2], device=dev)
        self._level_heights = torch.tensor(variant.level_centre_heights, device=dev)
        self._spawn_cells = torch.tensor(PI_SPAWN_CELLS, device=dev)
        assert self._spawn_cells.shape[0] >= N, (
            f"spawn table holds {self._spawn_cells.shape[0]} cells, variant needs {N}"
        )

        self.block_pos_w = torch.zeros(E, N, 3, device=dev)
        self.block_quat_w = torch.zeros(E, N, 4, device=dev)
        self.levels_complete = torch.zeros(E, K, device=dev, dtype=torch.bool)
        self.progress = torch.zeros(E, device=dev)
        self._gate = SuccessGate(E, dev, vel_threshold=cfg.block_vel_threshold)
        self._contact_gate = ContactReleaseGate()
        self._grip_sensor = cast(
            ContactSensor, env.scene.sensors[GRIP_CONTACT_SENSOR_NAME]
        )

        self.metrics["levels_frac"] = torch.zeros(E, device=dev)
        self.metrics["episode_success"] = torch.zeros(E, device=dev)

    @property
    def command(self) -> torch.Tensor:
        """(E, 1) top level's lintel centre height, constant per task
        (``command_obs=False``)."""
        return torch.full(
            (self.num_envs, 1),
            self.variant.level_centre_heights[-1],
            device=self.device,
        )

    # Helpers used by observation/reward/termination terms. ----------------

    @property
    def success(self) -> torch.Tensor:
        """(E,) float: every level standing, blocks at rest, released."""
        return self._gate.success

    @property
    def episode_success(self) -> torch.Tensor:
        """(E,) float: success latched since the last reset."""
        return self._gate.episode_success

    # CommandTerm interface. ------------------------------------------------

    def _update_metrics(self) -> None:
        self.block_pos_w = torch.stack(
            [b.data.root_link_pos_w for b in self.blocks], dim=1
        )
        self.block_quat_w = torch.stack(
            [b.data.root_link_quat_w for b in self.blocks], dim=1
        )
        table_z = self._env.scene.env_origins[:, 2]

        robot = self._env.scene["robot"]
        pads_free = self._contact_gate(self._grip_sensor)  # pads exert no force
        held = self._held_block_mask(robot, gripping=~pads_free)
        self.levels_complete = pi_levels_complete(
            self.block_pos_w[:, :, 2],
            self.block_quat_w,
            table_z,
            self._level_heights,
            self.variant.height_tol,
            exclude=held,
        )
        self.progress = pi_progress(self.levels_complete)
        # One predicate for reward and termination: the exact level count,
        # not a float comparison on progress.
        task_done = self.levels_complete.all(dim=-1)

        speed = torch.stack(
            [b.data.root_link_lin_vel_w for b in self.blocks], dim=1
        ).norm(dim=-1)  # (E, N)
        self._gate.update(task_done, speed, released=pads_free)

        self.metrics["levels_frac"] = self.progress
        self.metrics["episode_success"] = self.episode_success

    def _held_block_mask(self, robot: Entity, gripping: torch.Tensor) -> torch.Tensor:
        """(E, N) bool marking the single block currently held in the gripper.

        Held = the pads exert grip force and the block is the one nearest the
        grasp site, within ``grip_reach``. Excluded from level credit so a
        lintel merely carried to a level's height is not mistaken for a
        placed one; it counts again the instant it is released.
        """
        grasp_pos = robot.data.site_pos_w[:, self.grasp_cfg.site_ids].squeeze(1)
        dist = (self.block_pos_w - grasp_pos.unsqueeze(1)).norm(dim=-1)  # (E, N)
        nearest = dist.argmin(dim=-1)  # (E,)
        env = torch.arange(self.num_envs, device=self.device)
        held_env = gripping & (dist[env, nearest] <= self.cfg.grip_reach)
        mask = torch.zeros_like(dist, dtype=torch.bool)
        mask[env, nearest] = held_env
        return mask

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        origins = self._env.scene.env_origins[env_ids]

        self._gate.reset(env_ids)
        self.levels_complete[env_ids] = False

        # Object i takes cell i with +/-1 cm jitter (one rand draw per block,
        # in index order), so an easier task's reset is a prefix of a harder
        # task's (1Pi -> 2Pis -> 3Pis). Blocks spawn standing, so
        # the uprights come for free and each lintel costs one knock-over.
        xy, spawn_yaw = sample_fixed_spawns(self._spawn_cells, n, self.num_blocks)
        write_upright_spawn(self.blocks, xy, spawn_yaw, self._half_h, origins, env_ids)

    def _update_command(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        # No marker over the build zone itself: a plane at the top level's
        # height occluded everything under it, which is the whole structure.
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        for batch in env_indices:
            origin = self._env.scene.env_origins[batch].cpu().numpy()
            self._draw_goal_pi(visualizer, batch, origin)

    def _draw_goal_pi(
        self, visualizer: DebugVisualizer, batch: int, origin: np.ndarray
    ) -> None:
        """Ghost trilithon per level: base in the build-zone middle, FreePos
        directly right of the robot.

        Position-free success, so the ghost is one possible solution, not
        the required spot. Trilithon form (the T stays legal by criterion,
        not shown); tinted off ``levels_complete``, so a level greens when
        accepted anywhere. Buildable directly (see :func:`goal_pi_parts`).
        """
        half = np.array(BLOCK_HALF_EXTENTS) + MARKER_INFLATE
        success = bool(self.success[batch] > 0.5)
        reached = self.levels_complete[batch].tolist()
        for level, part, center, mat in goal_pi_parts(
            self.variant.levels, self.variant.free_pos
        ):
            color = marker_color(_GOAL_PI_RGBA, bool(reached[level]), success)
            center = origin + center
            label = f"goal_pi_{batch}_{level}_{part}"
            visualizer.add_box(
                center=center, size=half, mat=mat, color=color, label=label
            )
            # Red edges mark it as a virtual target, not a real block
            # (same convention as the domino slot markers).
            add_box_wireframe(
                visualizer,
                center=center,
                half_extents=half,
                mat=mat,
                color=OUTLINE_RGBA,
                label_prefix=f"{label}_outline",
            )
