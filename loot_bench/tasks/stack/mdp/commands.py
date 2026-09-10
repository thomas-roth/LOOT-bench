"""Task command for the stack category.

There is no goal pose to sample — the target height is a per-task constant —
so at every resample the command only resets the success gate and scatters
the blocks upright on a jittered grid (collision-free by construction,
seed-prefix safe: no shared draws, then per-block draws in index order via
``ScatterGrid.sample_poses``).

Every step it recomputes the height-only success criterion (``success.py``
kernels) and feeds it through the shared at-rest + released gate.
"""

from __future__ import annotations

import math
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
from loot_bench.tasks.common.scatter import ScatterGrid, write_upright_spawn
from loot_bench.tasks.common.success import OBJECT_VEL_THRESHOLD, SuccessGate
from loot_bench.tasks.stack.mdp.success import (
    block_top_heights,
    stack_progress,
    stack_reached,
)
from loot_bench.tasks.stack.variants import StackVariantCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

# Debug-vis plane color while the target height is not reached yet.
_PLANE_RGBA = (0.95, 0.95, 0.95, 1.0)

SHOW_GOAL_STACK = False
"""Draw the ghost target stack (:meth:`StackCommand._draw_goal_stack`).

Switched on by ``scripts/play.py`` only: a build guide for interactive play,
not something a teleop recording should show."""

GRIP_REACH = 0.08
"""Max distance (m) from the grasp site for a block to count as held.

Shared with ``scripts/split_demos.py``, which reproduces the held-block
exclusion offline and would otherwise recompute a different reward."""

# Clearance (m) between the scatter region and the guide tower, so the guide
# never overlaps the blocks it describes.
_GOAL_STACK_MARGIN = 0.06
_GOAL_STACK_RGBA = (0.55, 0.75, 1.0, 1.0)  # base (pending) color, alpha unused


def get_command(env: ManagerBasedRlEnv, name: str) -> StackCommand:
    return cast(StackCommand, env.command_manager.get_term(name))


@dataclass(kw_only=True)
class StackCommandCfg(CommandTermCfg):
    block_names: tuple[str, ...]
    """All block entities, in the variant's (tail-extending) index order."""
    variant: StackVariantCfg
    grasp_cfg: SceneEntityCfg
    """Robot grasp site, used to find the held block; site set by wiring."""
    workspace_x: tuple[float, float] = (0.21, 0.44)
    workspace_y: tuple[float, float] = (-0.21, 0.21)
    block_vel_threshold: float = OBJECT_VEL_THRESHOLD
    """Max linear speed (m/s) of any block to count success."""
    grip_reach: float = GRIP_REACH

    def build(self, env: ManagerBasedRlEnv) -> StackCommand:
        return StackCommand(self, env)


class StackCommand(CommandTerm):
    cfg: StackCommandCfg

    def __init__(self, cfg: StackCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        variant = cfg.variant
        self.variant = variant
        self.blocks: list[Entity] = [env.scene[n] for n in cfg.block_names]
        self.num_blocks = len(self.blocks)

        # The managers resolve a term's ``asset_cfg``; this command-owned one
        # is not, so resolve it here to get ``grasp_cfg.site_ids``.
        self.grasp_cfg = cfg.grasp_cfg
        self.grasp_cfg.resolve(env.scene)

        E, N = self.num_envs, self.num_blocks
        dev = self.device

        halves = [b.half_extents for b in variant.blocks]
        self._half_h = torch.tensor([h[2] for h in halves], device=dev)
        # Bounding-circle radius of the widest footprint (collision-free
        # scatter must hold at any yaw).
        object_radius = max(math.hypot(h[0], h[1]) for h in halves)
        self._scatter = ScatterGrid.tiling(
            cfg.workspace_x, cfg.workspace_y, object_radius=object_radius, device=dev
        )
        assert self._scatter.num_cells >= N

        self.block_pos_w = torch.zeros(E, N, 3, device=dev)
        self.block_quat_w = torch.zeros(E, N, 4, device=dev)
        self.block_reached = torch.zeros(E, N, device=dev, dtype=torch.bool)
        self.progress = torch.zeros(E, device=dev)
        self._gate = SuccessGate(E, dev, vel_threshold=cfg.block_vel_threshold)
        self._contact_gate = ContactReleaseGate()
        self._grip_sensor = cast(
            ContactSensor, env.scene.sensors[GRIP_CONTACT_SENSOR_NAME]
        )

        self.metrics["max_height_frac"] = torch.zeros(E, device=dev)
        self.metrics["episode_success"] = torch.zeros(E, device=dev)

    @property
    def command(self) -> torch.Tensor:
        """(E, 1) target height — constant per task (``command_obs=False``)."""
        return torch.full(
            (self.num_envs, 1), self.variant.target_height, device=self.device
        )

    # Helpers used by observation/reward/termination terms. ----------------

    @property
    def success(self) -> torch.Tensor:
        """(E,) float: target height reached, blocks at rest, released."""
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
        target = self.variant.target_height
        top_z = block_top_heights(self.block_pos_w[:, :, 2], self._half_h)

        self.block_reached = stack_reached(
            top_z, table_z, target, self.variant.height_tol
        )
        robot = self._env.scene["robot"]
        pads_free = self._contact_gate(self._grip_sensor)  # pads exert no force
        held = self._held_block_mask(robot, gripping=~pads_free)
        task_done = self.block_reached.any(dim=-1)
        self.progress = stack_progress(top_z, table_z, target, exclude=held)

        speed = torch.stack(
            [b.data.root_link_lin_vel_w for b in self.blocks], dim=1
        ).norm(dim=-1)  # (E, N)
        released = pads_free
        self._gate.update(task_done, speed, released)

        self.metrics["max_height_frac"] = self.progress
        self.metrics["episode_success"] = self.episode_success

    def _held_block_mask(self, robot: Entity, gripping: torch.Tensor) -> torch.Tensor:
        """(E, N) bool marking the single block currently held in the gripper.

        Held = the pads exert grip force and the block is the one nearest the
        grasp site, within ``grip_reach``. Excluded from progress so a block
        merely lifted to the target height is not mistaken for a placed stack;
        it counts again the instant it is released.
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
        self.block_reached[env_ids] = False

        # No shared randomness; per-block draws in index order (seed-prefix
        # rule: an easier task's reset is a prefix of a harder task's).
        xy, spawn_yaw = self._scatter.sample_poses(n, self.num_blocks)
        write_upright_spawn(self.blocks, xy, spawn_yaw, self._half_h, origins, env_ids)

    def _update_command(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        cfg = self.cfg
        cx = (cfg.workspace_x[0] + cfg.workspace_x[1]) / 2
        cy = (cfg.workspace_y[0] + cfg.workspace_y[1]) / 2
        hx = (cfg.workspace_x[1] - cfg.workspace_x[0]) / 2
        hy = (cfg.workspace_y[1] - cfg.workspace_y[0]) / 2
        for batch in env_indices:
            # Thin translucent plane at the target height over the workspace.
            origin = self._env.scene.env_origins[batch].cpu().numpy()
            center = origin + np.array([cx, cy, self.variant.target_height])
            reached = bool(self.block_reached[batch].any())
            success = bool(self.success[batch] > 0.5)
            visualizer.add_box(
                center=center,
                size=np.array([hx, hy, 0.001]),
                mat=np.eye(3),
                color=marker_color(_PLANE_RGBA, reached, success),
                label=f"target_height_{batch}",
            )
            if SHOW_GOAL_STACK:
                self._draw_goal_stack(visualizer, batch, origin, cx, success)

    def _level_reached(self, batch: int) -> np.ndarray:
        """(target_blocks,) bool: the resting stack has reached that level.

        Read off ``progress``, so the guide fills up exactly like the reward —
        a block merely lifted to a level lights nothing.
        """
        variant = self.variant
        height = float(self.progress[batch]) * variant.target_height
        levels = variant.block_height * np.arange(1, variant.target_blocks + 1)
        return height >= levels - variant.height_tol

    def _draw_goal_stack(
        self,
        visualizer: DebugVisualizer,
        batch: int,
        origin: np.ndarray,
        cx: float,
        success: bool,
    ) -> None:
        """Ghost stack of ``target_blocks`` blocks beside the workspace.

        Stands beside the scatter region rather than in it: success is
        height-only, so the real stack's xy is free.
        """
        variant = self.variant
        half = np.array(variant.blocks[0].half_extents) + MARKER_INFLATE
        gy = self.cfg.workspace_y[1] + _GOAL_STACK_MARGIN + half[1]
        reached = self._level_reached(batch)
        for level in range(variant.target_blocks):
            center = origin + np.array([cx, gy, (level + 0.5) * variant.block_height])
            color = marker_color(_GOAL_STACK_RGBA, bool(reached[level]), success)
            visualizer.add_box(
                center=center,
                size=half,
                mat=np.eye(3),
                color=color,
                label=f"goal_stack_{batch}_{level}",
            )
            # Red edges mark it as a virtual target, not a real block (same
            # convention as the domino slot markers).
            add_box_wireframe(
                visualizer,
                center=center,
                half_extents=half,
                mat=np.eye(3),
                color=OUTLINE_RGBA,
                label_prefix=f"goal_stack_outline_{batch}_{level}",
            )
