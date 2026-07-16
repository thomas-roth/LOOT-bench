"""Task command for the stacking category.

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
from mjlab.managers.command_manager import CommandTerm
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from g3a_bench.tasks.common.commands import GripperGateCommandCfg, GripperOpenGate
from g3a_bench.tasks.common.debug_markers import marker_color
from g3a_bench.tasks.common.scatter import ScatterGrid
from g3a_bench.tasks.common.success import OBJECT_VEL_THRESHOLD, SuccessGate
from g3a_bench.tasks.stacking.mdp.success import (
    block_top_heights,
    stack_reached,
    stacking_progress,
    two_stacks_progress,
    two_stacks_success,
)
from g3a_bench.tasks.stacking.variants import (
    FORM_HALF_EXTENTS,
    MIN_STACK_SEPARATION,
    StackingVariantCfg,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

# Debug-vis plane color while the target height is not reached yet.
_PLANE_RGBA = (0.95, 0.95, 0.95, 1.0)


def get_command(env: ManagerBasedRlEnv, name: str) -> StackingCommand:
    return cast(StackingCommand, env.command_manager.get_term(name))


@dataclass(kw_only=True)
class StackingCommandCfg(GripperGateCommandCfg):
    block_names: tuple[str, ...]
    """All block entities, in the variant's (tail-extending) index order."""
    variant: StackingVariantCfg
    workspace_x: tuple[float, float] = (0.22, 0.42)
    workspace_y: tuple[float, float] = (-0.18, 0.18)
    block_vel_threshold: float = OBJECT_VEL_THRESHOLD
    """Max linear speed (m/s) of any block to count success."""

    def build(self, env: ManagerBasedRlEnv) -> StackingCommand:
        return StackingCommand(self, env)


class StackingCommand(CommandTerm):
    cfg: StackingCommandCfg

    def __init__(self, cfg: StackingCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        variant = cfg.variant
        self.variant = variant
        self.blocks: list[Entity] = [env.scene[n] for n in cfg.block_names]
        self.num_blocks = len(self.blocks)

        E, N = self.num_envs, self.num_blocks
        dev = self.device

        halves = [FORM_HALF_EXTENTS[b.form] for b in variant.blocks]
        self._half_h = torch.tensor([h[2] for h in halves], device=dev)
        # Bounding-circle radius of the widest footprint (collision-free
        # scatter must hold at any yaw).
        object_radius = max(math.hypot(h[0], h[1]) for h in halves)
        self._scatter = ScatterGrid.tiling(
            cfg.workspace_x, cfg.workspace_y, object_radius=object_radius, device=dev
        )
        assert self._scatter.num_cells >= N

        self.block_pos_w = torch.zeros(E, N, 3, device=dev)
        self.block_reached = torch.zeros(E, N, device=dev, dtype=torch.bool)
        self.progress = torch.zeros(E, device=dev)
        self._gate = SuccessGate(E, dev, vel_threshold=cfg.block_vel_threshold)
        self._gripper_gate = GripperOpenGate(cfg)

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
        """(E,) float: target height reached, blocks at rest, gripper open."""
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
        table_z = self._env.scene.env_origins[:, 2]
        target = self.variant.target_height
        top_z = block_top_heights(self.block_pos_w[:, :, 2], self._half_h)
        xy = self.block_pos_w[:, :, :2]

        self.block_reached = stack_reached(
            top_z, table_z, target, self.variant.height_tol
        )
        if self.variant.two_stacks:
            task_done = two_stacks_success(self.block_reached, xy, MIN_STACK_SEPARATION)
            self.progress = two_stacks_progress(
                top_z, xy, table_z, target, MIN_STACK_SEPARATION
            )
        else:
            task_done = self.block_reached.any(dim=-1)
            self.progress = stacking_progress(top_z, table_z, target)

        speed = torch.stack(
            [b.data.root_link_lin_vel_w for b in self.blocks], dim=1
        ).norm(dim=-1)  # (E, N)
        released = self._gripper_gate(self._env.scene["robot"])
        self._gate.update(task_done, speed, released)

        self.metrics["max_height_frac"] = self.progress
        self.metrics["episode_success"] = self.episode_success

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        dev = self.device
        origins = self._env.scene.env_origins[env_ids]

        self._gate.reset(env_ids)
        self.block_reached[env_ids] = False

        # No shared randomness; per-block draws in index order (seed-prefix
        # rule: an easier task's reset is a prefix of a harder task's).
        xy, spawn_yaw = self._scatter.sample_poses(n, self.num_blocks)

        zeros = torch.zeros(n, device=dev)
        for i, block in enumerate(self.blocks):
            z = (self._half_h[i] + 0.002 + origins[:, 2]).unsqueeze(-1)
            pos = torch.cat([xy[:, i, :] + origins[:, :2], z], dim=-1)
            quat = quat_from_euler_xyz(zeros, zeros, spawn_yaw[:, i])
            pose = torch.cat([pos, quat], dim=-1)
            block.write_root_link_pose_to_sim(pose, env_ids=env_ids)
            block.write_root_link_velocity_to_sim(
                torch.zeros(n, 6, device=dev), env_ids=env_ids
            )

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
