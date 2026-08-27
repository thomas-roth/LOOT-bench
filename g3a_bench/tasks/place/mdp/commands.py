"""Task command for the place category.

At every resample the command:

1. scatters the cube upright at a fresh spawn pose via the shared
   ``ScatterGrid`` (the same single-object-through-multi-object API every
   other category uses, even though there is only one object here and no
   collision to avoid),
2. draws a fresh target point uniformly in its own workspace region, disjoint
   in x from the spawn region (``spawn_x_range`` vs ``target_x_range``) so
   the cube can never spawn already at the target.

Each of the two draws is independently switched off by the variant
(``random_spawn`` / ``random_target``, see ``variants.py``); a switched-off
axis writes the region's midpoint (``FIXED_SPAWN_XY`` / ``FIXED_TARGET_XY``)
every reset instead.

Every step it recomputes the cube-to-target 3D distance and turns it into a
continuous ``progress`` (``mdp/success.py::place_progress``) and a pure-
position ``success`` test (``place_reached``), gated by the shared at-rest +
released conditions (``SuccessGate``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.sensor import ContactSensor

from g3a_bench.tasks.common.commands import (
    GRIP_CONTACT_SENSOR_NAME,
    ContactReleaseGate,
    GripperGateCommandCfg,
    GripperOpenGate,
)
from g3a_bench.tasks.common.debug_markers import marker_color
from g3a_bench.tasks.common.scatter import ScatterGrid, write_upright_spawn
from g3a_bench.tasks.common.success import OBJECT_VEL_THRESHOLD, SuccessGate
from g3a_bench.tasks.place.mdp.success import place_progress, place_reached
from g3a_bench.tasks.place.variants import CUBE_HALF_EXTENTS, PlaceVariantCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

_TARGET_RGBA = (0.95, 0.95, 0.2, 1.0)

SPAWN_X_RANGE = (0.20, 0.28)
TARGET_X_RANGE = (0.36, 0.44)
WORKSPACE_Y = (-0.18, 0.18)


def _midpoint(r: tuple[float, float]) -> float:
    return 0.5 * (r[0] + r[1])


FIXED_SPAWN_XY = (_midpoint(SPAWN_X_RANGE), _midpoint(WORKSPACE_Y))
FIXED_TARGET_XY = (_midpoint(TARGET_X_RANGE), _midpoint(WORKSPACE_Y))
"""Poses the ``Fixed*`` variants pin to. Derived from the ranges above rather
than written out, so the fixed pose can never drift out of the region its
random counterpart samples."""

FIXED_SPAWN_YAW = 0.0
"""A pinned spawn is fully deterministic -- position *and* orientation."""

MAX_TRAVEL_DIST = math.hypot(
    TARGET_X_RANGE[1] - SPAWN_X_RANGE[0], WORKSPACE_Y[1] - WORKSPACE_Y[0]
)
"""Largest possible spawn->target separation given the disjoint x-ranges
above (opposite y corners); normalizes ``place_progress`` to a fixed,
non-per-episode constant (analogous to how ``stack_progress`` normalizes by
the fixed ``target_height``). NOTE: derived from the *default* range tuples
above, not from ``PlaceCommandCfg``'s (overridable) fields -- if those cfg
fields are ever overridden away from the defaults, recompute this constant
to match (same caveat as domino's ``HARDEST_COUNT``-anchored math)."""


def get_command(env: ManagerBasedRlEnv, name: str) -> PlaceCommand:
    return cast(PlaceCommand, env.command_manager.get_term(name))


@dataclass(kw_only=True)
class PlaceCommandCfg(GripperGateCommandCfg):
    cube_name: str
    variant: PlaceVariantCfg
    spawn_x_range: tuple[float, float] = SPAWN_X_RANGE
    spawn_y_range: tuple[float, float] = WORKSPACE_Y
    target_x_range: tuple[float, float] = TARGET_X_RANGE
    target_y_range: tuple[float, float] = WORKSPACE_Y
    pos_threshold: float = 0.02
    """Max 3D distance between the cube and the target to count as placed
    (x margin)."""
    cube_vel_threshold: float = OBJECT_VEL_THRESHOLD
    """Max linear speed (m/s) of the cube to count success."""

    def build(self, env: ManagerBasedRlEnv) -> PlaceCommand:
        return PlaceCommand(self, env)


class PlaceCommand(CommandTerm):
    cfg: PlaceCommandCfg

    def __init__(self, cfg: PlaceCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.variant = cfg.variant
        self.cube: Entity = env.scene[cfg.cube_name]
        E, dev = self.num_envs, self.device

        self._half_h = CUBE_HALF_EXTENTS[2]
        object_radius = math.hypot(CUBE_HALF_EXTENTS[0], CUBE_HALF_EXTENTS[1])
        self._scatter = ScatterGrid.tiling(
            cfg.spawn_x_range,
            cfg.spawn_y_range,
            object_radius=object_radius,
            device=dev,
        )

        self.target_pos_w = torch.zeros(E, 3, device=dev)
        self.cube_pos_w = torch.zeros(E, 3, device=dev)
        self.cube_quat_w = torch.zeros(E, 4, device=dev)
        self.dist_to_target = torch.zeros(E, device=dev)
        self.reached = torch.zeros(E, device=dev, dtype=torch.bool)
        self.progress = torch.zeros(E, device=dev)
        self._gate = SuccessGate(E, dev, vel_threshold=cfg.cube_vel_threshold)
        self._gripper_gate = GripperOpenGate(cfg)
        self._contact_gate = ContactReleaseGate()
        self._grip_sensor = cast(
            ContactSensor, env.scene.sensors[GRIP_CONTACT_SENSOR_NAME]
        )

        self.metrics["dist_to_target"] = torch.zeros(E, device=dev)
        self.metrics["reached"] = torch.zeros(E, device=dev)
        self.metrics["episode_success"] = torch.zeros(E, device=dev)

    @property
    def command(self) -> torch.Tensor:
        """(E, 3): env-local target position."""
        return self.target_pos_w - self._env.scene.env_origins

    # Helpers used by observation/reward/termination terms. ----------------

    @property
    def success(self) -> torch.Tensor:
        """(E,) float: cube at target, at rest, gripper released."""
        return self._gate.success

    @property
    def episode_success(self) -> torch.Tensor:
        """(E,) float: success latched since the last reset."""
        return self._gate.episode_success

    # CommandTerm interface. ------------------------------------------------

    def _update_metrics(self) -> None:
        self.cube_pos_w = self.cube.data.root_link_pos_w
        self.cube_quat_w = self.cube.data.root_link_quat_w
        self.dist_to_target = torch.norm(self.cube_pos_w - self.target_pos_w, dim=-1)

        margin = self.variant.margin
        self.reached = place_reached(
            self.dist_to_target, self.cfg.pos_threshold * margin
        )
        self.progress = place_progress(self.dist_to_target, MAX_TRAVEL_DIST)

        speed = self.cube.data.root_link_lin_vel_w.norm(dim=-1)
        released = self._gripper_gate(self._env.scene["robot"]) & self._contact_gate(
            self._grip_sensor
        )
        self._gate.update(self.reached, speed, released)

        # No pinning of progress on success frames: the category once raised it
        # to 1.0 there, so that a "progress + success - reward_max" reward read
        # exactly 0 in the absorbing state. tasks/common/rewards.py now owns
        # that exactness for every category -- shaped_task_reward returns a
        # literal zero wherever the gate has latched, whatever progress says.
        self.metrics["dist_to_target"] = self.dist_to_target
        self.metrics["reached"] = self.reached.float()
        self.metrics["episode_success"] = self.episode_success

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        dev = self.device
        origins = self._env.scene.env_origins[env_ids]

        self._gate.reset(env_ids)

        # Cube spawn: single-object scatter draw (same API every other
        # category uses; count=1 here, no collision to avoid), or the pinned
        # pose. The fixed branch consumes no RNG -- the seed-prefix rule the
        # scatter upholds only concerns the sampled variants.
        if self.variant.random_spawn:
            xy, spawn_yaw = self._scatter.sample_poses(n, 1)
        else:
            xy = torch.tensor(FIXED_SPAWN_XY, device=dev).expand(n, 1, 2)
            spawn_yaw = torch.full((n, 1), FIXED_SPAWN_YAW, device=dev)
        write_upright_spawn(
            [self.cube],
            xy,
            spawn_yaw,
            torch.full((1,), self._half_h, device=dev),
            origins,
            env_ids,
        )

        # Target: independent uniform draw in its own disjoint region, or the
        # pinned point.
        if self.variant.random_target:
            x0, x1 = self.cfg.target_x_range
            y0, y1 = self.cfg.target_y_range
            u = torch.rand(n, 2, device=dev)
            tx = x0 + u[:, 0] * (x1 - x0)
            ty = y0 + u[:, 1] * (y1 - y0)
        else:
            tx = torch.full((n,), FIXED_TARGET_XY[0], device=dev)
            ty = torch.full((n,), FIXED_TARGET_XY[1], device=dev)
        tz = torch.full((n,), self._half_h, device=dev)
        self.target_pos_w[env_ids] = torch.stack([tx, ty, tz], dim=-1) + origins

    def _update_command(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        radius = self.cfg.pos_threshold * self.variant.margin
        for batch in env_indices:
            center = self.target_pos_w[batch].cpu().numpy()
            reached = bool(self.reached[batch])
            success = bool(self.success[batch] > 0.5)
            visualizer.add_sphere(
                center=center,
                radius=radius,
                color=marker_color(_TARGET_RGBA, reached, success),
                label=f"target_{batch}",
            )
