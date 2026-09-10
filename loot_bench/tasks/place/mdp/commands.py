"""Place command. Samples the cube spawn and target, tests placement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.sensor import ContactSensor

from loot_bench.tasks.common.commands import (
    GRIP_CONTACT_SENSOR_NAME,
    ContactReleaseGate,
)
from loot_bench.tasks.common.debug_markers import marker_color
from loot_bench.tasks.common.fixed_spawns import (
    SPAWN_JITTER,
    SPAWN_REACH_RADIUS,
    SPAWN_TABLE_HALF_WIDTH,
    SPAWN_X_MIN,
)
from loot_bench.tasks.common.scatter import write_upright_spawn
from loot_bench.tasks.common.success import OBJECT_VEL_THRESHOLD, SuccessGate
from loot_bench.tasks.place.mdp.success import place_progress, place_reached
from loot_bench.tasks.place.variants import CUBE_HALF_EXTENTS, PlaceVariantCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

_TARGET_RGBA = (0.95, 0.95, 0.2, 1.0)

PLACE_X_MAX = 0.45
"""Far bound (m) of the uniform envelope draw; the near bound is shared."""

FIXED_SPAWN_XY = (0.25, 0.0)
FIXED_TARGET_XY = (0.40, 0.0)
"""Poses the Fixed variants pin to: spawn at x = 0.25, target at x = 0.40."""

FIXED_SPAWN_YAW = 0.0
"""A pinned spawn is fully deterministic: position and orientation."""

PLACE_OBJECT_RADIUS = math.hypot(CUBE_HALF_EXTENTS[0], CUBE_HALF_EXTENTS[1])
"""Bounding-circle radius (m) of the cube, in the table plane."""

PLACE_MIN_SEPARATION = 0.10
"""Minimum cube-target separation (m), enforced by one deterministic push-apart."""

PLACE_Y_MAX = SPAWN_TABLE_HALF_WIDTH - PLACE_OBJECT_RADIUS - SPAWN_JITTER
"""Widest half-range (m) of the envelope draw, reached near the robot."""

MAX_TRAVEL_DIST = math.hypot(PLACE_X_MAX - SPAWN_X_MIN, 2.0 * PLACE_Y_MAX)
"""Largest spawn to target distance. Normalizes progress to a constant."""


def place_half_width(
    x: torch.Tensor, object_radius: float = PLACE_OBJECT_RADIUS
) -> torch.Tensor:
    """Half-range of ``y`` (m) at column ``x`` for the envelope draw.

    ``Y(x) = min(0.35 - r - jitter, sqrt(0.49**2 - x**2))``: the table's
    half-width bound near the robot, the radial reach bound far from it.
    """
    radial = torch.sqrt(torch.clamp(SPAWN_REACH_RADIUS**2 - x**2, min=0.0))
    cap = torch.tensor(
        SPAWN_TABLE_HALF_WIDTH - object_radius - SPAWN_JITTER,
        dtype=x.dtype,
        device=x.device,
    )
    return torch.minimum(cap, radial)


def place_xy_from_draws(
    u: torch.Tensor, object_radius: float = PLACE_OBJECT_RADIUS
) -> torch.Tensor:
    """Map unit draws ``(..., 2)`` to envelope positions ``(..., 2)``.

    ``x ~ U[0.20, 0.45]``, then ``y ~ U[-Y(x), +Y(x)]``: uniform per column,
    not per unit area (a mild bias toward the narrow far columns buys a
    fixed, reproducible draw count).
    """
    x = SPAWN_X_MIN + u[..., 0] * (PLACE_X_MAX - SPAWN_X_MIN)
    half = place_half_width(x, object_radius)
    y = (u[..., 1] * 2.0 - 1.0) * half
    return torch.stack([x, y], dim=-1)


def enforce_place_separation(
    cube_xy: torch.Tensor,
    target_xy: torch.Tensor,
    min_separation: float = PLACE_MIN_SEPARATION,
    *,
    move_cube: bool = True,
    move_target: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Push cube and target apart to at least ``min_separation`` (m).

    One deterministic push along the line joining them (a coincident pair
    splits along +x), so an episode can never start already solved — no
    rejection loop. The shortfall is split over the sides allowed to move:
    a pinned axis never moves, so a ``Fixed`` variant stays exactly pinned.
    """
    if not move_cube and not move_target:
        return cube_xy, target_xy
    delta = target_xy - cube_xy
    dist = delta.norm(dim=-1, keepdim=True)
    shortfall = (min_separation - dist).clamp(min=0.0)
    fallback = torch.zeros_like(delta)
    fallback[..., 0] = 1.0
    unit = torch.where(dist > 1e-9, delta / dist.clamp(min=1e-9), fallback)
    push = unit * shortfall
    weights = float(move_cube) + float(move_target)
    return (
        cube_xy - push * (float(move_cube) / weights),
        target_xy + push * (float(move_target) / weights),
    )


def draw_place_poses(
    num_envs: int,
    random_spawn: bool,
    random_target: bool,
    *,
    device: str | torch.device = "cpu",
    object_radius: float = PLACE_OBJECT_RADIUS,
    min_separation: float = PLACE_MIN_SEPARATION,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw the cube pose and target over the shared envelope.

    Fixed draw counts in index order — cube ``rand(n, 3)`` (x, y, yaw),
    then target ``rand(n, 2)`` (x, y) — even for pinned axes, so the
    seed-prefix rule holds across variants. Pinned axes overwrite their
    draw with the fixed pose; the push-apart runs last either way.
    """
    u_cube = torch.rand(num_envs, 3, device=device)
    u_target = torch.rand(num_envs, 2, device=device)
    cube_xy = place_xy_from_draws(u_cube[:, :2], object_radius)
    cube_yaw = u_cube[:, 2] * 2.0 * math.pi
    target_xy = place_xy_from_draws(u_target, object_radius)
    if not random_spawn:
        cube_xy = torch.tensor(FIXED_SPAWN_XY, device=device).expand(num_envs, 2)
        cube_yaw = torch.zeros(num_envs, device=device)
    if not random_target:
        target_xy = torch.tensor(FIXED_TARGET_XY, device=device).expand(num_envs, 2)
    cube_xy, target_xy = enforce_place_separation(
        cube_xy,
        target_xy,
        min_separation,
        move_cube=random_spawn,
        move_target=random_target,
    )
    return cube_xy, cube_yaw, target_xy


def get_command(env: ManagerBasedRlEnv, name: str) -> PlaceCommand:
    return cast(PlaceCommand, env.command_manager.get_term(name))


@dataclass(kw_only=True)
class PlaceCommandCfg(CommandTermCfg):
    cube_name: str
    variant: PlaceVariantCfg
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

        self.target_pos_w = torch.zeros(E, 3, device=dev)
        self.cube_pos_w = torch.zeros(E, 3, device=dev)
        self.cube_quat_w = torch.zeros(E, 4, device=dev)
        self.dist_to_target = torch.zeros(E, device=dev)
        self.reached = torch.zeros(E, device=dev, dtype=torch.bool)
        self.progress = torch.zeros(E, device=dev)
        self._gate = SuccessGate(E, dev, vel_threshold=cfg.cube_vel_threshold)
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
        released = self._contact_gate(self._grip_sensor)
        self._gate.update(self.reached, speed, released)

        # shaped_task_reward handles the exact zero on latched success.
        self.metrics["dist_to_target"] = self.dist_to_target
        self.metrics["reached"] = self.reached.float()
        self.metrics["episode_success"] = self.episode_success

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        env_ids_len = len(env_ids)
        if env_ids_len == 0:
            return
        dev = self.device
        origins = self._env.scene.env_origins[env_ids]

        self._gate.reset(env_ids)

        # Fixed draw counts in index order (cube, then target), even for
        # pinned axes, so the seed-prefix rule holds across variants; one
        # deterministic push-apart keeps the minimum separation.
        cube_xy, spawn_yaw, target_xy = draw_place_poses(
            env_ids_len,
            self.variant.random_spawn,
            self.variant.random_target,
            device=dev,
        )
        write_upright_spawn(
            [self.cube],
            cube_xy.unsqueeze(1),
            spawn_yaw.unsqueeze(1),
            torch.full((1,), self._half_h, device=dev),
            origins,
            env_ids,
        )

        z_tensor = torch.full((env_ids_len,), self._half_h, device=dev)
        self.target_pos_w[env_ids] = (
            torch.cat([target_xy, z_tensor.unsqueeze(-1)], dim=-1) + origins
        )

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
