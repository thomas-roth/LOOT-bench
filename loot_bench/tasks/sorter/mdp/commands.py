"""Sorter command. Owns hole poses, spawns, and insertion latching."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
)

from loot_bench.tasks.common.colors import COLOR_VALUES
from loot_bench.tasks.common.commands import (
    GRIP_CONTACT_SENSOR_NAME,
    ContactReleaseGate,
)
from loot_bench.tasks.common.debug_markers import PENDING_ALPHA, marker_color
from loot_bench.tasks.common.fixed_spawns import SPAWN_JITTER, sample_fixed_spawns
from loot_bench.tasks.common.scatter import write_upright_spawn
from loot_bench.tasks.common.success import SuccessGate
from loot_bench.tasks.sorter.assets import (
    BOARD_POS,
    PIECE_SCALE,
    load_sorter_layout,
)
from loot_bench.tasks.sorter.mdp.success import (
    SorterLatch,
    peg_axial_lateral,
)
from loot_bench.tasks.sorter.variants import (
    HOLE_SLACK,
    SorterVariantCfg,
    allowed_faces,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

# Flat resting orientations of the cube, one per face down.
_FLAT_AXES = torch.tensor(
    [
        [1.0, 0.0, 0.0],  # identity ....... -z face down (circle)
        [1.0, 0.0, 0.0],  # roll pi ........ +z face down (star)
        [0.0, 1.0, 0.0],  # pitch +pi/2 .... +x face down (square)
        [0.0, 1.0, 0.0],  # pitch -pi/2 .... -x face down (rect)
        [1.0, 0.0, 0.0],  # roll -pi/2 ..... +y face down (hexagon)
        [1.0, 0.0, 0.0],  # roll +pi/2 ..... -y face down (triangle)
    ]
)
_FLAT_ANGLES = torch.tensor(
    [0.0, math.pi, math.pi / 2, -math.pi / 2, -math.pi / 2, math.pi / 2]
)
# Pose used by spawn fixed. Down face is the unused star socket.
_FIXED_FACE_INDEX = 1

SORTER_SPAWN_CELLS: tuple[tuple[float, float], ...] = (
    (0.21, 0.07),
    (0.21, -0.07),
    (0.21, 0.19),
    (0.21, -0.19),
    (0.21, 0.31),
)
"""Fixed piece-spawn cells, object ``i`` takes cell ``i``.

One column in front of the cube (whose near face sits at 0.265), so a
recording session starts with the pieces staged and the work area clear.
Sorter3's reset is a prefix of Sorter4's and Sorter4's a prefix of
Sorter5's. Minimum pairwise separation 0.120, above the gripper-derived
pitch for the largest piece.
"""

SORTER_CORRIDOR_HALF_WIDTH = 0.03
"""Half-width (m) of the rect hole's insertion approach corridor.

In ``fixed`` spawn the star face is down, which leaves the rect hole
facing -x, straight at the robot; its insertion approach is a horizontal
corridor along the centreline. The table keeps that corridor clear, which
is why there is no ``y = 0`` cell and why the fifth cell is asymmetric.
An asymmetric table is cheaper than a silently blocked insertion.
"""


def sorter_spawn_is_clear(x: float, y: float, object_radius: float) -> bool:
    """Whether a jittered ``(x, y)`` spawn pose is clear of the approach corridor."""
    return abs(y) >= SORTER_CORRIDOR_HALF_WIDTH + object_radius


def get_command(env: ManagerBasedRlEnv, name: str) -> SorterCommand:
    return cast(SorterCommand, env.command_manager.get_term(name))


@dataclass(kw_only=True)
class SorterCommandCfg(CommandTermCfg):
    cube_name: str
    piece_names: tuple[str, ...]
    """One piece entity per shape, in the variant's (tail-extending) index
    order."""
    variant: SorterVariantCfg
    hole_slack: float = HOLE_SLACK
    """Radial slack (m) added to every hole's bounding radius (x margin)."""

    def build(self, env: ManagerBasedRlEnv) -> SorterCommand:
        return SorterCommand(self, env)


class SorterCommand(CommandTerm):
    cfg: SorterCommandCfg

    def __init__(self, cfg: SorterCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        variant = cfg.variant
        self.variant = variant
        self.cube: Entity = env.scene[cfg.cube_name]
        self.pieces: list[Entity] = [env.scene[n] for n in cfg.piece_names]
        self.num_pieces = N = len(self.pieces)

        E, dev = self.num_envs, self.device

        layout = load_sorter_layout()
        self._cube_half = layout.half_size
        holes = [layout.holes[s] for s in variant.shapes]
        self._mouth_local = torch.tensor(
            [h.mouth for h in holes], device=dev, dtype=torch.float32
        )
        self._axis_local = torch.tensor(
            [h.axis_out for h in holes], device=dev, dtype=torch.float32
        )
        # Meshes are scaled uniformly by PIECE_SCALE (see assets), so the sim
        # length is the layout half_len times that factor.
        piece_half = torch.tensor(
            [layout.pieces[s].half_len * PIECE_SCALE[s] for s in variant.shapes],
            device=dev,
        )
        hole_radius = (
            torch.tensor([h.bounding_radius for h in holes], device=dev)
            + cfg.hole_slack * variant.margin
        )
        wall_t = torch.tensor([h.wall_thickness for h in holes], device=dev)
        self._latch = SorterLatch(
            E,
            N,
            dev,
            hole_radius=hole_radius,
            piece_half=piece_half,
            s_cross=-(wall_t + piece_half),
        )

        self._flat_quats = quat_from_angle_axis(
            _FLAT_ANGLES.to(dev), _FLAT_AXES.to(dev)
        )  # (6, 4)
        self._allowed_faces = torch.tensor(
            allowed_faces(variant.shapes), device=dev, dtype=torch.long
        )
        self._all_faces = torch.arange(6, device=dev, dtype=torch.long)

        self._spawn_cells = torch.tensor(SORTER_SPAWN_CELLS, device=dev)
        assert self._spawn_cells.shape[0] >= N, (
            f"spawn table holds {self._spawn_cells.shape[0]} cells, variant needs {N}"
        )

        self.socket_mouth_w = torch.zeros(E, N, 3, device=dev)
        self.socket_axis_w = torch.zeros(E, N, 3, device=dev)
        self.cube_pos_w = torch.zeros(E, 3, device=dev)
        self.cube_quat_w = torch.zeros(E, 4, device=dev)
        self.piece_pos_w = torch.zeros(E, N, 3, device=dev)
        self.piece_quat_w = torch.zeros(E, N, 4, device=dev)
        self.progress = torch.zeros(E, device=dev)
        self._gate = SuccessGate(E, dev, vel_threshold=math.inf)
        self._contact_gate = ContactReleaseGate()
        self._grip_sensor = cast(
            ContactSensor, env.scene.sensors[GRIP_CONTACT_SENSOR_NAME]
        )

        self.metrics["placed_frac"] = torch.zeros(E, device=dev)
        self.metrics["episode_success"] = torch.zeros(E, device=dev)

    @property
    def command(self) -> torch.Tensor:
        """(E, N*6): per hole mouth xyz and axis xyz, in piece order."""
        mouth = self.socket_mouth_w - self._env.scene.env_origins.unsqueeze(1)
        return torch.cat([mouth, self.socket_axis_w], dim=-1).reshape(self.num_envs, -1)

    # Helpers used by observation/reward/termination terms. ----------------

    @property
    def success(self) -> torch.Tensor:
        """(E,) float: every piece latched, released."""
        return self._gate.success

    @property
    def episode_success(self) -> torch.Tensor:
        """(E,) float: success latched since the last reset."""
        return self._gate.episode_success

    @property
    def latched(self) -> torch.Tensor:
        """(E, N) bool: pieces latched inside the cube."""
        return self._latch.latched

    # CommandTerm interface. ------------------------------------------------

    def _update_metrics(self) -> None:
        E, N = self.num_envs, self.num_pieces
        self.cube_pos_w = self.cube.data.root_link_pos_w
        self.cube_quat_w = self.cube.data.root_link_quat_w
        q = self.cube_quat_w.unsqueeze(1).expand(E, N, 4)
        self.socket_mouth_w = self.cube_pos_w.unsqueeze(1) + quat_apply(
            q, self._mouth_local.unsqueeze(0).expand(E, N, 3)
        )
        self.socket_axis_w = quat_apply(
            q, self._axis_local.unsqueeze(0).expand(E, N, 3)
        )

        self.piece_pos_w = torch.stack(
            [p.data.root_link_pos_w for p in self.pieces], dim=1
        )
        self.piece_quat_w = torch.stack(
            [p.data.root_link_quat_w for p in self.pieces], dim=1
        )

        axial, lateral = peg_axial_lateral(
            self.piece_pos_w, self.socket_mouth_w, self.socket_axis_w
        )
        latched = self._latch.update(axial, lateral)
        all_latched = latched.all(dim=1)
        self.progress = latched.float().mean(dim=1)

        # The at-rest gate is disabled (vel_threshold=inf): a latched piece
        # stays scored regardless of motion, so skip the speed computation.
        speed = torch.zeros(E, device=self.device)
        released = self._contact_gate(self._grip_sensor)
        self._gate.update(all_latched, speed, released)

        self.metrics["placed_frac"] = self.progress
        self.metrics["episode_success"] = self.episode_success

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        dev = self.device
        variant = self.variant
        N = self.num_pieces
        origins = self._env.scene.env_origins[env_ids]

        self._gate.reset(env_ids)
        self._latch.reset(env_ids)

        # Cube pose: one shared draw, same count for every spawn mode.
        u = torch.rand(n, 4, device=dev)
        zeros = torch.zeros(n, device=dev)
        jitter = (u[:, :2] - 0.5) * 2.0 * SPAWN_JITTER
        cube_xy = origins[:, :2] + torch.tensor(BOARD_POS[:2], device=dev) + jitter
        cube_z = origins[:, 2] + BOARD_POS[2] + self._cube_half + 0.002
        cube_pos = torch.cat([cube_xy, cube_z.unsqueeze(-1)], dim=-1)

        if variant.spawn == "fixed":
            cube_quat = (
                self._flat_quats[_FIXED_FACE_INDEX]
                .unsqueeze(0)
                .expand(n, 4)
                .contiguous()
            )
        else:
            faces = (
                self._allowed_faces
                if variant.spawn == "no_rotation"
                else self._all_faces
            )
            num_faces = faces.shape[0]
            idx = (u[:, 2] * num_faces).long().clamp_max(num_faces - 1)
            face = faces[idx]
            yaw = quat_from_euler_xyz(zeros, zeros, u[:, 3] * 2 * math.pi)
            cube_quat = quat_mul(yaw, self._flat_quats[face])

        self.cube.write_root_link_pose_to_sim(
            torch.cat([cube_pos, cube_quat], dim=-1), env_ids=env_ids
        )
        self.cube.write_root_link_velocity_to_sim(
            torch.zeros(n, 6, device=dev), env_ids=env_ids
        )

        # Pieces: object i takes cell i with +/-1 cm jitter (one rand draw
        # per piece, in index order), so an easier task's reset is a prefix
        # of a harder task's.
        xy, spawn_yaw = sample_fixed_spawns(self._spawn_cells, n, N)
        write_upright_spawn(
            self.pieces, xy, spawn_yaw, self._latch.piece_half, origins, env_ids
        )

    def _update_command(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        for batch in env_indices:
            success = bool(self.success[batch] > 0.5)
            for i in range(self.num_pieces):
                mouth = self.socket_mouth_w[batch, i].cpu().numpy()
                done = bool(self.latched[batch, i])
                color = marker_color(
                    COLOR_VALUES[self.variant.colors[i]],
                    done,
                    success,
                    pending_alpha=PENDING_ALPHA,
                )
                visualizer.add_sphere(
                    center=mouth,
                    radius=0.015,
                    color=color,
                    label=f"mouth_{batch}_{i}",
                )
