"""Teleport-based success-reward check for the five task categories.

See tests/tasks/README.md.
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path
from typing import cast

import numpy as np
import torch
from _render import domain_of, save_env_render
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_from_angle_axis, quat_from_euler_xyz, quat_mul

import loot_bench  # noqa: F401  (registers all tasks)
from loot_bench.tasks.ball.mdp.commands import BallCommand
from loot_bench.tasks.ball.variants import (
    BALL_RADIUS,
    BALL_SPAWN_X,
    BALL_VARIANTS,
    TARGET_X,
)
from loot_bench.tasks.common.actions import delta_action_from_poses
from loot_bench.tasks.common.env_cfg import CONTROL_STEP_S
from loot_bench.tasks.common.success import bore_axis_cos_tilt
from loot_bench.tasks.domino.mdp.commands import DominoCommand
from loot_bench.tasks.place.mdp.commands import PlaceCommand
from loot_bench.tasks.shape_sorter.assets import BOARD_POS, load_sorter_layout
from loot_bench.tasks.shape_sorter.mdp.commands import ShapeSorterCommand
from loot_bench.tasks.stack.mdp.commands import GUIDE_UPRIGHT_SPACING, PiCommand
from loot_bench.tasks.stack.variants import (
    BLOCK_HALF_EXTENTS,
    BLOCK_LENGTH,
    LEVEL_HEIGHT,
    TILT_COS_MAX,
)

_NUM_ENVS = 8
_STEPS = 10
_KNOCKOVER_SECONDS = 2.0
"""Enough for a 3-domino chain reaction to propagate and settle."""

_KNOCKOVER_STEPS = round(_KNOCKOVER_SECONDS / CONTROL_STEP_S)
"""Derived from the control step rather than written, so a rate change moves
the budget instead of silently shortening the wait (ADR-016 took it from
0.02 s to 0.05 s)."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env(task_id: str, device: str, *, play: bool = True) -> ManagerBasedRlEnv:
    cfg = load_env_cfg(task_id, play=play)
    # Play cfgs keep the "success" termination (live_bridge demo playback
    # terminates on success); pop it here, otherwise the env auto-resets the
    # instant our teleported state scores, wiping the state before we read
    # the fraction below.
    cfg.terminations.pop("success", None)
    cfg.scene.num_envs = _NUM_ENVS
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.reset()
    return env


def _zeros(env: ManagerBasedRlEnv, device: str) -> torch.Tensor:
    return torch.zeros((_NUM_ENVS, env.action_manager.action.shape[1]), device=device)


def _upright_quat(n: int, device: str) -> torch.Tensor:
    z = torch.zeros(n, device=device)
    return quat_from_euler_xyz(z, z, z)


def _reward_convention_error(env: ManagerBasedRlEnv, zeros: torch.Tensor) -> str | None:
    """Check the reward convention on a live env holding a scored state.

    ``tasks/common/rewards.py`` promises a reward in ``[-1, 0]`` that reads
    exactly ``0`` only at success. ``tests/tasks/test_rewards.py`` pins the
    kernel; this pins the wiring, which is the half a unit test cannot reach:
    that each category's command really does feed its own success and progress
    buffers to that kernel, and that the reward manager publishes the result
    unscaled.

    Takes one step of its own, because the reward lags the command by exactly
    one step. Mjlab runs the reward manager before the command manager, so
    the reward a step returns was computed from the success buffer as it stood
    *before* that step. Snapshotting success first and then stepping is what
    makes the comparison line up. Reading both after the same step instead
    looks correct on a settled scene and silently disagrees on a moving one,
    which is how a stale-by-one reward hides.

    ``_make_env`` pops the success termination, so a scored env stays scored
    instead of resetting out from under the read.
    """
    command = env.command_manager.get_term("task")
    success = command.success.bool().clone()  # type: ignore[attr-defined]
    env.step(zeros)
    reward = env.reward_buf

    lo, hi = float(reward.min()), float(reward.max())
    if hi > 0.0 or lo < -1.0:
        return f"reward {lo:.4f}..{hi:.4f} outside [-1, 0]"
    scored = reward == 0.0
    if not bool((scored == success).all()):
        return (
            f"reward==0 on {int(scored.sum())} envs but success on "
            f"{int(success.sum())} of {reward.numel()}"
        )
    return None


def _render_goal(env: ManagerBasedRlEnv, task_id: str, render: bool) -> None:
    """Save a picture of the teleported goal state for *task_id*.

    Called at the scored instant (after the step loop that computes the success
    fraction) so the image shows the exact state the metric evaluated. A render
    failure is reported as a warning and never aborts the check.
    """
    if not render:
        return
    path = Path("out/render") / domain_of(task_id) / f"{task_id}_goal.png"
    warn = save_env_render(env, path)
    if warn is not None:
        print(f"        (render skipped {task_id}: {warn})")


# ---------------------------------------------------------------------------
# Pi (the Stack category)
# ---------------------------------------------------------------------------

_PI_XY = (0.28, 0.12)
"""Where the checks teleport the pi (success is position-free).

Beside the build zone, not in it: the robot's home gripper hangs over the
zone's centre at ~(0.30, 0, 0.20), and a three-level pi is 0.255 m tall,
so teleporting it there puts lintel 3 inside the wrist and the solver
returns NaN for the whole env, robot included."""

_SETTLE_STEPS = round(2.0 / CONTROL_STEP_S)
"""Steps a teleported trilithon is left to settle before it is measured.

Longer than the generic ``_STEPS``: the acceptance gate (ADR-020) reads the
settled per-lintel centre error and tilt off this run, and a three-level
trilithon takes more than half a second to stop creeping."""

_DRIVE_STEPS = 100
"""IK drive budget for lowering the grasp site to the level-1 lintel height."""


def _lintel_quat(n: int, device: str) -> torch.Tensor:
    """Block long axis (local z) along world x: a 90 deg rotation about y."""
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).expand(n, 3)
    return quat_from_angle_axis(torch.full((n,), math.pi / 2, device=device), y_axis)


def _write_block(block, pos: torch.Tensor, quat: torch.Tensor, device: str) -> None:
    block.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1))
    block.write_root_link_velocity_to_sim(torch.zeros(pos.shape[0], 6, device=device))


def _teleport_trilithons(
    cmd: PiCommand,
    env: ManagerBasedRlEnv,
    device: str,
    *,
    xy: tuple[float, float],
) -> None:
    """Build every level as a trilithon at one xy, each on top of the last.

    Level ``k`` uses blocks ``3k``, ``3k + 1`` (uprights, spaced
    ``GUIDE_UPRIGHT_SPACING`` apart along x) and ``3k + 2`` (the lintel,
    lying across them along x), i.e. the recording protocol's tier order.
    """
    origins = env.scene.env_origins
    hx, _hy, hz = BLOCK_HALF_EXTENTS
    base_xy = origins[:, :2] + torch.tensor(xy, device=device)
    upright = _upright_quat(cmd.num_envs, device)
    lintel = _lintel_quat(cmd.num_envs, device)
    for level in range(cmd.variant.levels):
        base_z = origins[:, 2] + level * LEVEL_HEIGHT
        blocks = cmd.blocks[3 * level : 3 * level + 3]
        for block, sign in zip(blocks[:2], (-1.0, 1.0), strict=True):
            off = torch.tensor([sign * GUIDE_UPRIGHT_SPACING / 2, 0.0], device=device)
            pos = torch.cat([base_xy + off, (base_z + hz).unsqueeze(-1)], dim=-1)
            _write_block(block, pos, upright, device)
        # Lying, the block's local x (half extent hx) is vertical.
        pos = torch.cat([base_xy, (base_z + BLOCK_LENGTH + hx).unsqueeze(-1)], dim=-1)
        _write_block(blocks[2], pos, lintel, device)


def _report_settle(cmd: PiCommand, env: ManagerBasedRlEnv) -> None:
    """Print the acceptance-gate numbers per lintel after settling.

    ``HEIGHT_TOL`` and ``TILT_COS_MAX`` are provisional until measured
    (ADR-020): the centre error is the slack one, the tilt is the tight one,
    because differential settle across an upright pair tilts the lintel and
    that error compounds upward through the levels standing on it.
    """
    table_z = env.scene.env_origins[:, 2]
    up = torch.tensor([0.0, 0.0, 1.0], device=cmd.device).unsqueeze(0)
    up = up.expand(cmd.num_envs, 3)
    for level in range(cmd.variant.levels):
        block = cmd.blocks[3 * level + 2]
        height = block.data.root_link_pos_w[:, 2] - table_z
        err = (height - cmd.variant.level_centre_height(level + 1)).abs()
        cos = bore_axis_cos_tilt(block.data.root_link_quat_w, up)
        print(
            f"        lintel {level + 1}: centre error max {err.max() * 1000:.2f} mm "
            f"(HEIGHT_TOL {cmd.variant.height_tol * 1000:.1f} mm), "
            f"|cos| max {cos.max():.4f} (TILT_COS_MAX {TILT_COS_MAX})"
        )


def _check_pi_3levels(device: str, *, render: bool = False) -> str | None:
    """Three teleported trilithons must score after settling."""
    task_id = "LOOT-Stack-3Pis-Dk1"
    env = _make_env(task_id, device)
    cmd = cast(PiCommand, env.command_manager.get_term("task"))
    zeros = _zeros(env, device)
    env.step(zeros)
    _teleport_trilithons(cmd, env, device, xy=_PI_XY)
    for _ in range(_SETTLE_STEPS):
        env.step(zeros)
    _report_settle(cmd, env)
    frac = cmd.progress.mean().item()
    reward_err = _reward_convention_error(env, zeros)
    _render_goal(env, task_id, render)
    env.close()
    return f"progress={frac:.3f} (expected >0.9)" if frac <= 0.9 else reward_err


_GRIP_CLOSE = -2.0  # last action dim, raw value that fully closes the gripper
_GRIP_OPEN = 2.0
_GRIP_STEPS = 60  # sim steps for the fingers to close on the pinned block


def _check_pi_held_lintel_excluded(device: str, *, render: bool = False) -> str | None:
    """A lintel gripped at level height must NOT earn level credit.

    Exercises the held-block exclusion (``PiCommand._held_block_mask`` +
    ``pi_levels_complete(exclude=...)``) against the real grip-contact
    sensor, in the recording protocol's own geometry: the grasp site is
    driven down to the level-1 lintel height, blocks 1 and 2 are stood up
    as trilithon uprights under it, and block 0 is pinned between the
    closing pads end-to-end (long axis along the closing axis, so the pads
    hang outside its ends beside the uprights). While the pads are loaded
    the lintel is held and level 1 must not count; once they open it rests
    on the uprights and must count. The lintel is re-pinned every step, so
    it has to have something under it: a released block written into free
    air falls 12 mm inside one control step and reads under tolerance.
    """
    task_id = "LOOT-Stack-1Pi-Dk1"
    env = _make_env(task_id, device)
    cmd = cast(PiCommand, env.command_manager.get_term("task"))
    robot = env.scene["robot"]
    ne, dim = _NUM_ENVS, env.action_manager.action.shape[1]
    origins = env.scene.env_origins
    site_ids = cmd.grasp_cfg.site_ids
    env.step(_zeros(env, device))

    def action(grip: float, delta: torch.Tensor | None = None) -> torch.Tensor:
        a = torch.zeros((ne, dim), device=device)
        if delta is not None:
            a[:, :-1] = delta
        a[:, -1] = grip
        return a

    def site_pose() -> tuple[torch.Tensor, torch.Tensor]:
        return (
            robot.data.site_pos_w[:, site_ids].squeeze(1),
            robot.data.site_quat_w[:, site_ids].squeeze(1),
        )

    # Lower the open gripper to the lintel height over the build zone.
    home_quat = site_pose()[1].clone()
    lintel_z = origins[:, 2] + cmd.variant.level_centre_height(1)
    target = torch.cat(
        [
            origins[:, :2] + torch.tensor(_PI_XY, device=device),
            lintel_z.unsqueeze(-1),
        ],
        dim=-1,
    )
    for _ in range(_DRIVE_STEPS):
        pos, quat = site_pose()
        env.step(
            action(_GRIP_OPEN, delta_action_from_poses(pos, quat, target, home_quat))
        )
    residual = float((site_pose()[0] - target).norm(dim=-1).max())
    if residual > 0.01:
        return f"grasp site never reached the lintel height (residual {residual * 1000:.1f} mm)"

    # Two uprights under the lintel's ends. The DK1 closes along site y,
    # which at the home yaw is world x, so the uprights go along x.
    upright = _upright_quat(ne, device)
    for block, sign in zip(cmd.blocks[1:3], (-1.0, 1.0), strict=True):
        off = torch.tensor([sign * GUIDE_UPRIGHT_SPACING / 2, 0.0], device=device)
        pos = torch.cat(
            [
                target[:, :2] + off,
                (origins[:, 2] + BLOCK_HALF_EXTENTS[2]).unsqueeze(-1),
            ],
            dim=-1,
        )
        _write_block(block, pos, upright, device)

    # End-to-end grip: local z (the long axis) onto site y, the closing axis.
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=device).unsqueeze(0).expand(ne, 3)
    flat = quat_from_angle_axis(torch.full((ne,), math.pi / 2, device=device), x_axis)

    def pin_lintel_to_grasp() -> None:
        pos, quat = site_pose()
        pos = torch.cat([pos[:, :2], lintel_z.unsqueeze(-1)], dim=-1)
        _write_block(cmd.blocks[0], pos, quat_mul(quat, flat), device)

    for _ in range(_GRIP_STEPS):
        pin_lintel_to_grasp()
        env.step(action(_GRIP_CLOSE))
    held_progress = cmd.progress.mean().item()
    force = cmd._grip_sensor.data.force.norm(dim=-1).amax(dim=-1)  # pyright: ignore[reportOptionalMemberAccess]
    gripping = bool((~cmd._contact_gate(cmd._grip_sensor)).all())

    _render_goal(env, task_id, render)

    # Open the gripper at the same pose: the lintel must count again.
    for _ in range(20):
        pin_lintel_to_grasp()
        env.step(action(_GRIP_OPEN))
    released_progress = cmd.progress.mean().item()
    env.close()

    if not gripping:
        return f"grip sensor never registered a grasp (max force={force.mean():.2f} N)"
    if held_progress > 0.1:
        return (
            f"held lintel counted toward level credit: held={held_progress:.3f} "
            "(expected 0.0, nothing else is horizontal)"
        )
    if released_progress < 0.9:
        return (
            f"released lintel did not count: released={released_progress:.3f} "
            "(expected 1.0)"
        )
    return None


def _check_pi_empty_gripper_no_grip(device: str, *, render: bool = False) -> str | None:
    """An empty, fully-closed gripper must not register grip force.

    Guards the finger<->finger contact exclusion in the DK1 model: without it
    the closed empty fingers self-collide, the grip sensor reads force, and
    ``_held_block_mask`` would wrongly exclude any resting block within
    ``grip_reach`` (e.g. a freshly placed lintel). Every block is sunk out
    of reach here, so any force can only be finger self-contact.
    """
    env = _make_env("LOOT-Stack-1Pi-Dk1", device)
    cmd = cast(PiCommand, env.command_manager.get_term("task"))
    ne, dim = _NUM_ENVS, env.action_manager.action.shape[1]
    quat = _upright_quat(ne, device)
    for block in cmd.blocks:
        sunk = env.scene.env_origins.clone()
        sunk[:, 2] -= 1.0
        _write_block(block, sunk, quat, device)
    close = torch.zeros((ne, dim), device=device)
    close[:, -1] = _GRIP_CLOSE
    for _ in range(_GRIP_STEPS):
        env.step(close)
    force = cmd._grip_sensor.data.force.norm(dim=-1).amax(dim=-1)  # pyright: ignore[reportOptionalMemberAccess]
    gripping = bool((~cmd._contact_gate(cmd._grip_sensor)).any())
    env.close()
    if gripping:
        return (
            f"empty closed gripper registered grip (max force={force.max():.2f} N); "
            "finger self-contact would cause false held-block exclusion"
        )
    return None


# ---------------------------------------------------------------------------
# Place
# ---------------------------------------------------------------------------


def _check_place(task_id: str, device: str, *, render: bool = False) -> str | None:
    """Teleport the cube onto the target.

    Parameterized by task id because the place category's four variants only
    differ in which of the two draws (cube spawn, target) is pinned. The
    success criterion is the same for all of them.
    """
    env = _make_env(task_id, device)
    cmd = cast(PlaceCommand, env.command_manager.get_term("task"))
    zeros = _zeros(env, device)
    env.step(zeros)
    # The target already sits at the cube's resting height, so teleporting the
    # cube onto it needs no z correction.
    quat = _upright_quat(_NUM_ENVS, device)
    cmd.cube.write_root_link_pose_to_sim(
        torch.cat([cmd.target_pos_w, quat], dim=-1),
    )
    cmd.cube.write_root_link_velocity_to_sim(torch.zeros(_NUM_ENVS, 6, device=device))
    for _ in range(_STEPS):
        env.step(zeros)
    frac = cmd.reached.float().mean().item()
    reward_err = _reward_convention_error(env, zeros)
    _render_goal(env, task_id, render)
    env.close()
    return f"reached_fraction={frac:.3f} (expected >0.9)" if frac <= 0.9 else reward_err


# ---------------------------------------------------------------------------
# Domino
# ---------------------------------------------------------------------------


def _teleport_dominoes_upright(cmd: DominoCommand, device: str) -> None:
    """Place every domino at its own slot goal, upright and row-aligned."""
    zeros = torch.zeros(cmd.num_envs, device=device)
    quat = quat_from_euler_xyz(zeros, zeros, cmd.row_yaw)
    for i, domino in enumerate(cmd.dominoes):
        pos = cmd.slot_goal_pos[:, i]
        domino.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1))
        domino.write_root_link_velocity_to_sim(
            torch.zeros(cmd.num_envs, 6, device=device)
        )


def _tip_first_domino(cmd: DominoCommand, env: ManagerBasedRlEnv, device: str) -> None:
    """Tilt only domino 0, ~25 deg toward slot 1, and let physics do the rest.

    The chain criterion requires the whole row to topple along the row axis,
    all leaning the same way, in row order. So instead of teleporting every
    domino into a "fallen" pose, this seeds only the first domino past its
    tip-over point and simulates the chain reaction.

    A positive pitch about the domino's body y-axis (composed after the
    upright/row-yaw quat, i.e. in the body frame) tips its body z-axis toward
    its body x-axis. Upright, it points along the row toward slot 1. A
    domino tips over once past ~atan(hx/hz) =~ 12 deg, so 25 deg guarantees a
    fall without an initial velocity kick.
    """
    n = cmd.num_envs
    origins = env.scene.env_origins
    zeros = torch.zeros(n, device=device)
    upright = quat_from_euler_xyz(zeros, zeros, cmd.row_yaw)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).expand(n, 3)
    theta = math.radians(25.0)
    tip = quat_from_angle_axis(torch.full((n,), theta, device=device), y_axis)
    tilted = quat_mul(upright, tip)

    hx, _hy, hz = cmd.variant.half_extents[0]
    pos = cmd.slot_goal_pos[:, 0].clone()
    pos[:, 2] = origins[:, 2] + hz * math.cos(theta) + hx * math.sin(theta) + 0.002

    domino = cmd.dominoes[0]
    domino.write_root_link_pose_to_sim(torch.cat([pos, tilted], dim=-1))
    domino.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=device))


def _check_domino_3pieces(device: str, *, render: bool = False) -> str | None:
    task_id = "LOOT-Domino-3Pieces-Dk1"
    env = _make_env(task_id, device)
    cmd = cast(DominoCommand, env.command_manager.get_term("task"))
    zeros = _zeros(env, device)
    env.step(zeros)
    _teleport_dominoes_upright(cmd, device)
    for _ in range(_STEPS):
        env.step(zeros)
    frac = cmd.progress.mean().item()
    reward_err = _reward_convention_error(env, zeros)
    _render_goal(env, task_id, render)
    env.close()
    return f"placed_fraction={frac:.3f} (expected >0.9)" if frac <= 0.9 else reward_err


def _check_domino_knockover(device: str, *, render: bool = False) -> str | None:
    """Build the row, then tip domino 0 and let the chain reaction propagate.

    Verifies the new chain semantics physically: only the first domino is
    seeded past its tip point, so the rest must fall from the actual contact
    cascade, along the row axis, in row order, for ``progress`` to reach 1.0.
    """
    task_id = "LOOT-Domino-3Pieces-KnockOver-Dk1"
    env = _make_env(task_id, device)
    cmd = cast(DominoCommand, env.command_manager.get_term("task"))
    zeros = _zeros(env, device)
    env.step(zeros)
    _teleport_dominoes_upright(cmd, device)
    env.step(zeros)  # all placed simultaneously -> phase latches to 1
    _tip_first_domino(cmd, env, device)
    for _ in range(_KNOCKOVER_STEPS):
        env.step(zeros)
    frac = cmd.progress.mean().item()
    reward_err = _reward_convention_error(env, zeros)
    _render_goal(env, task_id, render)
    env.close()
    return f"progress={frac:.3f} (expected >0.9)" if frac <= 0.9 else reward_err


# ---------------------------------------------------------------------------
# Shape sorter
# ---------------------------------------------------------------------------


def _vec_to_up_quat(ax: np.ndarray, device: str) -> torch.Tensor:
    """(4,) wxyz quaternion that rotates local vector *ax* onto world +z."""
    up = np.array([0.0, 0.0, 1.0])
    d = float(np.clip(np.dot(ax, up), -1.0, 1.0))
    if d > 0.9999:
        rot_axis, angle = np.array([1.0, 0.0, 0.0]), 0.0
    elif d < -0.9999:
        rot_axis, angle = np.array([1.0, 0.0, 0.0]), math.pi
    else:
        rot_axis = np.cross(ax, up)
        rot_axis /= np.linalg.norm(rot_axis)
        angle = math.acos(d)
    ca2, sa2 = math.cos(angle / 2), math.sin(angle / 2)
    return torch.tensor(
        [ca2, sa2 * rot_axis[0], sa2 * rot_axis[1], sa2 * rot_axis[2]],
        device=device,
        dtype=torch.float32,
    )


def _check_shape_sorter_full(device: str, *, render: bool = False) -> str | None:
    """Per socket: reorient the cube hole-up, two-station insert; all latch.

    The cube can only present one hole up at a time, so pieces are latched in
    sequence. The latch is monotonic (never clears except on reset).

    Each piece is parked out of the cube right after its own crossing is
    registered: three teleported pieces don't all fit in the small hollow
    interior at once and the latch not clearing on relocation means this
    doesn't affect what's being verified.
    """
    task_id = "LOOT-Sorter-3Shapes-Full-Dk1"
    env = _make_env(task_id, device)
    cmd = cast(ShapeSorterCommand, env.command_manager.get_term("task"))
    zeros = _zeros(env, device)
    env.step(zeros)

    layout = load_sorter_layout()
    holes = [layout.holes[s] for s in cmd.variant.shapes]
    piece_half = torch.tensor(
        [layout.pieces[s].half_len for s in cmd.variant.shapes], device=device
    )
    wall_t = torch.tensor([h.wall_thickness for h in holes], device=device)
    s_cross = -(wall_t + piece_half)

    board_pos = torch.tensor(BOARD_POS, device=device, dtype=torch.float32)
    origins = env.scene.env_origins

    for s in range(cmd.num_pieces):
        axis_local = np.array(holes[s].axis_out)
        quat = _vec_to_up_quat(axis_local, device).unsqueeze(0).expand(_NUM_ENVS, -1)
        pos = origins[:, :2] + board_pos[:2]
        z = origins[:, 2] + board_pos[2] + layout.half_size + 0.001
        cube_pos = torch.cat([pos, z.unsqueeze(-1)], dim=-1)
        cmd.cube.write_root_link_pose_to_sim(torch.cat([cube_pos, quat], dim=-1))
        cmd.cube.write_root_link_velocity_to_sim(
            torch.zeros(_NUM_ENVS, 6, device=device)
        )
        env.step(zeros)  # refresh socket_mouth_w/socket_axis_w from the new pose

        mouth = cmd.socket_mouth_w[:, s]
        axis = cmd.socket_axis_w[:, s]
        upright = _upright_quat(_NUM_ENVS, device)

        station1 = mouth + axis * (piece_half[s] + 0.005)
        cmd.pieces[s].write_root_link_pose_to_sim(
            torch.cat([station1, upright], dim=-1)
        )
        cmd.pieces[s].write_root_link_velocity_to_sim(
            torch.zeros(_NUM_ENVS, 6, device=device)
        )
        env.step(zeros)

        station2 = mouth + axis * (s_cross[s] - 0.005)
        cmd.pieces[s].write_root_link_pose_to_sim(
            torch.cat([station2, upright], dim=-1)
        )
        cmd.pieces[s].write_root_link_velocity_to_sim(
            torch.zeros(_NUM_ENVS, 6, device=device)
        )
        env.step(zeros)  # registers the crossing for piece s

        # Clear the cavity for the next piece (see docstring): the latch
        # already fired and is monotonic, so relocating doesn't unscore it.
        park = origins + torch.tensor([0.0, 0.0, -1.0], device=device)
        cmd.pieces[s].write_root_link_pose_to_sim(torch.cat([park, upright], dim=-1))
        cmd.pieces[s].write_root_link_velocity_to_sim(
            torch.zeros(_NUM_ENVS, 6, device=device)
        )
        env.step(zeros)

    for _ in range(_STEPS):
        env.step(zeros)
    frac = cmd.progress.mean().item()
    reward_err = _reward_convention_error(env, zeros)
    _render_goal(env, task_id, render)
    env.close()
    return f"latched_fraction={frac:.3f} (expected >0.9)" if frac <= 0.9 else reward_err


# ---------------------------------------------------------------------------
# Ball
# ---------------------------------------------------------------------------


def _teleport_ball(cmd: BallCommand, env: ManagerBasedRlEnv, device: str) -> None:
    """Place ball at target XY at its natural resting height."""
    ball_pos = cmd.target_pos_w.clone()
    if cmd.variant.task == "minigolf":
        # Ball resting on terrain inside the pit; avoid a large bounce.
        ball_pos[:, 2] = env.scene.env_origins[:, 2] + BALL_RADIUS + 0.001
    else:
        ball_pos[:, 2] = cmd.target_z + BALL_RADIUS
    quat = _upright_quat(_NUM_ENVS, device)
    cmd.ball.write_root_link_pose_to_sim(torch.cat([ball_pos, quat], dim=-1))
    cmd.ball.write_root_link_velocity_to_sim(torch.zeros(_NUM_ENVS, 6, device=device))


def _check_ball_at_target(
    device: str, task_id: str, *, render: bool = False
) -> str | None:
    env = _make_env(task_id, device)
    cmd = cast(BallCommand, env.command_manager.get_term("task"))
    zeros = _zeros(env, device)
    env.step(zeros)
    _teleport_ball(cmd, env, device)
    for _ in range(_STEPS):
        env.step(zeros)
    frac = cmd.at_target.mean().item()
    reward_err = _reward_convention_error(env, zeros)
    _render_goal(env, task_id, render)
    env.close()
    return (
        f"at_target_fraction={frac:.3f} (expected >0.9)" if frac <= 0.9 else reward_err
    )


def _segment_intersects_aabb(
    p0: tuple[float, float],
    p1: tuple[float, float],
    box_min: tuple[float, float],
    box_max: tuple[float, float],
) -> bool:
    """2D segment-vs-axis-aligned-box intersection (slab method)."""
    d = (p1[0] - p0[0], p1[1] - p0[1])
    tmin, tmax = 0.0, 1.0
    for axis in range(2):
        o, dd = p0[axis], d[axis]
        lo, hi = box_min[axis], box_max[axis]
        if abs(dd) < 1e-12:
            if o < lo or o > hi:
                return False
            continue
        t1, t2 = (lo - o) / dd, (hi - o) / dd
        if t1 > t2:
            t1, t2 = t2, t1
        tmin, tmax = max(tmin, t1), min(tmax, t2)
        if tmin > tmax:
            return False
    return True


def _check_minigolf_using_wall_blocks_direct_shots() -> str | None:
    """Pure-math check (no simulation): every straight line from a spawn
    corner to the pit crosses the blocker box, so a direct shot is never
    geometrically possible.

    Blocker geometry (must match ``ball/assets.py._add_blocker``):
    box centred at (0.58, 0.09), half extents (0.012, 0.13). Spawn rectangle
    matches ``BallCommandCfg``'s defaults (``BALL_SPAWN_X`` +/- 0.03 x,
    +/- 0.08 y).
    """
    variant = BALL_VARIANTS["MinigolfUsingWall"]
    pit = (TARGET_X, variant.pit_y)
    spawn_x = (BALL_SPAWN_X - 0.03, BALL_SPAWN_X + 0.03)
    spawn_y = (-0.08, 0.08)
    blocker_pos, blocker_half = (0.58, 0.09), (0.012, 0.13)
    box_min = (blocker_pos[0] - blocker_half[0], blocker_pos[1] - blocker_half[1])
    box_max = (blocker_pos[0] + blocker_half[0], blocker_pos[1] + blocker_half[1])

    misses = [
        (sx, sy)
        for sx in spawn_x
        for sy in spawn_y
        if not _segment_intersects_aabb((sx, sy), pit, box_min, box_max)
    ]
    if misses:
        return f"spawn corners with a clear line to the pit: {misses}"
    return None


def _check_ball_minigolf_using_wall(device: str, *, render: bool = False) -> str | None:
    task_id = "LOOT-Ball-Minigolf-UsingWall-Dk1"
    err = _check_ball_at_target(device, task_id, render=render)
    if err is not None:
        return err
    return _check_minigolf_using_wall_blocks_direct_shots()


# ---------------------------------------------------------------------------
# Test table
# ---------------------------------------------------------------------------

_PLACE_TASK_IDS: dict[str, str] = {
    "place_rands_randt": "LOOT-Place-RandS-RandT-Dk1",
    "place_rands_fixedt": "LOOT-Place-RandS-FixedT-Dk1",
    "place_fixeds_randt": "LOOT-Place-FixedS-RandT-Dk1",
    "place_fixeds_fixedt": "LOOT-Place-FixedS-FixedT-Dk1",
}

_CHECKS: list[tuple[str, str]] = [
    ("Place     RandS-RandT  ", "place_rands_randt"),
    ("Place     RandS-FixedT ", "place_rands_fixedt"),
    ("Place     FixedS-RandT ", "place_fixeds_randt"),
    ("Place     FixedS-FixedT", "place_fixeds_fixedt"),
    ("Stack     3Pis      ", "pi_3levels"),
    ("Stack     1Pi-HeldLintel", "pi_held_lintel"),
    ("Stack     1Pi-EmptyGrip", "pi_empty_gripper"),
    ("Domino    3Pieces       ", "domino_3pieces"),
    ("Domino    3Pieces-KnockOver", "domino_knockover"),
    ("Sorter    3Shapes-Full ", "shape_sorter_full"),
    ("Ball      Minigolf-Direct", "ball_minigolf_direct"),
    ("Ball      Minigolf-UsingWall", "ball_minigolf_using_wall"),
    ("Ball      Throw-U       ", "ball_throw_u"),
]


def _dispatch(key: str, device: str, *, render: bool = False) -> str | None:
    if key.startswith("place_"):
        return _check_place(_PLACE_TASK_IDS[key], device, render=render)
    if key == "pi_3levels":
        return _check_pi_3levels(device, render=render)
    if key == "pi_held_lintel":
        return _check_pi_held_lintel_excluded(device, render=render)
    if key == "pi_empty_gripper":
        return _check_pi_empty_gripper_no_grip(device, render=render)
    if key == "domino_3pieces":
        return _check_domino_3pieces(device, render=render)
    if key == "domino_knockover":
        return _check_domino_knockover(device, render=render)
    if key == "shape_sorter_full":
        return _check_shape_sorter_full(device, render=render)
    if key == "ball_minigolf_direct":
        return _check_ball_at_target(
            device, "LOOT-Ball-Minigolf-Direct-Dk1", render=render
        )
    if key == "ball_minigolf_using_wall":
        return _check_ball_minigolf_using_wall(device, render=render)
    if key == "ball_throw_u":
        return _check_ball_at_target(device, "LOOT-Ball-Throw-U-Dk1", render=render)
    return f"Unknown check key: {key}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify success metrics fire for representative LOOT tasks."
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU device.")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip saving a goal-state picture per check to out/render/<domain>/.",
    )
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    render = not args.no_render
    print(f"Device: {device}   Checks: {len(_CHECKS)}")
    if render:
        print("Goal pictures: out/render/<domain>/")
    print()

    failures: dict[str, str] = {}
    for label, key in _CHECKS:
        try:
            err = _dispatch(key, device, render=render)
        except Exception as exc:
            err = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        if err is None:
            print(f"  PASS  {label}")
        else:
            print(f"  FAIL  {label}  ({err})")
            failures[label] = err

    passed = len(_CHECKS) - len(failures)
    print()
    print(f"Result: {passed}/{len(_CHECKS)} passed, {len(failures)} failed")

    if failures:
        print()
        for label, tb in failures.items():
            if "\n" in tb:
                print(f"--- {label.strip()} ---")
                print(tb)
        sys.exit(1)


if __name__ == "__main__":
    main()
