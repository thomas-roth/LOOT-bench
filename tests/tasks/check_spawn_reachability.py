"""IK reachability check over every spawn cell. See tests/tasks/README.md.

For every cell of the stack, domino and sorter spawn tables — plus the
sorter cube pin and Place's fixed spawn/target pins — at all four corners
of its jitter box (pins are exact), this drives the arm with the
environment's own differential-IK action term
(:func:`delta_action_from_poses`) to a hover above the corner with the
gripper straight down, and reports the worst position residual.

It runs as a script rather than a unit test because it needs the compiled
model. Only one task is built: the robot and the table are shared by every
category, so one compiled model covers all tables.
"""

from __future__ import annotations

import argparse
import math
import sys

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg

import loot_bench  # noqa: F401  (registers all tasks)
from loot_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from loot_bench.tasks.common.actions import delta_action_from_poses
from loot_bench.tasks.common.fixed_spawns import SPAWN_JITTER
from loot_bench.tasks.domino.mdp.commands import DOMINO_SPAWN_CELLS
from loot_bench.tasks.place.mdp.commands import FIXED_SPAWN_XY, FIXED_TARGET_XY
from loot_bench.tasks.shape_sorter.assets import BOARD_POS
from loot_bench.tasks.shape_sorter.mdp.commands import SORTER_SPAWN_CELLS
from loot_bench.tasks.stack.mdp.commands import STACK_SPAWN_CELLS

_BUILD_TASK = "LOOT-Stack3-Easy-Dk1"
"""Task the check builds (robot and table are shared by every category)."""

_HOVER = 0.05
"""[m] grasp-site height above the table surface.

Above the tallest spawn (a Hard block tops out at 0.06 m standing, but the
site hovers over the cell, not over a block) and below the reach dome's
ceiling (0.20 m even at the far cells), so a solve here means the gripper
can descend onto anything staged in the cell.
"""

_MAX_STEPS = 200
"""IK drive budget per corner."""

_SETTLE_RESIDUAL = 5e-4
"""[m] position residual that stops the drive early."""

_PASS_RESIDUAL = 5e-3
"""[m] worst residual the check accepts."""


def _targets() -> list[tuple[str, float, float]]:
    """``(label, x, y)`` over every cell at all four jitter corners."""
    out: list[tuple[str, float, float]] = []
    cube_pin: tuple[float, float] = (float(BOARD_POS[0]), float(BOARD_POS[1]))
    tables: list[tuple[str, tuple[tuple[float, float], ...]]] = [
        ("stack", STACK_SPAWN_CELLS),
        ("domino", DOMINO_SPAWN_CELLS),
        ("sorter", SORTER_SPAWN_CELLS),
        ("sorter-cube", (cube_pin,)),
    ]
    for name, cells in tables:
        for i, (cx, cy) in enumerate(cells):
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    out.append(
                        (
                            f"{name}[{i}]({sx:+.0f},{sy:+.0f})",
                            cx + sx * SPAWN_JITTER,
                            cy + sy * SPAWN_JITTER,
                        )
                    )
    for name, pin in (
        ("place-spawn", FIXED_SPAWN_XY),
        ("place-target", FIXED_TARGET_XY),
    ):
        out.append((name, pin[0], pin[1]))
    return out


def _solve(
    env: ManagerBasedRlEnv,
    site_ids: list[int] | slice,
    home_quat: torch.Tensor,
    tx: float,
    ty: float,
    tz: float,
    device: str,
) -> tuple[float, int]:
    """Drive the grasp site to ``(tx, ty, tz)``; return best residual, steps."""
    env.reset()
    zeros = torch.zeros((1, env.action_manager.action.shape[1]), device=device)
    env.step(zeros)  # settle buffers after the reset
    robot = env.scene["robot"]
    origins = env.scene.env_origins
    target_pos = torch.tensor([[tx, ty, tz]], device=device)
    best = math.inf
    steps = 0
    for step in range(_MAX_STEPS):
        pos = robot.data.site_pos_w[:, site_ids].squeeze(1) - origins
        quat = robot.data.site_quat_w[:, site_ids].squeeze(1)
        residual = float((pos - target_pos).norm())
        best = min(best, residual)
        steps = step + 1
        if residual <= _SETTLE_RESIDUAL:
            break
        delta = delta_action_from_poses(pos, quat, target_pos, home_quat)
        action = torch.cat([delta, torch.zeros(1, 1, device=device)], dim=-1)
        env.step(action)
    return best, steps


def main() -> None:
    parser = argparse.ArgumentParser(description="IK reachability over spawn cells.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU device.")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_env_cfg(_BUILD_TASK)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.reset()

    grasp_site = get_unimanual_dk1_interface().grasp_site
    site_cfg = SceneEntityCfg("robot", site_names=(grasp_site,))
    site_cfg.resolve(env.scene)
    site_ids = site_cfg.site_ids
    robot = env.scene["robot"]
    home_quat = robot.data.site_quat_w[:, site_ids].squeeze(1).clone()
    table_z = float(env.scene.env_origins[0, 2])

    targets = _targets()
    print(f"Device: {device}   Task: {_BUILD_TASK}   Targets: {len(targets)}")
    print()

    worst: dict[str, tuple[float, str]] = {}
    overall = (0.0, "")
    for label, x, y in targets:
        residual, steps = _solve(
            env, site_ids, home_quat, x, y, table_z + _HOVER, device
        )
        group = label.split("[")[0]
        if residual > worst.get(group, (0.0, ""))[0]:
            worst[group] = (residual, label)
        if residual > overall[0]:
            overall = (residual, label)
        print(f"  {residual * 1000:7.3f} mm  {steps:3d} steps  {label}")

    env.close()
    print()
    for group in sorted(worst):
        residual, label = worst[group]
        print(f"  worst {group}: {residual * 1000:.3f} mm at {label}")
    print(f"  worst overall: {overall[0] * 1000:.3f} mm at {overall[1]}")
    print()
    if overall[0] > _PASS_RESIDUAL:
        print(
            f"Result: FAIL (worst {overall[0] * 1000:.3f} mm > {_PASS_RESIDUAL * 1000:.0f} mm)"
        )
        sys.exit(1)
    print(f"Result: PASS (worst {overall[0] * 1000:.3f} mm)")


if __name__ == "__main__":
    main()
