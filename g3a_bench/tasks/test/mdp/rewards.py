from __future__ import annotations

import torch
from mjlab.entity import Entity

from g3a_bench.tasks.test.mdp.utils import get_scene_command

SUCCESS_TOLERANCE = 0.04


def _root_pos(
    env,
    entity_name: str,
) -> torch.Tensor:
    """Return root body position.

    Shape: (num_envs, 3)
    """
    entity: Entity = env.scene[entity_name]
    return entity.data.root_link_pos_w


def cube_success(
    env,
    tolerance: float = SUCCESS_TOLERANCE,
) -> torch.Tensor:
    """OGBench-style cube success."""
    cmd = get_scene_command(env)
    cube_pos = _root_pos(env, "cube")
    distance = torch.norm(cube_pos - cmd.target_cube_pos, dim=-1)
    return distance <= tolerance


def button_successes(env) -> torch.Tensor:
    """OGBench-style button successes. Shape: (num_envs, 2)"""
    cmd = get_scene_command(env)
    return cmd.scene_button_states == cmd.target_button_states


def drawer_success(
    env,
    tolerance: float = SUCCESS_TOLERANCE,
) -> torch.Tensor:
    """OGBench-style drawer success."""
    cmd = get_scene_command(env)
    drawer_pos = cmd._drawer.data.joint_pos[:, cmd._drawer_joint_slot[0]]
    return torch.abs(drawer_pos - cmd.target_drawer_pos) <= tolerance


def window_success(
    env,
    tolerance: float = SUCCESS_TOLERANCE,
) -> torch.Tensor:
    """OGBench-style window success."""
    cmd = get_scene_command(env)
    window_pos = cmd._window.data.joint_pos[:, cmd._window_joint_slot[0]]
    return torch.abs(window_pos - cmd.target_window_pos) <= tolerance


def scene_successes(env) -> torch.Tensor:
    """Return all OGBench-style success checks.

    Shape: (num_envs, 5)

    Columns:
      0 cube
      1 button 0
      2 button 1
      3 drawer
      4 window
    """
    cube = cube_success(env).unsqueeze(-1)
    buttons = button_successes(env)
    drawer = drawer_success(env).unsqueeze(-1)
    window = window_success(env).unsqueeze(-1)

    return torch.cat(
        [
            cube,
            buttons,
            drawer,
            window,
        ],
        dim=-1,
    )


def scene_ogbench_reward(env) -> torch.Tensor:
    """OGBench-style sparse Scene reward.

    For one cube, two buttons, drawer, and window:

      reward = number_of_successes - 5

    Reward range:
      -5 = nothing correct
       0 = everything correct
    """
    successes = scene_successes(env).float()
    num_successes = successes.sum(dim=-1)
    total_successes = successes.shape[-1]

    return num_successes - total_successes


def scene_success(env) -> torch.Tensor:
    """Full task success.

    True only if all object-level successes are true.
    """
    return scene_successes(env).all(dim=-1)
