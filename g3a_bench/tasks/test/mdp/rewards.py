from __future__ import annotations

import torch
from mjlab.entity import Entity

SUCCESS_TOLERANCE = 0.04


def _joint_pos_by_name(
    env,
    entity_name: str,
    joint_name: str,
) -> torch.Tensor:
    """Return one joint position.

    Shape: (num_envs,)
    """
    entity: Entity = env.scene[entity_name]
    joint_id = entity.joint_names.index(joint_name)
    return entity.data.joint_pos[:, joint_id]


def _root_pos(
    env,
    entity_name: str,
) -> torch.Tensor:
    """Return root body position.

    Shape: (num_envs, 3)
    """
    entity: Entity = env.scene[entity_name]
    return entity.data.root_link_pos_w


def _ensure_scene_targets(env):
    """Create temporary fixed targets if they do not exist yet.

    This is Step 4 target logic.
    Later Step 6 should replace this with a real command/task system.
    """
    num_envs = env.scene.num_envs
    device = env.device

    if not hasattr(env, "scene_target_cube_pos"):
        env.scene_target_cube_pos = torch.tensor(
            [0.35, 0.05, 0.02],
            device=device,
        ).repeat(num_envs, 1)

    if not hasattr(env, "scene_target_button_states"):
        env.scene_target_button_states = torch.ones(
            (num_envs, 2),
            device=device,
        )

    if not hasattr(env, "scene_target_drawer_pos"):
        env.scene_target_drawer_pos = torch.full(
            (num_envs,),
            -0.16,
            device=device,
        )

    if not hasattr(env, "scene_target_window_pos"):
        env.scene_target_window_pos = torch.full(
            (num_envs,),
            0.2,
            device=device,
        )


def cube_success(
    env,
    tolerance: float = SUCCESS_TOLERANCE,
) -> torch.Tensor:
    """OGBench-style cube success.

    Success if cube position is within tolerance of target cube position.
    """
    _ensure_scene_targets(env)

    cube_pos = _root_pos(env, "cube")
    target_pos = env.scene_target_cube_pos

    distance = torch.norm(cube_pos - target_pos, dim=-1)
    return distance <= tolerance


def button_successes(env) -> torch.Tensor:
    """OGBench-style button successes.

    Shape: (num_envs, 2)
    """
    _ensure_scene_targets(env)

    if not hasattr(env, "scene_button_states"):
        return torch.zeros(
            (env.scene.num_envs, 2),
            device=env.device,
            dtype=torch.bool,
        )

    return env.scene_button_states == env.scene_target_button_states


def drawer_success(
    env,
    tolerance: float = SUCCESS_TOLERANCE,
) -> torch.Tensor:
    """OGBench-style drawer success."""
    _ensure_scene_targets(env)

    drawer_pos = _joint_pos_by_name(
        env,
        entity_name="drawer",
        joint_name="drawer_slide",
    )

    return torch.abs(drawer_pos - env.scene_target_drawer_pos) <= tolerance


def window_success(
    env,
    tolerance: float = SUCCESS_TOLERANCE,
) -> torch.Tensor:
    """OGBench-style window success."""
    _ensure_scene_targets(env)

    window_pos = _joint_pos_by_name(
        env,
        entity_name="window",
        joint_name="window_slide",
    )

    return torch.abs(window_pos - env.scene_target_window_pos) <= tolerance


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
