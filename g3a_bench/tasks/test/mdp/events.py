from __future__ import annotations

import torch

from .utils import (
    apply_drawer_window_locks,
    detect_new_button_presses,
    get_button_positions,
    toggle_button_states,
)


def reset_scene_logic(env, env_ids=None):
    """Initialize logical Scene states at reset."""
    num_envs = env.scene.num_envs

    env.scene_button_states = torch.zeros((num_envs, 2), device=env.device)
    env.prev_button_positions = get_button_positions(env).clone()

    apply_drawer_window_locks(env)


def update_scene_logic(env, env_ids=None):
    """Update logical button states after each step."""
    if not hasattr(env, "scene_button_states"):
        reset_scene_logic(env)

    cur_button_positions = get_button_positions(env)

    new_presses = detect_new_button_presses(
        env.prev_button_positions,
        cur_button_positions,
    )

    env.scene_button_states = toggle_button_states(
        env.scene_button_states,
        new_presses,
    )

    env.prev_button_positions = cur_button_positions.clone()

    apply_drawer_window_locks(env)
