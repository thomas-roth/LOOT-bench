from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from g3a_bench.tasks.scene.mdp.commands import SceneTaskCommand

BUTTON_PRESS_THRESHOLD = -0.02


def get_scene_command(env) -> SceneTaskCommand:
    from g3a_bench.tasks.scene.mdp.commands import SceneTaskCommand  # noqa: I001 lazy: avoids circular init

    return cast(SceneTaskCommand, env.command_manager.get_term("scene_task"))


def detect_new_button_presses(
    prev_button_pos: torch.Tensor,
    cur_button_pos: torch.Tensor,
    threshold: float = BUTTON_PRESS_THRESHOLD,
) -> torch.Tensor:
    """Detect falling-edge button presses.

    A press happens when:
      previous position > threshold
      current position <= threshold

    Shape: (num_envs, 2)
    """
    was_not_pressed = prev_button_pos > threshold
    is_pressed = cur_button_pos <= threshold
    return was_not_pressed & is_pressed
