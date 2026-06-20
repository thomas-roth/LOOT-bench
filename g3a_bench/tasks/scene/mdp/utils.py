from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity

if TYPE_CHECKING:
    from g3a_bench.tasks.scene.mdp.commands import SceneTaskCommand

BUTTON_PRESS_THRESHOLD = -0.02


def get_scene_command(env) -> SceneTaskCommand:
    from g3a_bench.tasks.scene.mdp.commands import SceneTaskCommand  # noqa: I001 lazy: avoids circular init

    return cast(SceneTaskCommand, env.command_manager.get_term("scene_task"))


def get_button_positions(env) -> torch.Tensor:
    """Return physical button joint positions.

    Shape: (num_envs, 2)
    """
    buttons: Entity = env.scene["buttons"]
    joint_names = buttons.joint_names

    button_0_id = joint_names.index("buttonbox_joint_0")
    button_1_id = joint_names.index("buttonbox_joint_1")

    return buttons.data.joint_pos[:, [button_0_id, button_1_id]]


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


def toggle_button_states(
    button_states: torch.Tensor,
    new_presses: torch.Tensor,
) -> torch.Tensor:
    """Toggle logical button states where a new press occurred.

    button_states: 0/1 tensor
    new_presses: boolean tensor

    Shape: (num_envs, 2)
    """
    return torch.where(
        new_presses,
        1.0 - button_states,
        button_states,
    )
