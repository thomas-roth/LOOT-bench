from __future__ import annotations

import torch
from mjlab.entity import Entity

BUTTON_PRESS_THRESHOLD = -0.02


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


LOCKED_DAMPING = 1e6
UNLOCKED_DAMPING = 2.0


def apply_drawer_window_locks(env):
    import mujoco
    import numpy as np
    import torch

    mj_model = env.sim.mj_model
    wp_model = env.sim.wp_model

    drawer_joint_id = mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "drawer_slide",
    )
    window_joint_id = mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "window_slide",
    )

    drawer_dof_id = mj_model.jnt_dofadr[drawer_joint_id]
    window_dof_id = mj_model.jnt_dofadr[window_joint_id]

    button_states = env.scene_button_states

    drawer_unlocked = button_states[:, 0] > 0.5
    window_unlocked = button_states[:, 1] > 0.5

    drawer_damping = torch.where(
        drawer_unlocked,
        torch.full_like(button_states[:, 0], UNLOCKED_DAMPING),
        torch.full_like(button_states[:, 0], LOCKED_DAMPING),
    )

    window_damping = torch.where(
        window_unlocked,
        torch.full_like(button_states[:, 1], UNLOCKED_DAMPING),
        torch.full_like(button_states[:, 1], LOCKED_DAMPING),
    )

    damping_np = wp_model.dof_damping.numpy().copy()

    damping_np[:, drawer_dof_id] = drawer_damping.detach().cpu().numpy()
    damping_np[:, window_dof_id] = window_damping.detach().cpu().numpy()

    wp_model.dof_damping.assign(damping_np.astype(np.float32))
