"""Termination terms for the cube task category."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

from g3a_bench.tasks.cube.mdp.commands import get_cube_command


def arrangement_complete(
    env: ManagerBasedRlEnv, command_name: str, hold_time: float = 1.0
) -> torch.Tensor:
    """All slots placed continuously for `hold_time` seconds."""
    command = get_cube_command(env, command_name)
    return command.success_steps * env.step_dt >= hold_time
