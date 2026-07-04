"""Termination terms for the cube task category."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

from g3a_bench.tasks.cube.mdp.commands import get_cube_command


def arrangement_complete(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Position, velocity, and gripper conditions all met."""
    return get_cube_command(env, command_name).success.bool()
