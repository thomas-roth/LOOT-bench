"""Termination terms for the cube task category."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from g3a_bench.tasks.cube.mdp.commands import PieceArrangementCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def arrangement_complete(
    env: ManagerBasedRlEnv,
    command_name: str,
    hold_time: float = 1.0,
) -> torch.Tensor:
    """All slots placed continuously for `hold_time` seconds."""
    command = cast(PieceArrangementCommand, env.command_manager.get_term(command_name))
    return command.success_steps * env.step_dt >= hold_time
