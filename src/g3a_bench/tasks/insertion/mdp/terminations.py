"""Termination terms for the object-insertion task category."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from g3a_bench.tasks.insertion.mdp.commands import InsertionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def insertion_complete(
    env: ManagerBasedRlEnv,
    command_name: str,
    hold_time: float = 0.5,
) -> torch.Tensor:
    """All pegs inserted continuously for ``hold_time`` seconds."""
    command = cast(InsertionCommand, env.command_manager.get_term(command_name))
    return command.success_steps * env.step_dt >= hold_time
