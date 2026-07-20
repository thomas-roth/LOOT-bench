"""Termination terms for the peg-insertion task category (archived, unregistered)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from g3a_bench.tasks.peg_insertion.mdp.commands import get_command as _command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def insertion_complete(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """All pegs inserted, at rest, and released."""
    return _command(env, command_name).success.bool()
