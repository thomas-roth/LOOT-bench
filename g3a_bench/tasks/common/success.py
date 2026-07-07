"""Success gating shared by all task categories.

The category-specific success criteria (the pure placement tests) live in
each category's ``mdp/success.py``; this module holds only what every
category shares: the at-rest speed threshold and the :class:`SuccessGate`
that combines a task's criterion with the universal at-rest + released
conditions. No mjlab or environment imports — plain tensor math, unit-tested
in ``tests/tasks/test_success.py``.
"""

from __future__ import annotations

import torch

OBJECT_VEL_THRESHOLD = 0.05
"""Max linear speed (m/s) of a scored object to count success."""


class SuccessGate:
    """At-rest + released gating with a per-episode success latch.

    ``update`` combines the task's success criterion with two universal
    conditions — the scored objects have (nearly) stopped moving (speed at or
    below ``vel_threshold``, inclusive) and the robot has let go — and
    latches ``episode_success`` until :meth:`reset`. The release signal is
    injected because it varies per category (gripper-joint opening vs.
    ball-to-gripper distance).
    """

    def __init__(
        self, num_envs: int, device: str | torch.device, *, vel_threshold: float
    ):
        self.vel_threshold = vel_threshold
        self.success = torch.zeros(num_envs, device=device)
        self.episode_success = torch.zeros(num_envs, device=device)

    def update(
        self, task_done: torch.Tensor, speed: torch.Tensor, released: torch.Tensor
    ) -> torch.Tensor:
        """Advance one step; returns the (E,) float success signal.

        Args:
            task_done: (E,) bool result of the task's success criterion.
            speed: (E,) or (E, K) linear speeds; with multiple scored objects
                the slowest-common denominator (max over K) must be at rest.
            released: (E,) bool.
        """
        if speed.dim() > 1:
            speed = speed.amax(dim=-1)
        at_rest = speed <= self.vel_threshold
        self.success = (task_done & at_rest & released).float()
        self.episode_success = torch.maximum(self.episode_success, self.success)
        return self.success

    def reset(self, env_ids: torch.Tensor) -> None:
        self.success[env_ids] = 0.0
        self.episode_success[env_ids] = 0.0
