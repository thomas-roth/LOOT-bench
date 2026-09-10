"""Success gating shared by all task categories

Contains the :class:`SuccessGate` that combines a task's criterion with the
universal at-rest + released conditions, plus the bore/hole geometry kernels
(``bore_axial_lateral``/``bore_axis_cos_tilt``) shared by the shape-sorter and
(archived) peg-insertion categories.
Only tensor math (unit-tested in ``tests/tasks/test_success.py``), no mjlab imports.

The category-specific success criteria (the pure placement tests) live in
each category's ``mdp/success.py``.
"""

from __future__ import annotations

import torch

# TODO: check with real robot
OBJECT_VEL_THRESHOLD = 0.01
"""Max linear speed (m/s) of a scored object to count success."""


def bore_axial_lateral(
    pos: torch.Tensor, mouth: torch.Tensor, axis_out: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose object-to-mouth offsets along/across a bore/hole axis.

    Shared by the shape-sorter and (archived) peg-insertion categories, whose
    peg/hole geometry is otherwise identical.

    Args:
        pos: (..., 3) object centre positions.
        mouth: (..., 3) bore/hole mouth positions.
        axis_out: (..., 3) unit outward bore/hole axes.

    Returns:
        ``(axial, lateral)``, each (...,): signed distance along the axis
        (+ is outside the bore/hole) and distance from its centreline.
    """
    rel = pos - mouth
    axial = torch.sum(rel * axis_out, dim=-1)
    lateral = torch.norm(rel - axial.unsqueeze(-1) * axis_out, dim=-1)
    return axial, lateral


def bore_axis_cos_tilt(quat_wxyz: torch.Tensor, axis_out: torch.Tensor) -> torch.Tensor:
    """|cos| of the angle between an object's body z-axis and a bore/hole axis.

    Sign-free: an object inserted "upside down" is equally aligned.

    Args:
        quat_wxyz: (..., 4) object orientations.
        axis_out: (..., 3) unit bore/hole axes.

    Returns:
        (...,) in [0, 1].
    """
    w, x, y, z = quat_wxyz.unbind(-1)
    obj_axis = torch.stack(
        [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=-1
    )
    return torch.sum(obj_axis * axis_out, dim=-1).abs()


class SuccessGate:
    """At-rest + released gating with a per-episode success latch.

    ``update`` combines the task's success criterion with two universal
    conditions:
    - the scored objects have (nearly) stopped moving (speed at or
    below ``vel_threshold``, inclusive)
    - the robot has let go

    It latches ``episode_success`` until :meth:`reset`.

    The release signal is injected because it varies per category
    (finger-pad contact force vs. ball-to-grasp-site distance).
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
