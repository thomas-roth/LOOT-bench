"""Pure success criteria for the ball category.

Plain tensor math, no mjlab or environment imports: the command term
(``commands.py``) acts as the adapter that gathers sim buffers and calls
these, so the success semantics live (and are unit-tested in
``tests/tasks/test_success.py``) in exactly one place.

Not to be confused with the *termination term* ``terminations.ball_at_target``,
which additionally folds in the at-rest and released gating via the command's
:class:`~loot_bench.tasks.common.success.SuccessGate`.
"""

from __future__ import annotations

import torch


def ball_in_target_zone(
    ball_pos: torch.Tensor,
    target_pos: torch.Tensor,
    *,
    radius: float,
    surface_z: float,
    is_hole: bool,
    ball_radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """xy-radius target test for the ball.

    Hole targets additionally require the ball to have dropped below the
    platform surface; flat targets require it to rest *on* the surface (not
    flying over it).

    Args:
        ball_pos: (E, 3).
        target_pos: (E, 3).

    Returns:
        ``(at_target, d_xy)``: (E,) bool and the (E,) xy distance (for
        metrics).
    """
    d_xy = torch.norm(ball_pos[:, :2] - target_pos[:, :2], dim=-1)
    within = d_xy < radius
    if is_hole:
        at_target = within & (ball_pos[:, 2] < surface_z)
    else:
        at_target = within & (ball_pos[:, 2] < surface_z + 3 * ball_radius)
    return at_target, d_xy
