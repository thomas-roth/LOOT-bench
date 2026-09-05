"""Pure success criteria for the place category.

Plain tensor math, no mjlab or environment imports: the command term
(``commands.py``) is the adapter that gathers sim buffers and calls these,
so the success semantics live (and are unit-tested in
``tests/tasks/test_success.py``) in exactly one place.

Success is *position only*: no orientation term, unlike domino's
row-yaw-aligned upright check. This is "move to a location," not
"move and orient."
"""

from __future__ import annotations

import torch


def place_reached(dist_to_target: torch.Tensor, pos_threshold: float) -> torch.Tensor:
    """Cube centre within ``pos_threshold`` of the target (3D, position-only).

    Args:
        dist_to_target: (E,) Euclidean distance, cube centre to target point.
        pos_threshold: max distance (exclusive) to count as placed.

    Returns:
        (E,) bool.
    """
    return dist_to_target < pos_threshold


def place_progress(
    dist_to_target: torch.Tensor, max_travel_dist: float
) -> torch.Tensor:
    """Continuous distance-based shaping, in [0, 1].

    1.0 exactly at the target, decreasing linearly to 0.0 at
    ``max_travel_dist`` away and clamped beyond it. Genuinely continuous even
    for a single object. Contrast domino's ``slot_at_goal.mean(dim=1)``,
    which only reads as continuous because there are several tiles.

    Args:
        dist_to_target: (E,) Euclidean distance, cube centre to target point.
        max_travel_dist: fixed normalization constant (the largest possible
            spawn->target separation given the task's workspace geometry;
            see ``mdp/commands.py::MAX_TRAVEL_DIST``), not per-episode state.

    Returns:
        (E,) in [0, 1].
    """
    return (1.0 - dist_to_target / max_travel_dist).clamp(0.0, 1.0)
