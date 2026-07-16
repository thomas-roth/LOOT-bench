"""Pure success criteria for the stacking category.

Plain tensor math, no mjlab or environment imports: the command term
(``commands.py``) acts as the adapter that gathers sim buffers and calls
these, so the success semantics live (and are unit-tested in
``tests/tasks/test_success.py``) in exactly one place.

Success is *height only*: the top face of some block must reach the target
height above the table, regardless of xy position or which blocks form the
stack. TwoStacks additionally requires a second reached block far enough away
in xy to belong to a separate stack.
"""

from __future__ import annotations

import torch


def _heights(top_z: torch.Tensor, table_z: torch.Tensor | float) -> torch.Tensor:
    """Top-face heights above the table, (E, N)."""
    if isinstance(table_z, torch.Tensor):
        table_z = table_z.unsqueeze(-1)
    return top_z - table_z


def block_top_heights(pos_z: torch.Tensor, half_h: torch.Tensor) -> torch.Tensor:
    """Top-face z per block.

    Args:
        pos_z: (E, N) block-centre z (world).
        half_h: (N,) block half heights.

    Returns:
        (E, N) top-face z. Only exact for upright blocks; a toppled block
        reads lower, which is the desired conservative behavior.
    """
    return pos_z + half_h


def stack_reached(
    top_z: torch.Tensor,
    table_z: torch.Tensor | float,
    target_height: float,
    tol: float,
) -> torch.Tensor:
    """Per-block "top face at the target height" test.

    Args:
        top_z: (E, N) top-face z (world).
        table_z: (E,) or scalar table-surface z (world).
        target_height: goal height above the table.
        tol: max |height - target_height| (inclusive).

    Returns:
        (E, N) bool.
    """
    return (_heights(top_z, table_z) - target_height).abs() <= tol


def two_stacks_success(
    reached: torch.Tensor, xy: torch.Tensor, min_separation: float
) -> torch.Tensor:
    """Two reached blocks separated enough to top two distinct stacks.

    Args:
        reached: (E, N) bool per-block :func:`stack_reached` result.
        xy: (E, N, 2) block xy positions.
        min_separation: min xy distance between the two blocks (inclusive).

    Returns:
        (E,) bool.
    """
    dist = torch.cdist(xy, xy)  # (E, N, N)
    pair = reached.unsqueeze(-1) & reached.unsqueeze(-2) & (dist >= min_separation)
    return pair.any(dim=-1).any(dim=-1)


def stacking_progress(
    top_z: torch.Tensor, table_z: torch.Tensor | float, target_height: float
) -> torch.Tensor:
    """Fraction of the target height reached by the tallest block, (E,) in [0, 1]."""
    best = _heights(top_z, table_z).amax(dim=-1)
    return (best / target_height).clamp(0.0, 1.0)


def two_stacks_progress(
    top_z: torch.Tensor,
    xy: torch.Tensor,
    table_z: torch.Tensor | float,
    target_height: float,
    min_separation: float,
) -> torch.Tensor:
    """Mean height fraction of the two best-separated stack leaders, (E,).

    The leader is the tallest block; the second stack is represented by the
    tallest block at least ``min_separation`` away from the leader in xy
    (fraction 0 while no block is separated enough).
    """
    frac = (_heights(top_z, table_z) / target_height).clamp(0.0, 1.0)  # (E, N)
    lead = frac.argmax(dim=-1)  # (E,)
    env = torch.arange(xy.shape[0], device=xy.device)
    lead_xy = xy[env, lead]  # (E, 2)
    dist = (xy - lead_xy.unsqueeze(1)).norm(dim=-1)  # (E, N)
    second = torch.where(dist >= min_separation, frac, 0.0).amax(dim=-1)
    return (frac[env, lead] + second) / 2.0
