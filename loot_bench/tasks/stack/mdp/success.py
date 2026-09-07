"""Pure success criteria for the stack category.

Plain tensor math, no mjlab or environment imports: the command term
(``commands.py``) acts as the adapter that gathers sim buffers and calls
these, so the success semantics live (and are unit-tested in
``tests/tasks/test_success.py``) in exactly one place.

Success is *height only*: the top face of some block must reach the target
height above the table, regardless of xy position or which blocks form the
stack. All blocks of a variant share one height, so only a stack of exactly
``target_blocks`` blocks lands within tolerance.
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


def stack_progress(
    top_z: torch.Tensor,
    table_z: torch.Tensor | float,
    target_height: float,
    exclude: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fraction of the target height reached by the tallest block, (E,) in [0, 1].

    ``exclude`` (E, N) bool zeroes out blocks currently held in the gripper, so
    a block merely lifted to the target height earns no progress until it is
    placed and released (ADR-017; the mask is built in ``commands.py``).
    """
    frac = (_heights(top_z, table_z) / target_height).clamp(0.0, 1.0)
    if exclude is not None:
        frac = frac.masked_fill(exclude, 0.0)
    return frac.amax(dim=-1)
