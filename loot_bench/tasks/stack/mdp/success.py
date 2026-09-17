"""Pure success criteria for the pi (stack) category.

Plain tensor math, no mjlab or environment imports: the command term
(``commands.py``) is the adapter that gathers sim buffers and calls these,
so the success semantics live (and are unit-tested in
``tests/tasks/test_success.py``) in exactly one place, and
``loot-split-demos`` recomputes them offline through the same functions.

Success tests the **lintel**, never the structure under it: level ``k`` is
complete when some block is horizontal, unheld, and has its centre within
tolerance of ``k * LEVEL_HEIGHT - BLOCK_WIDTH / 2`` above the table. A
trilithon and a T both satisfy it; the criterion encodes neither
(ADR-020).
"""

from __future__ import annotations

import torch

from loot_bench.tasks.common.success import bore_axis_cos_tilt
from loot_bench.tasks.stack.variants import TILT_COS_MAX


def block_is_horizontal(
    quat_wxyz: torch.Tensor, max_cos: float = TILT_COS_MAX
) -> torch.Tensor:
    """Whether each block's long axis lies within the tilt gate of level.

    ``bore_axis_cos_tilt`` against world z gives |cos(long axis, up)|: an
    upright block reads ~1.0, a level lintel ~0.0, and horizontal means at or
    below ``max_cos``.

    The gate is not a "looks level enough" threshold. A lintel spanning two
    piles of unequal top height rests at their mean and is tilted by
    ``atan(dz / (L - w))``; the gate is what bounds ``dz`` and so excludes
    those builds. At the 0.26 (15 deg) the other categories use, four such
    upright/flat builds fit inside the block budget and ``HEIGHT_TOL``; at
    0.10 (5.7 deg) none do. See ADR-020 and the import-time assertion in
    ``variants.py``, which is why the constant is imported from there and
    not redefined here.

    What the gate does *not* do: a support leaning at an intermediate angle
    presents a continuum of top heights, so it can carry a near-perfectly
    level lintel to a target height. That is a known, accepted limit
    (ADR-020); do not try to close it here.

    Body quaternions come from ``data.xquat``, which MuJoCo integrates
    continuously, so this is safe to read off the block pose. ADR-007's
    sign-flip instability is a grasp-*site* problem (a matrix-to-quaternion
    conversion on a tie) and does not apply.

    Args:
        quat_wxyz: (..., 4) block orientations.
        max_cos: the tilt gate, |cos| inclusive.

    Returns:
        (...,) bool.
    """
    up = torch.zeros_like(quat_wxyz[..., :3])
    up[..., 2] = 1.0
    return bore_axis_cos_tilt(quat_wxyz, up) <= max_cos


def pi_levels_complete(
    centre_z: torch.Tensor,
    quat_wxyz: torch.Tensor,
    table_z: torch.Tensor | float,
    level_heights: torch.Tensor,
    tol: float,
    exclude: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-level "some block is a lintel at this level's height" test.

    Level ``k`` is complete when some block is horizontal, not held, and has
    its centre within ``tol`` of ``level_heights[k]`` *above the table*
    (``table_z`` is subtracted, so the targets are table-relative like every
    other category's). The centre is measured, not the top face: a lintel on
    two uneven uprights sits at their mean, so its centre is exact geometry,
    while a box-AABB top at the gate's tilt limit reads 3.2 mm high.

    ``exclude`` zeroes blocks currently held in the gripper (ADR-017): a
    lintel dangling at the right height earns nothing until it is released,
    and counts again the instant it is let go.

    Args:
        centre_z: (E, N) block centre z, world.
        quat_wxyz: (E, N, 4) block orientations.
        table_z: (E,) or scalar table-surface z, world.
        level_heights: (K,) target centre heights above the table.
        tol: max |height - target| (inclusive).
        exclude: optional (E, N) bool held-block mask.

    Returns:
        (E, K) bool.
    """
    if isinstance(table_z, torch.Tensor):
        table_z = table_z.unsqueeze(-1)
    height = centre_z - table_z  # (E, N)
    lintel = block_is_horizontal(quat_wxyz)  # (E, N)
    if exclude is not None:
        lintel = lintel & ~exclude
    at_level = (height.unsqueeze(-1) - level_heights).abs() <= tol  # (E, N, K)
    return (at_level & lintel.unsqueeze(-1)).any(dim=1)


def pi_progress(levels_complete: torch.Tensor) -> torch.Tensor:
    """Fraction of levels standing right now, (E,) in [0, 1].

    Instantaneous, not latched: a collapse drops it back, because the work
    really was undone and the agent should see that. The success gate's
    per-episode latch (``SuccessGate.episode_success``) is a separate thing.
    """
    return levels_complete.float().mean(dim=-1)
