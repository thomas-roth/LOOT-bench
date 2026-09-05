"""Pure success criteria for the domino category.

Plain tensor math, no mjlab or environment imports: the command term
(``commands.py``) acts as the adapter that gathers sim buffers and calls
these, so the success semantics live (and are unit-tested in
``tests/tasks/test_success.py``) in exactly one place.

Quaternions are wxyz, matching mjlab.
"""

from __future__ import annotations

import math

import torch


def domino_upright_aligned(
    quat_wxyz: torch.Tensor, row_yaw: torch.Tensor, threshold_deg: float
) -> torch.Tensor:
    """Upright + row-alignment check for dominoes.

    A domino counts as standing when its body z-axis is within
    ``threshold_deg`` of world z *and* its body x-axis (the thin side) is
    within ``threshold_deg`` of the row direction mod pi, so the dominoes
    can knock each other over regardless of which way they face.

    Args:
        quat_wxyz: (E, G, 4) piece orientations.
        row_yaw: (E,) row direction.
        threshold_deg: max tilt and yaw misalignment.

    Returns:
        (E, G) bool.
    """
    w, x, y, z = quat_wxyz.unbind(-1)
    cos_tol = math.cos(math.radians(threshold_deg))
    # Body z-axis dot world z.
    up_cos = 1.0 - 2.0 * (x * x + y * y)
    # Body x-axis in the world xy-plane.
    bx_x = 1.0 - 2.0 * (y * y + z * z)
    bx_y = 2.0 * (x * y + w * z)
    dir_x = torch.cos(row_yaw)[:, None]
    dir_y = torch.sin(row_yaw)[:, None]
    align = (bx_x * dir_x + bx_y * dir_y).abs()
    return (up_cos > cos_tol) & (align > cos_tol)


def domino_fallen_along_row(
    quat_wxyz: torch.Tensor, row_yaw: torch.Tensor, tilt_deg: float = 45.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-domino "toppled along the row" test (chain semantics).

    A domino counts as chain-fallen when it has tilted past ``tilt_deg``
    AND its body z-axis leans along the row axis (within 45 deg of it in
    the horizontal plane) rather than sideways.

    Args:
        quat_wxyz: (E, G, 4) piece orientations.
        row_yaw: (E,) row direction.
        tilt_deg: tilt beyond which a domino counts as fallen.

    Returns:
        ``(fallen, sign)``: (E, G) bool, and (E, G) lean direction along
        the row (+1 toward +row, -1 toward -row, 0 where not chain-fallen).
    """
    w, x, y, z = quat_wxyz.unbind(-1)
    # Body z-axis in world coordinates.
    zx = 2.0 * (x * z + w * y)
    zy = 2.0 * (y * z - w * x)
    zz = 1.0 - 2.0 * (x * x + y * y)
    tilted = zz < math.cos(math.radians(tilt_deg))
    proj = zx * torch.cos(row_yaw)[:, None] + zy * torch.sin(row_yaw)[:, None]
    horiz = torch.sqrt(zx * zx + zy * zy).clamp_min(1e-9)
    along = proj.abs() / horiz > math.cos(math.radians(45.0))
    fallen = tilted & along
    sign = torch.where(fallen, torch.sign(proj), torch.zeros_like(proj))
    return fallen, sign


def chain_topple_success(
    fallen: torch.Tensor, sign: torch.Tensor, fall_order: torch.Tensor
) -> torch.Tensor:
    """All dominoes chain-fallen the same way, in row order.

    Args:
        fallen: (E, G) bool from :func:`domino_fallen_along_row`.
        sign: (E, G) lean directions from the same call.
        fall_order: (E, G) step at which each domino first counted as
            chain-fallen (latched by the command; huge value = never).

    Returns:
        (E,) bool: every domino fallen, all leaning the same way, and
        first-fall steps monotonic along the row in that direction (ties
        allowed: neighbors may settle within the same step).
    """
    all_fallen = fallen.all(dim=-1)
    same_dir = (sign == sign[:, :1]).all(dim=-1)
    d = fall_order.diff(dim=-1)
    ordered = torch.where(sign[:, 0] > 0, (d >= 0).all(dim=-1), (d <= 0).all(dim=-1))
    return all_fallen & same_dir & ordered
