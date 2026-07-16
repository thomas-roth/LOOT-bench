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
    within ``threshold_deg`` of the row direction mod pi — so the dominoes
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


def domino_fallen(quat_wxyz: torch.Tensor, tilt_deg: float = 45.0) -> torch.Tensor:
    """Per-domino "toppled" test.

    A standing domino's body z-axis is close to world z (``up_cos`` close to
    1); once tilted past ``tilt_deg`` it counts as fallen, regardless of
    which way it fell.

    Args:
        quat_wxyz: (E, G, 4) piece orientations.
        tilt_deg: tilt beyond which a domino counts as fallen.

    Returns:
        (E, G) bool.
    """
    _, x, y, _ = quat_wxyz.unbind(-1)
    up_cos = 1.0 - 2.0 * (x * x + y * y)
    return up_cos < math.cos(math.radians(tilt_deg))
