"""Pure success criteria for the peg-insertion category.

ARCHIVED (unregistered, untested): see ``README.md``.

Plain tensor math — functions of (positions, quaternions, thresholds)
returning bore-frame decompositions and bool insertion tests. No mjlab or
environment imports: the command term (``commands.py``) acts as the adapter
that gathers sim buffers and calls these.

Quaternions are wxyz, matching mjlab.
"""

from __future__ import annotations

import torch


def peg_axial_lateral(
    pos: torch.Tensor, mouth: torch.Tensor, axis_out: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose peg-to-mouth offsets along/across the bore axis.

    Args:
        pos: (..., 3) peg centre positions.
        mouth: (..., 3) bore mouth positions.
        axis_out: (..., 3) unit outward bore axes.

    Returns:
        ``(axial, lateral)``, each (...,): signed distance along the axis
        (+ is outside the bore) and distance from the bore centreline.
    """
    rel = pos - mouth
    axial = torch.sum(rel * axis_out, dim=-1)
    lateral = torch.norm(rel - axial.unsqueeze(-1) * axis_out, dim=-1)
    return axial, lateral


def peg_axis_cos_tilt(quat_wxyz: torch.Tensor, axis_out: torch.Tensor) -> torch.Tensor:
    """|cos| of the angle between a peg's body z-axis and the bore axis.

    Sign-free: a peg inserted "upside down" is equally aligned.

    Args:
        quat_wxyz: (..., 4) peg orientations.
        axis_out: (..., 3) unit bore axes.

    Returns:
        (...,) in [0, 1].
    """
    w, x, y, z = quat_wxyz.unbind(-1)
    peg_axis = torch.stack(
        [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=-1
    )
    return torch.sum(peg_axis * axis_out, dim=-1).abs()


def peg_inserted(
    axial: torch.Tensor,
    lateral: torch.Tensor,
    cos_tilt: torch.Tensor,
    *,
    half_len: float,
    insert_depth: float,
    lateral_tol: float,
    tilt_cos: float,
) -> torch.Tensor:
    """Instantaneous insertion test for fixed-bore pegs.

    The peg tip (centre depth plus the projected half length) must reach
    ``insert_depth`` below the mouth while the peg stays centred and aligned.

    Returns:
        bool tensor of the broadcast shape of the inputs.
    """
    tip_depth = -axial + half_len * cos_tilt
    return (tip_depth >= insert_depth) & (lateral < lateral_tol) & (cos_tilt > tilt_cos)
