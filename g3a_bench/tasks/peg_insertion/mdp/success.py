"""Archived peg-insertion success criteria. Unregistered, see README.
Pure tensor math, no sim.
"""

from __future__ import annotations

import torch

from g3a_bench.tasks.common.success import (
    bore_axial_lateral as peg_axial_lateral,
)
from g3a_bench.tasks.common.success import (
    bore_axis_cos_tilt as peg_axis_cos_tilt,
)

__all__ = ["peg_axial_lateral", "peg_axis_cos_tilt", "peg_inserted"]


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
    """Tip reaches insert_depth below mouth, centred and aligned."""
    tip_depth = -axial + half_len * cos_tilt
    return (tip_depth >= insert_depth) & (lateral < lateral_tol) & (cos_tilt > tilt_cos)
