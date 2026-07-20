"""Pure success criteria for the peg-insertion category.

ARCHIVED (unregistered, untested): see ``README.md``.

Plain tensor math — functions of (positions, quaternions, thresholds)
returning bore-frame decompositions and bool insertion tests. No mjlab or
environment imports: the command term (``commands.py``) acts as the adapter
that gathers sim buffers and calls these.

``peg_axial_lateral``/``peg_axis_cos_tilt`` are the shared bore/hole geometry
kernels (``tasks/common/success.py``), also used by ``shape_sorter``.
``peg_inserted`` (the instantaneous insertion-depth test) is specific to this
category. Quaternions are wxyz, matching mjlab.
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
    """Instantaneous insertion test for fixed-bore pegs.

    The peg tip (centre depth plus the projected half length) must reach
    ``insert_depth`` below the mouth while the peg stays centred and aligned.

    Returns:
        bool tensor of the broadcast shape of the inputs.
    """
    tip_depth = -axial + half_len * cos_tilt
    return (tip_depth >= insert_depth) & (lateral < lateral_tol) & (cos_tilt > tilt_cos)
