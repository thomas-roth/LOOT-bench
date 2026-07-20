"""Pure success criteria for the shape-sorter category.

Plain tensor math, no mjlab or environment imports: the command term
(``commands.py``) acts as the adapter that gathers sim buffers and calls
these, so the success semantics live (and are unit-tested in
``tests/tasks/test_success.py``) in exactly one place.

``peg_axial_lateral``/``peg_axis_cos_tilt`` are the shared bore/hole geometry
kernels (``tasks/common/success.py``), also used by the archived
``peg_insertion`` category. ``SorterLatch`` is specific to this category (the
insertion-depth test in ``peg_insertion/mdp/success.py`` differs). Quaternions
are wxyz, matching mjlab.
"""

from __future__ import annotations

import torch

from g3a_bench.tasks.common.success import (
    bore_axial_lateral as peg_axial_lateral,
)
from g3a_bench.tasks.common.success import (
    bore_axis_cos_tilt as peg_axis_cos_tilt,
)

__all__ = ["peg_axial_lateral", "peg_axis_cos_tilt", "SorterLatch"]


class SorterLatch:
    """Latched mouth-plane crossing for the shape sorter.

    A piece scores the moment it passes through its own hole into the cube:
    it must be inside the insertion channel above the crossing plane on the
    previous step and at or below it now, roughly axis-aligned. The latch
    holds until :meth:`reset`, so it survives the piece tumbling around
    inside the hollow cube.

    All per-socket tensors are (S,); ``axial`` is measured from the mouth
    along the outward axis (+ is outside).
    """

    # Channel slack beyond the exact crossing window (m).
    _BELOW_MARGIN = 0.02
    _ABOVE_MARGIN = 0.01

    def __init__(
        self,
        num_envs: int,
        num_sockets: int,
        device: str | torch.device,
        *,
        hole_radius: torch.Tensor,
        piece_half: torch.Tensor,
        s_cross: torch.Tensor,
        tilt_cos: float,
    ):
        self.hole_radius = hole_radius.to(device)
        self.piece_half = piece_half.to(device)
        self.s_cross = s_cross.to(device)
        self.tilt_cos = tilt_cos
        self.latched = torch.zeros(
            num_envs, num_sockets, dtype=torch.bool, device=device
        )
        # Far outside, out of channel: the first step can never latch.
        self._prev_axial = torch.full((num_envs, num_sockets), 1.0, device=device)
        self._prev_in_channel = torch.zeros(
            num_envs, num_sockets, dtype=torch.bool, device=device
        )

    def update(
        self, axial: torch.Tensor, lateral: torch.Tensor, cos_tilt: torch.Tensor
    ) -> torch.Tensor:
        """Advance one step; returns the (E, S) bool latch state."""
        in_channel = (
            (lateral < self.hole_radius)
            & (axial > self.s_cross - self._BELOW_MARGIN)
            & (axial < self.piece_half + self._ABOVE_MARGIN)
        )
        crossed = (
            self._prev_in_channel
            & in_channel
            & (self._prev_axial > self.s_cross)
            & (axial <= self.s_cross)
            & (cos_tilt > self.tilt_cos)
        )
        self.latched |= crossed
        self._prev_axial = axial.clone()
        self._prev_in_channel = in_channel
        return self.latched

    def reset(self, env_ids: torch.Tensor) -> None:
        self.latched[env_ids] = False
        self._prev_in_channel[env_ids] = False
        self._prev_axial[env_ids] = 1.0  # far outside: no spurious crossing
