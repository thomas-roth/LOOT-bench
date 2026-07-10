"""Shared MLP backbone for every QC-FQL network.

All three QC-FQL networks (the flow-matching actor, the one-step distilled
actor, and the critic ensemble) are plain MLPs in the reference
implementation (`qc/utils/networks.py::MLP`); this file is the single
PyTorch equivalent they all build on. Getting it numerically identical to
the flax original matters more than it looks: three framework *defaults*
silently differ between flax and torch, and each one alone is enough to
break agreement with the reference:

- weight init: flax Dense defaults to `variance_scaling(1.0, 'fan_avg',
  'uniform')`; torch Linear defaults to Kaiming-uniform (fan_in only).
- GELU: flax defaults to the tanh approximation; torch defaults to exact erf.
- LayerNorm epsilon: flax 1e-6, torch 1e-5.

Consumers: `networks/value.py`, `networks/actor_vector_field.py`, and the
gold-data tests (which load flax weights into `layers`/`norms` by index, so
those attribute names are part of the test contract).
"""

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def uniform_fan_avg_init(tensor: torch.Tensor, scale: float = 1.0):
    """flax's `variance_scaling(scale, 'fan_avg', 'uniform')`, which torch lacks.

    Uniform on [-b, b] with b = sqrt(3 * scale / fan_avg),
    fan_avg = (fan_in + fan_out) / 2.
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    fan_avg = (fan_in + fan_out) / 2.0
    bound = math.sqrt(3.0 * scale / max(fan_avg, 1.0))
    nn.init.uniform_(tensor, -bound, bound)


class MLP(nn.Module):
    """Dense stack matching `qc/utils/networks.py::MLP`.

    Unlike flax (which infers input width lazily on first call),
    `in_features` must be given up front. `hidden_dims` includes the output
    layer, exactly like the flax original -- callers append their output
    dimension to it rather than passing it separately.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: Sequence[int],
        activate_final: bool = False,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.activate_final = activate_final
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        prev = in_features
        for dim in hidden_dims:
            layer = nn.Linear(prev, dim)
            uniform_fan_avg_init(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
            # eps=1e-6 is flax's LayerNorm default; torch's 1e-5 produces
            # ~1e-4-scale output drift, caught by the gold tests.
            # A norm is allocated for every layer (Identity when disabled)
            # to keep layers/norms zip-able, but the final layer's norm is
            # never applied -- flax only creates norms for activated layers,
            # which is why the test-side weight loaders skip the last slot.
            self.norms.append(
                nn.LayerNorm(dim, eps=1e-6) if layer_norm else nn.Identity()
            )
            prev = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms, strict=True)):
            x = layer(x)
            # flax order: activation first, then norm; final layer stays raw
            # unless activate_final (critic outputs raw Q, actor raw velocity).
            if i < len(self.layers) - 1 or self.activate_final:
                x = F.gelu(x, approximate="tanh")
                x = norm(x)
        return x
