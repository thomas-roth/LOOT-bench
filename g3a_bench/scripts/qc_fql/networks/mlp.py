import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def uniform_fan_avg_init(tensor: torch.Tensor, scale: float = 1.0):
    """JAX variance_scaling(scale, 'fan_avg', 'uniform') init.

    bound = sqrt(3 * scale / fan_avg) where fan_avg = (fan_in + fan_out) / 2
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    fan_avg = (fan_in + fan_out) / 2.0
    bound = math.sqrt(3.0 * scale / max(fan_avg, 1.0))
    nn.init.uniform_(tensor, -bound, bound)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dims: Sequence[int],
        activation: type[nn.Module] = nn.GELU,
        activate_final: bool = False,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.activate_final = activate_final
        self.activation = activation
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        prev = in_features
        for dim in hidden_dims:
            layer = nn.Linear(prev, dim)
            uniform_fan_avg_init(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
            self.norms.append(nn.LayerNorm(dim) if layer_norm else nn.Identity())
            prev = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms, strict=True)):
            x = layer(x)
            if i < len(self.layers) - 1 or self.activate_final:
                x = F.gelu(x, approximate="tanh")
                x = norm(x)
        return x
