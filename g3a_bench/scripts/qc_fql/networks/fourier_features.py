import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Sinusoidal timestep embedding.

    Ports only the `learnable=False` branch of JAX `FourierFeatures` — the
    `learnable=True` branch is never exercised by QC-FQL (`ActorVectorField`
    always constructs `FourierFeatures(fourier_feature_dim)`, leaving
    `learnable` at its default `False`), so it raises instead of guessing at
    an untested translation.
    """

    freq: torch.Tensor

    def __init__(self, output_size: int = 64):
        super().__init__()
        self.output_size = output_size
        half_dim = output_size // 2
        freq = math.log(10000) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -freq)
        self.register_buffer("freq", freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = x * self.freq
        return torch.cat([torch.cos(f), torch.sin(f)], dim=-1)
