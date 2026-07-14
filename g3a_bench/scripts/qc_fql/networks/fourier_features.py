"""Sinusoidal embedding for the flow-matching time variable.

The flow-matching actor (`networks/actor_vector_field.py`) conditions on a
scalar denoising time t in [0, 1]. Fed raw, one scalar carries little signal
into a wide MLP; this module lifts it to `output_size` sinusoids of
geometrically spaced frequencies (the DDPM/Transformer positional-encoding
construction), matching `qc/utils/networks.py::FourierFeatures`. Only used
when `use_fourier_features=True` in the config -- the default config feeds
raw t instead, so this path is exercised by tests but off in practice.

Port scope: the reference class also has a `learnable=True` variant (a
trained projection matrix). No QC-FQL code path constructs it -- the only
call site is `ActorVectorField.setup`, which passes a single positional arg
-- so it is deliberately not ported rather than shipped untested.
"""

import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    # Annotation tells pyright register_buffer's attribute is a Tensor,
    # not the Tensor|Module union it otherwise infers on nn.Module.
    freq: torch.Tensor

    def __init__(self, output_size: int = 64):
        super().__init__()
        self.output_size = output_size
        half_dim = output_size // 2
        # freq[i] = 10000^(-i/(half_dim-1)): identical arithmetic to the flax
        # original, just computed once here instead of on every call (it
        # depends only on output_size). A buffer (not Parameter) so it moves
        # with .to(device) but never trains.
        freq = math.log(10000) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -freq)
        self.register_buffer("freq", freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (..., 1) * (half_dim,) broadcasts to (..., half_dim); cos-then-sin
        # concat order matches the reference (order matters for weight compat).
        f = x * self.freq
        return torch.cat([torch.cos(f), torch.sin(f)], dim=-1)
