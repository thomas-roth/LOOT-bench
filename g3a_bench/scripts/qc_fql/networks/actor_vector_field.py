from collections.abc import Sequence

import torch
import torch.nn as nn
from networks.fourier_features import FourierFeatures
from networks.mlp import MLP


class ActorVectorField(nn.Module):
    """Flow-matching policy network: maps (obs, action, time) -> velocity.

    JAX infers the inner MLP's input width lazily from whatever gets
    concatenated on the first call, so the same class can be used both with
    and without `times` depending on how each instance happens to be called.
    PyTorch needs that width fixed at construction time, so `use_time` pins
    down which of the two calling conventions (`actor_bc_flow`: always with
    `times`; `actor_onestep_flow`: never with `times`) this instance uses.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        layer_norm: bool = False,
        use_time: bool = True,
        use_fourier_features: bool = False,
        fourier_feature_dim: int = 64,
    ):
        super().__init__()
        self.use_time = use_time
        self.use_fourier_features = use_fourier_features
        self.ff = (
            FourierFeatures(fourier_feature_dim)
            if (use_time and use_fourier_features)
            else None
        )

        time_dim = 0
        if use_time:
            time_dim = fourier_feature_dim if use_fourier_features else 1

        self.mlp = MLP(
            in_features=obs_dim + action_dim + time_dim,
            hidden_dims=(*hidden_dims, action_dim),
            activate_final=False,
            layer_norm=layer_norm,
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        times: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_time:
            if times is None:
                raise ValueError("This ActorVectorField instance requires `times`.")
            if self.ff is not None:
                times = self.ff(times)
            assert times is not None
            inputs = torch.cat([observations, actions, times], dim=-1)
        else:
            if times is not None:
                raise ValueError(
                    "This ActorVectorField instance does not accept `times`."
                )
            inputs = torch.cat([observations, actions], dim=-1)
        return self.mlp(inputs)
