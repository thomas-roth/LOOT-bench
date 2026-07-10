"""Flow-matching policy network: predicts a velocity toward the data action.

QC-FQL's actor is not a Gaussian policy -- it's a conditional flow: given an
observation, a noisy action `x_t`, and a denoising time `t`, it predicts the
velocity `x_1 - x_0` that carries noise toward the demonstrated action
(chunk). Euler-integrating this field from pure noise reproduces an action.
Port of `qc/utils/networks.py::ActorVectorField`.

The agent builds TWO instances of this class with different call signatures:
- `actor_bc_flow`: the real vector field, always called WITH `times`.
- `actor_onestep_flow`: the distilled one-shot policy (noise -> action in a
  single forward pass, no integration), always called WITHOUT `times`.

flax gets away with one class for both because Dense infers its input width
from whatever is concatenated on the first call; a torch Linear cannot, so
`use_time` fixes the width at construction and `forward` enforces that the
instance is only ever used in its declared mode (a wrong mix would otherwise
surface as an unrelated-looking matmul shape error).

The `encoder` machinery of the reference class is not ported: this project
is state-observation-only, and every encoder path in `qc/agents/acfql.py`
is gated behind `config['encoder'] is not None`, which is never the case here.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn

from g3a_bench.scripts.qc_fql.networks.fourier_features import FourierFeatures
from g3a_bench.scripts.qc_fql.networks.mlp import MLP


class ActorVectorField(nn.Module):
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

        # Input width is obs + action + however time enters: absent (0),
        # raw scalar (1), or Fourier-embedded (fourier_feature_dim).
        time_dim = 0
        if use_time:
            time_dim = fourier_feature_dim if use_fourier_features else 1

        # `mlp` as the attribute name mirrors the flax module tree, so gold
        # tests can address weights as `<module>/mlp/Dense_i/...`.
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
            # For pyright: the None case raised above, ff never returns None.
            assert times is not None
            inputs = torch.cat([observations, actions, times], dim=-1)
        else:
            if times is not None:
                raise ValueError(
                    "This ActorVectorField instance does not accept `times`."
                )
            inputs = torch.cat([observations, actions], dim=-1)
        return self.mlp(inputs)
