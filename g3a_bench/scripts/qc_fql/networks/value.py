from collections.abc import Sequence

import torch
import torch.nn as nn
from networks.mlp import MLP


class Value(nn.Module):
    """Ensemble value/critic network: Q(obs, action) via `num_ensembles` heads.

    JAX builds this with `nn.vmap` over `num_ensembles` independently
    initialized copies of the same MLP (ensemble axis at output axis 0); a
    plain Python loop over `num_ensembles` separately-constructed heads,
    stacked with `torch.stack(dim=0)`, computes the identical per-head
    function -- confirmed by running the real JAX class and inspecting its
    param shapes directly rather than assuming. JAX also has a structural
    fork for `num_ensembles == 1` (no vmap, output is unwrapped `(batch,)`
    instead of `(1, batch)`) -- ported as-is below, even though QC-FQL's
    config always uses `num_ensembles=2` in practice.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: Sequence[int],
        layer_norm: bool = True,
        num_ensembles: int = 2,
        use_actions: bool = True,
    ):
        super().__init__()
        self.num_ensembles = num_ensembles
        self.use_actions = use_actions
        self.heads = nn.ModuleList(
            [
                MLP(
                    in_features=in_features,
                    hidden_dims=(*hidden_dims, 1),
                    activate_final=False,
                    layer_norm=layer_norm,
                )
                for _ in range(num_ensembles)
            ]
        )

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.use_actions:
            assert actions is not None, "This Value instance requires `actions`."
            inputs = torch.cat([observations, actions], dim=-1)
        else:
            assert actions is None, "This Value instance does not accept `actions`."
            inputs = observations

        outs = [head(inputs).squeeze(-1) for head in self.heads]
        if self.num_ensembles == 1:
            return outs[0]
        return torch.stack(outs, dim=0)
