"""Ensemble Q-critic: Q(obs, action-chunk) with independent heads.

QC-FQL regresses `num_qs` (default 2) independently initialized critics
toward the same TD target and aggregates them (mean or min, per `q_agg`) --
the standard double-Q trick against overestimation. Port of
`qc/utils/networks.py::Value`.

The reference builds the ensemble with `flax.nn.vmap` (params get a leading
ensemble axis; one batched forward evaluates all heads). torch has no
drop-in equivalent, so this is a `ModuleList` of separate MLPs looped in
`forward` -- mathematically identical per head, just not fused. The output
layout is kept vmap-compatible: heads stacked on axis 0, i.e. shape
`(num_ensembles, batch)`, because the agent aggregates with `.min(dim=0)`/
`.mean(dim=0)` exactly like the reference's `axis=0` reductions.

One reference quirk preserved on purpose: with `num_ensembles == 1` flax
skips vmap entirely and returns an unstacked `(batch,)` -- so does this
class. QC-FQL never runs that configuration, but silently changing the
contract would make the gold data unexplainable.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn

from g3a_bench.scripts.qc_fql.networks.mlp import MLP


class Value(nn.Module):
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
        # Same construction-time pinning as ActorVectorField.use_time: flax
        # branches on `actions is None` per call; a torch Linear's width is
        # fixed, so the instance declares its mode once and forward enforces it.
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

        # squeeze(-1) drops each head's width-1 output layer, as in the
        # reference; stacking on dim 0 reproduces vmap's out_axes=0 layout.
        outs = [head(inputs).squeeze(-1) for head in self.heads]
        if self.num_ensembles == 1:
            return outs[0]
        return torch.stack(outs, dim=0)
