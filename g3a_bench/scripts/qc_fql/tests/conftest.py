"""Weight loaders shared by the gold-data tests.

The whole verification method of this port is: initialize the torch network,
overwrite its parameters with the flax reference's saved parameters, feed
the reference's saved inputs, and demand matching outputs. These helpers do
the overwrite. Layout facts they encode (all observable in the gen_gold
.npz files):

- flax Dense kernels are (in, out); torch Linear weights are (out, in) --
  hence the transpose on every kernel copy.
- flax names submodules positionally (Dense_0, LayerNorm_0, ...) in
  creation order, which matches our MLP's `layers`/`norms` list order.
- flax only creates LayerNorm params for layers that apply one (all but the
  last, unless activate_final); our MLP allocates a norm slot per layer for
  zip-ability, so the loaders stop at the flax count -- the extra slot has
  no gold data and is never applied in forward anyway.
- flax's nn.vmap ensemble (Value with num_ensembles > 1) stores every param
  with a leading ensemble axis; sliced per head here. With num_ensembles ==
  1 the reference skips vmap and the axis disappears.
"""

from typing import cast

import numpy as np
import torch
import torch.nn as nn

from g3a_bench.scripts.qc_fql.networks.mlp import MLP
from g3a_bench.scripts.qc_fql.networks.value import Value


def load_jax_mlp_weights(mlp: MLP, gold: np.lib.npyio.NpzFile, prefix: str = ""):
    for i, module in enumerate(mlp.layers):
        layer = cast(nn.Linear, module)
        jax_kernel = gold[f"{prefix}Dense_{i}/kernel"]
        jax_bias = gold[f"{prefix}Dense_{i}/bias"]
        layer.weight.data.copy_(torch.from_numpy(jax_kernel.T))
        layer.bias.data.copy_(torch.from_numpy(jax_bias))

    num_applied_norms = len(mlp.layers) - 1 + int(mlp.activate_final)
    for i, module in enumerate(mlp.norms[:num_applied_norms]):
        if not isinstance(module, nn.LayerNorm):
            continue
        # flax calls them scale/bias; torch calls them weight/bias.
        module.weight.data.copy_(torch.from_numpy(gold[f"{prefix}LayerNorm_{i}/scale"]))
        module.bias.data.copy_(torch.from_numpy(gold[f"{prefix}LayerNorm_{i}/bias"]))


def load_jax_value_weights(
    value: Value, gold: np.lib.npyio.NpzFile, prefix: str = "value_net/"
):
    num_ensembles = len(value.heads)
    for i, head_module in enumerate(value.heads):
        head = cast(MLP, head_module)
        for j, module in enumerate(head.layers):
            layer = cast(nn.Linear, module)
            jax_kernel = gold[f"{prefix}Dense_{j}/kernel"]
            jax_bias = gold[f"{prefix}Dense_{j}/bias"]
            if num_ensembles > 1:
                # Slice head i off the vmap ensemble axis.
                jax_kernel = jax_kernel[i]
                jax_bias = jax_bias[i]
            layer.weight.data.copy_(torch.from_numpy(jax_kernel.T))
            layer.bias.data.copy_(torch.from_numpy(jax_bias))

        num_applied_norms = len(head.layers) - 1 + int(head.activate_final)
        for j, module in enumerate(head.norms[:num_applied_norms]):
            if not isinstance(module, nn.LayerNorm):
                continue
            jax_scale = gold[f"{prefix}LayerNorm_{j}/scale"]
            jax_bias = gold[f"{prefix}LayerNorm_{j}/bias"]
            if num_ensembles > 1:
                jax_scale = jax_scale[i]
                jax_bias = jax_bias[i]
            module.weight.data.copy_(torch.from_numpy(jax_scale))
            module.bias.data.copy_(torch.from_numpy(jax_bias))
