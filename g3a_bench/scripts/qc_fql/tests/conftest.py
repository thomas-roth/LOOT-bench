"""Shared weight-loading helpers for JAX->PyTorch gold-data tests."""

import os
import sys
from typing import cast

import numpy as np
import torch
import torch.nn as nn

# Make the qc_fql package importable from tests/
_qc_fql_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _qc_fql_dir not in sys.path:
    sys.path.insert(0, _qc_fql_dir)

from networks.mlp import MLP  # noqa: E402
from networks.value import Value  # noqa: E402


def load_jax_mlp_weights(mlp: MLP, gold: np.lib.npyio.NpzFile, prefix: str = ""):
    """Copy JAX Dense weights into an MLP's Linear layers in order.

    JAX stores Dense kernels as ``(in_features, out_features)``.
    PyTorch ``nn.Linear`` stores weights as ``(out_features, in_features)``.
    So we transpose the kernel on load.
    """
    for i, module in enumerate(mlp.layers):
        layer = cast(nn.Linear, module)
        jax_kernel = gold[f"{prefix}Dense_{i}/kernel"]
        jax_bias = gold[f"{prefix}Dense_{i}/bias"]
        layer.weight.data.copy_(torch.from_numpy(jax_kernel.T))
        layer.bias.data.copy_(torch.from_numpy(jax_bias))


def load_jax_value_weights(
    value: Value, gold: np.lib.npyio.NpzFile, prefix: str = "value_net/"
):
    """Copy JAX ensemble `Value` weights into a PyTorch `Value`'s heads.

    JAX's `nn.vmap` ensemble stacks an extra leading axis of size
    `num_ensembles` on every param (`Dense_i/kernel`: `(num_ensembles, in,
    out)`) -- sliced per head below. With `num_ensembles == 1`, vmap isn't
    applied at all, so params look like a plain (unstacked) MLP's.
    """
    num_ensembles = len(value.heads)
    for i, head_module in enumerate(value.heads):
        head = cast(MLP, head_module)
        for j, module in enumerate(head.layers):
            layer = cast(nn.Linear, module)
            jax_kernel = gold[f"{prefix}Dense_{j}/kernel"]
            jax_bias = gold[f"{prefix}Dense_{j}/bias"]
            if num_ensembles > 1:
                jax_kernel = jax_kernel[i]
                jax_bias = jax_bias[i]
            layer.weight.data.copy_(torch.from_numpy(jax_kernel.T))
            layer.bias.data.copy_(torch.from_numpy(jax_bias))

        # JAX only creates a LayerNorm for layers that actually apply one
        # (all but the last, unless `activate_final`); our MLP always
        # allocates one per layer for simplicity, but leaves the last
        # unused, so only the JAX-matching prefix has gold data to load.
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
