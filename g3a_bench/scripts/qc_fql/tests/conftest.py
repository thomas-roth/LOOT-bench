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
