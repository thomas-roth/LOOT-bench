# pyright: reportMissingImports=false
"""Generate JAX reference outputs for the MLP module, from the REAL qc class.

Imports `qc/utils/networks.py::MLP` directly (like every other gen_gold_*
script) instead of hand-retyping a lookalike module -- an earlier version of
this script did the latter, which only proves the port matches a
transcription, not the reference.

Two cases:
- plain:      layer_norm=False (the actor's configuration)
- layer_norm: layer_norm=True (the critic's configuration) with RANDOMIZED
  LayerNorm scale/bias. Both flax and torch initialize LayerNorm to
  scale=1/bias=0, so default-init gold data cannot detect a loader that
  silently skips norm params; randomizing makes the test discriminating.
  This case also pins the flax LayerNorm epsilon (1e-6, vs torch's default
  1e-5) -- a real bug class that previously escaped the MLP test because
  only layer_norm=False was covered.

Run from the repo root with:
    uv run python g3a_bench/scripts/qc_fql/tests/gen_gold_mlp.py
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

_QC_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "qc"
    )
)
if _QC_DIR not in sys.path:
    sys.path.insert(0, _QC_DIR)

from utils.networks import MLP  # noqa: E402


def _flatten(params, prefix=""):
    flat = {}
    for k, v in params.items():
        if isinstance(v, dict):
            flat.update(_flatten(v, prefix + k + "/"))
        else:
            flat[prefix + k] = np.array(v)
    return flat


def _gen_case(name, key, layer_norm):
    key, k_x, k_init = jax.random.split(key, 3)

    batch_size, in_features = 4, 16
    hidden_dims = (64, 32)
    x = jax.random.normal(k_x, (batch_size, in_features))

    model = MLP(hidden_dims=hidden_dims, layer_norm=layer_norm)
    params = dict(model.init(k_init, x))

    if layer_norm:
        # Randomize LayerNorm params away from their (identical-in-both-
        # frameworks) defaults so the torch-side loader is actually tested.
        rng = np.random.default_rng(7)
        for pname, leaf in params["params"].items():
            if pname.startswith("LayerNorm"):
                leaf["scale"] = jnp.asarray(
                    rng.normal(1.0, 0.2, np.shape(leaf["scale"])), dtype=jnp.float32
                )
                leaf["bias"] = jnp.asarray(
                    rng.normal(0.0, 0.2, np.shape(leaf["bias"])), dtype=jnp.float32
                )

    output = model.apply(params, x)

    out_dir = os.path.join(os.path.dirname(__file__), "gold_data")
    os.makedirs(out_dir, exist_ok=True)

    out = {
        "input": np.array(x),
        "output": np.array(output),
        "in_features": in_features,
        "hidden_dims": np.array(hidden_dims),
        "layer_norm": layer_norm,
    }
    out.update(_flatten(params["params"]))

    np.savez(os.path.join(out_dir, f"mlp_{name}.npz"), **out)
    print(f"Saved gold data to {out_dir}/mlp_{name}.npz")
    print(f"  output shape: {output.shape}")
    return key


def main():
    key = jax.random.PRNGKey(42)
    key = _gen_case("plain", key, layer_norm=False)
    _gen_case("layer_norm", key, layer_norm=True)


if __name__ == "__main__":
    main()
