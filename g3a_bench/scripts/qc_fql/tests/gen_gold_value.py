# pyright: reportMissingImports=false
"""Generate JAX reference outputs for the Value (ensemble critic) module.

Run from the repo root with:
    uv run python g3a_bench/scripts/qc_fql/tests/gen_gold_value.py
"""

import os
import sys

import jax
import numpy as np

_QC_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "qc"
    )
)
if _QC_DIR not in sys.path:
    sys.path.insert(0, _QC_DIR)

from utils.networks import Value  # noqa: E402


def _flatten(params, prefix=""):
    flat = {}
    for k, v in params.items():
        if isinstance(v, dict):
            flat.update(_flatten(v, prefix + k + "/"))
        else:
            flat[prefix + k] = np.array(v)
    return flat


def _gen_case(name, key, obs_dim, action_dim, hidden_dims, layer_norm, num_ensembles):
    key, k_obs, k_act, k_init = jax.random.split(key, 4)
    batch_size = 4

    obs = jax.random.normal(k_obs, (batch_size, obs_dim))
    act = jax.random.normal(k_act, (batch_size, action_dim))

    model = Value(
        hidden_dims=tuple(hidden_dims),
        layer_norm=layer_norm,
        num_ensembles=num_ensembles,
    )
    params = model.init(k_init, obs, act)
    output = model.apply(params, obs, act)

    out_dir = os.path.join(os.path.dirname(__file__), "gold_data")
    os.makedirs(out_dir, exist_ok=True)

    out = {
        "input_observations": np.array(obs),
        "input_actions": np.array(act),
        "output": np.array(output),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dims": np.array(hidden_dims),
        "layer_norm": layer_norm,
        "num_ensembles": num_ensembles,
    }
    out.update(_flatten(params["params"]))

    np.savez(os.path.join(out_dir, f"value_{name}.npz"), **out)
    print(f"Saved gold data to {out_dir}/value_{name}.npz")
    print(f"  output shape: {output.shape}")
    return key


def main():
    key = jax.random.PRNGKey(42)

    key = _gen_case(
        "ensemble",
        key,
        obs_dim=5,
        action_dim=3,
        hidden_dims=(8, 4),
        layer_norm=True,
        num_ensembles=2,
    )
    _gen_case(
        "single",
        key,
        obs_dim=5,
        action_dim=3,
        hidden_dims=(8, 4),
        layer_norm=False,
        num_ensembles=1,
    )


if __name__ == "__main__":
    main()
