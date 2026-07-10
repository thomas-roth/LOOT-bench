# pyright: reportMissingImports=false
"""Dump reference outputs from the REAL qc ActorVectorField for test_actor_vector_field.

Three cases, matching the three ways the agent exercises the class: with a
raw scalar time (actor_bc_flow, default config), with Fourier-embedded time
(actor_bc_flow, use_fourier_features=True), and with no time at all
(actor_onestep_flow). flax builds a differently-shaped inner MLP for each,
which is exactly the lazy-shape behavior the torch port pins down with its
`use_time` constructor flag -- so each case needs its own gold file.

Run from the repo root:
    uv run python g3a_bench/scripts/qc_fql/tests/gen_gold_actor_vector_field.py
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

from utils.networks import ActorVectorField  # noqa: E402


def _flatten(params, prefix=""):
    flat = {}
    for k, v in params.items():
        if isinstance(v, dict):
            flat.update(_flatten(v, prefix + k + "/"))
        else:
            flat[prefix + k] = np.array(v)
    return flat


def _gen_case(
    name,
    key,
    obs_dim,
    action_dim,
    hidden_dims,
    use_time,
    use_fourier_features,
    fourier_feature_dim,
):
    key, k_obs, k_act, k_t, k_init = jax.random.split(key, 5)
    batch_size = 4

    obs = jax.random.normal(k_obs, (batch_size, obs_dim))
    act = jax.random.normal(k_act, (batch_size, action_dim))

    model = ActorVectorField(
        hidden_dims=tuple(hidden_dims),
        action_dim=action_dim,
        use_fourier_features=use_fourier_features,
        fourier_feature_dim=fourier_feature_dim,
    )

    if use_time:
        t = jax.random.uniform(k_t, (batch_size, 1))
        params = model.init(k_init, obs, act, t)
        output = model.apply(params, obs, act, t)
    else:
        t = None
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
        "use_time": use_time,
        "use_fourier_features": use_fourier_features,
        "fourier_feature_dim": fourier_feature_dim,
    }
    if t is not None:
        out["input_times"] = np.array(t)
    out.update(_flatten(params["params"]))

    np.savez(os.path.join(out_dir, f"actor_vector_field_{name}.npz"), **out)
    print(f"Saved gold data to {out_dir}/actor_vector_field_{name}.npz")
    print(f"  output shape: {output.shape}")
    return key


def main():
    key = jax.random.PRNGKey(42)

    key = _gen_case(
        "plain",
        key,
        obs_dim=5,
        action_dim=3,
        hidden_dims=(8, 4),
        use_time=True,
        use_fourier_features=False,
        fourier_feature_dim=64,
    )
    key = _gen_case(
        "fourier",
        key,
        obs_dim=5,
        action_dim=3,
        hidden_dims=(8, 4),
        use_time=True,
        use_fourier_features=True,
        fourier_feature_dim=16,
    )
    _gen_case(
        "no_time",
        key,
        obs_dim=5,
        action_dim=3,
        hidden_dims=(8, 4),
        use_time=False,
        use_fourier_features=False,
        fourier_feature_dim=64,
    )


if __name__ == "__main__":
    main()
