"""Generate JAX reference outputs for the MLP module.

Run from the repo root with:
    uv run python g3a_bench/scripts/qc_fql/tests/gen_gold_mlp.py
"""

import os

import flax.linen as nn
import jax
import numpy as np


def main():
    key = jax.random.PRNGKey(42)
    key, k1, k2 = jax.random.split(key, 3)

    B, in_features = 4, 16
    hidden_dims = [64, 32]
    x = jax.random.normal(k1, (B, in_features))

    class TestMLP(nn.Module):
        hidden_dims: tuple

        @nn.compact
        def __call__(self, x):
            for i, size in enumerate(self.hidden_dims):
                x = nn.Dense(
                    size,
                    kernel_init=nn.initializers.variance_scaling(
                        1.0, "fan_avg", "uniform"
                    ),
                )(x)
                if i < len(self.hidden_dims) - 1:
                    x = nn.gelu(x)
            return x

    model = TestMLP(hidden_dims=tuple(hidden_dims))
    params = model.init(k2, x)
    output = model.apply(params, x)

    out_dir = os.path.join(os.path.dirname(__file__), "gold_data")
    os.makedirs(out_dir, exist_ok=True)

    flat_params = {}
    for k, v in params["params"].items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                flat_params[f"{k}/{k2}"] = np.array(v2)
        else:
            flat_params[k] = np.array(v)

    np.savez(
        os.path.join(out_dir, "mlp.npz"),
        input=np.array(x),
        output=np.array(output),
        **flat_params,
    )
    print(f"Saved gold data to {out_dir}/mlp.npz")
    print(f"  input shape: {x.shape}")
    print(f"  output shape: {output.shape}")


if __name__ == "__main__":
    main()
