# pyright: reportMissingImports=false
"""Generate JAX reference outputs for the FourierFeatures module.

Run from the repo root with:
    uv run python g3a_bench/scripts/qc_fql/tests/gen_gold_fourier_features.py
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

from utils.networks import FourierFeatures  # noqa: E402


def main():
    key = jax.random.PRNGKey(42)
    key, k1, k2 = jax.random.split(key, 3)

    batch_size, output_size = 4, 64
    t = jax.random.uniform(k1, (batch_size, 1))

    model = FourierFeatures(output_size=output_size)
    params = model.init(k2, t)
    output = model.apply(params, t)

    out_dir = os.path.join(os.path.dirname(__file__), "gold_data")
    os.makedirs(out_dir, exist_ok=True)

    np.savez(
        os.path.join(out_dir, "fourier_features.npz"),
        input=np.array(t),
        output=np.array(output),
        output_size=output_size,
    )
    print(f"Saved gold data to {out_dir}/fourier_features.npz")
    print(f"  input shape: {t.shape}")
    print(f"  output shape: {output.shape}")


if __name__ == "__main__":
    main()
