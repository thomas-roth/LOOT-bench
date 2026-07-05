# pyright: reportMissingImports=false
"""Generate JAX reference outputs for ACFQLAgent.critic_loss / actor_loss.

`critic_loss`/`actor_loss` draw their own randomness internally via
`jax.random.split`/`normal`/`uniform`. JAX's PRNG is a pure, deterministic
function of its key, so replicating the exact split/sample sequence used
inside `total_loss` -> `critic_loss`/`actor_loss` (see `agents/acfql.py`)
with the SAME top-level rng key reproduces the identical noise arrays as
plain numpy -- which we save for injection into the PyTorch port -- while
calling the real `agent.critic_loss`/`agent.actor_loss` with that same key
gives the matching reference loss. No source modification needed.

Only `actor_type="distill-ddpg"` is covered (the config default and the
scope of the original plan for this step); `best-of-n`'s sample_actions path
is ported but not gold-tested here.

Run from the repo root with:
    uv run python g3a_bench/scripts/qc_fql/tests/gen_gold_agent.py
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

from agents.acfql import ACFQLAgent, get_config  # noqa: E402

BATCH_SIZE = 4
HORIZON_LENGTH = 3
OBS_DIM = 5
ACTION_DIM = 3
ACTOR_HIDDEN_DIMS = (16, 16)
VALUE_HIDDEN_DIMS = (16, 16)
CREATE_SEED = 42
UPDATE_RNG_SEED = 7


def _flatten(params, prefix=""):
    flat = {}
    for k, v in params.items():
        if isinstance(v, dict):
            flat.update(_flatten(v, prefix + k + "/"))
        else:
            flat[prefix + k] = np.array(v)
    return flat


def main():
    config = get_config()
    config["horizon_length"] = HORIZON_LENGTH
    config["action_chunking"] = True
    config["actor_type"] = "distill-ddpg"
    config["encoder"] = None
    config["actor_hidden_dims"] = ACTOR_HIDDEN_DIMS
    config["value_hidden_dims"] = VALUE_HIDDEN_DIMS

    rng = np.random.default_rng(0)
    # Unbatched single example, matching what `Dataset.sample(())` actually
    # produces in `main.py` (an empty-tuple size draws one scalar index, so
    # `ex_observations.shape == (obs_dim,)` with NO leading batch dim -- this
    # is what makes `sample_actions`'s `len(config['ob_dims'])` batch-shape
    # inference work: `ob_dims` must have length 1, not 2).
    ex_obs = rng.standard_normal(OBS_DIM).astype(np.float32)
    ex_act = rng.standard_normal(ACTION_DIM).astype(np.float32)

    agent = ACFQLAgent.create(CREATE_SEED, ex_obs, ex_act, config)
    full_action_dim = ACTION_DIM * HORIZON_LENGTH

    batch = {
        "observations": rng.standard_normal((BATCH_SIZE, OBS_DIM)).astype(np.float32),
        "actions": rng.standard_normal((BATCH_SIZE, HORIZON_LENGTH, ACTION_DIM)).astype(
            np.float32
        ),
        "rewards": rng.standard_normal((BATCH_SIZE, HORIZON_LENGTH)).astype(np.float32),
        "masks": np.ones((BATCH_SIZE, HORIZON_LENGTH), dtype=np.float32),
        "valid": np.ones((BATCH_SIZE, HORIZON_LENGTH), dtype=np.float32),
        "next_observations": rng.standard_normal(
            (BATCH_SIZE, HORIZON_LENGTH, OBS_DIM)
        ).astype(np.float32),
    }
    jax_batch = {k: jnp.array(v) for k, v in batch.items()}

    # --- Replicate total_loss's rng-splitting sequence exactly ---
    top_rng = jax.random.PRNGKey(UPDATE_RNG_SEED)
    _rng, actor_rng, critic_rng = jax.random.split(top_rng, 3)

    # critic_loss: `rng, sample_rng = jax.random.split(rng)`, then
    # `sample_actions` (distill-ddpg) draws `jax.random.normal(rng, (B, full_action_dim))`
    _c_rng, sample_rng = jax.random.split(critic_rng)
    next_action_noise = jax.random.normal(sample_rng, (BATCH_SIZE, full_action_dim))

    # actor_loss: `rng, x_rng, t_rng = jax.random.split(rng, 3)`, then (distill-ddpg)
    # `rng, noise_rng = jax.random.split(rng)`
    a_rng, x_rng, t_rng = jax.random.split(actor_rng, 3)
    x_0 = jax.random.normal(x_rng, (BATCH_SIZE, full_action_dim))
    t = jax.random.uniform(t_rng, (BATCH_SIZE, 1))
    _a_rng2, noise_rng = jax.random.split(a_rng)
    distill_noise = jax.random.normal(noise_rng, (BATCH_SIZE, full_action_dim))

    # --- Real JAX loss values, using the SAME keys (deterministically reproduces
    # the same internal draws as the manual replication above) ---
    critic_loss, critic_info = agent.critic_loss(
        jax_batch, agent.network.params, critic_rng
    )
    actor_loss, actor_info = agent.actor_loss(
        jax_batch, agent.network.params, actor_rng
    )

    out_dir = os.path.join(os.path.dirname(__file__), "gold_data")
    os.makedirs(out_dir, exist_ok=True)

    out = {f"batch_{k}": v for k, v in batch.items()}
    out.update(
        {
            "obs_dim": OBS_DIM,
            "action_dim": ACTION_DIM,
            "horizon_length": HORIZON_LENGTH,
            "actor_hidden_dims": np.array(ACTOR_HIDDEN_DIMS),
            "value_hidden_dims": np.array(VALUE_HIDDEN_DIMS),
            "next_action_noise": np.array(next_action_noise),
            "x_0": np.array(x_0),
            "t": np.array(t),
            "distill_noise": np.array(distill_noise),
            "critic_loss": float(critic_loss),
            "q_mean": float(critic_info["q_mean"]),
            "q_max": float(critic_info["q_max"]),
            "q_min": float(critic_info["q_min"]),
            "actor_loss": float(actor_loss),
            "bc_flow_loss": float(actor_info["bc_flow_loss"]),
            "distill_loss": float(actor_info["distill_loss"]),
        }
    )
    out.update(_flatten(agent.network.params))

    np.savez(os.path.join(out_dir, "agent_loss.npz"), **out)
    print(f"Saved gold data to {out_dir}/agent_loss.npz")
    print(f"  critic_loss={float(critic_loss):.6f}  actor_loss={float(actor_loss):.6f}")
    print(
        f"  bc_flow_loss={float(actor_info['bc_flow_loss']):.6f}  distill_loss={float(actor_info['distill_loss']):.6f}"
    )


if __name__ == "__main__":
    main()
