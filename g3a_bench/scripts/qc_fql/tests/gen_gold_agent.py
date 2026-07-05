# pyright: reportMissingImports=false
"""Dump reference losses + the exact noise draws from the REAL qc ACFQLAgent.

The hard part of gold-testing the agent: its losses draw randomness
internally (flow-matching x_0 and t, distillation noise, next-action noise),
and jax.random can never agree with torch.randn. But jax's PRNG is a pure
function of its key, so this script (a) replays the reference's exact
key-split sequence OUTSIDE the agent to extract the very arrays its losses
will draw, saving them as plain numpy, and (b) calls the real
`agent.critic_loss`/`actor_loss` with the same keys for the matching loss
values. The torch test then injects the saved arrays through the port's
noise-override arguments -- both sides compute on identical randomness with
zero modification to the reference source.

Only `actor_type="distill-ddpg"` (the config default) is captured;
best-of-n has no gold trace and is covered by shape/bounds tests only.

Run from the repo root:
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
    # Example arrays must be UNBATCHED -- shape (obs_dim,), no leading 1.
    # The reference's `main.py` gets them from `Dataset.sample(())` (empty-
    # tuple size = scalar index = no batch dim), and its `sample_actions`
    # infers batch shape from `len(config['ob_dims'])`; a (1, obs_dim)
    # example silently corrupts that and dies in a shape error far away.
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

    # Mirror the reference's key-split tree line by line (total_loss splits
    # top -> (actor_rng, critic_rng); each loss splits further before each
    # draw -- see agents/acfql.py). Any drift here from the reference's
    # actual split order silently desynchronizes the extracted arrays from
    # the draws inside the loss calls below.
    top_rng = jax.random.PRNGKey(UPDATE_RNG_SEED)
    _rng, actor_rng, critic_rng = jax.random.split(top_rng, 3)

    # critic_loss internals: split once, then sample_actions (distill-ddpg)
    # draws the next-action noise.
    _c_rng, sample_rng = jax.random.split(critic_rng)
    next_action_noise = jax.random.normal(sample_rng, (BATCH_SIZE, full_action_dim))

    # actor_loss internals: 3-way split for (x_0, t), then one more split
    # for the distillation noise.
    a_rng, x_rng, t_rng = jax.random.split(actor_rng, 3)
    x_0 = jax.random.normal(x_rng, (BATCH_SIZE, full_action_dim))
    t = jax.random.uniform(t_rng, (BATCH_SIZE, 1))
    _a_rng2, noise_rng = jax.random.split(a_rng)
    distill_noise = jax.random.normal(noise_rng, (BATCH_SIZE, full_action_dim))

    # The real reference losses, with the same keys: their internal draws
    # are exactly the arrays extracted above (PRNG purity).
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
