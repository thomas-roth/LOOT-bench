"""All QC-FQL hyperparameters, one dataclass, defaults copied from the reference.

Port of `qc/agents/acfql.py::get_config()`. The reference uses
`ml_collections.ConfigDict` because it plugs into absl's CLI-flag system and
survives jax pytree boundaries; neither applies here, so a plain dataclass
gives the same attribute access with none of the machinery. Every default
below is the reference's default verbatim -- change them per-run (CLI flags
in `scripts/train_qcfql.py`), not here, or gold-data agreement stops meaning
anything.

Fields the reference declares but this port drops (each verified by grepping
all of `acfql.py`, not just `get_config()`): `agent_name` (only used by the
reference's multi-agent registry), `normalize_q_loss` (declared, never read
anywhere), `encoder` (always None for state-only tasks; the network-side
plumbing was dropped with it).
"""

from dataclasses import dataclass, field


# kw_only lets the one required field sit above the defaulted ones without
# tripping dataclass's no-default-after-default rule.
@dataclass(kw_only=True)
class QCFQLConfig:
    # Chunk length. A reference "will be set" placeholder, but never
    # auto-computed -- the caller always knows it -- so it's required here
    # rather than Optional (Optional would push None-checks into every use).
    horizon_length: int

    lr: float = 3e-4
    batch_size: int = 256
    actor_hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    value_hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    layer_norm: bool = True  # critic MLPs
    actor_layer_norm: bool = False  # both actor MLPs
    discount: float = 0.99
    tau: float = 0.005  # Polyak rate for the target critic
    q_agg: str = "mean"  # "mean" | "min", ensemble aggregation
    alpha: float = 100.0  # distillation (BC) coefficient; env-specific tuning knob
    num_qs: int = 2  # critic ensemble size
    flow_steps: int = 10  # Euler steps when integrating the BC flow
    action_chunking: bool = True  # False = plain n-step returns, no chunk actions
    actor_type: str = "distill-ddpg"  # "distill-ddpg" | "best-of-n"
    actor_num_samples: int = 32  # best-of-n only
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64
    weight_decay: float = 0.0  # >0 switches Adam -> AdamW, as in the reference

    # Inferred from example data by QCFQLAgent.create(), never by the caller
    # (mirrors the reference's create() writing into its config dict).
    ob_dims: tuple[int, ...] | None = field(default=None, init=False)
    action_dim: int | None = field(default=None, init=False)
