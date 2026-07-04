from dataclasses import dataclass, field


@dataclass
class QCFQLConfig:
    """Port of `qc/agents/acfql.py::get_config()`.

    `ml_collections.ConfigDict` (absl-flags CLI integration + JAX-pytree-safe
    placeholders) is replaced by a plain dataclass -- our code doesn't use
    absl or jax pytrees, so a dataclass gives identical attribute access and
    defaults with none of that machinery.

    Dropped fields (confirmed by grepping the whole of `acfql.py`, not just
    `get_config()`): `agent_name` (only used by `main.py`'s multi-agent
    registry lookup -- we build exactly one agent), `normalize_q_loss`
    (defined but never read anywhere), `encoder` (real logic in JAX, but
    always `None` for state-based tasks, and the encoder plumbing itself was
    already dropped from `ActorVectorField`/`Value`).

    `ob_dims`/`action_dim` are left `None` here and set by `QCFQLAgent.create()`,
    matching how JAX's `create()` sets `config['ob_dims']`/`config['action_dim']`
    on the fly.
    """

    lr: float = 3e-4
    batch_size: int = 256
    actor_hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    value_hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    layer_norm: bool = True
    actor_layer_norm: bool = False
    discount: float = 0.99
    tau: float = 0.005
    q_agg: str = "mean"
    alpha: float = 100.0
    num_qs: int = 2
    flow_steps: int = 10
    horizon_length: int | None = None
    action_chunking: bool = True
    actor_type: str = "distill-ddpg"
    actor_num_samples: int = 32
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64
    weight_decay: float = 0.0

    # Set by QCFQLAgent.create(), not by the caller.
    ob_dims: tuple[int, ...] | None = field(default=None, init=False)
    action_dim: int | None = field(default=None, init=False)
