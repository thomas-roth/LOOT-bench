# QC-FQL PyTorch Port

Port of **QC-FQL** (Action-Chunked Flow Q-Learning) from JAX/Flax to PyTorch,
integrated into g3a-bench for offline-RL pretraining + online fine-tuning on
mjlab tasks (stacking, domino, shape sorter, ball).

## Quick start

```bash
# Offline pretraining on a recorded demo dataset:
uv run g3a-train-qcfql --dataset-path demos.npz --offline-steps 1000000 \
    --checkpoint-path agent.pt

# Offline + online fine-tuning:
uv run g3a-train-qcfql --dataset-path demos.npz --offline-steps 1000000 \
    --online-steps 50000 --task G3a-Stacking-Stack3-Dk1_uni --checkpoint-path agent.pt

# All knobs:
uv run g3a-train-qcfql --help
```

The dataset `.npz` must come from `scripts/live_bridge/lerobot_to_qc.py`
(default `physical` action normalization — do not use `--action-norm data`).
**As of now, no recorded demo dataset exists for `G3a-Stacking-Stack3-Dk1_uni`** —
recording demos via `live_bridge` and converting them is a manual prerequisite.

```bash
uv run python g3a_bench/scripts/live_bridge/lerobot_to_qc.py \
    --success-dir out/demos/success/<stamp> --output demos.npz
```

## What it does

Offline-RL algorithm with three components:
- **Flow-matching actor** (`ActorVectorField`): denoises noise into an action
  chunk (H future actions at once) by learning the velocity toward demonstrated
  actions.
- **One-shot distilled actor** (`actor_onestep_flow`): maps noise → action
  directly (1 forward pass instead of H Euler steps).
- **Ensemble Q-critic** (`Value`, 2 heads): scores action chunks, pushes the
  actor uphill on Q-values (distill-ddpg mode, default).

## Directory layout

```
qc_fql/
├── networks/     MLP, FourierFeatures, ActorVectorField, Value
├── data/         Dataset (offline chunked sampling), ReplayBuffer (online)
├── agent/        QCFQLConfig, QCFQLAgent (losses, update, sample_actions)
├── scripts/      evaluate.py, train_qcfql.py (CLI), mjlab_env.py (adapter)
└── tests/        gen_gold_*.py + test_*.py + gold_data/*.npz
```

## Tests

```bash
uv run python -m pytest g3a_bench/scripts/qc_fql/tests/ -v

# Regenerate gold data (requires dev deps: jax, flax, distrax, ml_collections):
for gen in mlp fourier_features actor_vector_field value dataset replay_buffer agent; do
  uv run python g3a_bench/scripts/qc_fql/tests/gen_gold_$gen.py
done
```

## Key design points

- **Single-env only** (matches the JAX reference). Multi-env vectorized
  fine-tuning is out of scope.
- **State-based observations only** (no visual encoders). Adding images means
  re-adding the `encoder` branch to `ActorVectorField`/`Value` + CNN.
- **Action chunking** flattens `(B, H, act_dim)` → `(B, H*act_dim)` before
  network input.
- **Gradient isolation** in `actor_loss`: the critic scores the actor's actions
  with `requires_grad_(False)` so the Q-push term only trains `actor_onestep_flow`,
  not the critic itself (verified by a dedicated gradient test).
- Verified against the real JAX reference via gold-data tests (bit-for-bit
  `allclose(atol=1e-5)` on all network and loss outputs).
