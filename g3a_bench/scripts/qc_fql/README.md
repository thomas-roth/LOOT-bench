# QC-FQL PyTorch Port

## Goal

Port **QC-FQL** (the paper's name for `ACFQLAgent` — Action-Chunked Flow
Q-Learning) from JAX/Flax to PyTorch, then integrate it into g3a-bench so it
can train on mjlab tasks (cube stacking, insertion, ball) instead of the
current PPO/rsl_rl pipeline.

End state: `uv run g3a-train-qcfql <task_id> --dataset-path <demos.npz>`
pretrains offline on recorded demonstrations, then fine-tunes online in mjlab
sim.

## What QC-FQL does

Offline-RL-then-online-fine-tune algorithm. A flow-matching actor denoises
noise into an action *chunk* (like a small diffusion policy), and an ensemble
Q-critic scores/distills those chunks. "Action chunking" means the policy
predicts `horizon_length` future actions at once instead of one step at a
time.

## Ground truth

The real JAX implementation lives locally at `../../../qc/` (sibling repo,
not this one). It's used two ways:
1. **Directly imported** by `tests/gen_gold_*.py` scripts to produce reference
   outputs (preferred — don't hand-retype JAX modules, import the real class).
2. **Read as a reference** when writing the PyTorch training/integration code
   that has no JAX equivalent to import (e.g. the mjlab env loop).

Files needed from `qc/`:
- `agents/acfql.py` — the agent (`critic_loss`, `actor_loss`, `sample_actions`,
  `compute_flow_actions`, `create`, `get_config`)
- `utils/networks.py` — `MLP`, `FourierFeatures`, `Value`, `ActorVectorField`,
  `default_init`, `ensemblize`
- `utils/datasets.py` — `Dataset`, `ReplayBuffer`, `sample_sequence`,
  `add_history`
- `main.py` — offline-pretrain-then-online-finetune reference (replay-buffer-only
  online phase)
- `main_online.py` — alternate online loop that mixes offline dataset +
  replay buffer every step (`utd_ratio`, `start_training` warmup) — pick one
  design for Phase 4, don't blend them silently
- `evaluation.py` — eval loop reference

Not needed: `utils/encoders.py` (no vision inputs), `utils/flax_utils.py`
(JAX TrainState/ModuleDict, replaced by plain PyTorch modules/optimizers),
`rlpd_*`, `envs/*` (d4rl/ogbench/robomimic wrappers — not our env).

## Method: gold-data TDD

Every module is ported the same way:
1. Write/import the JAX reference, run it on a fixed seed, dump inputs +
   outputs + params to `tests/gold_data/<name>.npz`.
2. Build the PyTorch module, load the JAX params into it (transpose `Linear`
   weights — JAX `Dense.kernel` is `(in, out)`, PyTorch `Linear.weight` is
   `(out, in)`).
3. Run the same inputs through, assert `torch.allclose(..., atol=1e-5)`
   against the gold output.
4. Fix until green, then move to the next module.

This is what makes the port trustworthy — it isn't "does this look right,"
it's "does this match the real algorithm's actual numbers." It has already
caught real bugs during planning (see the step list below).

## Directory layout

```
qc_fql/
├── networks/        MLP, FourierFeatures, ActorVectorField, Value
├── data/             Dataset (offline, n-step/chunked sampling), ReplayBuffer (online)
├── agents/           QCFQLConfig, QCFQLAgent (losses, update, sample_actions)
├── scripts/          evaluate.py, train_qcfql.py
└── tests/            conftest.py (weight loaders) + gen_gold_*.py + test_*.py + gold_data/*.npz
```

## Porting order

Networks first (nothing else works without them) → Data (feeds the agent,
independent of it) → Agent (consumes both) → Eval/training loop →
g3a-bench/mjlab integration last (highest risk, least testable, so it should
touch a working, verified stack).

| # | Step | Status |
|---|---|---|
| 1 | MLP | ✅ done |
| 2 | FourierFeatures | not started |
| 3 | ActorVectorField | not started |
| 4 | Value | not started |
| 5 | Dataset | not started |
| 6 | ReplayBuffer | not started |
| 7 | Config | not started |
| 8 | Agent (QCFQLAgent) | not started |
| 9 | Evaluate loop | not started |
| 10 | Training script | not started |
| 11 | g3a-bench / mjlab integration | not started |

For each step below: what's copied verbatim from JAX, what's changed because
PyTorch ≠ JAX, and why that change doesn't alter the algorithm (or, where it
does, that's called out explicitly).

### 1. MLP — done

**Copied:** loop over `hidden_dims`, `Dense` → activation → (LayerNorm) per
layer except the last (unless `activate_final`), GELU, optional LayerNorm.

**Changed:**
- `nn.initializers.variance_scaling(1.0,'fan_avg','uniform')` → hand-written
  `uniform_fan_avg_init`. *Why valid:* PyTorch has no built-in fan-avg
  initializer (only fan-in/fan-out `kaiming_*`), so the formula
  (`bound = sqrt(3·scale/fan_avg)`, uniform in `[-bound, bound]`) was
  re-derived from JAX's variance-scaling definition, not approximated —
  verified bit-for-bit by the gold test.
- `nn.gelu(x)` (JAX default `approximate=True`) → `F.gelu(x,
  approximate='tanh')` explicit. *Why valid:* PyTorch's own default is the
  exact (erf-based) GELU, so JAX's default had to be requested explicitly —
  same formula, not a behavior change, confirmed by the passing test.
- Weight-layout transpose (`kernel.T`) is a **test-loading detail only**, not
  an architectural change — real training never touches JAX weights, this
  only matters when copying gold params in for verification.

**Known loose end:** `mlp.py`'s `activation` constructor param is currently
unused dead code (forward hardcodes `F.gelu`). Harmless since it defaults to
matching behavior, but worth deleting next time this file is touched.

### 2. FourierFeatures

**Copied:** the sinusoidal timestep-embedding formula exactly — `half_dim =
output_size//2`, `f = exp(arange(half_dim) * -log(10000)/(half_dim-1))`,
`x*f`, then `concat([cos(f), sin(f)])`.

**Changed:**
- `jnp.arange/exp/log` → `torch.arange/exp/log` — direct 1:1 API
  translation, no semantic difference.
- The frequency vector `f` is a fixed constant (for `learnable=False`) —
  worth computing once in `__init__` and storing via `register_buffer`
  instead of recomputing every forward call. *Why valid:* it's the same
  deterministic arithmetic either way; registering as a buffer only changes
  *when* it's computed, not the value, and it moves with `.to(device)`
  automatically.
- `learnable=True` branch (a learned projection matrix via `self.param(...)`)
  → `nn.Parameter` initialized with `normal_(0, 0.2)`. Low priority: QC-FQL's
  config never sets `learnable=True`, so this can be stubbed and added later
  if needed.

### 3. ActorVectorField

**Copied:** conditioning structure — concat `[obs, actions]` (or `[obs,
actions, time_embedding]` if times given) → single inner MLP → velocity
output.

**Changed:**
- `encoder` param dropped entirely. *Why valid:* state-only observations
  (design decision), and in JAX `encoder=None` is already a no-op passthrough
  — omitting the parameter changes nothing behaviorally.
- Flax's `setup()` (lazy shape inference — `nn.Dense` infers `in_features`
  from the first actual input) → PyTorch `__init__` needs `in_features`
  (obs_dim) passed explicitly up front. *Why valid:* obs_dim/action_dim are
  static and known before training starts, so nothing is lost — it's just
  supplied once at construction instead of inferred on first call.
- Time tensor shape: JAX derives `ex_times = ex_actions[..., :1]` → a `(B,1)`
  shape must be matched explicitly in the PyTorch constructor's
  example-batch handling.

### 4. Value

**Copied:** per-head architecture (`MLP(*hidden_dims, 1)`, no final
activation), concat `[obs, actions]`, `squeeze(-1)` on output.

**Changed:**
- `nn.vmap`-based `ensemblize(MLP, num_qs)` (auto-parallelizes N
  independently-initialized copies over a synthetic axis) →
  `nn.ModuleList([MLP(...) for _ in range(num_qs)])`, looped in `forward`,
  `torch.stack(outputs, dim=0)`. *Why valid:* `nn.vmap` with
  `variable_axes={'params':0}, split_rngs={'params':True}` means "N
  independently initialized copies of the same architecture applied to the
  same input" — a Python loop over N separately-constructed MLPs computes
  the identical function per head; only the execution strategy (batched XLA
  vmap vs. eager loop) differs, not the math. Confirmed per-head by loading
  each JAX ensemble slice into the matching `ModuleList[i]` and comparing.

### 5. Dataset

**Copied:** the `sample_sequence` math exactly — index gathering
(`idxs[:,None] + arange(seq_len)`), cumulative discounted reward
(`rewards[t] = rewards[t-1] + raw[t]*discount**t`), running
`masks`/`terminals` via min/max, `valid[t] = 1 - terminals[t-1]`, next-action
index clipped to `size-1`.

**Changed:**
- `FrozenDict` (immutable, JAX-pytree-compatible dict subclass — required so
  it can pass through `jax.jit`/`tree_map` boundaries) → plain mutable
  dict-backed class. *Why valid:* nothing here is jitted or traced in
  PyTorch; immutability existed to satisfy JAX's functional-purity
  requirement, not because the algorithm needs it. A regular dict with the
  same `[]`/`.items()` interface is behaviorally identical for our purposes.
- `np.random.randint` used directly — the JAX version already uses plain
  NumPy here too, not `jax.random`; this file was never really "JAX" in
  either version, one less thing to worry about.
- **Fix required:** an earlier draft set `observations == full_observations`
  (both the `(B,T,obs_dim)` sequence). The real source returns
  `observations = data['observations'][idxs]` — just the **starting** obs,
  `(B, obs_dim)` — with `full_observations` as the separate `(B,T,obs_dim)`
  version. Must be corrected here, since `critic_loss`/`actor_loss` consume
  `batch['observations']` expecting the non-repeated shape.

### 6. ReplayBuffer

**Copied:** circular-buffer semantics — preallocate `(size, *shape)` zero
arrays, `add_transition` writes at `pointer`, `pointer = (pointer+1) %
max_size`, `size = max(pointer, size)`; `create_from_initial_dataset`
preloads offline data into the buffer prefix.

**Changed:**
- `jax.tree_util.tree_map` over the transition dict (JAX's generic
  "apply-per-leaf" utility, needed because it also works on arbitrary nested
  pytrees) → a plain `for k, v in transition.items()` loop. *Why valid:* our
  transition dict is flat (fixed known keys) — `tree_map` on a flat dict
  degenerates to exactly this loop, no generality lost.

### 7. Config

**Copied:** every hyperparameter name and default value from `get_config()`
verbatim (lr, batch_size, actor/value hidden dims, layer_norm flags,
discount, tau, q_agg, alpha, num_qs, flow_steps, horizon_length,
action_chunking, actor_type, actor_num_samples, use_fourier_features,
fourier_feature_dim, weight_decay).

**Changed:**
- `ml_collections.ConfigDict` (supports `absl.flags` CLI integration and
  JAX-pytree-safe placeholders like `config_dict.placeholder(int)` for
  "set later" fields) → plain Python `@dataclass` with `Optional[...] = None`
  for `ob_dims`/`action_dim` (set in `create()`). *Why valid:* the
  flag-system/pytree integration is JAX/absl-specific tooling our PyTorch
  code doesn't use; a dataclass gives identical attribute-access semantics
  and defaults with none of that machinery needed.
- Drop `encoder` field (no vision).

### 8. Agent — the one with a real structural correction

**Copied:** all loss math verbatim — TD target `reward + discount^H · mask ·
next_q`, flow-matching interpolation `x_t=(1-t)x_0+t·x_1`, target velocity
`x_1-x_0`, distillation loss, `q_loss = -mean(Q(obs, actor_actions))`, Euler
integration in `compute_flow_actions`, both `sample_actions` branches
(`distill-ddpg`, `best-of-n`).

**Changed — mechanical, low-risk:**
- `flax.struct.PyTreeNode` immutable agent + `agent.replace(...)` functional
  update → ordinary mutable `nn.Module`, `optimizer.step()` mutates
  parameters in place. *Why valid:* JAX's immutability exists because
  `jit`/`grad` require pure functions with explicit state in/out; PyTorch's
  autograd is imperative by design — in-place mutation via an optimizer is
  the standard, semantically identical way to express the same update rule.
- Explicit `rng` key-splitting at every stochastic call → PyTorch global RNG
  (or an explicit `torch.Generator`). *Why valid:* key-splitting exists for
  reproducibility under JAX's functional/parallel execution model; not
  required outside JAX, so implicit global RNG state is the idiomatic
  equivalent (same *distribution* of behavior — bit-identical only matters
  for the gold test, see below, not for real training).
- `jax.lax.scan` over `utd_ratio` in `batch_update` → plain Python `for` loop
  calling `update()` repeatedly. *Why valid:* `scan` and a for-loop compute
  the identical sequence of updates; `scan` only exists for XLA compilation
  speed, which PyTorch's eager mode doesn't need.
- `copy.deepcopy(critic_def)` for the target critic → same call, same
  meaning, in PyTorch.

**Changed — this one is NOT just idiom, it's a real design correction to
make *before* writing code:**
JAX's `total_loss = critic_loss + actor_loss` is optimized as **one combined
scalar through one shared optimizer** (`network_tx`, a single `optax.adam`
over the entire `ModuleDict` of critic + actor_bc_flow + actor_onestep_flow),
via a single `jax.grad(loss_fn)(self.params)` call — confirmed directly in
`utils/flax_utils.py`'s `apply_loss_fn`. This means the actor's `q_loss` term
(`-mean(critic(obs, actor_actions))`) **also contributes gradient into the
critic's own parameters**, computed against the *same pre-update* params as
`critic_loss`, in the same step.

An earlier draft used **two separate optimizers** (critic-only, actor-only)
with two sequential `.backward()+.step()` calls. That is *not* equivalent:
(a) the actor's `q_loss` gradient into critic params gets silently dropped
since critic isn't in the actor optimizer's param list, and (b) if critic is
stepped first, the actor loss ends up evaluated against post-update critic
params instead of the pre-update ones JAX uses.

**Correct port:** build **one optimizer** over
`list(critic.parameters()) + list(actor_bc_flow.parameters()) +
list(actor_onestep_flow.parameters())`, compute `total_loss = critic_loss +
actor_loss` once, single `zero_grad()`/`backward()`/`step()`. This is a case
where the JAX structure must be copied, not simplified.

**Required addition (testing-only, doesn't change real behavior):**
`critic_loss`/`actor_loss`/`sample_actions` draw their own randomness
internally (`x_0`, `t`, distillation noise). JAX's `jax.random` and
PyTorch's `torch.randn` will never produce the same numbers, so comparing
loss scalars directly will fail for reasons unrelated to correctness. Add
optional injected-noise arguments to these methods so the gold test can feed
identical draws to both sides. When omitted at real training time, the
methods sample their own noise exactly as JAX does — the addition is inert
in production.

### 9. Evaluate loop

**Copied:** almost directly from `qc/evaluation.py` — action-chunk queue
(sample once per `horizon_length` steps, execute the whole chunk),
`env.reset()`/`env.step()` gym-style loop, success/return/length stats.

**Changed:** essentially nothing algorithmic — matched the draft closely
when checked. Real changes come from swapping the env interface in step 11.

### 10. Training script

**Copied:** two real reference structures exist in `qc/` — pick one:
- `main.py`: offline pretrain (loop of `agent.update` on
  `dataset.sample_sequence`), *then* online fine-tune sampling only from the
  replay buffer (`agent.batch_update`).
- `main_online.py`: no separate pretrain phase — every online step mixes
  **half a batch from the offline dataset + half from the replay buffer**
  (`utd_ratio`-scaled), plus a random-action warmup (`start_training`)
  before the policy is used.

**Decision needed before writing this file:** which structure. Both are
"valid" — they're different training regimes from the same paper's
codebase, not a JAX-vs-PyTorch translation question. Single-env constrains
*env stepping*, not this choice.

### 11. g3a-bench / mjlab integration — no JAX equivalent, no allclose net

Nothing here is a JAX→PyTorch translation — it's genuinely new code, because
mjlab's `ManagerBasedRlEnv` is not a gym-style env. Verified directly against
mjlab's source: `step()` **auto-resets terminated envs inside the call**
(when `cfg.auto_reset=True`) and returns torch tensors, not numpy;
observations come back as `self.obs_buf`, a dict keyed by observation-group
name, not a flat array. The `if done: env.reset()` pattern and `obs[i]`
indexing from the JAX reference scripts don't apply — this step has to be
written fresh against mjlab's actual API and verified by running training
and watching return go up, not by any offline test.

## Key decisions

- **Single-env only** for now, matching the original JAX `main.py`/
  `main_online.py`. Multi-env vectorized fine-tuning is a future refactor —
  don't build it prematurely into Phase 4/5.
- Pure NumPy for `Dataset`/`ReplayBuffer`; convert to torch only at agent
  input (matches JAX, which is also plain NumPy for these).
- No visual encoders — state-based observations only.
- Action chunking folds `horizon_length` into the action dim:
  `(B, T, act_dim) → (B, T*act_dim)`.
- Value ensemble = `nn.ModuleList` of independent `MLP` heads, looped
  manually (JAX uses `nn.vmap`, which PyTorch has no clean equivalent for).
- Polyak target update on the critic only.
- Weight init: custom `uniform_fan_avg_init` matching JAX
  `variance_scaling(1.0, 'fan_avg', 'uniform')` — **not** PyTorch's default
  `kaiming_uniform_`.
- Activation: `F.gelu(x, approximate='tanh')` matches JAX `nn.gelu` default.
- `lerobot_to_qc.py` (in `scripts/live_bridge/`) already produces the exact
  schema `Dataset.create()` expects (`observations, next_observations,
  actions, rewards, terminals, masks`) — no adapter needed there.

## Housekeeping

- There's a stray empty `agent/` directory (singular) alongside the real
  `agents/` package — leftover, safe to delete once noticed.

## Running tests

```bash
# from g3a-bench repo root
PYTHONPATH="g3a_bench/scripts/qc_fql:$PYTHONPATH" uv run python -m pytest g3a_bench/scripts/qc_fql/tests/ -v

# regenerate gold data for a module (needs jax+flax, installed via `uv pip install`,
# NOT added to pyproject.toml/uv.lock — they're dev-only for gold-data generation)
uv run python g3a_bench/scripts/qc_fql/tests/gen_gold_mlp.py
```
