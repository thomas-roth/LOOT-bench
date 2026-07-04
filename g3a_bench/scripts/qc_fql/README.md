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
| 2 | FourierFeatures | ✅ done |
| 3 | ActorVectorField | ✅ done |
| 4 | Value | ✅ done |
| 5 | Dataset | ✅ done |
| 6 | ReplayBuffer | ✅ done |
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
- `nn.LayerNorm(dim)` → `nn.LayerNorm(dim, eps=1e-6)`. *Why required:* flax's
  `nn.LayerNorm` defaults to `epsilon=1e-6`; PyTorch's default is `1e-5`. This
  wasn't caught when MLP was first ported because `test_mlp.py` only
  exercises `layer_norm=False` — it surfaced later (step 4, `Value`) as a
  ~2e-4 output mismatch once a `layer_norm=True` gold case actually ran.
  Fixed at the source since it's a real numerical bug, not a test-loading
  quirk.

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

### 3. ActorVectorField — done

**Copied:** conditioning structure — concat `[obs, actions]` (or `[obs,
actions, time_embedding]` if times given) → single inner MLP (`activate_final
=False`) → velocity output.

**Changed:**
- `encoder` param dropped entirely. *Why valid:* state-only observations
  (design decision), and in JAX `encoder=None` is already a no-op passthrough
  — omitting the parameter changes nothing behaviorally.
- Flax's `setup()` (lazy shape inference — `nn.Dense` infers `in_features`
  from the first actual input) → PyTorch `__init__` needs `in_features`
  fixed up front (`obs_dim + action_dim [+ time_width]`). *Why valid:*
  obs_dim/action_dim are static and known before training starts.
- **The one real design addition:** `qc/agents/acfql.py` constructs the same
  `ActorVectorField` class two different ways — `actor_bc_flow` is always
  called *with* `times` (`ex_observations, full_actions, ex_times`),
  `actor_onestep_flow` is always called *without* it
  (`ex_observations, full_actions`). JAX doesn't need to decide this ahead of
  time since `nn.Dense` infers the width lazily on first call; PyTorch does,
  so a `use_time: bool` constructor flag was added (not present in the JAX
  dataclass) to pin the inner MLP's input width at construction —
  `use_time=True` for `actor_bc_flow`, `use_time=False` for
  `actor_onestep_flow`, mirroring exactly how each is called in practice.
  `forward()` asserts `times is None` iff `not use_time`, surfacing a clear
  error instead of a silent shape mismatch.
- Verified directly (not guessed) that param names nest as `mlp/Dense_i/...`
  in flax, so the *existing* `conftest.py::load_jax_mlp_weights` helper is
  reused as-is with `prefix="mlp/"` — no new weight-loading code needed.

### 4. Value — done

**Copied:** per-head architecture (`MLP(*hidden_dims, 1)`, `activate_final=
False`), concat `[obs, actions]`, `squeeze(-1)` on each head's output.

**Changed:**
- `nn.vmap`-based `ensemblize(MLP, num_qs)` (auto-parallelizes N
  independently-initialized copies over a synthetic axis) →
  `nn.ModuleList([MLP(...) for _ in range(num_qs)])`, looped in `forward`,
  `torch.stack(outputs, dim=0)`. *Why valid:* verified directly by running
  the real JAX class — `nn.vmap` with `variable_axes={'params':0},
  split_rngs={'params':True}, out_axes=0` produces per-head params with an
  extra leading axis of size `num_ensembles` (e.g. `Dense_0/kernel:
  (2, in, out)`) and an output shaped `(num_ensembles, batch)` — a Python
  loop over N separately-constructed MLPs, stacked at `dim=0`, computes the
  identical per-head function; only the execution strategy (batched XLA vmap
  vs. eager loop) differs, not the math.
- **Structural fork ported as-is:** with `num_ensembles == 1`, JAX skips
  `ensemblize` entirely (`if self.num_ensembles > 1: ...`) — no ensemble axis
  at all, output is plain `(batch,)` instead of `(1, batch)`. QC-FQL's config
  always uses `num_ensembles=2`, so this never triggers in practice, but the
  port preserves the fork rather than normalizing it away.
- Same lazy-shape issue as `ActorVectorField` (step 3): since ACFQL's critic
  is always called with `actions` provided, added the same kind of
  `use_actions: bool = True` constructor flag to pin the input width, with
  `forward()` asserting consistency instead of silently branching on shape.
- **Bug found by this step's gold test, not obvious from inspection alone:**
  JAX only creates a `LayerNorm` for layers that actually use one (all but
  the last, since `activate_final=False`); our `MLP` always allocates one
  per layer (dead weight on the last, since `forward()` never applies it
  there) for structural simplicity. The weight-loading test tried to load a
  `LayerNorm_2` that JAX never created — fixed in the test loader
  (`conftest.py::load_jax_value_weights`), not in `MLP` itself, since it's a
  test-infra mismatch, not a forward-pass correctness bug.
- **Real numerical bug found by this step, fixed at the source:** flax's
  `nn.LayerNorm` epsilon (`1e-6`) differs from PyTorch's default (`1e-5`) —
  see the MLP section above. This one *is* a correctness fix to `mlp.py`,
  not a test-only issue, since it changes real forward-pass output whenever
  `layer_norm=True` (which the real critic config always uses).

### 5. Dataset — done

**Copied verbatim** (the entire `sample_sequence` body is pure NumPy in the
original — nothing JAX-specific to translate): index gathering
(`idxs[:,None] + arange(seq_len)`), cumulative discounted reward
(`rewards[t] = rewards[t-1] + raw[t]*discount**t`), running
`masks`/`terminals` via min/max, `valid[t] = 1 - terminals[t-1]`, next-action
index clipped to `size-1`. `__getitem__`/`items()` are implemented so the
copied `self['x']` / `self.items()` calls stay a true verbatim copy rather
than transcribed.

**Changed:**
- `FrozenDict` (immutable, JAX-pytree-compatible dict subclass — required so
  it can pass through `jax.jit`/`tree_map` boundaries) → plain class wrapping
  `self.data: dict`. *Why valid:* nothing here is jitted or traced in
  PyTorch; immutability existed for JAX's functional-purity contract, not the
  algorithm. `create()` still calls `arr.setflags(write=False)` (plain numpy,
  unchanged) to protect raw arrays; `__init__` stays unfrozen so step-6
  `ReplayBuffer` can write.
- `get_size` (`tree_map` max over possibly-nested leaves) → `len(data[
  'observations'])` — our fields are always flat, equal-length arrays.
- **Dropped as grep-confirmed dead code for ACFQL:** `frame_stack`/`p_aug`/
  `augment`/`random_crop`/`return_next_actions`/`terminal_locs`/
  `initial_locs`/`add_history`, and the visual `len(shape)==5` transpose
  branch inside `sample_sequence`. All support pixel/frame-stacked obs;
  never referenced by `agents/acfql.py`, `main.py`, or `main_online.py`.
- **The `observations` vs `full_observations` distinction** (previously
  flagged as a bug in an early draft): the real source returns
  `observations = data['observations'][idxs]` — just the **starting** obs,
  `(B, obs_dim)` — with `full_observations` as the separate `(B,T,obs_dim)`
  sequence. `critic_loss`/`actor_loss` consume `batch['observations']`
  expecting the non-repeated shape. The gold test confirms our port produces
  `(8,16)` vs `(8,5,16)` respectively.
- `return dict(...)` → dict-literal (repo's ruff `C408`); mechanical, no
  behavior change.

**Testing note:** `sample_sequence` draws a single `np.random.randint`
internally, so the gold generator and the test both `np.random.seed(0)`
immediately before calling it — identical seed + identical single draw ⇒
identical indices ⇒ deterministic downstream, letting `np.allclose` compare
all 9 output keys against the real `qc` Dataset.

### 6. ReplayBuffer — done

**Copied:** circular-buffer semantics — preallocate `(size, *shape)` zero
arrays, `add_transition` writes at `pointer`, `pointer = (pointer+1) %
max_size`, `size = max(pointer, size)`; `create_from_initial_dataset`
preloads offline data into the buffer prefix + `size = pointer = len(init)`;
`clear`. `sample_sequence` is **inherited unchanged** from `Dataset` (samples
within `self.size`, the current fill count).

**Changed:**
- `jax.tree_util.tree_map` over the transition dict (JAX's generic
  "apply-per-leaf" utility, needed because it also works on arbitrary nested
  pytrees) → plain per-key loops (`{k: fn(v) for k, v in d.items()}` /
  `for k in self.data: self.data[k][pointer] = transition[k]`). *Why valid:*
  our dicts are flat (fixed known keys) — `tree_map` on a flat dict
  degenerates to exactly this loop, no generality lost.
- `get_size(self._dict)` (tree-max) → `len(data["observations"])`, same as
  step 5. Buffers stay writable because `create`/`create_from_initial_dataset`
  go through `__init__` (unfrozen), never `Dataset.create` (which freezes).

**Faithful quirk preserved (not fixed):** after wrapping a size-`N` buffer,
`size` caps at `N-1`, never `N` — because `size = max(pointer, size)` and
`pointer` wraps `N→0` before it can equal `N`, so the last slot is written but
never counted/sampled. This is an off-by-one in the *original* qc code; the
gold test locks in the exact `pointer=3, size=9` state after 13 adds into a
size-10 buffer, so the port reproduces it rather than silently diverging.

**Testing note:** the gold generator replays a fixed, saved sequence of
transitions through the real `qc` `ReplayBuffer` and dumps the final buffer
arrays + `pointer`/`size`/`max_size`; the port replays the identical inputs
and must match exactly (`np.array_equal`). Also covers
`create_from_initial_dataset`, inherited `sample_sequence` (seeded, same
determinism trick as step 5), and `clear`.

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

# regenerate gold data for a module (needs jax+flax+distrax, installed via
# `uv pip install`, NOT added to pyproject.toml/uv.lock — dev-only, for
# gold-data generation. distrax is only needed because `qc/utils/networks.py`
# imports it at module level for other classes we don't use.)
uv run python g3a_bench/scripts/qc_fql/tests/gen_gold_mlp.py
```
