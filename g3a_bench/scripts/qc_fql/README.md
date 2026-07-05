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
├── agent/            QCFQLConfig, QCFQLAgent (losses, update, sample_actions) --
│                     singular, NOT `agents/`: qc's own `agents/` package
│                     would collide on the top-level import name otherwise
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
| 7 | Config | ✅ done |
| 8 | Agent (QCFQLAgent) | ✅ done |
| 9 | Evaluate loop | ✅ done |
| 10 | Training script | ✅ done |
| 11 | g3a-bench / mjlab integration | ✅ done (cube, DK1 unimanual) |

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

**~~Known loose end~~ (resolved in the audit pass):** `mlp.py`'s unused
`activation` constructor param was removed (forward hardcodes the
flax-matching `F.gelu(approximate="tanh")`; nothing ever passed the param).

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

### 7. Config — done

**Copied:** every consumed hyperparameter name and default value from
`get_config()` verbatim (lr, batch_size, actor/value hidden dims, layer_norm
flags, discount, tau, q_agg, alpha, num_qs, flow_steps, horizon_length,
action_chunking, actor_type, actor_num_samples, use_fourier_features,
fourier_feature_dim, weight_decay). `ob_dims`/`action_dim` are dataclass
fields with `init=False`, set later by `QCFQLAgent.create()` — matching how
JAX's `create()` assigns `config['ob_dims']`/`config['action_dim']` on the fly.

**Changed:**
- `ml_collections.ConfigDict` (supports `absl.flags` CLI integration and
  JAX-pytree-safe placeholders like `config_dict.placeholder(int)` for
  "set later" fields) → plain Python `@dataclass`. *Why valid:* the
  flag-system/pytree integration is JAX/absl-specific tooling our PyTorch
  code doesn't use; a dataclass gives identical attribute-access semantics
  and defaults with none of that machinery needed.
- Dropped three fields, each confirmed by grepping the *entire* `acfql.py`
  (not just `get_config()`) rather than assuming: `agent_name` (never read
  inside `acfql.py` — only used by `main.py`'s multi-agent registry lookup;
  we build exactly one agent), `normalize_q_loss` (defined but never read
  anywhere in the file — genuinely dead), `encoder` (real logic in JAX, but
  always `None` for state-based tasks, and the plumbing itself was already
  dropped from `ActorVectorField`/`Value` in steps 3–4).

**No gold-data test for this step** — it's a pure config container with no
numerical computation to verify against JAX; instantiation/defaults were
smoke-tested directly instead (matches the original plan, which also had no
dedicated test for the Config task, only for the Agent).

### 8. Agent — done

**Copied:** all loss math verbatim — TD target `reward + discount^H · mask ·
next_q`, flow-matching interpolation `x_t=(1-t)x_0+t·x_1`, target velocity
`x_1-x_0`, distillation loss, `q_loss = -mean(Q(obs, actor_actions))`, Euler
integration in `compute_flow_actions`, both `sample_actions` branches
(`distill-ddpg`, `best-of-n`). Polyak update `target = tau·current +
(1-tau)·target`.

**Changed — mechanical, low-risk:**
- `flax.struct.PyTreeNode` immutable agent + `agent.replace(...)` functional
  update → ordinary mutable `nn.Module`, `optimizer.step()` mutates
  parameters in place. `update()` returns only the info dict, not a new agent.
- Explicit `rng` key-splitting at every stochastic call → PyTorch global RNG.
- `jax.lax.scan` over `utd_ratio` in `batch_update` → plain Python `for` loop
  calling `update()` repeatedly.
- `copy.deepcopy(critic_def)` for the target critic → same call, same
  meaning, in PyTorch.
- (All four: same reasoning as prior steps — JAX's version exists for
  jit/pytree/XLA-compilation requirements our eager PyTorch code doesn't have.)

**A previous draft of this README claimed "the actor's `q_loss` term also
contributes gradient into the critic's own parameters, requiring one shared
optimizer." That claim was wrong — corrected here after re-reading
`utils/flax_utils.py::TrainState.__call__`'s own docstring:**
> "the default behavior is to stop the gradients, and you need to explicitly
> provide the parameters to flow the gradients."

`critic_loss`'s critic call passes `params=grad_params` (gradient flows in —
this is critic_loss's own regression target). `actor_loss`'s critic call
(`qs = self.network.select('critic')(obs, actions=actor_actions)`) does
**not** pass `params=grad_params` — it uses the closed-over frozen snapshot,
so `jax.grad` (which only differentiates w.r.t. the actual `grad_params`
*argument*) contributes **zero** gradient to the critic's own weights from
that call. Gradient only flows backward through `actor_actions` into
`actor_onestep_flow`. This is the standard actor-critic pattern — e.g.
OpenAI SpinningUp's DDPG does the exact same thing, explicitly:
```python
# Freeze Q-network so you don't waste computational effort
# computing gradients for it during the policy learning step.
for p in ac.q.parameters():
    p.requires_grad = False
...
loss_pi.backward()
pi_optimizer.step()
for p in ac.q.parameters():
    p.requires_grad = True
```

**The actual (corrected) requirement:** PyTorch's autograd has no per-call
"use this frozen snapshot instead" switch like JAX's `params=None` default —
gradient flows through every differentiable op unless explicitly blocked.
So `actor_loss` explicitly toggles `requires_grad_(False)` on the critic's
parameters around that one call, then restores it — otherwise a single
shared optimizer *would* incorrectly apply the actor's gradient signal to
the critic's own weights (verified directly: `test_critic_only_gets_gradient_
from_its_own_loss` confirms zero critic gradient, nonzero
`actor_onestep_flow` gradient, after `actor_loss.backward()`). One optimizer
over all params (critic + actor_bc_flow + actor_onestep_flow), one combined
`total_loss = critic_loss + actor_loss`, single `backward()`/`step()` —
matching JAX's single joint `apply_loss_fn` call structurally, now for the
right reason.

**Required addition (testing-only, doesn't change real behavior):**
`critic_loss`/`actor_loss`/`sample_actions` draw their own randomness
internally (`x_0`, `t`, distillation noise, next-action noise). JAX's
`jax.random` and PyTorch's `torch.randn` never produce the same numbers, so
comparing loss scalars directly would fail for reasons unrelated to
correctness. Added optional injected-noise arguments to these methods —
inert in production (omitted, each samples its own noise exactly as JAX
does). The gold generator replicates JAX's exact `rng`-splitting sequence
from `total_loss`/`critic_loss`/`actor_loss` *outside* the agent (JAX's PRNG
is a pure function of its key, so this reproduces the identical arrays as
plain NumPy) and calls the real `agent.critic_loss`/`actor_loss` with the
same keys for the reference values — no source modification needed.

**Bugs/gotchas found while writing the gold generator, not the port itself:**
- `ex_observations`/`ex_actions` passed to `create()` must be genuinely
  *unbatched* — shape `(obs_dim,)`, no leading `1`. This matches what
  `main.py`'s `Dataset.sample(())` actually produces (an empty-tuple size
  draws one scalar index — the step-5 finding). Passing `(1, obs_dim)`
  instead breaks `sample_actions`'s `len(config['ob_dims'])` batch-shape
  inference in a way that only surfaces as a cryptic JAX concatenate error.
- Critic loss's `q` is `(num_ensembles, batch)`, `target_q` is `(batch,)` —
  JAX's `jnp.square(q - target_q)` broadcasts this intentionally (each
  critic head regresses toward the same target). `F.mse_loss` produces the
  same numeric result but emits a broadcasting `UserWarning`; switched to
  the literal `(q - target_q) ** 2` to match the source exactly and silence
  the warning.
- **Package name collision, unrelated to any of the above:** this repo's own
  `agents/` package and `qc/agents/` (real ACFQL source) share the same
  top-level name. Since Python caches imports by top-level name in
  `sys.modules`, whichever gets imported first within a pytest session wins
  — `test_dataset.py` (collected alphabetically before `test_qcfql_agent.py`)
  imports `gen_gold_dataset`, which inserts `qc/` at the front of `sys.path`,
  so `agents.config` resolves to `qc/agents/` (no `config` submodule there)
  instead of ours. **Fixed by renaming our package to `agent/` (singular)** —
  which, it turns out, was the originally-planned name all along (the
  Housekeeping note below about a stray empty `agent/` dir was this same
  collision, just not yet understood).
- `q_agg` fallback direction is inconsistent in the *source itself*:
  `critic_loss` checks `if q_agg == 'min': ... else: mean`, `sample_actions`'s
  `best-of-n` branch checks `if q_agg == 'mean': ... else: min` — opposite
  defaults, functionally identical only because the actual config default is
  `'mean'`. Ported both literally rather than unifying them.

**Coverage:** `critic_loss`/`actor_loss` are gold-tested bit-for-bit against
the real JAX agent (`actor_type="distill-ddpg"`, the config default — matches
the scope of the original plan for this step). `update`/`batch_update`/
`sample_actions` (both `distill-ddpg` and `best-of-n`)/`compute_flow_actions`
are smoke-tested (run end-to-end, checked shapes/bounds/finiteness/that the
Polyak update actually moves the target) — no reference trace exists for
these to compare against numerically.

### 9. Evaluate loop — done

**Copied:** the action-chunk queue logic exactly — sample once per
`horizon_length` steps (`len(action_queue)==0`), execute one action at a
time, `np.clip(action, -1, 1)` before stepping; the generic `flatten(info)`-
based stat aggregation (averages whatever keys the env's terminal `info`
dict happens to contain, without hard-coding any key name); final mean over
episodes. Assumes the same gym-style single env as the source
(`env.reset()`/`env.step()`), not mjlab's vectorized API — that gap is
step 11's, not this one's.

**Dropped** (confirmed unused by every real caller of `evaluate()`, or dead
in the source itself — checked both `main.py` call sites and
`main_online.py`'s, which all do `eval_info, _, _ = evaluate(...)`):
`supply_rng`/explicit `rng` threading (same reasoning as every prior step —
PyTorch's implicit global RNG removes the need); `trajs`/`renders`/video
recording (always discarded via `_, _`); `eval_temperature` (never
referenced in the source's own function body — dead there too);
`eval_gaussian`/`action_shape`/`observation_shape` (real code paths, but
never passed a non-default value by any caller); gripper-contact-length
tracking (tied to the original paper's specific robot/sensor conventions,
not applicable to mjlab).

**Added, not from source:** `return`/`length` tracking (sum of reward, step
count). The source relies *entirely* on the env's terminal `info` dict for
stats and never computes these itself — a real gap for a general eval loop,
trivial to add, called out explicitly as an addition rather than blended in
silently.

**Testing approach:** this is orchestration/control-flow, not numerical
computation — no JAX gold data to compare bit-for-bit. Tested against a
small synthetic gym-style env (fixed episode length, fixed reward, reports
`success` in terminal info) using the real, already-tested `QCFQLAgent`:
confirms the action queue samples exactly `ceil(episode_length /
horizon_length)` times per episode (not once per step), and that
return/length/`success` all aggregate correctly across episodes.

### 10. Training script — done

**Resolved: `main.py`, not `main_online.py` — not actually a choice.**
Checked which agent each script wires up by default
(`config_flags.DEFINE_config_file('agent', ...)`):
```python
# main.py:         'agents/acfql.py'   -> ACFQLAgent (our agent)
# main_online.py:  'agents/acrlpd.py'  -> ACRLPDAgent (a different agent)
```
and `qc/agents/__init__.py` confirms these are two separate classes
(`agents = dict(acfql=ACFQLAgent, acrlpd=ACRLPDAgent)`). The qc repo's own
README's "Reproducing paper results" section spells out the same thing:
QC-FQL (and QC/BFN-n/BFN/FQL-n/FQL) all use `main.py`; RLPD/RLPD-AC/QC-RLPD
use `main_online.py`. `main_online.py`'s mixed offline+online batch sampling
was never a design option for QC-FQL — it belongs to a different agent this
port doesn't touch.

**Copied from `main.py`:** the offline loop (`dataset.sample_sequence` →
`agent.update`, periodic eval/log); the online loop's action-chunk queue
(same logic as `evaluate.py`, now writing to a replay buffer instead of just
executing); `masks = 1.0 - terminated` (**not** `1.0 - done`) — bootstrapping
disabled only on true termination, not truncation, preserved exactly and
covered by a dedicated test; the `utd_ratio` batch reshape
(`jax.tree.map(lambda x: x.reshape((utd_ratio, batch_size) + x.shape[1:]), batch)`
→ per-key `.reshape((utd_ratio, batch_size) + shape[1:])`) so `batch_update`
gets `utd_ratio` sequential mini-batches; `ReplayBuffer.create_from_initial_dataset`
seeding the buffer with offline data before online steps begin.

**Dropped** (confirmed environment/dataset-specific to the original paper's
benchmark suite — d4rl/ogbench/robomimic — none of which apply to g3a-bench):
`process_train_dataset`'s `dataset_proportion`/`is_robomimic_env` reward-penalty/
`sparse` reward handling, antmaze/robomimic online reward adjustment,
`ogbench_dataset_dir`/`dataset_replace_interval`, `save_all_online_states`;
`agents[config['agent_name']]` registry lookup → directly `QCFQLAgent.create(...)`;
`example_batch = train_dataset.sample(())` → `dataset.data["observations"][0]`
directly (same reasoning as steps 5/8); `wandb`/`CsvLogger`/absl `flags` →
`argparse` + plain `print` (real experiment-tracking integration is out of
scope here, can be wired to g3a-bench's existing wandb conventions during
step 11); `save_agent`'s pickle+flax-serialization → plain
`torch.save(agent.state_dict(), ...)`.

**Structural decision, not a translation question:** like `evaluate.py`, the
online phase needs a gym-style env that doesn't exist yet (step 11).
`train_online()` is fully implemented and tested against the same synthetic
env as step 9; `main()`'s CLI runs the offline phase for real (needs no env
at all) and raises a clear `NotImplementedError` if `--online-steps > 0` is
requested without one — honest about the gap rather than writing speculative
mjlab-specific code we already know would need rewriting once step 11 nails
down the real API (auto-reset, tensor/dict observations).

### 11. g3a-bench / mjlab integration — done (cube tasks, DK1 unimanual)

Nothing here is a JAX→PyTorch translation — genuinely new code, verified by
*running it against the real env*, not `allclose`. Scope: cube task family,
DK1 unimanual robot (per user decision) — action denormalization constants
are DK1-unimanual-specific, not yet generalized to other robots/categories.

`scripts/mjlab_env.py::MjlabSingleEnv` wraps `ManagerBasedRlEnv` (`num_envs=1`)
as the gym-style single-env interface `evaluate.py`/`train_online.py` already
assume — no changes needed to either of those files. Every design fact below
was checked directly against the real, running env (see the module's own
docstring for the exact probes), not assumed from reading docs:

- **Observations**: `{"actor": Tensor(1,53), "critic": Tensor(1,53)}` for
  cube tasks — two groups (not `"policy"`), each already a flat, concatenated
  vector. Uses `"actor"` (the noisy/corrupted group — matches what a
  deployed policy actually perceives; `"critic"` is clean-state-only).
- **Action denormalization**: mjlab does not clip/normalize actions —
  `target = raw_action * scale + default_joint_pos`, with `scale` varying
  >700× across the 7 DK1 joints. QC-FQL's policy outputs `[-1,1]`.
  `g3a_bench/scripts/live_bridge/mapper.py::PHYSICAL_RAW_ACTION_MIN/MAX`
  (the fixed DK1 joint range in raw_action space — a robot constant, not
  tied to any specific recording) is used to un-normalize before
  `env.step()`. Confirmed **live**, not assumed, that the real action
  term's `target_names` (`['joint1',...,'joint6','gripper_left']`) and its
  `scale`/`offset` tensors are bit-identical to `mapper.py`'s
  `ACTION_SCALE`/`HOME_JOINT_POS` — no silent joint-ordering mismatch.
  `lerobot_to_qc.py` already defaults to this same `"physical"` normalization
  method for recorded demos (not dataset-derived min/max), so the policy and
  the demos agree on what `[-1,1]` means.
- **`auto_reset=True`** (kept, per decision): confirmed live that on episode
  end, `step()`'s returned observation is already the *fresh post-reset* one
  for the next episode, not the true terminal state (`masks=1-terminated`
  already neutralizes this correctly for bootstrapping — same limitation
  `qc`'s own vectorized envs have). Since `evaluate.py`/`train_online.py`
  expect standard gym semantics (call `.reset()` after `done`), an
  unconditional external reset would trigger a **redundant second reset**.
  `MjlabSingleEnv` caches the auto-reset observation and returns it from the
  next `.reset()` call instead — contained entirely in this one file, so
  nothing else needed to change, and the existing synthetic-env tests for
  `evaluate.py`/`train_online.py` (which aren't auto-resetting) stay valid.
- **`terminated`/`time_outs`** map directly to gym's `terminated`/`truncated`
  (confirmed: `time_out` is its own termination term with `Time Out=True`;
  `success`/`ee_ground_collision` are `Time Out=False`).
- **Success signal**: no framework-level key exists in `extras` — surfaced
  via `env.command_manager.get_term("arrangement").episode_success`
  (cube-specific), included in `info["success"]` so `evaluate.py`'s existing
  generic `flatten(info)` aggregation picks it up with no changes there either.

**Verified by actually running it** (not just unit-testing the wrapper):
constructed the real `G3a-Cube-Stack3-Dk1_uni` env, ran a fresh (untrained)
`QCFQLAgent` through the *unmodified* `evaluate()` for a full episode —
2175 steps, terminated via time-out, produced sane `{return, length,
success}` stats, no errors. That's the actual verification this step needs;
`allclose` doesn't apply to any of it.

**Still open** (deliberately out of scope for this pass, not hidden gaps):
no converted demo `.npz` exists yet for this task — `train_offline` can't be
run for real until one is recorded (via `live_bridge`) and converted (via
`lerobot_to_qc.py`, default `physical` norm method — already correct,
nothing to change there). Action denormalization is DK1-unimanual-specific;
extending to other robots/categories (insertion, ball, bimanual) means
locating/deriving their own physical action-range constants the same way.

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

- ~~Stray empty `agent/` directory~~ — resolved in step 8: this was the
  correct name all along, needed to avoid colliding with qc's own `agents/`
  package name in `sys.modules`. The plural `agents/` package (config.py,
  qcfql_agent.py) was moved into it.
- Since the audit pass, all port-internal imports are package-qualified
  (`g3a_bench.scripts.qc_fql.*` — `g3a_bench` is installed editable), so the
  old `PYTHONPATH` hack is gone and a top-level name collision with qc's
  `agents/` is structurally impossible. The `agent/` (singular) name is kept
  anyway; renaming back would only reintroduce confusion.

## Independent audit (Fable pass)

A separate session re-verified this port per `FABLE_TASK.md`, without
trusting the account above. What it did:

- **Regenerated all gold data from scratch** from the real `qc/` source and
  re-ran the suite against it — everything reproduced, including the agent
  loss reference values bit-for-bit.
- **Break-tested the tests**: GELU tanh→erf swap, LayerNorm eps 1e-6→1e-5,
  and dropping the `discount**H` exponent were each deliberately introduced
  and confirmed caught; the gradient-isolation test was confirmed to fail
  when the `requires_grad_` toggle in `actor_loss` is removed (and that no
  loss-value test can catch that removal — forward values are identical).
- **Fixed `gen_gold_mlp.py`** to import the real `qc` MLP instead of a
  hand-retyped lookalike, and added a `layer_norm=True` gold case with
  randomized norm params (the eps bug class had previously escaped the MLP
  test because only `layer_norm=False` was covered; default norm params
  can't detect a loader that skips them).
- **Removed one vacuous test** (`test_online_steps_without_env_raises` —
  despite its name it never tested that anything raises; the raising
  behavior is covered by `test_main_raises_for_online_without_env`).
- **Removed dead code** (`MLP.activation` param), **rewrote every comment/
  docstring** from re-derived understanding, and **converted all imports to
  package-qualified form** (killing the PYTHONPATH hack).
- **Wired up `g3a-train-qcfql`** as a console script and verified it
  end-to-end (synthetic dataset → offline training with decreasing losses →
  checkpoint saved).

Evidence tiers, honestly: `critic_loss`/`actor_loss` and all network/data
modules are verified against freshly regenerated reference outputs with
break-tested sensitivity. `update`/`batch_update`/`sample_actions`/
`compute_flow_actions` have no extractable reference trace (stateful/fresh
noise) and are covered behaviorally (shapes, bounds, finiteness, Polyak
target actually moves, gradient isolation). The mjlab adapter is new code
tested against the real env, not against any reference.

## Usage (analogous to the PPO workflow)

```bash
# Offline pretraining on a recorded demo dataset (the PPO equivalent is
# `uv run g3a-train <task_id>`):
uv run g3a-train-qcfql --dataset-path path/to/demos.npz \
    --offline-steps 1000000 --horizon-length 5 --alpha 100 \
    --checkpoint-path qcfql_agent.pt

# All knobs: uv run g3a-train-qcfql --help
```

The dataset `.npz` must come from `live_bridge/lerobot_to_qc.py` (its
default `physical` action normalization matches what the mjlab adapter
un-normalizes with — do not use `--action-norm data` for datasets meant for
this pipeline). **As of the audit pass, no recorded demo dataset exists yet
for `G3a-Cube-Stack3-Dk1_uni`** — recording demos and converting them is
the remaining manual prerequisite before a real training run.

Not supported (by design, not omission): online fine-tuning via the CLI
(raises until a real dataset exists; `train_online()` itself works — call it
directly with a `MjlabSingleEnv`), and multi-env vectorized training (this
is a single-env algorithm, matching the reference).

## Running tests

```bash
# from the g3a-bench repo root (no PYTHONPATH needed since the audit pass)
uv run python -m pytest g3a_bench/scripts/qc_fql/tests/ -v

# regenerate gold data (dev-only deps: jax, flax, distrax, ml_collections —
# installed via `uv pip install ...`, deliberately NOT in pyproject.toml.
# GOTCHA: any `uv sync` prunes them from the venv; just reinstall after.)
for gen in mlp fourier_features actor_vector_field value dataset replay_buffer agent; do
  uv run python g3a_bench/scripts/qc_fql/tests/gen_gold_$gen.py
done
```
