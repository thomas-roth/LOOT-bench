# RL Decisions — Reference for Max

## What this is

A walk-through of every finding in the review doc (`rl_mjlab_config_review.md`).
For each finding: what the problem is, how it's created, why it's a problem
under PPO, and why QC-FQL handles it differently (or doesn't). Followed by a
concise reference table of all RL decisions across both pipelines.

---

## Part 1 — The core concept: exploration vs imitation

**All of F1–F5 are about the same thing: an exploration problem.**

PPO starts with a **random policy** and must discover success by trial and
error. That's hard when every exploratory step is penalised harder than doing
nothing. The policy converges to "sit still" before it ever finds success.

QC-FQL starts by **imitating a human who already solved the task**. It never
has to explore. The actor's loss is `‖actor(obs) - demo_action‖²` — it just
copies. Rewards are irrelevant to that loss.

This distinction is the answer to every "why is it a problem for PPO but not
QC-FQL?" question below.

---

## Part 2 — The reward math (so you can see the actual numbers)

### Cube task per-step rewards after dt=0.02 scaling

| Reward term | Weight | Raw value (0..1) | After dt=0.02 |
|---|---|---|---|
| `staged_next_slot` | 1.0 | Gaussian 0..2 → 0..2 | 0..0.04 |
| `placed_slots` | 2.0 | fraction 0..1 → 0..1 | 0..0.04 |
| `success` | 2.0 | binary 0 or 1 | 0 or **0.04** |
| `action_rate_l2` | -0.01 | ~0..0.1 | ~0 |
| `joint_pos_limits` | -10.0 | up to 1.0 | up to **-0.2** |
| `joint_vel_hinge` | -0.01→-1.0 | up to 1.0 | up to -0.0002→-0.02 |

### What this means per episode (cube, 500 steps)

| Scenario | Per-step reward (avg) | Episode total |
|---|---|---|
| Sit still (no penalties) | ~0 | **~0** |
| Move but at joint limits | -0.2 | **-100** |
| Succeed (1 success step) | -0.2 for 499 steps + 0.04 | **-99.96** |

PPO compares these totals and picks "sit still" — it's the highest return.

### The -5..0 scenario (another branch, scene task)

If rewards are -5..0 with 0 on success, it's even worse:

| Scenario | Per-step reward | Episode total |
|---|---|---|
| Sit still (still incurs penalty?) | -0.1 | -50 |
| Succeed (1 zero-reward step) | -0.1 for 499 steps + 0 | **-49.9** |

Still negative. PPO still picks sitting still or dying fast (F2).

### What QC-FQL actually does with these rewards

The actor (BC flow matching) **ignores rewards completely**:
```
actor_loss = ‖actor_bc_flow(obs, noise, t) - (demo_action - noise)‖²
```
No reward term. It only depends on `(obs, action)` pairs from the demonstration.

The critic **does use rewards**:
```
target_q = reward(H-step sum) + γ^H · Q'(s_H, chunk')
```
But the critic only provides *relative* Q-values. Even if Q-values are all
negative (because rewards are -5..0), the ranking is still correct:
- Q(good_action) = -1.5
- Q(bad_action) = -3.0
- The Q-push (`-q.mean()` in distill-ddpg) pushes toward -1.5 > -3.0

So QC-FQL works correctly with any reward structure — even all-negative or
all-zero rewards — because the actor never uses the reward directly.

### One caveat

If rewards are extremely noisy or inconsistent in the data, the critic's
Q-estimates can be imprecise. This affects the **online fine-tuning** phase
(distill-ddpg's Q-push), where the actor is slightly adjusted toward
higher-Q actions. But the actor is already initialised from BC, so this is a
small correction — not "find success from scratch."

---

## Part 3 — Finding-by-finding walk-through

---

### F1 — Success terminates the episode, making the bonus ~0.04

**What the code does:**
- `success_reward` returns 1.0 while the arrangement is complete
- `success` termination fires the same step the flag first goes up
- Reward is dt-scaled: 2.0 × 1.0 × 0.02 = 0.04 (one step only)

**Why it's a problem (PPO):**
PPO must explore to find success. Every exploratory step that hits joint limits
gives -0.2. A successful episode returns -99.96. Sitting still returns ~0.
PPO picks sitting still. This is **the core exploration problem**.

**For QC-FQL:**
Does not explore. The actor clones demonstrator actions (no reward in its loss).
The critic learns from stored rewards (works fine with any values).
The problem doesn't exist because there's no exploration.

---

### F2 — Net-negative rewards turn failure terminations into escape hatches

**What the code does:**
- Insertion: `ee_werkbox_collision` (>=10N on tray) → termination
- Ball: `out_of_bounds` (ball >1.2m) → termination
- Cube: no failure terminations (but also has `ee_ground_collision`)

**Why it's a problem (PPO):**
Every step is net-negative (F1). Dying fast = fewer penalties = higher return.
PPO learns to seek out failure terminations to stop the bleeding.

**For QC-FQL:**
Offline: actor clones demonstrator actions (teleop doesn't seek failures).
Online: the critic sees failure terminations during rollouts. But the actor
starts from a policy that already succeeds — it doesn't need to explore into
failure territory.

---

### F3 — Joint-velocity hinge curriculum ramps before success is found

**What the code does:**
```python
stages = [
    {"step": 0,      "weight": -0.01},
    {"step": 12000,  "weight": -0.1},     # iter 500
    {"step": 24000,  "weight": -1.0},     # iter 1000
]
```

**Why it's a problem (PPO):**
Copied from lift-cube where the policy is succeeding by iter 500 (dense
rewards). Here the policy hasn't seen a single success by iter 1000. At
iter 1000, the penalty is -1.0 per fast joint rotation — exploration is
killed, the policy freezes.

**For QC-FQL:**
Offline: no hinge penalty in the loss. The actor trains on demo actions
which already move at reasonable speeds. Online: the environment does apply
the penalty, but the actor already moves well and doesn't need to explore.

---

### F4 — Success weight differs across domains (2.0 vs 3.0)

**Current code:**
```
cube:      2.0
insertion: 2.0
ball:      3.0
```
(The review claims 5.0 for insertion/ball — numbers may differ by branch.)

**For PPO:** With the tiny dt-scaled values (~0.04), the difference is
negligible anyway.

**For QC-FQL:** Irrelevant. The critic learns from whatever rewards are
stored, it doesn't tune success weights.

---

### F5 — γ = 0.99 for episodes up to 2500 steps

**What the code does:**
```python
gamma = 0.99  # all three rl_cfg.py files
```

**Why it's a problem (PPO):**
γ = 0.99 means a reward at step 2000 is worth 0.99^2000 ≈ 2×10⁻⁹ at the
start of the episode — effectively zero. The critic cannot propagate the
success signal back to early decisions. Standard practice for long-horizon
sparse tasks: γ = 0.995..0.999.

**For QC-FQL:**
Uses n-step returns with horizon_length H=5. The critic target is
`reward(H-step sum) + γ^H · Q'(s_H, chunk')`. There is no 2000-step
bootstrap chain — just 5 steps at a time.

---

### F6 — EE-contact protection exists only in insertion?

**Actual code:**
- Cube: yes (`ee_ground_collision`, lines 176-184)
- Insertion: yes (`ee_ground_collision` and `ee_werkbox_collision`)
- Ball: no

The review says cube doesn't have it, but it does. Ball genuinely lacks it.

**For QC-FQL:** Low impact. Human demos don't grind the floor.

---

### F7 — Domain randomization is thin

**Current:**
- Sliding friction only: uniform (0.6, 1.5), startup only
- No mass/friction/actuator randomization

**Lift-cube reference:** 0.3-1.5 sliding + torsional + rolling (log-uniform)

**For both:** If sim-to-real is the goal, this is thin. Actionable.

---

### F8 — Solver iterations differ across domains

**Current:**
```
cube:      10 iterations, 20 ls_iterations
ball:      12 iterations, 20 ls_iterations
insertion: 15 iterations, 25 ls_iterations
```

**What solver iterations are:**
MuJoCo's constraint solver runs N Newton iterations per physics step to resolve
contacts. More iterations = more accurate contact resolution = more stable
simulation, but slower. Fewer iterations = faster but contacts may jitter.

Ball (single sphere on a plane) needs fewer iterations than cube (multi-body
stacking with 1.5 cm tolerances). Having ball > cube is backwards — but if
both are stable, it doesn't matter.

**For both:** Same concern. Simulation fidelity, not algorithm-specific.

---

### F9 — Wrist action scales of 2.3-6.6 rad with Gaussian std 1.0

**Current:**
```
scale = 0.25 × effort / stiffness  → joint6 = 6.60 rad per unit action
init_std = 1.0, clip_actions = None
```

A 1σ random action commands ±6.6 rad on the wrist. The ctrlrange saturates
it, but early PPO exploration slams the wrist against limits, triggering
`joint_pos_limits` penalty (-0.2 per step).

**For QC-FQL:** Actions are clamped to [-1,1] by `torch.clamp`. During
offline training: no wrist slamming (demo actions are smooth). During online
fine-tuning: the actor starts from a good policy, so it doesn't randomly
explore ±6.6 rad.

---

### F10 — JitteredCellGrid scatter isn't truly collision-free

Cube pieces can interpenetrate by ~8 mm on reset. MuJoCo resolves it.
Minor reset-state noise for both algorithms.

---

### F11 — Minor inconsistencies

- `max_iterations`: cube 10k / insertion 15k / ball 15k — no rationale
- Ball episode fixed 12s for golf and push (push may need longer)
- `ball_velocity` noise ±0.05 m/s vs `joint_vel` ±1.5 rad/s — optimistic
- Gripper effort derate 0.3 (DK1) vs 0.1 (YAM) — same practice, different number
- `save_interval=100` over 15k iterations → 150 checkpoints pushed to W&B

---

## Part 4 — About the dataset: does QC-FQL need only successful demos?

### Original QC repo
**No filtering.** The original QC (`qc/main.py`) uses whatever
`ogbench.make_env_and_datasets()` returns — all episodes, success and failure.
The algorithm handles both: the actor imitates good actions (BC) and ignores
bad ones (they contribute to the MSE but don't dominate), the critic learns
from whatever rewards exist.

### Our pipeline (g3a-bench)
`lerobot_to_qc.py` defaults to `only_successes=True`. This is our design
choice, not a requirement of the algorithm. You can pass `--all-episodes` to
include failures. The algorithm works fine either way.

### `live_record.py` recording
The operator labels each episode:
- Numpad 1 → save as SUCCESS (auto-saved on task completion)
- Numpad 2 → save as FAILURE (e.g. ground collision)
- Numpad 3 → discard (bad demo)

So success/failure **is** split during recording. Whether both go into the
training `.npz` depends on the `only_successes` flag in `lerobot_to_qc.py`.

---

## Part 5 — Concise RL decisions reference

### PPO (current config in `tasks/*/rl_cfg.py`)

| Decision | Value | Sparse problem? | Why |
|---|---|---|---|
| Actor/critic MLP | (512, 256, 128), ELU | No | Architecture is fine |
| Distribution | Gaussian, init_std=1.0 | Yes (F9) | Huge wrist scales + no clip |
| LR | 1e-3, adaptive | No | |
| γ / λ | 0.99 / 0.95 | **Yes** (F5) | Can't propagate 2000 steps |
| entropy_coef | 0.005 | No | |
| num_steps_per_env | 24 | No | But critic carries all the burden |
| Success weight | 2.0 (cube), 2.0 (ins), 3.0 (ball) | **Yes** (F1) | 0.04 after dt-scaling |
| Penalties | up to -0.2/step | **Yes** (F1) | Swamp the sparse success bonus |
| Success termination | Yes | **Yes** (F1) | Collects bonus once, then dies |
| Failure termination | varies | **Yes** (F2) | Becomes an escape hatch |
| Hinge curriculum | 0→-1.0 by iter 1000 | **Yes** (F3) | Freezes policy before success |
| DR | Sliding friction only | No | Thin for S2R |

**Verdict:** PPO cannot train sparse tasks with this config. It's tuned for
dense rewards where every step gives positive signal.

### QC-FQL (in `scripts/qc_fql/`)

| Decision | Value | Sparse problem? | Why |
|---|---|---|---|
| Actor training | Flow-matching BC | **No** | Imitates (obs → action), no reward needed |
| Critic training | n-step TD (H=5) | **No** | Short bootstrap, works with any reward values |
| Online fine-tuning | distill-ddpg Q-push | **No** | Starts from BC policy; Q-push is small correction |
| Offline data source | `.npz` (demos) | **No** | Human already explored for us |
| Discount γ | 0.99 (config.py) | **No** | Only used over H=5, not 2000 steps |
| Success filtering | `only_successes=True` (our pipeline) | **No** | Not required by algorithm |

**Verdict:** QC-FQL was designed for exactly this scenario. The review's
F1–F5 describe an exploration problem that doesn't exist for QC-FQL.
F7 (DR) and F9 (wrist scales) are the only findings that matter.

---

## Part 6 — Recommended actions

### For PPO (only if training from scratch is needed):
1. Multiply success weight by 50× (to offset dt-scaling)
2. Don't terminate on success (let bonus accrue)
3. Add large terminal penalty on failure (-100)
4. Set γ = 0.995
5. Slow hinge curriculum to max_iterations
6. Keep at least one dense shaping term (staged_reward)

### For QC-FQL (the designed pipeline):
1. **No code changes needed for F1–F5** — they describe an exploration
   problem we don't have
2. Optionally broaden DR (F7) if sim-to-real matters
3. Optionally adjust ball episode length (F11)
4. Everything else is operational or cosmetic

---

*Generated 2026-07-06. The review doc was written against PPO configs in
`tasks/*/rl_cfg.py` / `*_env_cfg.py`. QC-FQL lives in `scripts/qc_fql/`.*
