# RL / mjlab configuration — overview and sanity check

A written survey of the RL- and mjlab-related configuration decisions across
the three task domains (`cube`, `insertion`, `ball`), how they relate to each
other and to the robot interface, followed by flagged findings. **This is a
review document — nothing was changed.** Findings are things worth a second
look, not certain bugs; each names the file so a human can decide.

Reviewed at mjlab 1.4.0 (pinned), branch `dev-task`, 2026-07-06.

---

## Part 1 — Overview

### 1.1 PPO runner (shared, `tasks/common/rl_cfg.py`)

All three domains use one runner factory,
`manipulation_ppo_runner_cfg(experiment_name, max_iterations)`. It is a
verbatim copy of mjlab's own `yam_lift_cube_ppo_runner_cfg`
(`mjlab/tasks/manipulation/config/yam/rl_cfg.py`) except for
`max_iterations`:

| Setting | Value | mjlab YAM lift-cube |
| --- | --- | --- |
| actor / critic MLP | (512, 256, 128), ELU, obs normalization on | same |
| policy distribution | Gaussian, scalar `init_std=1.0` | same |
| `learning_rate` | 1e-3, `schedule="adaptive"`, `desired_kl=0.01` | same |
| `gamma` / `lam` | 0.99 / 0.95 | same |
| `clip_param` / `entropy_coef` | 0.2 / 0.005 | same |
| `num_learning_epochs` / `num_mini_batches` | 5 / 4 | same |
| `num_steps_per_env` | 24 (= 0.48 s of experience per env per update) | same |
| `max_iterations` | cube 10 000; insertion 15 000; ball 15 000 | 5 000 |
| `save_interval` | 100 | same |

Left at mjlab defaults (not set explicitly): `seed=42`,
`obs_groups={"actor": ("actor",), "critic": ("critic",)}` (matches the two
observation groups every env cfg defines), `clip_actions=None` (no action
clipping before scaling), wandb logging with `upload_model=True`.

The per-domain `rl_cfg.py` files are one-line wrappers that only set
`max_iterations`. So the entire PPO hyperparameter set is intentionally
identical across domains and inherited from mjlab's dense-reward reference
task — see findings F1/F3/F5 for where that inheritance is questionable now
that the reward design is sparse.

### 1.2 Env cfg assembly (`make_<cat>_env_cfg`)

Structure is deliberately mirrored; `tasks/common/env_cfg.py` holds the
byte-identical parts (robot reset events, fixed-prop pinning, fingertip
friction DR, the three regularization rewards, the `joint_vel_hinge`
curriculum, play overrides). Per-domain values:

| Setting | cube | insertion | ball | mjlab lift-cube |
| --- | --- | --- | --- | --- |
| `decimation` | 4 | 4 | 4 | 4 |
| `sim.mujoco.timestep` | 0.005 | 0.005 | 0.005 | 0.005 |
| control rate | 50 Hz | 50 Hz | 50 Hz | 50 Hz |
| `episode_length_s` | 10 · goal pieces (10–50 s) | 10 + 8 · sockets; sorter flat 30 s | flat 12 s | 20 s |
| `scene.num_envs` | 1 (CLI-overridden, e.g. 2048) | 1 | 1 | 1 |
| `env_spacing` | 1.5 | 2.0 | 2.5 | 1.0 |
| solver `iterations`/`ls_iterations` | 10 / 20 | 15 / 25 | 12 / 20 | 10 / 20 |
| `impratio` / `cone` | 10 / elliptic | 10 / elliptic | 10 / elliptic | 10 / elliptic |
| `nconmax` | 55 + 80·pieces (+60 maze) | 120 + 120·pegs (+200 two-sheet); sorter 2000 | 150 | 55 |
| `njmax` | 600 + 60·pieces (+120 maze) | 900 + 90·pegs (+400 two-sheet); sorter 8000 | 800 | 600 |
| contact sensors | — | `ee_werkbox_collision` (EE subtree vs. parts tray) | — | `ee_ground_collision` (EE vs. terrain) |
| terrain | plane | plane | plane | plane |

Episode lengths scale with task size in cube/insertion but are flat in ball;
sim sizes scale with entity count. The sorter's 2000/8000 headroom is for the
~90 convex mesh parts of the STL cube (commented in the cfg).

**Actions** — identical in all three: a single `JointPositionActionCfg` over
all actuators, `use_default_offset=True` (policy outputs deltas around the
home keyframe), `scale` filled per robot by the wiring (below). No `clip` on
the action term and `clip_actions=None` in the runner.

**Observations** — same skeleton everywhere, two groups `actor` (with noise)
and `critic` (same term objects, noise off — standard asymmetric setup):

| Term | cube | insertion | ball | noise (uniform) |
| --- | --- | --- | --- | --- |
| `joint_pos` (rel) | ✓ | ✓ | ✓ | ±0.01 |
| `joint_vel` (rel) | ✓ | ✓ | ✓ | ±1.5 |
| object(s) rel. grasp site | `pieces_rel_grasp` | `pegs_rel_grasp` (+ `sheet_rel_grasp`) | `ball_rel_grasp` | ±0.01 |
| object → goal delta | `slot_goal_deltas` | `socket_deltas` | `ball_to_target` | ±0.01 |
| progress flags | `slot_placed` | `socket_inserted` | — | none |
| object velocity | — | — | `ball_velocity` | ±0.05 |
| `generated_commands` | ✓ (goals + domino yaw) | sorter only (live mouths + target one-hot) | — (duplicate of `ball_to_target`) | none |
| `last_action` | ✓ | ✓ | ✓ | none |

The joint noise magnitudes are mjlab's lift-cube values verbatim.

**Rewards** — the deliberate design decision of this benchmark (documented in
comments in `insertion_env_cfg.py` / `ball_env_cfg.py`): *no dense distance
shaping*. Task signal is only (near-)binary outcome terms, intended to be
paired with a GELLO-teleop demonstration pipeline (offline-to-online RL).
`scripts/train.py --reward-mode success|progress` zeroes the terms outside
the chosen tier; regularization is always active.

| Term (weight) | cube | insertion | ball |
| --- | --- | --- | --- |
| discrete progress | `placed_slots` (2.0) — fraction of slots placed | `inserted_count` (2.0) — scored fraction | — (both modes identical) |
| sparse success | `success` (**2.0**) | `success` (**5.0**) | `success` (**5.0**) |
| `action_rate_l2` | −0.01 | −0.01 | −0.01 |
| `joint_pos_limits` | −10.0 | −10.0 | −10.0 |
| `joint_vel_hinge` (max_vel 0.5) | −0.01 → curriculum | −0.01 → curriculum | −0.01 → curriculum |

Success in every domain is a *gated conjunction* computed by the command
term: task condition (placed / inserted / at target) AND object(s) at rest
(≤ 0.05–0.06 m/s) AND released (gripper joints open beyond ⅔ stroke in
cube/insertion, grasp site ≥ 8 cm away in ball). mjlab's reward manager
scales all rewards by the step dt (0.02 s).

**Terminations** — `time_out` everywhere plus `success` (episode ends the
step success first holds). Extra, per domain: insertion `ee_werkbox_collision`
(≥ 10 N on the parts tray); ball `out_of_bounds` (ball > 1.2 m from origin),
`off_platform` (hole variants), `wall_collision` (throw variants). Cube has
no failure termination beyond timeout.

**Events / domain randomization** — the same three everywhere: pin robot base
+ joints to the home keyframe, pin fixed props to `env_origins +
init_state.pos`, and one startup randomization of finger-pad *sliding*
friction, uniform (0.6, 1.5). Free objects are placed by the command term's
`_resample_command` (jittered-grid scatter, `resampling_time_range = 1e9` so
goals are fixed per episode). There is no mass, actuator-gain, or object
friction randomization anywhere.

**Curriculum** — one shared term: `joint_vel_hinge` weight −0.01 → −0.1 at
env step 12 000 (= iteration 500) → −1.0 at step 24 000 (= iteration 1000).
Copied unchanged from mjlab lift-cube.

**Play overrides** (`apply_play_overrides`) — infinite episode, no obs noise,
no curriculum, success termination removed.

### 1.3 Robot interface and action scaling

Tasks are robot-agnostic; `wire_manipulator_into_env` fills the placeholders
(robot entity, action scale, grasp-site names, gripper-gate joints/threshold,
finger-pad geoms for DR, EE subtree for contact sensors, viewer body) from a
`ManipulatorInterface`. Only the unimanual DK1 is currently registered.

Action scale follows mjlab's asset-zoo convention exactly:
`scale = 0.25 · effort_limit / stiffness` per actuator, with PD gains from
effective inertia (`stiffness = I_eff · ω²`, ω = 2 Hz arm / 1 Hz gripper,
damping ratio 2.0 — same constants as mjlab's YAM). Resulting DK1 scales
(rad per unit action, m for the gripper):

| joint1 | joint2 | joint3 | joint4 | joint5 | joint6 | gripper |
| --- | --- | --- | --- | --- | --- | --- |
| 0.19 | 0.11 | 0.25 | 0.84 | 2.33 | 6.60 | 0.0092 |

For reference, mjlab's YAM ends up with joint6 ≈ 5.5 by the same formula, so
the large wrist scales are inherited convention, not a DK1-specific mistake
(see F9). Motor data: DM4340/DM4310 with DK1-datasheet velocity limits
(defined but unused by the position actuator cfg — same as in mjlab's YAM),
gripper effort derated to 0.3 × 114 N for sim stability (YAM derates its
gripper by 0.1 ×). Collision config mirrors YAM's gripper-only scheme (plus a
self-collision bit), pad friction `(1, 5e-3, 5e-4)`, condim 6, priority 1 —
values identical to the YAM asset. `soft_joint_pos_limit_factor=0.9` matches
YAM.

The gripper-open success gate is a benchmark-wide convention
(`_GRIPPER_OPEN_FRACTION = 2/3` of joint stroke in `robots/interface.py`) that each robot converts to a
threshold from its own joint range.

---

## Part 2 — Flagged findings

Ordered roughly by expected impact. "Flag" means: decide consciously, don't
necessarily change.

### F1 (high) — Success terminates the episode, so the sparse bonus is worth ~one step of dt-scaled reward

All domains reward `success` *while the condition holds*, but also register
`success` as a termination, which fires the same step the flag first goes up.
With mjlab's dt-scaling, the entire task signal of an episode is therefore a
single step of `weight · dt`: **≈ 0.1** for insertion/ball (5 · 0.02) and
**≈ 0.04** for cube (2 · 0.02). Meanwhile the always-on regularization terms
accumulate every step (e.g. `joint_pos_limits` at −10, hinge at up to −1).
Under `--reward-mode success` (no progress terms) the total return of a
*successful* episode can easily be more negative than that of a do-nothing
episode. With γ = 0.99 the one-step bonus is also discounted by up to
0.99^600–2400 ≈ 2e-3–3e-11 from the start of the episode.

The comments in `insertion_env_cfg.py:199` / `ball_env_cfg.py:124` say this
design assumes a demonstration pipeline (GELLO → offline-to-online RL), but
the configs as registered run plain RSL-RL PPO from scratch
(`ManipulationOnPolicyRunner`). Until that pipeline exists, ball and
insertion (and cube under `--reward-mode success`) are, in practice, very
unlikely to train. Options a human might consider: don't terminate on
success and let the bonus accrue over a hold window (mjlab lift-cube does not
terminate on success), give the success reward a much larger weight to offset
dt-scaling, or gate the benchmark on the demo pipeline landing.
Files: `*_env_cfg.py` (rewards + terminations), `common/env_cfg.py`.

### F2 (high) — With net-negative step rewards, failure terminations become escape hatches

Related to F1: whenever the expected per-step reward is negative (sparse
modes, or any time before the policy finds the task signal), *any*
termination raises the return by cutting future penalties. Ball's
`out_of_bounds` (knock the ball 1.2 m away) and insertion's
`ee_werkbox_collision` (shove the tray with > 10 N) then act as *positive*
incentives — PPO reliably discovers "end the episode early" optima in this
regime. mjlab's lift-cube avoids this because its dense rewards keep the
per-step reward positive. A common mitigation is a negative terminal penalty
on failure terminations, or keeping a small dense positive "alive/progress"
signal. Files: `ball_env_cfg.py:139`, `insertion_env_cfg.py:216`.

### F3 (medium-high) — `joint_vel_hinge` curriculum ramps to −1.0 at iteration 1000 regardless of task progress

The stage schedule (−0.01 → −0.1 @ 500 it → −1.0 @ 1000 it,
`common/env_cfg.py:90`) is copied from mjlab lift-cube, where dense shaping
guarantees the policy is already succeeding by then. Here, with sparse
rewards and 10–15 k iteration budgets, the ramp finishes at 7–10 % of
training, likely *before* the policy has ever seen a success. From that point
the dominant gradient is "move slowly", which compounds F1/F2 into a freeze
local optimum. Consider tying the ramp to a success metric or stretching it
to the actual `max_iterations`. Also note the schedule is expressed in raw
env steps (`500 * 24`) — it silently shifts if `num_steps_per_env` ever
changes.

### F4 (medium) — Success weight differs across domains for no stated reason

Cube's `success` weight is **2.0** (`cube_env_cfg.py:136`); insertion and
ball use **5.0**. Cube also terminates on success, so its entire sparse
signal is half/five-halves of the others' with no comment explaining why.
Either align them or document the asymmetry (e.g. "cube leans on
`placed_slots`"). Under `--reward-mode success` — where `placed_slots` is
zeroed — cube's task signal is strictly weaker than the other domains'.

### F5 (medium) — γ = 0.99 for horizons up to 2 500 steps

γ = 0.99 gives an effective credit-assignment horizon of ~100 steps (2 s at
50 Hz). Episodes are 500 steps (ball) to 2 500 steps (Domino5, 50 s), and the
only reward arrives at the very end (F1). Standard practice for long-horizon
sparse manipulation is γ ≈ 0.995–0.999 (or shorter episodes). The value was
inherited from the 20 s dense-reward lift-cube task where it is appropriate.
Same file: `common/rl_cfg.py:38`. Relatedly, `num_steps_per_env=24` (0.48 s)
means the critic bootstraps across ~20–100 rollout fragments per episode —
fine in itself, but it makes the critic the only carrier of the sparse signal,
which raises the stakes on γ.

### F6 (medium) — EE-contact protection exists only in insertion

mjlab's lift-cube reference terminates on EE–ground contact
(`ee_ground_collision`). Here, insertion has the analogous
`ee_werkbox_collision` (vs. the parts tray only), while **cube and ball have
no EE contact termination at all** — the wrist/fingers can grind the terrain
plane freely (the table is visual-only; the terrain is the contact surface).
If that's deliberate (e.g. pushing dominoes flat against the surface needs
near-surface motion), a comment would help; otherwise cube/ball lost a guard
in the copy that insertion kept. Files: `cube_env_cfg.py` /`ball_env_cfg.py`
(terminations), vs. `insertion_env_cfg.py:185`.

### F7 (low-medium) — Domain randomization is a strict subset of mjlab's reference

Only sliding friction of the finger pads is randomized, and over a narrower
range (0.6–1.5 vs. lift-cube's 0.3–1.5); lift-cube additionally randomizes
torsional and rolling friction (log-uniform). No object mass/friction or
actuator randomization anywhere. Fine for a pure-sim benchmark, but if
sim-to-real (the GELLO/live-bridge direction suggests it), this is thin —
and the narrowing from 0.3 to 0.6 looks like an unexplained edit of the
copied value. File: `common/env_cfg.py:51`.

### F8 (low) — Solver iteration counts differ across domains without a stated reason

cube 10/20, ball 12/20, insertion 15/25 (`sim=` blocks in each
`*_env_cfg.py`). Insertion's bump is plausible (tight-clearance contacts),
but ball — a single sphere on a plane, the *least* contact-rich domain —
running more iterations than cube (multi-body stacking with tiny 1.5 cm
tolerances) reads backwards. Suspect these were tuned ad hoc at different
times. Harmless if all are sufficient, but stacking stability is exactly
where too-few solver iterations shows up as jitter that breaks the
`pos_threshold=0.015` / vel ≤ 0.05 success gate.

### F9 (low) — Wrist action scales of 2.3–6.6 rad with unclipped Gaussian actions

`0.25 · effort/stiffness` yields joint5 ≈ 2.33, joint6 ≈ 6.60 rad per unit
action on the DK1 (YAM: joint6 ≈ 5.5 — inherited mjlab convention). With
`init_std=1.0` and no clipping (`clip_actions=None`, no action-term `clip`),
a 1σ exploration sample commands a ±6.6 rad wrist delta; the builtin position
actuator's ctrlrange is what actually saturates it. This works in mjlab's own
tasks, so it's flagged only as "know that early exploration slams the wrist
against its limits" — which also feeds the `joint_pos_limits` (−10) penalty
and can dominate early returns. File: `robots/trlc_dk1/dk1_actuators.py:105`.

### F10 (low) — "Collision-free by construction" scatter isn't quite, for rotated boxes

`JitteredCellGrid` (`common/commands.py:54`) samples distinct cells and
jitters by 40 % of the cell half-extent. For cube's workspace grid the cell
pitch is 0.06 m; adjacent picks can be as close as 0.036 m apart. A `cube`
piece (half-extent 0.02) at random yaw has a 0.028 m footprint circumradius
and a `rect` (0.03 × 0.02) 0.036 m — so two neighbors can spawn
interpenetrating by a few mm and pop apart on reset. Not fatal (MuJoCo
resolves it), but the docstring's guarantee is overstated and the pop adds
reset-state noise. Worth either shrinking the jitter/piece sizes relative to
the pitch or softening the claim.

### F11 (notes) — Smaller inconsistencies, listed for completeness

- **`max_iterations` split** (cube 10 k vs. insertion/ball 15 k) has no
  documented rationale; with F1 unresolved the numbers are moot anyway.
- **Ball episode length is flat 12 s** for golf (ball must roll 0.85 m and
  settle) and push (robot must guide it 0.68 m and retreat 8 cm) alike; cube
  and insertion scale their budgets with task size. 12 s for PushHole feels
  tight relative to how generous cube is (10 s per piece).
- **Ball obs-noise philosophy differs**: world-frame `ball_velocity` gets
  ±0.05 m/s noise while `joint_vel` gets ±1.5 rad/s (locomotion-derived);
  perfect low-noise object velocity is optimistic if a camera pipeline is the
  eventual source.
- **Gripper effort derate factor** is 0.3 on DK1 vs. 0.1 on the YAM asset it
  cites as "same practice" (`dk1_actuators.py:89`) — same practice, different
  number; fine if intentional, worth one clarifying word.
- **Operational**: `save_interval=100` over 15 k iterations = 150 checkpoints,
  and mjlab's default `upload_model=True` pushes each to W&B — storage adds
  up across 26 registered tasks.
- **Sorter sim arrays** (`njmax=8000`, `nconmax=2000`) are per-world; at 2048
  envs this is a large constraint buffer — expected, just budget GPU memory
  accordingly.

---

## Summary

The configuration is highly disciplined: one PPO recipe, one env-cfg
skeleton, one robot seam, all traceable to mjlab's YAM lift-cube reference,
with per-domain values isolated exactly where the CLAUDE.md conventions say
they should be. The cluster of findings that matters is F1–F5: the benchmark
swapped lift-cube's *dense* reward design for a *sparse* one (deliberately,
pending a demonstration pipeline) but kept the dense-reward training
hyperparameters — success-termination, dt-scaled tiny bonuses, the aggressive
hinge curriculum, γ = 0.99, entropy 0.005 — and those defaults are tuned for
a regime this benchmark is no longer in. Everything else is minor drift
(F6–F10) worth a comment or a small conscious decision.
