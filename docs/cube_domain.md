# The Cube Domain

`cube` covers long-horizon piece arrangement: pick scattered pieces off the
table and stack them or stand them up as a row of dominos. It's the
procedurally-generated counterpart of the OGBench cube tasks, with more
variation axes (form, color, count, distractors, arrangement type).

Source lives in [`g3a_bench/tasks/cube/`](../g3a_bench/tasks/cube/), following
the standard task-category layout from [`CLAUDE.md`](../CLAUDE.md).

---

## At a glance

| Concept | What it means |
| ------- | ------------- |
| **Piece** | A free-body scene entity (`piece_0`, `piece_1`, …). One of four forms: `cube`, `cylinder`, `rect` (flat cuboid), `domino` (tall thin cuboid). |
| **Variant** | A frozen config fixing the piece set + goal arrangement (e.g. "stack 3 red cubes"). |
| **Slot** | One position in the goal arrangement: bottom-to-top within a stack, or left-to-right in a domino row. |
| **Arrangement** | `stack` (pieces on top of each other) or `domino` (pieces upright in a row, spaced for the domino effect). |
| **Goal pieces vs. distractors** | Goal pieces fill slots; distractors are extra pieces with no slot, present only to make perception harder. |

Task ids look like `G3a-Cube-<Variant>-<Robot>`, e.g. `G3a-Cube-Stack3-Dk1_uni`.

---

## File map

| File | Responsibility |
| ---- | -------------- |
| [`variants.py`](../g3a_bench/tasks/cube/variants.py) | `Form`, `FORM_HALF_EXTENTS`, `PieceCfg`, `CubeVariantCfg`, and the `CUBE_VARIANTS` dict. |
| [`assets.py`](../g3a_bench/tasks/cube/assets.py) | `get_piece_spec` — procedural `MjSpec` per piece (box or cylinder geom + free joint). The `PALETTE` of named colors. |
| [`cube_env_cfg.py`](../g3a_bench/tasks/cube/cube_env_cfg.py) | `make_cube_env_cfg(variant, robot, play)` — assembles the full `ManagerBasedRlEnvCfg`. |
| [`mdp/commands.py`](../g3a_bench/tasks/cube/mdp/commands.py) | `PieceArrangementCommand(Cfg)` — samples goals, scatters pieces, tracks per-slot placement and success. |
| [`mdp/observations.py`](../g3a_bench/tasks/cube/mdp/observations.py) | `pieces_rel_grasp`, `slot_goal_deltas`, `slot_placed_flags`. |
| [`mdp/rewards.py`](../g3a_bench/tasks/cube/mdp/rewards.py) | `placed_slots_reward`, `success_reward`. |
| [`mdp/terminations.py`](../g3a_bench/tasks/cube/mdp/terminations.py) | `arrangement_complete`. |
| [`mdp/__init__.py`](../g3a_bench/tasks/cube/mdp/__init__.py) | Re-exports the MDP terms. |
| [`rl_cfg.py`](../g3a_bench/tasks/cube/rl_cfg.py) | `cube_ppo_runner_cfg` — RSL-RL PPO runner config. |
| [`config/__init__.py`](../g3a_bench/tasks/cube/config/__init__.py) | Registers every variant × robot into the mjlab registry. |
| [`README.md`](../g3a_bench/tasks/cube/README.md) | Short in-tree summary + variant table. |

---

## How a variant is defined

A variant is a frozen `CubeVariantCfg` ([`variants.py`](../g3a_bench/tasks/cube/variants.py))
listing `stacks` (a tuple of stacks, each a tuple of `PieceCfg` bottom-to-top),
optional `distractors`, and flags like `interchangeable` and `domino_spacing`.
Derived properties give the goal-piece list, total piece list, and the
slot→stack mapping.

`FORM_HALF_EXTENTS` and `form_height()` are the single source of geometry —
the command derives stacking heights from them, so goals never drift from
the actual piece size.

The nine registered variants range from `Stack3` (3 red cubes) through
`Stack3Distract`, `TwoStacks`, mixed forms, to `Domino5` / `Domino5Distract`.
Full table in [`README.md`](../g3a_bench/tasks/cube/README.md).

---

## The command: where the task logic lives

`PieceArrangementCommand` ([`mdp/commands.py`](../g3a_bench/tasks/cube/mdp/commands.py))
is a mjlab `CommandTerm`. Resampling is locked to the episode
(`resampling_time_range = (1e9, 1e9)`), so it only fires on env reset. On
reset (`_resample_command`):

1. **Goal poses.** For stacks, base positions come from a coarse jittered
   grid (with a minimum separation); slot z's are cumulative form heights.
   For dominos, a row direction (`row_yaw`, biased lateral so the row fits in
   x) and center are sampled, and slots step out along the row at
   `domino_spacing`.
2. **Slot assignment.** Interchangeable pieces (same form+color) get their
   slot assignment permuted, so identical pieces can fill each other's slots.
3. **Scatter.** All pieces (goal + distractors) land on distinct cells of a
   jittered grid over the workspace — collision-free by construction, fully
   vectorized, random yaw.

Each step (`_update_metrics`) recomputes per-slot distance to goal. A slot is
**placed** when within `pos_threshold` (1.5 cm); domino slots additionally
require the piece upright and rotated so its thin axis lies along the row
(else toppling wouldn't propagate). Success additionally requires every goal
piece moving slowly and the gripper open — termination, the `success`
reward, and the `episode_success` metric all read the same condition.

The command exposes goal positions (env-origin relative, plus row cos/sin
for dominos) as the policy observation; `slot_piece_pos()` is consumed by
reward/observation terms.

---

## Command internals

A line-by-line tour of [`mdp/commands.py`](../g3a_bench/tasks/cube/mdp/commands.py)
for anyone editing the task logic.

### Lifecycle hooks

The command manager drives four hooks on `PieceArrangementCommand`:

- `_resample_command(env_ids)` — env reset only (resampling is pinned off).
  Sets up a fresh episode: goal poses, slot assignment, scatter.
- `_update_metrics()` — every step. Recomputes placement and success.
- `_update_command()` — every step; no-op here (goals are fixed per episode).
- `command` (property) — the tensor handed to the policy as an observation.

Shape convention: **E** = num_envs, **G** = num_slots (goal pieces), **N** =
num_pieces (goal + distractors). Everything is batched over E, no per-env
Python loops.

### Cfg knobs (`PieceArrangementCommandCfg`)

- `entity_names` — scene entities; the first `num_goal_pieces` are goal
  pieces in slot order, the rest distractors. `cube_env_cfg.py` relies on
  this ordering (builds `piece_0…` goal-first).
- `workspace_x/y` — the table rectangle goals and scatter live in.
- `pos_threshold` (1.5 cm) — placement tolerance.
- `upright_threshold_deg` (15°) — tilt and yaw-alignment tolerance for
  dominos.
- `piece_vel_threshold` (0.05 m/s) — max speed of any goal piece to count as
  at rest.
- `gripper_open_threshold` (-0.015) — min `gripper_left` joint position to
  count as open (roughly ≥2/3 open).

### `__init__` — state + precomputation

Per-env episode state (mutated each reset/step): `slot_goal_pos` (E,G,3,
world), `piece_for_slot` (E,G, which piece fills each slot; identity unless
interchangeable), `row_yaw` (E,), cached `piece_pos_w` (E,N,3), and
`slot_at_goal`, `success`, `episode_success`. `_gripper_joint_idx` is
resolved lazily on the first `_update_metrics` call (joint data isn't
available yet in `__init__`).

Variant-only constants, computed once:
- `_slot_goal_z` — per-slot height. For a **stack** it's cumulative (`z +
  h/2`, then `z += h`) so piece *k* rests on those below; for a **domino**
  every slot sits at its own `h/2` on the ground. Derived from
  `form_height()`.
- `_slot_to_stack` — maps each slot to its stack, for broadcasting a stack's
  base xy to all its slots.
- `_groups` — interchangeability groups. Goal slots are bucketed by `(form,
  color)`; buckets with >1 member become permutable. A unique color order
  (e.g. `Stack3Ordered`) yields only size-1 buckets, so this is empty.
- `_cells` + `_cell_jitter` — a 3×6 grid of cell centers over the workspace
  plus per-axis jitter (40% of the half-cell). `assert _cells.shape[0] >= N`
  guarantees a distinct cell per piece, which is what keeps the scatter
  collision-free.

The `metrics` dict (`position_error`, `num_placed`, `all_placed`,
`episode_success`) is mjlab's logging channel, visible in training curves.

### `command` property

Subtracts `env_origins` so goals are env-local (parallel envs are
world-shifted; the policy must be origin-agnostic), flattens to (E, G·3),
and for dominos appends `(cos row_yaw, sin row_yaw)` — avoids the
wraparound discontinuity a raw angle would have.

### Helpers

- `_gather_slot(per_piece)` — generic (E,N,D) → (E,G,D) gather via
  `piece_for_slot`. Backs `slot_piece_pos()` and the domino quaternion/
  velocity gathers below.
- `slot_piece_pos()` — position of the piece assigned to each slot. Lets
  rewards/observations ask "where is slot k's piece?" without caring which
  physical piece fills it.

### `_update_metrics` — per-step success detection

1. Refresh `piece_pos_w`, gather each slot's assigned-piece position,
   compute Euclidean `error` to goal, threshold → `at_goal` (E,G).
2. **Domino extra checks.** A domino near the goal xy isn't placed unless it
   stands correctly. Two quaternion-derived tests (q is wxyz):
   - `up_cos = 1 − 2(x²+y²)` is the world-z component of the body z-axis;
     must exceed `cos(15°)` → upright, not toppled.
   - `bx = (1−2(y²+z²), 2(xy+wz))` is the body x-axis (the thin axis —
     domino half-extents `(0.0075, 0.02, 0.035)`, x smallest) projected to
     xy. Its `abs` dot with the row direction must exceed `cos(15°)` → the
     thin face points along the row so a falling domino strikes its
     neighbour. `abs` makes it mod-π.
   - All three (`at_goal & upright & aligned`) must hold.
3. **Velocity and gripper gates.** Every goal piece's linear speed must be
   below `piece_vel_threshold`, and the `gripper_left` joint position above
   `gripper_open_threshold` (open).
4. Success bookkeeping:
   - `self.success = all_placed & pieces_still & gripper_open` — the full
     success condition, recomputed each step. Termination, `success_reward`,
     and `episode_success` all read this same value.
   - `episode_success = max(episode_success, success)` — latches whether
     the episode ever succeeded, for logging.
   - `all_placed` (position only) stays separate, feeding `placed_slots`
     (the progress reward) and the `all_placed`/`num_placed` metrics.

### `_resample_command` — the reset

Runs only on the reset `env_ids` (`n` of them); clears the counters, then
three phases:

1. **Goal poses.**
   - *Stack:* bases drawn from a coarse 2×3 grid; `argsort(rand)` picks
     `_num_stacks` distinct cells (vectorized sample-without-replacement),
     jittered ±0.015. `bases[:, _slot_to_stack]` broadcasts each base to its
     slots; z comes from the cumulative heights.
   - *Domino:* yaw sampled near ±90° (`sign·π/2 ± 0.6 rad`) so the row runs
     mostly lateral (along y) and fits the narrow x range. The row's
     bounding extent is computed and the center sampled only within the
     shrunk `[lo, hi]` box so the whole row stays in bounds (`clamp_min(0)`
     guards a degenerate range). Slots step out symmetrically at
     `domino_spacing`; `row_yaw` is stored.
   - Both add `origins` to lift env-local goals into world frame.
2. **Slot assignment.** Start identity, then permute each interchangeable
   group (`argsort(rand)`) — this is why two red cubes can satisfy each
   other's slots.
3. **Scatter.** Pick N distinct grid cells per env, jitter within each,
   random yaw, then write each piece's pose to sim: xy from the cell +
   origin, z = `half_height + 2 mm + origin_z` (resting on the table),
   zeroed velocity. Distinct cells mean no spawn overlap and no settling
   explosion.

### `_debug_vis_impl`

Draws a sphere at each slot goal — green when placed, yellow otherwise.
Visible in `g3a-play`.

### Domino spacing caveat

The domino "fits in workspace" math assumes `domino_spacing · (G−1)` plus
margin is smaller than the workspace. For a longer row or larger spacing the
`clamp_min(0.0)` would silently collapse all dominos onto one center. Fine
for `Domino5` (4 × 0.05 = 0.2 m vs the 0.36 m y-range).

---

## Observations, rewards, terminations

**Observations** (state-based; identical actor and critic groups, see
[`cube_env_cfg.py`](../g3a_bench/tasks/cube/cube_env_cfg.py)): joint pos/vel
+ last action (mjlab built-ins), plus `pieces_rel_grasp` (every piece
relative to the grasp site), `slot_goal_deltas` (goal minus assigned-piece
position per slot), `slot_placed_flags`, and the raw command. The actor
group has observation noise; the critic group does not.

**Rewards** ([`mdp/rewards.py`](../g3a_bench/tasks/cube/mdp/rewards.py)):
- `placed_slots` — fraction of placed slots (progress + keep-placed),
  position only.
- `success` — bonus while position, piece velocity, and gripper conditions
  all hold (same condition as the termination).
- Penalties (mjlab built-ins): action rate, joint position limits, joint
  velocity hinge (ramped up by a curriculum).

**Termination** ([`mdp/terminations.py`](../g3a_bench/tasks/cube/mdp/terminations.py)):
`arrangement_complete` fires when every slot is placed, all goal pieces are
moving slowly, and the gripper is open. Also: `time_out`. In `play` mode the
success termination is removed so sessions run indefinitely.

---

## How it links to the rest of the repo

### Robot-agnostic by design
`make_cube_env_cfg` takes a `ManipulatorInterface`
([`robots/interface.py`](../g3a_bench/robots/interface.py)) and reads every
robot-specific value from it — grasp site, end-effector subtree, finger-pad
geom regex, action scale, viewer body. Nothing in the cube code hardcodes
the DK1. Currently only the unimanual DK1
([`robots/trlc_dk1/`](../g3a_bench/robots/trlc_dk1/)) is registered; adding
a robot is one more interface instance.

### Built on mjlab
The cube domain only defines the task; the train/play loop is mjlab's. It
uses mjlab managers (`ManagerBasedRlEnvCfg`, observation/reward/event/
command managers), mjlab's built-in MDP terms (`mdp.joint_pos_rel`,
`mdp.action_rate_l2`, …), the manipulation MDP helpers
(`joint_velocity_hinge_penalty`, `illegal_contact`, `reward_curriculum`),
and the RSL-RL PPO runner. mjlab is pinned to 1.4.0 — see the API-drift
notes in [`CLAUDE.md`](../CLAUDE.md).

### Automatic registration chain
A single `import g3a_bench` registers everything:

```
g3a_bench/__init__.py        →  import g3a_bench.tasks
tasks/__init__.py            →  import_packages(...)  (blacklists "utils", ".mdp")
tasks/cube/__init__.py       →  import_packages(...)  → imports config/
tasks/cube/config/__init__.py→  register_mjlab_task(...) per variant × robot
```

See [`g3a_bench/__init__.py`](../g3a_bench/__init__.py),
[`tasks/__init__.py`](../g3a_bench/tasks/__init__.py), and
[`config/__init__.py`](../g3a_bench/tasks/cube/config/__init__.py). The
`.mdp` blacklist keeps the MDP modules from being scanned as task configs.

### Where it sits among categories
`cube` is one of three categories alongside
[`insertion`](../g3a_bench/tasks/insertion/) (peg insertion + shape sorter)
and [`ball`](../g3a_bench/tasks/ball/) (golf / push / throw). All three
share the same layout, so cube is the reference template for the others.

---

## Running it

```bash
# Train (registration happens when g3a_bench is imported by the wrapper)
uv run g3a-train G3a-Cube-Stack3-Dk1_uni --env.scene.num-envs 1024

# Play a checkpoint
uv run g3a-play  G3a-Cube-Stack3-Dk1_uni
```

Episode length scales with the number of goal pieces (`10 s ×
num_goal_pieces`). Verification scripts that exercise registration and the
success/reward path live in [`tests/tasks/`](../tests/tasks/).
