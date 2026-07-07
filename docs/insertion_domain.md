# The Insertion Domain

`insertion` covers tight-tolerance object insertion: pick pegs out of a parts
tray and insert them into matching bores, slide a sheet to line up a
through-hole, or push a shape through the right face of a free-standing shape
sorter. Three kinds of task share one command/observation/reward stack:

- **`peg`** — a single peg into a single tight-fitting bore on a fixed board;
  the bore is either vertical (mouth up) or horizontal (mouth facing the
  robot, on a cantilevered tube).
- **`two_sheet`** — the hole only lines up once a movable top sheet has been
  slid over the fixed bottom sheet; then the peg goes through both.
- **`sorter`** — a free-standing hollow cube (one shaped hole per face) plus
  one piece per socket. The cube spawns at a random flat orientation, so the
  robot may first have to reorient it; a single commanded piece must be
  pushed through its matching hole, the other pieces are distractors.

Source lives in [`g3a_bench/tasks/insertion/`](../g3a_bench/tasks/insertion/),
following the standard task-category layout from [`CLAUDE.md`](../CLAUDE.md).

---

## At a glance

| Concept | What it means |
| ------- | ------------- |
| **Peg / piece** | A free-body scene entity (`peg_0`, `peg_1`, …). Peg/two-sheet: a convex prism (`circle`, `square`, `rect`, `triangle`, `hexagon`). Sorter: an STL piece (adds `star`). |
| **Socket** | One bore/hole plus the peg colour assigned to it. Peg `i` must go into socket `i` — matching is enforced by index, not by physical exclusion. |
| **Bore** | The cavity a peg slides into. Built procedurally as a ring of wall boxes (MuJoCo has no boolean subtraction); the sorter's holes come from STL assets instead. |
| **Mouth / axis** | Each socket's opening position and unit outward axis. The command decomposes every peg pose into *axial* (along the axis, + is outside) and *lateral* (off-centreline) components against them. |
| **Variant** | A frozen config fixing the kind, socket set and insertion tolerances (e.g. "cylinder peg, vertical bore"). |
| **Distractor** | In the sorter, the non-commanded pieces act as distractors. Pegs map 1:1 onto sockets — there are no extra clutter pegs. |

Task ids look like `G3a-Insertion-<Variant>-<Robot>`, e.g.
`G3a-Insertion-PegCylinder-Dk1_uni`.

---

## File map

| File | Responsibility |
| ---- | -------------- |
| [`variants.py`](../g3a_bench/tasks/insertion/variants.py) | `Shape`/`Orientation`/`Kind`, `ShapeGeom` + `SHAPE_GEOMS`, `bore_apothem`, `SocketCfg`, `InsertionVariantCfg`, and the `INSERTION_VARIANTS` dict. |
| [`assets.py`](../g3a_bench/tasks/insertion/assets.py) | `socket_layout` (single source for wall geometry *and* goal frames), the procedural `get_*_spec` builders, the STL sorter loaders, tray/board placement constants, the `PALETTE`. |
| [`insertion_env_cfg.py`](../g3a_bench/tasks/insertion/insertion_env_cfg.py) | `make_insertion_env_cfg(variant, play)` — builds the insertion-shaped parts and hands them to the shared workbench builder ([`tasks/common/env_cfg.py`](../g3a_bench/tasks/common/env_cfg.py)). Also `REWARD_MODE_TERMS`. |
| [`mdp/commands.py`](../g3a_bench/tasks/insertion/mdp/commands.py) | `InsertionCommand(Cfg)` — bore frames, peg scatter, sheet/cube placement, target selection, and the insertion test. |
| [`mdp/observations.py`](../g3a_bench/tasks/insertion/mdp/observations.py) | `pegs_rel_grasp`, `socket_deltas`, `socket_inserted_flags`, `sheet_rel_grasp`. |
| [`mdp/rewards.py`](../g3a_bench/tasks/insertion/mdp/rewards.py) | `inserted_count_reward`, `success_reward`. |
| [`mdp/terminations.py`](../g3a_bench/tasks/insertion/mdp/terminations.py) | `insertion_complete`. |
| [`mdp/__init__.py`](../g3a_bench/tasks/insertion/mdp/__init__.py) | Re-exports the MDP terms. |
| [`config/__init__.py`](../g3a_bench/tasks/insertion/config/__init__.py) | One `register_category(...)` call ([`tasks/common/registration.py`](../g3a_bench/tasks/common/registration.py)): registers every variant × robot with the shared PPO cfg and records the reward modes. |
| [`xmls/`](../g3a_bench/tasks/insertion/xmls/) | Committed STL sorter assets: `sorter.xml`, `piece_<shape>.xml`, and `sorter_layout.json` — all generated once by `make_sorter_assets.py`. |
| [`decor/`](../g3a_bench/tasks/insertion/decor/) | `workstation_decor.xml` + STL meshes: visual-only workstation dressing shared by all variants, plus the parts tray ("Werkbox") whose walls do collide. |
| [`README.md`](../g3a_bench/tasks/insertion/README.md) | Short in-tree summary + variant table. |

---

## How a variant is defined

A variant is a frozen `InsertionVariantCfg`
([`variants.py`](../g3a_bench/tasks/insertion/variants.py)) fixing the `kind`,
the `sockets` tuple (one `SocketCfg` — shape, colour, orientation — per peg),
and the insertion tolerances: `insert_depth` (2.6 cm), `lateral_tol` (1 cm),
`tilt_tol_deg` (20°), `bore_depth` (4.5 cm). Derived properties give the peg
shape/colour lists in socket order.

For `peg`/`two_sheet`, `SHAPE_GEOMS` is the single source of cross-section
geometry: each shape is described by `nsides`/`circumradius` (or `rect_half`),
and the matching bore's apothem is `peg_inradius + CLEARANCE` (1.5 mm) — a
tight fit. Note the caveat in `variants.py`: shape matching for *scoring* is
by index (peg `i` vs. socket `i`), and the sizes do **not** guarantee physical
exclusion for every pair (the hexagon peg fits through the circle bore).

For `sorter`, geometry comes from the committed STL assets and
`xmls/sorter_layout.json` instead; `SHAPE_GEOMS` and the depth/tolerance
fields are ignored, and `star` exists only there.

The seven registered variants: `PegCylinder`, `PegCuboid`, `PegHex`,
`PegCylinderH` (horizontal bore), `TwoSheet`, `Sorter3`, `Sorter6`. Full
table in [`README.md`](../g3a_bench/tasks/insertion/README.md).

---

## Geometry: one source for walls and goals

`socket_layout(variant)` ([`assets.py`](../g3a_bench/tasks/insertion/assets.py))
returns a `SocketFrame` (mouth position + outward axis, board/cube-local) per
socket. `get_board_spec` builds the bore *walls* from those frames, and
`InsertionCommand` computes goal poses and the insertion test from the same
frames — so the collision geometry and the success metric can never drift
apart.

- **Bores are rings of boxes.** `_wall_boxes` surrounds the gap with `nsides`
  wall boxes (16 for `circle`, 4 for `rect`) at the bore apothem; a solid box
  has no cavity, so a hole must be a frame around a gap.
- **Boards are fixed props.** The `peg`/`two_sheet` board is a body with no
  free joint (mjlab wraps it as a mocap body), positioned via
  `EntityCfg.init_state.pos = BOARD_POS` (0.34, 0, 0) and pinned there each
  reset — never via the spec body `pos`.
- **Horizontal bore** (`PegCylinderH`): the tube cantilevers off a backing
  slab, mouth at `(-H_MOUTH_X, 0, H_BORE_CENTER_Z)` facing the robot along
  −x.
- **Two-sheet**: the fixed bottom sheet and the movable top sheet are both
  thin wall rings with the same hole. `SHEET_TOP_ROOT_Z`/`SHEET_MOUTH_Z`
  derive the resting height and through-hole mouth once, shared by
  `socket_layout` and the reset placement.
- **Sorter**: `sorter.xml` is a free-body cube whose holed walls and the star
  piece are decomposed into exact convex parts (MuJoCo collides meshes as
  their convex hull). The generation pipeline also emits
  `sorter_layout.json` — hole mouths/axes, wall thickness, bounding radii,
  piece dimensions — the single source for both collision geometry and the
  success metric.
- **Decor**: visual-only workstation dressing from
  `decor/workstation_decor.xml`, identical in every variant. Its `werkbox`
  child body (the parts tray) is the one part with real collision geometry;
  pegs spawn inside it every reset.

---

## The command: where the task logic lives

`InsertionCommand` ([`mdp/commands.py`](../g3a_bench/tasks/insertion/mdp/commands.py))
is a mjlab `CommandTerm` driving all three kinds. Resampling is locked to the
episode (`resampling_time_range = (1e9, 1e9)`), so it only fires on env reset.

On reset (`_resample_command`):

1. **Sorter only:** pick a commanded target piece uniformly, and drop the
   cube near the board spot with a random yaw composed onto one of its six
   flat resting orientations (any face down — including the commanded hole
   facing the table).
2. **Scatter.** All pegs/pieces spawn upright at random yaw on distinct cells
   inside the parts tray (a `ScatterGrid`;
   [`tasks/common/scatter.py`](../g3a_bench/tasks/common/scatter.py)).
3. **Two-sheet only:** the top sheet is placed with its hole offset 3–5 cm
   from the bottom hole at a random angle.

Each step (`_update_metrics`) decomposes every assigned peg's pose into
axial/lateral/tilt components against its socket's mouth and axis, and runs
the insertion test. For the *fixed* kinds that test is instantaneous; for the
sorter it is a **latched mouth-plane crossing** (details below). Success
additionally requires the scored pegs at rest and the gripper open — the
termination, the `success` reward, and the `episode_success` metric all read
the same [`SuccessGate`](../g3a_bench/tasks/common/success.py).

The command exposes the bore mouths (env-origin relative) and axes as the
policy observation; sorter variants append a one-hot selector of the
commanded target. `assigned_peg_pos()` / `scored_fraction()` are consumed by
observation/reward terms.

---

## Command internals

A tour of [`mdp/commands.py`](../g3a_bench/tasks/insertion/mdp/commands.py)
for anyone editing the task logic.

### Lifecycle hooks

- `_resample_command(env_ids)` — env reset only (resampling is pinned off).
  Places the cube/sheet, scatters pegs, picks the sorter target.
- `_update_metrics()` — every step. Refreshes bore frames (sorter) and
  recomputes insertion and success.
- `_update_command()` — every step; no-op here.
- `command` (property) — the tensor handed to the policy as an observation.

Shape convention: **E** = num_envs, **S** = num_sockets, **N** = num_pegs
(equal to S — pegs map 1:1 onto sockets). Everything is batched over E, no
per-env Python loops.

### Cfg knobs (`InsertionCommandCfg`)

- `board_name` / `peg_names` / `sheet_name` — scene entities; peg `i` is
  assigned to socket `i`, in variant socket order. `insertion_env_cfg.py`
  relies on this ordering (builds `peg_0…` in socket order).
- `variant` — the `InsertionVariantCfg`; supplies tolerances and the socket
  layout.
- `sheet_align_tol` (1 cm) — max top-sheet/bottom-hole xy offset to count as
  aligned (two-sheet only).
- `peg_vel_threshold` (0.05 m/s) — the shared at-rest threshold from
  [`tasks/common/success.py`](../g3a_bench/tasks/common/success.py).
- `gripper_joints` / `gripper_open_threshold` (inherited from
  [`GripperGateCommandCfg`](../g3a_bench/tasks/common/commands.py)) — the
  robot-specific "released" gate, injected by the manipulator wiring (DK1:
  `gripper_left` above ≈2/3 of its stroke).

### `__init__` — state + precomputation

From `socket_layout(variant)`: `_mouth_local` / `_axis_local` (S, 3), plus —
for the fixed kinds — per-socket shapes/colours/bases for the ghost-peg debug
vis (the sorter uses sphere-only mouth markers instead). The world-frame
buffers `socket_mouth_w` / `socket_axis_w` (E, S, 3) are then either

- **fixed kinds**: computed once — `env_origins + BOARD_POS + mouth_local`,
  identity orientation (the board is a fixed prop), or
- **sorter**: zero-initialised and re-derived from the cube's live root pose
  every step.

Success machinery: a `SuccessGate` (at-rest + released + episode latch) and a
`GripperOpenGate` (resolves the gripper joint indices lazily on first call —
joint data isn't available yet in `__init__`).

Sorter extras, computed once from `load_sorter_layout()`:

- `SorterLatch` parameters per socket — the lateral gate is the hole's
  bounding radius + 5 mm; the crossing plane sits at
  `s_cross = −(wall_thickness + piece_half)`, i.e. once the piece *centre*
  passes it, its tail has cleared the inner wall plane and the piece is fully
  inside the cube; tilt tolerance 30°.
- `target_idx` (E,) — the commanded piece per env.
- `_flat_quats` — the six flat resting orientations (each face down),
  composed with a random yaw on reset.

Spawn sampler: pegs use a jittered 2×4 `ScatterGrid` over the tray spawn
region; the bigger sorter pieces use fixed 60 mm-pitch cells
(`SORTER_PIECE_CELLS`, zero jitter) whose spacing rules out overlap at any
yaw. `assert num_cells >= N` keeps the scatter collision-free.

The `metrics` dict (`lateral_error`, `num_inserted`, `all_inserted`,
`episode_success`, plus `target_inserted` / `sheet_align_error` where
applicable) is mjlab's logging channel, visible in training curves.

### `command` property

Bore mouths with `env_origins` subtracted (parallel envs are world-shifted;
the policy must be origin-agnostic), flattened to (E, S·3), concatenated with
the flattened axes (E, S·3). Sorter appends the (E, S) one-hot target
selector — the cube pose itself is observable through the live mouths/axes.

### `_update_metrics` — per-step insertion + success

1. **Sorter only:** re-derive `socket_mouth_w` / `socket_axis_w` by rotating
   the cube-local frames with the cube's live root pose. Everything
   downstream (the `socket_deltas` obs, the command tensor, debug vis) reads
   these buffers, so they track a moving cube for free.
2. Decompose each assigned peg against its socket:
   `peg_axial_lateral` → signed axial distance (+ is outside the bore) and
   lateral off-centreline distance; `peg_axis_cos_tilt` → |cos| between the
   peg's body z-axis and the bore axis (sign-free, so an upside-down peg is
   equally aligned). All three are pure success criteria from
   [`mdp/success.py`](../g3a_bench/tasks/insertion/mdp/success.py).
3. **Insertion test.**
   - *Fixed kinds:* `peg_inserted` — the peg tip (centre depth + projected
     half length) at least `insert_depth` below the mouth, laterally within
     `lateral_tol`, tilt within `tilt_tol_deg`. Instantaneous: pull the peg
     out and the flag drops.
   - *Two-sheet extra:* `inserted` is additionally gated on the top sheet's
     hole sitting within `sheet_align_tol` of the bottom hole.
   - *Sorter:* `SorterLatch.update` — a piece scores the moment it moves,
     inside the insertion channel and roughly axis-aligned, from outside the
     crossing plane to at-or-past it **through its own hole**; the latch then
     survives the piece tumbling around inside the hollow cube. The
     previous-step state is initialised "far outside", so the first step
     after reset can never latch spuriously.
4. **Success gating.** Success criterion: all sockets filled (fixed kinds) or the
   single commanded target piece latched (sorter — the other pieces are
   clutter and may keep moving). `SuccessGate.update(task_done, speed,
   released)` adds the at-rest check (max speed over the scored pegs) and the
   open-gripper check, and latches `episode_success` for logging.

### `_resample_command` — the reset

Runs only on the reset `env_ids`; resets the gate and latch, then places
everything as described above (cube pose → pegs scatter → sheet offset).
All free-body writes go through `write_root_link_pose_to_sim` /
`write_root_link_velocity_to_sim` with zeroed velocity; spawn z is
`half_len + 2 mm` above the table so nothing starts intersecting.

### `_debug_vis_impl`

Fixed kinds: a ghost peg per socket at the minimum successful insertion (tip
at `insert_depth` below the mouth, extending one peg length out) — a box for
square/rect, a circumradius cylinder for the rest. Sorter: a sphere per mouth
(the STL shapes have no matching primitive and the mouths move with the
cube): green when inserted, red for the commanded hole, yellow otherwise.
Visible in `g3a-play`.

### Caveats

- **Index matching, not physical exclusion.** Peg `i` is only ever scored
  against socket `i`; nothing physically prevents e.g. the hexagon peg
  entering the circle bore. Making that impossible for every shape pair
  would need a coordinated resize of all shapes plus the board layout.
- **Tray constants must match the decor mesh.** `TRAY_POS` /
  `TRAY_SPAWN_HALF` are inset from the tray walls so upright pegs clear
  them; they must stay in sync with the tray footprint in
  `decor/make_decor_meshes.py` and `workstation_decor.xml`.

---

## Observations, rewards, terminations

**Observations** (state-based; identical actor and critic groups, see
[`insertion_env_cfg.py`](../g3a_bench/tasks/insertion/insertion_env_cfg.py)):
joint pos/vel + last action (mjlab built-ins), plus `pegs_rel_grasp` (every
peg relative to the grasp site), `socket_deltas` (mouth minus assigned-peg
position per socket), `socket_inserted` flags, and the raw command.
Two-sheet variants append `sheet_rel_grasp` (sheet relative to grasp and to
the bore) *after* the built-in terms, keeping the observation layout of the
other variants backward compatible. For the sorter, mouths and axes track
the cube's live pose, so the policy sees where the holes currently are while
it reorients the cube. The actor group has observation noise; the critic
group does not.

**Rewards** ([`mdp/rewards.py`](../g3a_bench/tasks/insertion/mdp/rewards.py))
— sparse by design, to pair with a demonstration pipeline (GELLO teleop →
offline-to-online RL) that supplies the exploration dense shaping would
otherwise provide:

- `inserted_count` — scored fraction: sockets filled (fixed kinds), or 1.0
  once the commanded target piece is latched (sorter).
- `success` — bonus while the full success condition holds (same condition
  as the termination).
- Penalties (mjlab built-ins): action rate, joint position limits, joint
  velocity hinge (ramped up by a curriculum).

`REWARD_MODE_TERMS` records two tiers for `g3a-train --reward-mode`:
`progress` (both task terms) and `success` (sparse outcome only).

**Terminations** ([`mdp/terminations.py`](../g3a_bench/tasks/insertion/mdp/terminations.py)):
`insertion_complete` fires on the gated success condition;
`ee_werkbox_collision` ends the episode if the end-effector hits the parts
tray with more than 10 N (a `ContactSensorCfg` whose robot-side pattern is
filled in by the wiring). Also: `time_out`. In `play` mode the success
termination is removed so sessions run indefinitely.

**Sim budget** (`make_insertion_env_cfg`): episode length is
`10 s + 8 s × num_sockets`, except the sorter's flat 30 s (one commanded
insert, but the cube may need reorienting first). Solver iterations and the
`nconmax`/`njmax` headroom are raised per kind — two stacked 16-box rings
(two-sheet) and ~90 convex mesh parts (sorter) need far more contacts than a
single bore.

---

## How it links to the rest of the repo

### Robot-agnostic by design
`make_insertion_env_cfg` leaves robot placeholders that
`wire_manipulator_into_env` ([`tasks/common/manipulator_wiring.py`](../g3a_bench/tasks/common/manipulator_wiring.py))
fills from a `ManipulatorInterface`
([`robots/interface.py`](../g3a_bench/robots/interface.py)) — grasp site,
end-effector subtree, finger-pad geom regex, action scale, viewer body, and
the robot-side pattern of the `ee_werkbox_collision` sensor. Nothing in the
insertion code hardcodes the DK1. Currently only the unimanual DK1
([`robots/trlc_dk1/`](../g3a_bench/robots/trlc_dk1/)) is registered.

### Built on mjlab + tasks/common
The insertion category only defines the task; the train/play loop is mjlab's,
and the shared skeleton (workbench, wiring, registration, PPO cfg, success
gating, scatter) lives in [`tasks/common/`](../g3a_bench/tasks/common/).
mjlab is pinned to 1.4.0 — see the API-drift notes in
[`CLAUDE.md`](../CLAUDE.md).

### Automatic registration chain
A single `import g3a_bench` registers everything:

```
g3a_bench/__init__.py             →  import g3a_bench.tasks
tasks/__init__.py                 →  import_packages(...)  (blacklists "utils", ".mdp")
tasks/insertion/__init__.py       →  import_packages(...)  → imports config/
tasks/insertion/config/__init__.py→  register_category(...) per variant × robot
```

### Where it sits among categories
`insertion` is one of three categories alongside
[`cube`](../g3a_bench/tasks/cube/) (piece stacking / dominoes — see
[`cube_domain.md`](cube_domain.md)) and
[`ball`](../g3a_bench/tasks/ball/) (golf / push / throw). All three share the
same layout.

---

## Running it

```bash
# Train (registration happens when g3a_bench is imported by the wrapper)
uv run g3a-train G3a-Insertion-PegCylinder-Dk1_uni --env.scene.num-envs 1024

# Play a checkpoint
uv run g3a-play  G3a-Insertion-PegCylinder-Dk1_uni
```

The pure insertion criteria and the latch are unit-tested in
[`tests/tasks/test_success.py`](../tests/tasks/); the verification scripts
there build every registered task and teleport-check the success metric
(including `TwoSheet`, `PegCylinderH`, `Sorter3`, `Sorter6` — the sorter
check drives the target piece through the mouth in two stations so the
latched crossing fires).
