# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`g3a-bench` is a suite of long-horizon robot-manipulation benchmark tasks for the
KIT "G3A Praktikum" course, built on the **mjlab** framework (MuJoCo + PyTorch
vectorized sim). It is an installable package (`g3a_bench`, hatchling,
flat layout) whose job is to **register tasks into mjlab's global registry** and
provide the robot assets and task definitions; the play loop is mjlab's, and
training is QC-FQL (offline-to-online RL), which lives in a **separate `qc`
checkout** reached by `PYTHONPATH` — this repo holds only the seam to it, in
`g3a_bench/qc/` (see `.scratch/qc-restart/`).

Five task categories, 41 tasks total, currently registered for one robot
(`-Dk1_uni`) → task ids look like `G3A-<Category>-<Task>[-<Variation>]-<Robot>`:
`tasks/stack` (stack blocks to a target height, registered as `Stack`),
`tasks/domino` (place and optionally knock over a row of dominoes),
`tasks/shape_sorter` (insert shapes into a sorter board, registered as
`Sorter`), `tasks/ball` (minigolf / throw a ball to a target), and
`tasks/place` (carry one cube to a sampled target spot, registered as
`Place`). Place is the one-cube sanity check the QC pipeline is brought up
on — a single `Easy` variant, 500 steps, no difficulty axis.
`tasks/peg_insertion` is an archived, unregistered category kept importable
for reference (see its README).

## Commands

Use `uv run` for everything — the venv is managed by uv. The environment is at .venv/.

```bash
# Install (torch build picked by platform in pyproject.toml, also installs pre-commit hooks)
bash install.sh            # or ./install.ps1 on Windows
uv sync                    # manual editable install (cu128 on Linux/Windows, cpu on macOS)

# Quality (CI = pre-commit; run before committing)
uv run ruff check          # lint
uv run ruff format         # format
uv run pyright             # type check (config in pyrightconfig.json, basic mode)
uv run pre-commit run --all-files

# Play a task (wraps mjlab's player after importing g3a_bench to register tasks)
uv run g3a-play G3A-Stack-Low-Dk1_uni

# List all registered task IDs (mjlab's built-ins included; --filter/--count narrow it)
uv run g3a-list-tasks
uv run g3a-list-tasks --filter Domino
uv run g3a-list-tasks --filter G3A --count

# Teleop recording -> the .npz QC-FQL trains on (several sessions at once)
uv run g3a-convert-demos --success-dir out/demos/success/<task>/<stamp> \
    --failure-dir out/demos/failure/<task>/<stamp> --output demos.npz

# Verification scripts (see tests/tasks/README.md), not installed as console scripts
uv run python tests/tasks/build_sim_registered_tasks.py
uv run python tests/tasks/check_success_reward.py

# Before/after pictures of the three QC-rebuild faults (see tests/qc/faults-visual/README.md)
uv run python tests/qc/faults-visual/run_all.py --cpu    # -> out/render/faults-visual/index.html

# Native-viewer checks (see tests/viewer/README.md), need a display, not for CI
uv run python tests/viewer/measure_viewer_fps.py
uv run python tests/viewer/check_viewer_shutdown.py
```

The pytest suite (`uv run pytest tests/`) unit-tests the pure kernels in
`tasks/common/` (`tests/tasks/`) and the DK1 collision contract on the
compiled model (`tests/trlc_dk1/`). See `tests/tasks/README.md` for
what each verification script covers; all accept `--cpu` to force CPU-only
execution. `tests/viewer/` holds the manual native-viewer checks (FPS cap,
shutdown segfault). `tests/qc/faults-visual/` reconstructs the three faults the QC
rebuild fixed and writes before/after pictures to `out/render/faults-visual/`; it
reads the two recorded sessions and changes nothing.

## Architecture

### Task category pattern (most work happens here)
Every category under `g3a_bench/tasks/<cat>/` follows the **same layout**, so
adding/editing a category means mirroring an existing one:

- `variants.py` — frozen dataclass `*VariantCfg` enumerating the task variations
  (forms, colors, targets…), collected in a `*_VARIANTS` dict keyed by name.
- `assets.py` — `mujoco.MjSpec` builders for the scene objects (`get_*_spec`).
  Most objects are built procedurally in code; the exception is
  shape_sorter's STL cube and pieces, loaded from committed XML/STL files
  (see the extra `xmls/` dir below).
- `mdp/` — the per-category MDP terms: `success.py` (the category's pure
  success criteria — simulator-free placement tests like `domino_upright_aligned`
  and `ball_in_target_zone`, plus category-specific latches like the sorter's
  `SorterLatch`; unit-tested in `tests/tasks/test_success.py`), `commands.py`
  (a `CommandTerm` subclass that samples goals, scatters objects on reset, and
  computes the success metric — an *adapter* that gathers sim buffers for the
  success criteria and the shared `tasks/common/success.py` / `scatter.py`
  helpers), `observations.py`, `rewards.py`, `terminations.py`.
  `mdp/__init__.py` re-exports the terms.
- `<cat>_env_cfg.py` — `make_<cat>_env_cfg(variant, play)` builds only the
  category-shaped parts (entities, command, task obs/reward/termination terms:
  `progress` + `success`, both weight 1.0, no shaping/penalties) and hands
  them to `tasks/common/env_cfg.make_manipulation_env_cfg`, which owns the
  shared skeleton (table, proprioceptive obs, actor/critic groups, reset
  events, viewer/sim defaults, the `play` strip-down).
- `config/__init__.py` — one `tasks/common/registration.register_category(...)`
  call: it loops robots × variants, wires the robot in (via
  `tasks/common/manipulator_wiring.wire_manipulator_into_env` with the
  category's `grasp_obs_terms`/`ee_contact_sensors`), and registers each task
  with mjlab's registry using the shared RSL-RL PPO cfg
  (`tasks/common/rl_cfg.manipulation_ppo_runner_cfg`, per-category
  `max_iterations`) — mjlab's registry requires an `rl_cfg`, but the actual
  training path is QC-FQL in the separate `qc` checkout, not RSL-RL PPO.
- `__init__.py` — empty (discovery happens one level up, see below).
- *(shape_sorter only)* `xmls/` — committed STL shape-sorter assets
  (`sorter.xml`, `piece_<shape>.xml`, STL meshes, `sorter_layout.json`),
  generated once by `xmls/make_sorter_assets.py`.

`tasks/common/` is the home of everything shared: the workbench builder, the
manipulator wiring, the registration seam + PPO cfg, the shared success
gating (`success.py`: `OBJECT_VEL_THRESHOLD` and the `SuccessGate` that
combines a category's success criterion with the at-rest + released
conditions), the `GripperGateCommandCfg`/`GripperOpenGate` "gripper released"
success gate (`commands.py`; the joints/threshold are robot-specific and
injected by the manipulator wiring), the `ScatterGrid` spawn sampler
(`tiling`/`sample_poses`, seed-prefix safe), the shared `colors.py` palette,
the table asset, and the debug markers.

**Registration is automatic via discovery**: `tasks/__init__.py` calls mjlab's
`import_packages` once (blacklist `["utils", ".mdp", "common", "peg_insertion"]`),
which walks the category packages, imports each `config` subpackage and
triggers `register_mjlab_task`. The `.mdp`/`common` blacklist entries keep
the MDP and shared-helper modules from being scanned as task configs;
`peg_insertion` is blacklisted because it's an archived, unregistered
category. Top-level `g3a_bench/__init__.py` imports `g3a_bench.tasks`, so a
single `import g3a_bench` registers everything.

### Robot-agnostic design
Tasks never hardcode robot details. `robots/interface.py` defines
`ManipulatorInterface` (name, entity-cfg factory, grasp site, gripper joints +
open threshold for the success gate, end-effector subtree, finger-pad geom
regex, viewer body, start pose). It carries no action scale: every task uses
`tasks/common/actions.NormalizedJointPositionAction`, where `[-1, 1]` spans
each joint's own travel and the numbers come from the compiled model. Each
robot package
exposes its own `get_<robot>_interface()` factory —
`robots/trlc_dk1/__init__.py` has `get_unimanual_dk1_interface()` and
`get_bimanual_dk1_interface()`, `robots/yam/__init__.py` has
`get_yam_interface()` (YAM comes from `mjlab.asset_zoo`). Currently only the
unimanual DK1 is registered against tasks (see `config/__init__.py` in each
task category). To support a new robot, add one `ManipulatorInterface`
instance — the env cfgs read everything they need from it.

### DK1 robot asset
The TRLC-DK1 has no public MJCF; it was converted from the official URDF
(github.com/robot-learning-co/trlc-dk1) into `robots/trlc_dk1/xml/dk1.xml`
(hand-written) plus per-color OBJ visual meshes split out of the source GLBs by
`robots/trlc_dk1/utils/convert_glb_meshes_to_obj.py` (a one-time tool, not part
of the runtime). `dk1_constants.py` exposes `get_unimanual_dk1_robot_cfg()` /
`get_bimanual_dk1_robot_cfg()` and `START_JOINT_POS`, the pose every episode
begins from and the pose the teleop operator aligns the leader arm to. It is
the home pose: q2=1.7, q3=1.7, q4=-1.5708 puts the downward gripper at
≈(0.30, 0, 0.20). The gripper is a
rack-and-pinion pair coupled by an equality constraint (only `gripper_left`
actuated).

## Conventions and gotchas

- **Torch build is platform-pinned, not extra-gated.** There are no `cpu`/`cu128`
  extras. `[tool.uv.sources]` in `pyproject.toml` pins `torch`/`torchvision` to
  the `pytorch-cu128` index on Linux/Windows and `pytorch-cpu` on macOS (no CUDA
  wheels exist there), baked into the base resolution. So a bare `uv sync` /
  `uv run` resolves the right wheel and **never churns torch back to the CPU
  wheel** — the old `--extra cu128` flag and the installers' `nvidia-smi` probing
  are gone. `torch-backend = "auto"` is *not* used: it's only honored by the
  `uv pip` interface, not `uv sync`/`uv lock` (uv issue #12994, unimplemented).
  Tradeoff: no lean CPU-only build on Linux/Windows — CPU-only boxes still run
  fine on the (larger, ~2.5 GB) cu128 wheels.
- **use the newest release of mjlab** and the API has drifted from older snippets.
  Check the installed source code in .venv. E.g. use
  `mjlab.utils.lab_api.math` (not `mjlab.utils.math`), `RslRlOnPolicyRunnerCfg`
  with separate actor/critic `RslRlModelCfg` (not `RslRlPpoRunnerCfg`), and
  `mjlab.viewer.debug_visualizer`.
- **One action unit is half a joint's travel.** Every task uses
  `tasks/common/actions.NormalizedJointPositionActionCfg`: `-1` commands a
  joint's lower limit, `+1` its upper, and the term reads those limits off the
  compiled model, so no file holds a copy. It also clamps the raw action to
  `[-1, 1]`, which binds teleop too — gravity compensation can otherwise leave
  the box near a joint limit and fly a trajectory no bounded policy can repeat.
  The teleop bridge inverts the same mapping
  (`live_bridge/mapper.action_scale_offset`), and `tests/tasks/test_actions.py`
  asserts the two agree. mjlab's plain `JointPositionActionCfg` uses a residual
  scale instead, which suits an unbounded PPO policy and makes the task
  unreachable for a tanh-bounded one.
- **Fixed props (boards, fields) are fixed-base entities** — a `MjSpec` body with
  *no* free joint, which mjlab auto-wraps as a mocap body. Position them with
  `EntityCfg(init_state=EntityCfg.InitialStateCfg(pos=...))`, **not** the spec
  body `pos` (mjlab overwrites a fixed-base root body's pose from `init_state`).
  Add a `reset_root_state_uniform` event with `asset_cfg=SceneEntityCfg(name)` so
  the prop is pinned to `env_origins + init_state.pos` each reset. Free bodies
  (pieces, pegs, ball) are placed each reset by the command's `_resample_command`.
- **Procedural geometry is the default**: holes/bores are built as rings of box
  "walls" (MuJoCo has no boolean subtraction). Non-primitive shapes
  (triangle/hex pegs) are convex-hull meshes added inline via
  `spec.add_mesh().uservert`. A *solid* box has no cavity, so a hole must be a
  frame/ring around a gap. The exception is shape_sorter's STL board and
  pieces: committed XML+STL assets loaded via `MjSpec.from_file`
  (`shape_sorter/xmls/`), regenerated by `xmls/make_sorter_assets.py` — don't
  edit the generated files by hand.
- **Keep geometry and the success metric from drifting** by deriving both from one
  source (e.g. `shape_sorter/assets.load_sorter_layout()` is the single source
  for both the cube's hole geometry and the command term's success metric).
- mujoco-heavy modules need `# pyright: reportAttributeAccessIssue=false` at the
  top (mujoco ships no stubs for the compiled `MjSpec`).
- ruff line-length is 88; format with double quotes. Lint+format+pyright run as
  pre-commit hooks on commit.
- Test renders are written to `out/render/<domain>/`; training logs to `logs/` and
  `wandb/`. The `examples/` dir holds reference mjlab code, not part of the package.

## Agent skills

### Issue tracker

Issues and PRDs live as local markdown under `.scratch/<feature-slug>/`; no external tracker. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root created lazily by `/domain-modeling`. See `docs/agents/domain.md`.
