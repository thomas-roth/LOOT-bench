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
`g3a_bench/qc/`.

Five task categories, 44 tasks total, currently registered for one robot
(`-Dk1_uni`) → task ids look like `G3A-<Category>-<Task>[-<Variation>]-<Robot>`:
`tasks/stack` (stack blocks to a target height, registered as `Stack`),
`tasks/domino` (place and optionally knock over a row of dominoes),
`tasks/shape_sorter` (insert shapes into a sorter board, registered as
`Sorter`), `tasks/ball` (minigolf / throw a ball to a target), and
`tasks/place` (carry one cube to a sampled target spot, registered as
`Place`). Place is the one-cube sanity check the QC pipeline is brought up
on — 500 steps, no difficulty axis; its four variants instead switch the two
randomness axes independently (`RandS`/`FixedS` for the cube spawn,
`RandT`/`FixedT` for the target), e.g. `G3A-Place-FixedS-RandT-Dk1_uni`.
`tasks/peg_insertion` is an archived, unregistered category kept importable
for reference (see its README).

**Read `docs/adr/` before changing anything it touches.** Those records are the
ground truth for *why* the repo is shaped this way; `CONTEXT.md` is the glossary
that names the concepts. If your work contradicts a record, say so and reopen
the decision — do not overturn it in silence.

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
uv run g3a-convert-demos --success-dir dataset/demos/<task>/<stamp> \
    --failure-dir dataset/demos/<task>/<stamp>_failure --output out/qc/demos.npz

# Verification scripts (see tests/tasks/README.md), not installed as console scripts
uv run python tests/tasks/build_sim_registered_tasks.py
uv run python tests/tasks/check_success_reward.py
uv run python tests/tasks/compare_sim_settings.py

# Native-viewer checks (see tests/viewer/README.md), need a display, not for CI
uv run python tests/viewer/measure_viewer_fps.py
uv run python tests/viewer/check_viewer_shutdown.py

# Read a demo set before spending a training run on it (ADR-012, ~2 minutes).
# Runs from the qc venv (it needs JAX), with both checkouts on the path.
PYTHONPATH=<g3a-bench>:<qc> <qc>/.venv/bin/python tests/qc/bc_gate.py out/qc/demos.npz

# Launch a QC-FQL run in the separate qc checkout (must run inside tmux)
./train_qc.sh G3A-Place-FixedS-FixedT-Dk1_uni out/qc/demos.npz --offline_steps 100000
run_scripts/G3A-Place/fixed-variants/fixed-s_fixed-t_100k-0.sh   # a 3-seed sweep
```

The pytest suite (`uv run pytest tests/`) unit-tests the pure kernels in
`tasks/common/` (`tests/tasks/`), the QC seam without a simulator
(`tests/qc/`: the converter, the transition assembly, the gym adapter, the
BC gate), the teleop bridge (`tests/live_bridge/`, no hardware needed), and
the DK1 collision contract on the compiled model (`tests/trlc_dk1/`). See
`tests/tasks/README.md` and `tests/live_bridge/README.md` for what each
script covers; all verification scripts accept `--cpu` to force CPU-only
execution. `tests/viewer/` holds the manual native-viewer checks (FPS cap,
shutdown segfault).

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
  category-shaped parts (entities, command, task observation/termination
  terms, and the **one** task reward — no shaping terms and no penalties, see
  ADR-006) and hands them to
  `tasks/common/env_cfg.make_manipulation_env_cfg`, which owns the shared
  skeleton (table, proprioceptive obs, actor/critic groups, reset events,
  `start_pose_noise`, viewer/sim defaults, the `play` strip-down).
- `config/__init__.py` — one `tasks/common/registration.register_category(...)`
  call: it loops robots × variants, wires the robot in (via
  `tasks/common/manipulator_wiring.wire_manipulator_into_env` with the
  category's `ee_contact_sensors` and its `grip_contact_sensor` /
  `object_contact_sensor`), and registers each task with mjlab's registry
  using the shared RSL-RL PPO cfg
  (`tasks/common/rl_cfg.manipulation_ppo_runner_cfg`, which takes an
  `experiment_name` and nothing else) — mjlab's registry requires an
  `rl_cfg`, but the actual training path is QC-FQL in the separate `qc`
  checkout, not RSL-RL PPO.
- `__init__.py` — empty (discovery happens one level up, see below).
- *(shape_sorter only)* `xmls/` — committed STL shape-sorter assets
  (`sorter.xml`, `piece_<shape>.xml`, STL meshes, `sorter_layout.json`),
  generated once by `xmls/make_sorter_assets.py`.

`tasks/common/` is the home of everything shared: the workbench builder
(`env_cfg.py`), the manipulator wiring, the registration seam + PPO cfg, the
normalized action term (`actions.py`), the one-reward convention
(`rewards.py`: `MIN_SHORTFALL` and `shaped_task_reward`), the shared
observation terms (`observations.py`: `ee_pose_w` and `rot6d_from_matrix`),
the absolute-pose reset (`events.py`: `reset_joints_to_pose`), the shared
success gating (`success.py`: `OBJECT_VEL_THRESHOLD` and the `SuccessGate`
that combines a category's success criterion with the at-rest + released
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

### The QC training seam (`g3a_bench/qc/`)
Training is QC-FQL and it runs from a **separate checkout of upstream `qc`**
(`ColinQiyangLi/qc`). This repo keeps only the seam (ADR-002). The in-repo
copy of the trainer is deleted; do not bring one back.

- `env.py` — `G3aEnv`, a single gymnasium env over one mjlab task. It is
  a seam and nothing else: no reward shaping, no observation rewriting, no
  action remapping (ADR-009). Four choices there are load-bearing: auto-reset
  is **off** (mjlab would return the *next* episode's first observation next to
  `done`), observation corruption is off, `terminated`/`truncated` pass through
  as mjlab reports them, and observations are **copied** out of simulator
  memory (mjlab returns a view the next step overwrites).
- `episode.py` — per-episode bookkeeping and the `info` dict `qc` reads.
  Success is **latched** over the episode, keyed off `reward == 0.0`. Contact
  telemetry is published under **flat** keys (`pad_contact_force`,
  `object_contact_force`); `qc/evaluation.py` consumes them by name, so the key
  names are a cross-repo contract with no version pin. After changing either
  side, run a short eval and check `eval/pad_contact_runs` is non-zero.
- `transitions.py` — recorded rows → `(s, a, r, s', done)`. `terminals` marks an
  **episode boundary**, `masks` marks whether to **bootstrap**; they are not the
  same column (ADR-010). Every episode end is a boundary, only a success is
  absorbing. A timed-out episode's final row is **dropped**, because the
  recorder stored no successor for it.
- `convert.py` — the `g3a-convert-demos` entry point. It repeats the record-time
  action conversion exactly (same function, same gravity compensation), sorts
  rows by `(episode, frame)`, and stamps the action `scale`/`offset`, the
  `start_pose_noise` the episodes were recorded under, and the task id into the
  `.npz` (ADR-011). It is the only module here that imports lerobot.
- `dataset.py` — `load_dataset(path, env)` reads a dataset back and **raises** on
  an action-convention, start-pose-noise, or task mismatch against the
  environment it is handed (ADR-011); an un-stamped `task` on an old file
  warns instead. It imports no lerobot on purpose: the `qc` venv loads
  datasets and must stay free of the recording stack.

`train_qc.sh` launches a run. It checks the things that are invisible in the
code and cost a whole run when wrong (ADR-004): `g3a-bench` reached by
`PYTHONPATH` and never pip-installed in the `qc` venv, a CUDA-12 torch,
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, a run directory off NFS, a wandb login,
and a tmux session. `run_scripts/<task>/` holds the per-experiment sweep
launchers that call it.

Two documents govern what to run: **ADR-012** says to gate a new demonstration
set with `tests/qc/bc_gate.py` before spending a training run, and **ADR-014**
lists the closed experiments that must not be run again (the discount, a dense
reward arm, `horizon_length=1`, a `flow_steps` sweep, more eval-reset coverage).
Read both before proposing an experiment.

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
- **mjlab is pinned to `>=1.5.1,<1.6`.** mjlab 1.6.0 refuses the DK1 collision
  specification, so no task builds on it (ADR-003). The upper bound is a hard
  constraint, not a preference: remove it only together with a DK1 asset that
  compiles on 1.6. Inside that range use the newest release, and read the
  installed source in `.venv` — the API has drifted from older snippets. E.g.
  use `mjlab.utils.lab_api.math` (not `mjlab.utils.math`),
  `RslRlOnPolicyRunnerCfg` with separate actor/critic `RslRlModelCfg` (not
  `RslRlPpoRunnerCfg`), and `mjlab.viewer.debug_visualizer`.
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
- **One task reward, in `[-1, 0]`, exactly `0` only at success** (ADR-006,
  `tasks/common/rewards.py`). Every task ends on success, so a positive
  per-step reward would pay a policy to hover at the goal and never touch it.
  A non-success step is floored at `-MIN_SHORTFALL` (1e-3), because QC's
  `--sparse` maps `(rewards != 0.0) * -1.0` and would relabel a true zero a
  success. The reward **lags the observation by one control step** on purpose:
  mjlab runs the reward and termination managers before the command manager,
  so success in step `k` is paid at step `k + 1`, the step that also
  terminates. The lag is the same when recording and when evaluating, so it
  cancels — never correct it on one side alone. Rewards are not scaled by the
  timestep (`scale_rewards_by_dt=False`), or the numbers in a recorded dataset
  would follow the decimation.
- **Observations carry no noise, on any term or group** (ADR-015). Observation
  noise is a robustness device for an on-policy run that collects its own data.
  Training is offline from recorded demonstrations, and the demonstrations came
  through the uncorrupted observation, so noise would split the offline and
  online halves apart. `actor` and `critic` are now the same terms. If
  sim-to-real robustness is ever wanted, put the noise back in **one** place
  (`make_manipulation_env_cfg`) and **record the demonstrations through it**.
- **`ee_pose` publishes a 6D rotation, not a quaternion** (ADR-007). The term is
  9 wide: position, then the first two columns of the rotation matrix
  (`tasks/common/observations.rot6d_from_matrix`). mjlab's `quat_from_matrix`
  picks its branch with an `argmax` over candidates that tie, and the DK1 grasp
  site sits on that tie, so float noise flipped the sign of one smooth motion on
  14 of 120 steps. Body poses keep their quaternion — `data.xquat` is integrated
  continuously and never runs the unstable branch.
- **Episodes start at the robot's home pose** (ADR-008).
  `tasks/common/events.reset_joints_to_pose` resets to an **absolute** pose
  (`ManipulatorInterface.reset_joint_pos`, the DK1's `START_JOINT_POS`) plus a
  uniform sample of half-width `start_pose_noise`, default `0.0`. mjlab's
  `reset_joints_by_offset` resets relative to the MJCF keyframe, 1.7 rad away
  on joints 2 and 3 — a policy trained on demos and evaluated from a state it
  had never seen. Sampled positions clamp to the **hard** joint limits, and the
  gripper is never sampled. `place` keeps the seam at
  `place_env_cfg.START_POSE_NOISE`; keep any value well under the recorder's
  0.2 rad alignment gate. Measure start-state variety with peak-to-peak range,
  never standard deviation.
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
- **`out/` is scratch, `dataset/` is kept.** `out/` is gitignored: the teleop
  recorder writes its sessions to `out/demos/success|failure/<task>/<stamp>/`, and
  `g3a-convert-demos` writes its `.npz` to `out/qc/`. A demo session that is worth
  keeping is moved into `dataset/demos/<task>/<stamp>/`, which is committed, with
  the failure half of the same session as the sibling
  `dataset/demos/<task>/<stamp>_failure/` (the `success/`+`failure/` split becomes a
  suffix there). Read demos from `dataset/demos/`; record new ones into `out/demos/`.

## Agent skills

### Issue tracker

Issues and PRDs live as local markdown under `.scratch/<feature-slug>/`; no external tracker. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root created lazily by `/domain-modeling`. See `docs/agents/domain.md`.
