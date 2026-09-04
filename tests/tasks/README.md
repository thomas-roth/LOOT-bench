# Registered-task verification

The scripts and pytest suite here verify changes to task code; run everything from the **repo root**.

---

## Unit tests for the pure task logic (`test_success.py`, `test_scatter.py`)

Pytest suites for the simulator-free task logic:

- `test_success.py`: the per-category success criteria in each
  `tasks/<cat>/mdp/success.py` (`domino_upright_aligned`,
  `domino_fallen_along_row`, `chain_topple_success`, `stack_reached`,
  `ball_in_target_zone`, `SorterLatch`, `place_reached`/`place_progress`)
  plus the shared gates: `SuccessGate`
  from `common/success.py` and `GripperOpenGate`/`ContactReleaseGate` from
  `common/commands.py`.
- `test_scatter.py`: the `ScatterGrid` spawn sampler in `common/scatter.py`
  (`tiling`'s cell layout and jitter bounds, `sample_poses`'s distinct-cells,
  determinism, and seed-prefix properties).

Run them after touching a category's `mdp/success.py`, `tasks/common/success.py`,
or `tasks/common/scatter.py`.

```bash
uv run pytest tests/tasks/ -v    # test all pytest suites inside
uv run pytest tests/tasks/test_success.py -v
uv run pytest tests/tasks/test_scatter.py -v
```

---

## Build, reset, step every task (`build_sim_registered_tasks.py`)

Builds every registered `G3A-*` task, resets it, and steps it a few times with zero actions.
A fast smoke test that the env, managers, and MJCF assemble without error.
Run it after touching any `*_env_cfg.py`, `assets.py`, or robot config.

By default it saves one full-HD scene picture per task (env 0) to `out/render/<domain>/<task_id>.png`.
`<domain>` is the task category parsed from the task id (`ball`, `stack`, `domino`, `sorter`, …), so new categories are picked up automatically.
Rendering is decoupled from PASS/FAIL: a missing GL context (headless CI) warns per task but never fails it.

```bash
uv run python tests/tasks/build_sim_registered_tasks.py
uv run python tests/tasks/build_sim_registered_tasks.py --cpu
uv run python tests/tasks/build_sim_registered_tasks.py --no-render
```

---

## Sim-settings comparison (`compare_sim_settings.py`)

Benchmarks the sim-setting profiles (current training settings, current play/demo
settings, MuJoCo solver defaults, and the train timestep with default solver caps)
against each other for speed and accuracy on one registered task.
Accuracy is measured as qpos deviation from a high-accuracy `reference` profile
along seeded, deterministically scripted scenarios sampled every 0.02 s of sim
time: a contact-free arm `sweep`, a teleported block tower at rest (`stack`,
reports the tower's absolute `drift`), and a `grasp` where a cube is pinned into
the closing gripper and then lifted and shaken (reports the absolute grip
`slip`; a dropped cube reads ~0.2 m). The contact scenarios need a
`G3A-Stack-*` task and are skipped elsewhere. Speed is headless `env.step()`
throughput after Warp warmup.
`--viewer` adds a native-viewer play-mode pass (1x real-time and max speed).
Needs a display, like the `tests/viewer/` checks.
Run it when reconsidering timestep / decimation / solver iterations, or on a new
machine to check that demo collection keeps up with real time under the training
physics.

```bash
uv run python tests/tasks/compare_sim_settings.py
uv run python tests/tasks/compare_sim_settings.py --cpu --sim-seconds 4
uv run python tests/tasks/compare_sim_settings.py --viewer
```

---

## Success-metric check (`check_success_reward.py`)

Teleports objects to their goals for one representative task per category, plus
structurally unusual variants (Stack Low-TwoStacks, Domino 3Pieces-KnockOver,
Ball Minigolf-UsingWall, Ball Throw-U), and checks that the command term's
success metric fires.
The Domino KnockOver check builds the row (teleport), then tips only the first
domino past its tip-over point and simulates the chain reaction, so it verifies
the row topples along the row axis, all leaning the same way, in row order,
not just that objects end up "fallen".
The shape-sorter check reorients the free cube so the target hole faces up, then drives the piece through the mouth in two stations so the latched outside-to-inside crossing fires.
This confirms the metric *can* fire when objects are placed correctly; it does not verify that reward shaping guides a policy there.
Run it after touching a category's `mdp/commands.py` or success-metric logic.

By default it saves a full-HD picture of the teleported goal state (with goal markers) to `out/render/<domain>/<task_id>_goal.png`.
The shot is taken at the scored instant, after the step loop computes the success fraction, so it shows the exact state the metric evaluated.
As in the build script, a render failure is a warning, not a check failure.

```bash
uv run python tests/tasks/check_success_reward.py
uv run python tests/tasks/check_success_reward.py --cpu
uv run python tests/tasks/check_success_reward.py --no-render
```
