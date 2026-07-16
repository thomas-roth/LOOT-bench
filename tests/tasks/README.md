# Registered-task verification

The scripts and pytest suite here verify changes to task code; run everything from the **repo root**.

---

## Unit tests for the pure task logic (`test_success.py`, `test_scatter.py`)

Pytest suites for the simulator-free task logic:

- `test_success.py` — the per-category success criteria in each
  `tasks/<cat>/mdp/success.py` (`domino_upright_aligned`, `domino_fallen`,
  `stack_reached`, `ball_in_target_zone`, `SorterLatch`) plus the shared
  gates: `SuccessGate` from `common/success.py` and `GripperOpenGate` from
  `common/commands.py`.
- `test_scatter.py` — the `ScatterGrid` spawn sampler in `common/scatter.py`
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

Builds every registered `G3a-*` task, resets it, and steps it a few times with zero actions.
A fast smoke test that the env, managers, and MJCF assemble without error.
Run it after touching any `*_env_cfg.py`, `assets.py`, or robot config.

By default it saves one full-HD scene picture per task (env 0) to `out/render/<domain>/<task_id>.png`.
`<domain>` is the task category parsed from the task id (`ball`, `stacking`, `domino`, `shapesorter`, …), so new categories are picked up automatically.
Rendering is decoupled from PASS/FAIL: a missing GL context (headless CI) warns per task but never fails it.

```bash
uv run python tests/tasks/build_sim_registered_tasks.py
uv run python tests/tasks/build_sim_registered_tasks.py --cpu
uv run python tests/tasks/build_sim_registered_tasks.py --no-render
```

---

## Success-metric check (`check_success_reward.py`)

Teleports objects to their goals for one representative task per category, plus structurally unusual variants (Stacking Stack3TwoStacks, Domino Domino3KnockOver, Ball MinigolfUsingWall, Ball ThrowU), and checks that the command term's success metric fires.
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
