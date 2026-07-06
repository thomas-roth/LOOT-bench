# Registered-task verification

Two scripts verify changes to task code; run both from the **repo root**.
Neither is a console script — invoke with `uv run python`.
Both accept `--cpu` to force CPU-only execution and `--no-render` to skip image output.

---

## Build, reset, step every task (`build_sim_registered_tasks.py`)

Builds every registered `G3a-*` task, resets it, and steps it a few times with zero actions.
A fast smoke test that the env, managers, and MJCF assemble without error.
Run it after touching any `*_env_cfg.py`, `assets.py`, or robot config.

By default it saves one full-HD scene picture per task (env 0) to `out/render/<domain>/<task_id>.png`.
`<domain>` is the task category parsed from the task id (`ball`, `cube`, `insertion`, …), so new categories are picked up automatically.
Rendering is decoupled from PASS/FAIL: a missing GL context (headless CI) warns per task but never fails it.

```bash
uv run python tests/tasks/build_sim_registered_tasks.py
uv run python tests/tasks/build_sim_registered_tasks.py --cpu
uv run python tests/tasks/build_sim_registered_tasks.py --no-render
```

---

## Success-metric check (`check_success_reward.py`)

Teleports objects to their goals for one representative task per category, plus structurally unusual variants (Domino5, Push3, Sorter3, Sorter6, TwoSheet, GolfHole, PushHole, PegCylinderH), and checks that the command term's success metric fires.
The sorter check reorients the free cube so the target hole faces up, then drives the target piece through the mouth in two stations so the latched outside-to-inside crossing fires.
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
