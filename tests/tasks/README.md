# Registered-task verification

These two scripts are the verification path for changes to task code; run both
from the **repo root**. Neither is installed as a console script: invoke them
with `uv run python`. Both accept `--cpu` to force CPU-only execution.

---

## Build, reset, step every task (`build_sim_registered_tasks.py`)

Builds every registered `G3a-*` task, resets it, and steps it a few times with
zero actions. Fast smoke test that the env, managers, and MJCF all assemble
without error. Run this after touching any `*_env_cfg.py`, `assets.py`, or
robot config.

```bash
uv run python tests/tasks/build_sim_registered_tasks.py
uv run python tests/tasks/build_sim_registered_tasks.py --cpu
```

---

## Success-metric check (`check_success_reward.py`)

Teleports objects to their goal positions for one representative task per
category (plus structurally unusual variants: Domino5, Sorter4, TwoSheet,
GolfHole, PushHole, PegCylinderH) and checks that the command term's success
metric fires. Confirms the metric *can* fire when objects are placed
correctly — it does not verify that reward shaping will guide a policy there.
Run this after touching a category's `mdp/commands.py` or success-metric logic.

```bash
uv run python tests/tasks/check_success_reward.py
uv run python tests/tasks/check_success_reward.py --cpu
```
