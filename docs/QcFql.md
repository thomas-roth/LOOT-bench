# QC-FQL training

This document describes the QC-FQL (offline-to-online RL) training path for
G3A tasks. It covers the OGBench-style env/dataset factory in
`g3a_bench.qc`, the dataset conversion tools, the training launcher, the
checkpoint evaluator, and the analysis scripts.

Training runs in the upstream
[`qc`](https://github.com/colinqiyangli/qc) repo (Q-chunking / QC-FQL), not
in mjlab's RSL-RL PPO loop. `g3a_bench` only provides the env and dataset
halves; `qc/main.py` runs the learning.

## Data flow

```
live_record.py  ─►  LeRobot dataset (success/ + failure/ roots)
      │
      ▼
lerobot_to_qc.py ─► QC-format .npz          (per session)
      │
      ▼
merge_datasets.py ─► combined .npz          (optional)
      │
      ▼
g3a_bench.qc.make_env_and_datasets ─► (env, eval_env, train_dataset, None)
      │
      ▼
qc/main.py  ─► checkpoints + eval.csv + offline/online_agent.csv
      │
      ├── train_qcfql.sh        (production launcher)
      ├── play_qcfql.py         (checkpoint evaluation)
      ├── dashboard.py          (interactive HTML plots)
      └── plot_training.py      (static PNG plots)
```

There is no validation split: `make_env_and_datasets` always returns `None`
for the validation dataset.

## The env/dataset factory (`g3a_bench.qc`)

`make_env_and_datasets(dataset_name, ...)` mirrors the useful part of
`ogbench.make_env_and_datasets`:

- Builds one or two `G3AMjlabEnv` wrappers (`env` for training, an
  independently-constructed `eval_env` for evaluation).
- Loads the training dataset from `--g3a_dataset_path` and validates it.
- Returns `(env, eval_env, train_dataset, None)`.

`G3AMjlabEnv` is a Gym-style single-env wrapper over one MJLab G3A task.
Key behaviours:

- **Actions are normalized to `[-1, 1]`.** Before stepping MJLab, they are
  mapped back to the DK1 physical raw-action range via
  `scripts/live_bridge/mapper.ActionNormalizer`. The normalizer is chosen by
  task id: `-Ik` tasks use the ik bounds (identity on the 6 arm-delta dims,
  physical range on the gripper); all others use the fixed physical DK1
  joint range.
- **Observations** come from the `actor` observation group (configured by
  `obs_group`), with the leading `num_envs=1` axis stripped.
- **`auto_reset=False`** on the underlying MJLab env. `qc/main.py` handles
  this, but any custom probe script must call `reset()` when an episode
  terminates, or the next `step()` raises.
- **Rewards** are per-step values in `[-reward_max, 0]`; see the reward
  convention section below.
- `info` carries `total.timesteps`, `success` (when the task command exposes
  it), and `gripper_contact` (see the gripper-contact section). On episode
  end it also carries an `episode` dict with `final_reward`, `return`,
  `length`, and `duration`.

### Dataset validation

`make_env_and_datasets` runs two checks that fail fast instead of silently
training a degraded policy:

1. **Shape check** (`validate_qc_dataset_against_env`): the dataset's
   observation and action shapes must match the env's spaces.
2. **Reward-convention check** (`validate_reward_convention`): the dataset's
   stamped `reward_max` must equal the env's. A mismatch is a `ValueError`;
   a dataset without the metadata is rejected as predating the negative
   reward convention; an env without an offset term (test doubles, ad-hoc
   cfgs) triggers a `RuntimeWarning` and passes.

## Reward convention

Every category term is normalized to `[0, 1]`. On top of those,
`make_manipulation_env_cfg` appends the shared `REWARD_OFFSET_TERM`
(`max_reward_offset`), a constant `-reward_max`, so the env's per-step
reward is:

```
sum(task terms) - reward_max      in [-reward_max, 0]
```

`reward_max` is the sum of the task's own term weights: `2.0` for the
`progress` + `success` categories (stack, domino, shape_sorter), `1.0` for
the success-only ones (place, ball). Reaching `0` means success.

- `scale_rewards_by_dt=False` so the offset is not scaled by the control dt.
- The convention matches what QC's defaults expect (OGBench `success - 1`,
  D4RL/RoboMimic `reward - 1`), so alpha/lr defaults land on the right
  scale.

The dataset must be produced under the same convention; that is what the
training-time check enforces.

## The `-Ik` task-space action variant

Each category can register both action parameterizations:

- **joint** (default, unchanged ids): one `JointPositionActionCfg` over
  every actuator.
- **ik** (`tasks/common/ik_action.py`): a 6-DoF end-effector delta-pose term
  through mjlab's damped-least-squares differential IK plus a separate
  1-DoF joint-position term for the gripper. Task ids get an `-Ik` suffix
  (e.g. `G3A-Stack-Low-Dk1_uni-Ik`). Only robots declaring `arm_actuator_names`
  support it.

The ik action layout is `[dx, dy, dz, drx, dry, drz, gripper]` (7-D). The
arm block is a relative delta scaled by the robot's
`ik_delta_pos_scale`/`ik_delta_ori_scale`; the gripper block behaves exactly
as under joint control. Datasets for `-Ik` tasks store the env action
directly and are normalized with the fixed ik bounds, regardless of the
`--action-norm` flag (a joint-space choice).

## Converting recordings to QC datasets

### `lerobot_to_qc.py`

Converts a `live_record.py` LeRobot dataset root into the QC transition
format. Takes the success root (required) and an optional failure root:

```bash
uv run python g3a_bench/scripts/live_bridge/lerobot_to_qc.py \
    --success-dir out/demos/success/<task>/<stamp> \   # required
    --failure-dir out/demos/failure/<task>/<stamp> \   # optional
    --output dataset.npz \
    --action-norm physical                             # or "data"
```

The output `.npz` holds the six QC training keys plus metadata:

| keys | purpose |
|---|---|
| `observations`, `actions`, `next_observations`, `rewards`, `terminals`, `masks` | passed to `Dataset.create()` |
| `episode_index`, `frame_index`, `timestamp`, `success`, `sim_done` | preserved for analysis |
| `action_min`, `action_max`, `action_norm_method` | denormalization at deployment |
| `reward_max`, `task_id` | stamped reward convention; checked at training |

Action normalization:

- `physical` (default): fixed bounds from the DK1 joint limits in raw-action
  space. Session-independent; safe to combine datasets.
- `data`: per-joint min/max of the dataset. Fills `[-1, 1]` fully but is
  session-specific. **Only `physical` (and `ik` for `-Ik` tasks) is
  accepted for training** — `load_qc_dataset` rejects `data`.
- `ik`: for `-Ik` tasks only; fixed identity-on-arm + physical-gripper
  bounds. Set automatically, ignoring `--action-norm`.

The script reconstructs the exact sim action sent to `env.step()` during
recording (the leader joints pass through
`mapper.lerobot_to_mjlab_raw_action(gravity_compensate=True)`; `-Ik`
recordings already store the env action) and derives `terminals` and `masks`
from episode boundaries + success. It rejects recordings that are not on the
current reward convention and episodes whose absorbing frame was never
scored a success (which would teach the critic that the goal is worthless).

### `merge_datasets.py`

Concatenates multiple `.npz` files into one. Rejects merges across different
`action_norm_method` or `reward_max` (different reward scales), renumbers
episode indices to stay unique, and carries the metadata through (a merged
file without `reward_max` would be rejected at training).

## Training

### `train_qcfql.sh` (launcher)

The production entry point. It keeps all live I/O on local disk and mirrors
the results to the NFS home periodically, so a home-mount hiccup costs one
sync instead of the whole run:

```bash
./train_qcfql.sh --env_name=G3A-Place-Easy-Dk1_uni \
    --g3a_dataset_path=dataset/demos/G3A-Place-Easy-Dk1_uni/..._qc_physical.npz \
    --g3a_device=auto \
    --seed=42 --horizon_length=5 --sparse=false --agent.alpha=300 \
    --offline_steps=1000000 --online_steps=1000000 --eval_episodes=50
```

All arguments pass straight to `qc/main.py`; `--save_dir` is injected and
cannot be overridden. Environment overrides:

| variable | default | meaning |
|---|---|---|
| `QC_DIR` | `../qc` | path to the qc checkout |
| `QC_RUN_ROOT` | `/tmp/$USER/qc-runs` | local scratch root for the run |
| `QC_MIRROR_DIR` | `$QC_DIR/runs` | where the run dir is mirrored |
| `QC_MIRROR_EVERY` | `600` | seconds between mirrors |
| `WANDB_MODE` | `online` | set `offline` to keep logs local (`wandb sync` later) |

Behaviours in the script:

- Runs in `$QC_DIR` with `PYTHONPATH` pointing at this repo — **do not
  `pip install` g3a-bench into the qc venv** (it drags in a different torch
  build; see the map scratch notes).
- Sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` so JAX and mjlab/Warp share
  one GPU without JAX grabbing ~75% of VRAM.
- Preflights wandb authentication when online: `wandb.init` runs *after* the
  env is built, so an unauthenticated online run would otherwise fail
  minutes in.
- Mirrors the whole run dir (checkpoints, CSVs, wandb, log) on an interval
  plus a final sync on exit. Mirror failure is never fatal.
- Warns (non-fatal) when no Kerberos ticket exists, since the mirror is the
  only part that touches the NFS home.

### Direct `qc/main.py` invocation

For quick runs without the mirror/wandb machinery:

```bash
cd ../qc && PYTHONPATH=../g3a-bench:$PYTHONPATH \
python main.py --env_name=G3A-Stack-Low-Dk1_uni \
    --g3a_dataset_path=demos.npz --g3a_device=auto \
    --offline_steps=100000 --online_steps=50000
```

On CPU-only machines: `--g3a_device=cpu` and `JAX_PLATFORMS=cpu` before
`python main.py`.

### Outputs

A run lives under the qc `--save_dir`:
`exp/qc/<wandb.project>/<run_group>/<env_name>/<exp_name>/` with `eval.csv`,
`offline_agent.csv`, `online_agent.csv`, and the checkpoints. Under
`train_qcfql.sh`, `--save_dir` is `<run_root>/exp`, and the run dir also
holds `train.log` and `wandb/`.

## Evaluation

### `play_qcfql.py`

Replays a QC checkpoint in the G3A MJLab sim and reports the success rate:

```bash
# from the qc repo (g3a-bench on PYTHONPATH)
cd ../qc
PYTHONPATH=../g3a-bench:$PYTHONPATH python ../g3a-bench/g3a_bench/scripts/play_qcfql.py \
    --checkpoint <path>/params_2000000.pkl \
    --task G3A-Stack-Low-Dk1_uni --episodes 10
```

- Accepts both `.pkl` checkpoints and the `.pkl.zip` ones from
  `dataset/models/`.
- `--horizon_length` and `--alpha` must match training.
- Requires a display-free environment; `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  is set inside the script.
- `--save_video out/eval.mp4` records an mp4 (needs `imageio[ffmpeg]`).

## Analysis scripts

Both read the CSV metrics QC flushes next to the checkpoints, so they work
while a run is still training. `--watch N` re-renders every N seconds.

### `dashboard.py`

Interactive Plotly HTML dashboard:

```bash
uv run python g3a_bench/scripts/dashboard.py <run_dir> [--watch 30] [--open]
```

Four panels: eval return + success rate, episode length + final reward,
critic Q-values, and losses (log scale), with a dashed line at the
offline→online boundary. `<run_dir>` may be the `save_dir` or any parent of
it (the newest matching dir is used). Needs plotly.

### `plot_training.py`

Static PNG from the same CSVs:

```bash
uv run python g3a_bench/scripts/plot_training.py <run_dir> [--watch 60]
```

Writes `<run_dir>/training.png` (default) and prints the last/best eval
success and return. Needs matplotlib.

## Gripper-contact metric

Upstream's `qc/evaluation.py` counts gripper contacts and logs
`num_gripper_contacts` / `avg_gripper_contact_length` in `eval.csv`. It
reads the `gripper_contact` key from `info`; if the key is absent the
counter is skipped silently (the columns read 0). `G3AMjlabEnv` exports the
peak finger-pad contact force at the top level of `info`:

- It prefers the **object-filtered** sensor (`grip_object_contact`) and
  falls back to the unfiltered gate sensor (`grip_contact`). The gate
  sensor reads ~2.4 N on pad-on-pad contact when the gripper closes on air,
  so it counts *grip attempts*; the object-filtered one answers "did the
  policy reach the object?". Only `place` registers the object-filtered
  sensor; the other categories carry just `grip_contact`, so there the
  number means "grip attempt".
- All categories except `ball` register the gate pad sensor
  (`grip_contact`), so the key is present for stack, domino, shape_sorter,
  and place. `ball` leaves it absent, which keeps upstream's old skip
  behaviour.
- The key is **flat** (top-level), not nested under `proprio`: upstream's
  nested branch thresholds `info["gripper_contact"]` even when it read the
  value out of `proprio`, so a proprio-only key raises `KeyError`.

The threshold upstream applies is the hardcoded `> 0.1` in
`qc/evaluation.py`, the same number as `MIN_GRIP_FORCE` in
`g3a_bench/tasks/common/commands.py`. Nothing enforces that agreement across
the repo boundary; if `MIN_GRIP_FORCE` moves, the two drift apart.

## Tests

`tests/test_qc_backend.py` unit-tests the `g3a_bench.qc` adapter with a fake
MJLab env:

- dataset loading (key stripping, uint8 preservation, missing-key and
  `data`-normalization rejection),
- obs/action shape validation,
- the reward-convention check (match, mismatch, missing metadata, env
  without an offset term),
- the Gym wrapper reset/step/render/get_state contract,
- the gripper-contact key (absent without sensors, peak force at the top
  level, object-filtered sensor winning over the gate sensor).

Run it with `uv run pytest tests/test_qc_backend.py`. The conversion side is
covered by `tests/live_bridge/` (`test_lerobot_to_qc.py`,
`test_merge_datasets.py`, `test_ik_action_mapping.py`).