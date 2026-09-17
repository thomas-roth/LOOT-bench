# QC seam tests

Unit and contract tests for the QC training seam (`loot_bench/qc/`), plus the
two standalone scripts that measure or feed it. Run everything from the
**repo root**.

Almost all of it is plain array work with no simulator. The exception is
`test_env.py`, which builds a real task. Nothing here needs the `qc` checkout
except `bc_gate.py`, which runs from the **qc venv** and never under pytest.

---

## Run all tests

```bash
uv run pytest tests/qc/ -v
```

Expected output: all green, 110 passed, about 15 seconds.

`conftest.py` imports `lerobot.datasets.lerobot_dataset` before anything else.
That is not a convenience import: lerobot and mjlab both load native
libraries, and the reverse order segfaults during collection. Do not remove it
and do not move it below another import.

---

## What each file tests

### `test_transitions.py`

The pure transition assembly (`qc/transitions.py`): recorded rows to
`(s, a, r, s', done)`. No lerobot, no simulator. Every failure mode here is
silent, so nothing raises when it goes wrong: a wrong `masks` column trains an
agent to like timeouts, and a wrong `terminals` column lets a sampled sequence
run out of one episode and into the next.

| Test class | What it checks |
|------------|----------------|
| `TestSuccessfulEpisode` | every row kept, final transition absorbing, interior rows bootstrap, `next_observations` is the following row |
| `TestTimeoutEpisode` | the unusable final row is dropped, the last kept row bootstraps into a real state |
| `TestEpisodeBoundaries` | `next_observations` never crosses into the next episode, every episode end is terminal, `masks` is zero only on success, interleaved episodes are rejected |
| `TestRewardConvention` | a positive reward, a reward below -1, and a success before the final row are all rejected |
| `TestOutputContract` | every key qc requires is published, scalar columns are 1-D, widths are preserved |

Run it after touching `qc/transitions.py` or the reward convention (ADR-006,
ADR-010).

```bash
uv run pytest tests/qc/test_transitions.py -v
```

---

### `test_convert.py`

The `loot-convert-demos` entry point (`qc/convert.py`) and the reader that
checks a dataset back against an environment (`qc/dataset.py`). The episode
structure rules are `test_transitions.py`'s; what is left here is everything
the converter adds around them.

| Test class | What it checks |
|------------|----------------|
| `TestActionColumn` | the recorded action is copied through, not re-derived, and an out-of-box row is rejected |
| `TestRowOrder` | rows are sorted by `(episode, frame)` before assembly, so a shuffled or interleaved recording still converts |
| `TestStamp` | every stamp key is published, and the stamped numbers match what the action terms apply |
| `TestStartPoseNoiseStamp` | the start-pose half-width the episodes were recorded under is stamped (ADR-008) |
| `TestStampCheck` | any stamp field that disagrees with the env is **rejected**, and so is a missing one |
| `TestSessionOffsetting` | episode indices that collide across sessions do not silently join two episodes |
| `TestRewardValidation` | a dataset on a stale reward convention is rejected |
| `TestReadingARecording` | a LeRobot session converts to an `.npz`, two sessions keep their episodes apart, and mixed tasks are rejected |
| `TestLoadDataset` | the `.npz` round-trips, and is checked against an env when one is given |

Run it after touching `qc/convert.py`, `qc/dataset.py`, or the stamp keys
(ADR-011).

```bash
uv run pytest tests/qc/test_convert.py -v
```

---

### `test_episode.py`

Per-episode bookkeeping and the `info` dict qc reads (`qc/episode.py`). Pure
Python, no simulator.

| Test class | What it checks |
|------------|----------------|
| `TestEpisodeStats` | length and return, success keyed off `reward == 0.0` and **latched** for the rest of the episode, all `info` values scalar, contact telemetry published under flat keys and omitted when the sensor is absent |

The flat key names (`pad_contact_force`, `object_contact_force`) are a
cross-repo contract with no version pin: `qc/evaluation.py` consumes them by
name. After changing either side, run a short eval and check that
`eval/pad_contact_runs` is non-zero. These tests cannot see that break.

```bash
uv run pytest tests/qc/test_episode.py -v
```

---

### `test_env.py`

Contract tests for the gym adapter (`qc/env.py`), run against a **real task**.
These build a simulator, so they dominate the runtime: the other 90 tests
finish in about 6 seconds, and these 20 take the suite to about 15. They earn
it. Every assertion is a promise qc relies on and cannot check for itself, and
the failure mode when one breaks is a flat training curve, not an exception.

| Test class | What it checks |
|------------|----------------|
| `TestTaskPrefix` | `is_loot_task` accepts `LOOT-` ids and rejects others |
| `TestSpaces` | the action space is the unit box, the observation space matches what `reset` returns |
| `TestStepContract` | the gym five-tuple, the reward convention, actions clipped rather than rejected, scalar `success`, flat contact keys, and that the returned observation is **not a view** on simulator memory |
| `TestObjectContact` | the object-filtered sensor, and that a gripper closing on air loads the pads only |
| `TestEpisodeBoundaries` | auto-reset is **off**, `reset` clears the stats, and `start_pose_noise` of zero repeats the same start while a non-zero value varies it |

`LOOT-Place-RandS-RandT-Dk1` is used for the object-contact tests because
Place scores exactly one free object. Stack declares no such sensor and is the
negative control.

```bash
uv run pytest tests/qc/test_env.py -v
```

---

### `test_bc_gate.py`

The BC gate's pure kernels. The gate itself needs JAX and the qc checkout, so
**only these kernels run under pytest**.

Each of these decisions is silent when it is wrong. A leaked held-out episode,
a window that walks across an episode boundary, or a standardizer dividing by
a constant dimension all give a number rather than an error, and the number is
what a run gets started on.

| Test class | What it checks |
|------------|----------------|
| `TestBuildChunks` | steps after a terminal inside the window are masked, targets are the following actions, a recording shorter than a chunk is rejected |
| `TestEpisodeBounds` | bounds are half-open and keyed by episode id, interleaved rows are rejected |
| `TestSplitEpisodes` | the split is a partition, it is seeded, and it never holds out everything |
| `TestTrainingStarts` | every window **touching** a held-out episode is dropped, not only the ones that start in it |
| `TestFirstFrames` | the opening row of each episode is read |
| `TestInteriorFrames` | the start pose and the chunk tail are skipped, each episode is capped, one long-enough episode is required |
| `TestStandardizer` | only the fitted rows are read, and a constant dimension is left alone |
| `TestLoadColumns` | a missing column is named in the error |

`tests/qc` carries no `__init__.py`, so pytest puts the directory on the path
and the gate imports by its bare name (`from bc_gate import ...`).

```bash
uv run pytest tests/qc/test_bc_gate.py -v
```

---

## The BC gate (`bc_gate.py`)

Not a pytest file. Fit the recorded actions with a plain regressor and read how
well the fit transfers to episodes it never saw. About two minutes, against
hours for a training run. See **ADR-012**, which says to run this before
spending a run on a new demonstration set.

It imports qc's own `MLP` body, so the regressor and the flow actor cannot
drift apart. That is why it runs from the **qc venv** with both checkouts on
`PYTHONPATH`, not under `uv run`:

```bash
PYTHONPATH=<loot-bench>:<qc> <qc>/.venv/bin/python tests/qc/bc_gate.py \
    out/qc/demos.npz
PYTHONPATH=<loot-bench>:<qc> <qc>/.venv/bin/python tests/qc/bc_gate.py \
    out/qc/demos.npz --holdout 10 --steps 100000 --out out/qc/bc
```

It prints numbers and nothing else: no pass, no fail, and no verdict exit
status. Useful flags: `--holdout` (default a fifth, at least one),
`--holdout-seed`, `--steps` (50000), `--horizon` (the action chunk length, 5),
`--hidden`, `--lr-schedule`, and `--out` to write `history.json`.

**There is no bar, and the old 0.0766 one is withdrawn** (ADR-012, amended
2026-09-12). It was never read on a set whose training outcome was known, and
against two sets whose outcome is known it ordered them backwards: the
`place_rand-s_rand-t` set that reaches 0.56 `eval/success` reads a held-out
interior error of 0.1946-0.2382, worse than `dominos_4-base` at 0.1327-0.1413,
which reaches none. Held-out interior error does not predict success.

Read the three numbers the gate prints against each other, and against the two
reference readings it prints below them:

- **held-out interior error** — how far the fit carries to unseen episodes;
- **fitted interior error** — how much of that is failure to transfer, versus
  the observation not determining the action at all;
- **the demos' own `|a[t+1] - a[t]|`** — the same statistic on the same scale,
  so the error can be compared with the motion it has to predict.

A set that reads far worse than both references, or whose fitted error sits
near its held-out error after the fit has converged, is saying something. A set
that lands between them is saying nothing yet, and is not a reason to
re-record.

---

## Synthetic dataset (`make_smoke_dataset.py`)

Not a test either. Writes an `.npz` the training path will accept, so the
launcher, qc's `LOOT-` branch, the dataset check, and the replay buffer's
shapes can be exercised end to end before a single demonstration exists.

**The rows are made up.** Random observations, random actions, and a reward
ramp that satisfies ADR-006. No number read off a run against it means
anything. The widths and the action stamp are read off the real environment,
so a dataset from here fails exactly where a stale recording would.

```bash
uv run python tests/qc/make_smoke_dataset.py --cpu \
    --task LOOT-Stack-1Pi-Dk1 --output out/qc/smoke.npz

RUN_DIR=/var/tmp/$USER/qc-runs WANDB_MODE=offline LOOT_ALLOW_NO_TMUX=1 \
    ./train_qc.sh LOOT-Stack-1Pi-Dk1 out/qc/smoke.npz \
    --offline_steps 2 --online_steps 0 --eval_episodes 1
```
