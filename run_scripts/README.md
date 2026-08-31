# run_scripts

Ready-made QC-FQL launches. Two kinds of file:

- **`sweep.sh`** — the generic alpha × seed sweep. Infrastructure: it owns the
  sweep and nothing about the experiment.
- **`<TASK_ID>/…`** — one file per experiment. A sweep wrapper pins the task,
  the dataset and the configuration of one arm; the older single-run scripts
  wrap `train_qc.sh` directly.

All of it is **Linux/cluster-only** (`train_qc.sh` needs bash, tmux,
`/var/tmp/$USER`, `stat -f`), and needs a `qc` checkout with `--wandb_tags`.

## Running a sweep

```bash
./run_scripts/G3A-Place/fixed-s_fixed-t_200k-0.sh
./run_scripts/G3A-Place/fixed-s_fixed-t_200k-0.sh --horizon_length 10
./run_scripts/G3A-Place/fixed-s_fixed-t_200k-0.sh --alphas 500,1000 --num-seeds 5 --dry-run
```

`sweep.sh --help` lists the options. `--dry-run` prints every `train_qc.sh`
invocation and starts none, use it before spending a night of GPU.

Defaults: alphas `1000,10000`, 3 random seeds, 3 runs at a time. **The
seeds are drawn once and reused for every alpha**, so the alphas are compared on
the same seeds and a difference between them is not a difference between seed
draws.

Runs go through a pool of `--max-parallel`, in alpha-major order and with no
barrier between alphas: a finished seed of one alpha starts a seed of the next
rather than leaving the card idle. A failed run does not stop the others.

## Writing a wrapper

```bash
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export G3A_RUN_SCRIPT="$(basename "${BASH_SOURCE[0]}" .sh)"

exec "$HERE/../sweep.sh" \
  <group-name> <TASK_ID> dataset/qc/<demos>.npz \
  "$@" \
  -- \
  --agent=agents/acfql.py \
  --offline_steps 100000
```

- **`"$@"`** forwards what you typed, so one wrapper serves every setting of
  that arm.
- **`--`** separates the two channels. Before it: your flags — passed to
  `main.py`, tagged, and appended to the group name. After it: the wrapper's
  fixed configuration — passed to `main.py` and neither tagged nor named.
  Without the split, the wrapper's own six lines would end up in every group
  name. Your flags are re-emitted *after* the wrapper's block, so they win.
- `sweep.sh` adds no training defaults of its own: whatever the wrapper does not
  set falls back to `main.py`'s, which are much longer runs with no video.

## Names and tags

One sweep is one wandb group, named after the wrapper plus the flags you typed:

```
./fixed-s_fixed-t_200k-0.sh --horizon_length 10
  ->  group  place_fixed-s_fixed-t_200k-0_horizon_length-10
  ->  tags   script-…, task-…, data-…, horizon_length-10, alpha-<a>, seed-<s>
  ->  logs   $RUN_DIR/<task>/<group>/alpha-<a>/seed-<s>/train.log
```

Alpha is a tag, not part of the name — the comparison the sweep exists for
stays inside one group. Names and tags are capped at 64 characters.

There is no timestamp in the name, so re-running the same wrapper with the same
flags **joins the previous group** (it warns first). That is how you add seeds
to an existing question; pass `--run-group NAME` when you want a separate one.

Tags reach wandb through `main.py`'s `--wandb_tags`: `wandb.init` is given
`tags=` explicitly, so the `WANDB_TAGS` environment variable cannot set them.
`train_qc.sh` sets `script-`/`task-`/`data-` by itself for runs launched
without a sweep.

## The older scripts

`critic-variants/` predates `sweep.sh`: eight single-seed runs that call
`train_qc.sh` directly, each pinning one critic configuration. They are the
record of a closed experiment (ADR-014) and are left as they were rather than
ported.
