# LOOT (Long-horizon Offline-to-Online Tabletop-manipulation) Benchmark

## Installation
1. Clone the repo:
   ```bash
   git clone git@github.com:thomas-roth/LOOT-bench.git
   cd LOOT-bench
   ```
2. Execute the install script depending on your operating system:
   
   Linux / macOS:
   ```bash
   bash install.sh
   ```
   Windows:
   ```powershell
   ./install.ps1
   ```
   The PyTorch build is selected automatically by platform (see `[tool.uv.sources]` in `pyproject.toml`): `cu128` on Linux/Windows, `cpu` on macOS.

## Task Commands

### Listing Tasks
The `loot-list-tasks` command lists all task IDs registered into mjlab (the `LOOT-*` benchmark tasks plus mjlab's built-ins).
Narrow it with a substring using `--filter`, or count the tasks using `--count`:
```shell
uv run loot-list-tasks
uv run loot-list-tasks --filter Domino
uv run loot-list-tasks --filter LOOT --count
```

### Playing a Task
The `loot-play` command starts mjlab's player on one task ID, so you can look at the scene and the robot. Give a checkpoint to see a policy, or `--agent zero` to hold the robot still.
```shell
uv run loot-play LOOT-Stack-2Henges-Dk1 --agent zero --viewer native --num-envs 1
```

## Data Collection

### Recording Demos
The `loot-record-demos` command records teleop demonstrations into a LeRobot dataset. A GELLO leader arm drives the simulation, and you label each episode a success or a failure with a key. The script requires the leader hardware and a display.
```shell
uv run loot-record-demos --task <TASK_ID>
```

For the keys, the savedir, the ports and the other bridge scripts, read `loot_bench/scripts/live_bridge/README.md`.

### Converting Demos
The `loot-convert-demos` command reads teleop recordings and converts them into an `.npz` file that QC-FQL trains on.

Give one or more success roots with `--success-dir`, optionally the sibling failure roots with `--failure-dir`, and the output file with `--output`:
```shell
uv run loot-convert-demos \
    --success-dir dataset/demos/<task>/<timestamp_success> \
    --failure-dir dataset/demos/<task>/<timestamp_failure> \
    --output dataset/qc/demos.npz
```

All recordings must come from the same task ID. The command writes that ID into the `.npz`, together with the action convention, so a training run can detect a dataset that does not agree with its environment.

### Splitting Demos
The `loot-split-demos` command turns one recording of a hard task into a demo set for each easier task in the same family, without re-simulating anything. Stack, Domino and Sorter all tail-extend: the easier task's objects are the first `k` of the harder one's, and the reset draws per object in index order, so the easier rollout is already inside the recording.

```shell
uv run loot-split-demos \
    --source-dir dataset/demos/<task>/<timestamp> \
    --source-task LOOT-Stack-3Henges-Dk1 \
    --output-dir dataset/demos_split
```

Every episode is truncated to the target's objects and cut on the frame the target's success would have been paid, with the reward recomputed through the environment's own success kernels. Pass `--target-task` (repeatable) to write only some of them. Each episode's own reward is recomputed first and checked against the recording, so a layout or threshold that no longer matches the environment fails immediately instead of writing a corrupt dataset.

### Fusing Demos
The `loot-fuse-demos` command merges several recorded LeRobot sessions into one LeRobot root (episodes reindexed, sidecars merged), for the consumers that take a single root: `loot-split-demos`, the replay readers, inspection. `loot-convert-demos` already fuses at `.npz` time; this covers everything upstream of it.
```shell
uv run loot-fuse-demos \
    --src out/demos/success/<task>/<timestamp_a> \
    --src out/demos/success/<task>/<timestamp_b> \
    --out out/demos/fused/<task>/<timestamp>
```

## Training

`run_scripts/` holds ready-made QC-FQL launches (Linux/cluster only, they need a `qc` checkout next to this repo). They consist of different run scripts, each pinning task, dataset, and configuration, as well as a generic sweep script to start a task experiment run with multiple alphas and seeds:
```shell
bash run_scripts/LOOT-Place/fixed-s_fixed-t_200k-0.sh
bash run_scripts/sweep.sh --help
```

The scripts are experiment records, so they change often. For the current list, look in `run_scripts/`. For the sweep options, the wandb config, and how to write a new script, read `run_scripts/README.md`.

## Development

### Code Quality Checks
The project uses [Ruff](https://www.astral.sh/ruff) for linting and formatting and [Pyright](https://microsoft.github.io/pyright) for type checking.

To run the linter on the current working directory (recursively):
```bash
uv run ruff check
```

To run the formatter on the current working directory (recursively):
```bash
uv run ruff format
```

To run the type checker on the `loot_bench` directory:
```bash
uv run pyright
```

### Pre-Commit Hooks
The project uses [pre-commit](https://pre-commit.com) to automatically run the code quality checks on staged Python files when committing.

To manually run the pre-commit hooks on staged files:
```bash
uv run pre-commit
```
To manually run the pre-commit hooks on all files:
```bash
uv run pre-commit run --all-files
```

