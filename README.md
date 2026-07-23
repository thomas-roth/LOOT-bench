# G3A-Bench

## Installation
1. Clone the repo:
   ```bash
   git clone git@github.com:thomas-roth/g3a-bench.git
   cd g3a-bench
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

## Training

Training is offline-to-online RL via **upstream [`qc`](https://github.com/colinqiyangli/qc)**
(Q-chunking / QC-FQL), not the mjlab RSL-RL PPO loop. `g3a_bench` only
provides an OGBench-style env/dataset factory (`g3a_bench.qc.make_env_and_datasets`);
`qc/main.py` runs the actual training loop. There is no validation split —
`val_dataset` is always `None`.

1. Record demos with `live_bridge` (see `g3a_bench/scripts/live_bridge/README.md`),
   then convert (+ merge, if more than one session) to a QC `.npz`:
   ```bash
   uv run python g3a_bench/scripts/live_bridge/lerobot_to_qc.py \
     --success-dir out/demos/success/<task>/<stamp> \
     --output demos.npz --action-norm physical
   # only "physical" action normalization is supported for training — "data" is rejected.

   uv run python g3a_bench/scripts/live_bridge/merge_datasets.py \
     demos_1.npz demos_2.npz --output demos.npz
   ```

2. One-time `qc/` environment setup (separate venv, sibling checkout):
   ```bash
   cd ../qc
   uv venv --python 3.12 && source .venv/bin/activate
   uv pip install -r requirements.txt
   uv pip install -e ../g3a-bench
   ```

3. Train (offline pretraining + online fine-tuning):
   ```bash
   cd ../qc && source .venv/bin/activate
   PYTHONPATH=../g3a-bench:$PYTHONPATH python main.py \
     --env_name=G3A-Stack-Low-Dk1_uni --g3a_dataset_path=demos.npz --g3a_device=auto \
     --seed=42 --horizon_length=5 --sparse=false --agent.alpha=300 \
     --offline_steps=1000000 --online_steps=1000000 --start_training=5000 \
     --eval_interval=100000 --eval_episodes=50 --save_interval=100000
   ```
   On CPU-only machines: `--g3a_device=cpu` and `JAX_PLATFORMS=cpu` before `python`.

## Code Quality Checks
The project uses [Ruff](https://www.astral.sh/ruff) for linting and formatting and [Pyright](https://microsoft.github.io/pyright) for type checking.

To run the linter on the current working directory (recursively):
```bash
uv run ruff check
```

To run the formatter on the current working directory (recursively):
```bash
uv run ruff format
```

To run the type checker on the `g3a_bench` directory:
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

## Example G3A-Stack-Low-Dk1_uni
View:
```shell
uv run g3a-play G3A-Stack-Low-Dk1_uni --agent zero --viewer native --num-envs 1 --no-terminations True
```



cd ../qc
source .venv/bin/activate
PYTHONPATH=../g3a-bench:$PYTHONPATH python main.py   --env_name=G3A-Place-Easy-Dk1_uni   --g3a_dataset_path=../g3a-bench/dataset/demos/G3A-Place-Easy-Dk1_uni/G3A-Place-Easy-Dk1_uni_qc_physical.npz   --g3a_device=auto   --seed=42 --horizon_length=5 --sparse=false --agent.alpha=300   --offline_steps=1000000 --online_steps=1000000 --start_training=5000   --eval_interval=100000 --eval_episodes=50 --save_interval=100000   --run_group=G3A-Place-Easy-Sanity