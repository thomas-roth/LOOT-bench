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

## Example G3a-Stacking-Stack3-Dk1_uni
View:
```shell
uv run g3a-play G3a-Stacking-Stack3-Dk1_uni --agent zero --viewer native --num-envs 1 --no-terminations True
```

Training is offline-to-online RL via QC-FQL, not the mjlab RSL-RL PPO loop
(see `g3a_bench/scripts/qc_fql/README.md`):
```shell
uv run g3a-train-qcfql --dataset-path demos.npz --offline-steps 1000000 \
    --task G3a-Stacking-Stack3-Dk1_uni --checkpoint-path agent.pt
```
