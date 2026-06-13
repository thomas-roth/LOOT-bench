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
   These scripts automatically install the appropriate PyTorch build based on the `nvidia-smi` CUDA version. Supported options are: `cu128` or `cpu`.

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

To run the type checker on the `src` directory:
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
