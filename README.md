# G3A-Bench

## Installation
```
git clone git@github.com:thomas-roth/g3a-bench.git
cd g3a-bench
bash install.sh
```

## Code Quality Checks
The project uses [Ruff](https://www.astral.sh/ruff) for linting and formatting and [Pyright](https://microsoft.github.io/pyright) for type checking.

To run the linter on the current working directory (recursively):
```
uv run ruff check
```

To run the formatter on the current working directory (recursively):
```
uv run ruff format
```

To run the type checker on the `src` directory:
```
uv run pyright
```

### Pre-Commit Hooks
The project uses [pre-commit](https://pre-commit.com) to automatically run the code quality checks on staged Python files when committing.

To manually run the pre-commit hooks on staged files:
```
uv run pre-commit
```
To manually run the pre-commit hooks on all files:
```
uv run pre-commit run --all-files
```
