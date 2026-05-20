# G3A-Bench

## Installation
```
git clone https://github.com/thomas-roth/g3a-bench.git
cd g3a-bench
bash install.sh
```

## Linting and type checking
Run the linter on all Python files:
```
uv run ruff check
```

Run the type checker on the `src` folder:
```
uv run pyright
```

These checks also run automatically on staged Python files when committing (via pre-commit).
