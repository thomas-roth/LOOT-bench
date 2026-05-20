# G3A-Bench

## Installation
```
uv sync
```

Install **PyTorch** CUDA 13.0 following the [official instructions](https://pytorch.org/get-started/locally/) but with uv
```
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### Keep local PyTorch wheels
When running scripts, skip auto-sync to avoid pruning local wheels:
```
uv run --no-sync main.py
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
