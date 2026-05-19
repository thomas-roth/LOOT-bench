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
By default `uv sync` may prune packages that are not in `pyproject.toml`/`uv.lock`. If you install PyTorch manually and want to keep it:
```
uv sync --no-prune
```

When running scripts, skip auto-sync to avoid pruning local wheels:
```
uv run --no-sync main.py
```