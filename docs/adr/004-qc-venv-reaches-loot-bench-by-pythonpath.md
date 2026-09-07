# The `qc` virtual environment reaches `loot_bench` by `PYTHONPATH`

The `qc` checkout must not `pip install loot-bench`. The install pulls a
CUDA-13 torch build, and JAX then aborts at import. The training process
reaches this package by `PYTHONPATH` instead. Several other environment
constraints travel with this one, because none of them is visible in the code.

## The constraints

- No install, `PYTHONPATH` only. Keep the `qc` environment CUDA-12 only.
- Set `XLA_PYTHON_CLIENT_PREALLOCATE=false`. JAX and mjlab's torch share one
  process and one 11 GB card. Without the flag, JAX takes about 75 % of the
  video memory at import and the simulator runs out.
- Keep `UV_CACHE_DIR` off `/dev/shm`. The site puts the cache on the RAM disk,
  where it filled 94 % of the disk and broke installs. It now points at local
  disk, which gave back 14 GB of RAM.
- Keep the run input and output off `$HOME`. It is NFSv4 with `sec=krb5p` and
  can refuse access in the middle of a run.
- Check the wandb login first. `wandb.init` runs after the environment and the
  dataset are built, so an unauthenticated online run dies minutes in.
- Expect LeRobot 0.6.0 to reject extra dataset metadata. Its `DatasetInfo`
  schema is closed, so per-episode seeds and similar facts need a separate
  sidecar file.
- Watch the import order. A standing warning says lerobot must load before
  mjlab, or the native import order causes a segmentation fault. No module in
  this package can obey it, because any `loot_bench` import runs
  `loot_bench/__init__.py` first, which registers the tasks and imports mjlab.
  Only the first import of the process can order the two, which is what
  `tests/qc/conftest.py` does. The fault did not reproduce on 2026-08-14 with
  the converter's import paths, so the warning is recorded in
  `loot_bench/qc/convert.py` and not acted on further. If a segmentation fault
  appears, move the ordering to the entry point.

## Consequences

Long training runs go in a tmux session. Each of these mistakes costs hours,
because they fail late rather than at startup.
