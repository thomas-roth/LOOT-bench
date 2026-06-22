# Import lerobot/HF-datasets before any test imports mjlab/g3a_bench, to avoid a
# native import-order segfault during collection.
import lerobot.datasets.lerobot_dataset  # noqa: F401
