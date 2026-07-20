"""G3A long-horizon manipulation benchmark on mjlab.

Importing this package registers all benchmark tasks with the mjlab task
registry.
"""

import g3a_bench.tasks  # noqa: F401
from g3a_bench.qc import make_env_and_datasets as make_env_and_datasets
