"""The seam between ``g3a-bench`` and the ``qc`` training code.

``qc`` reaches this package by ``PYTHONPATH``, never by installing
``g3a-bench`` into its venv: the install pulls a cu130 torch that makes JAX
abort at import.

Import order matters in the other direction too. Anything that touches
``lerobot`` must import it *before* mjlab, or the native import order
segfaults. This package imports mjlab, so a script that needs both has to
import lerobot first -- see ``g3a_bench/qc/convert.py``, which does.
"""

from g3a_bench.qc.env import TASK_PREFIX as TASK_PREFIX
from g3a_bench.qc.env import G3aEnv as G3aEnv
from g3a_bench.qc.env import is_g3a_task as is_g3a_task
from g3a_bench.qc.env import make_g3a_env as make_g3a_env
from g3a_bench.qc.episode import EpisodeStats as EpisodeStats
