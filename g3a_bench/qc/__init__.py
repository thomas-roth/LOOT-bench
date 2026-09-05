"""Seam between g3a-bench and qc training code.

qc reaches this package by PYTHONPATH (installing g3a-bench into qc's venv pulls
cu130 torch, making JAX abort at import).
Import lerobot before mjlab where both are needed to avoid segfault; only
convert needs lerobot.
"""

from g3a_bench.qc.dataset import STAMP_KEYS as STAMP_KEYS
from g3a_bench.qc.dataset import check_dataset_matches_env as check_dataset_matches_env
from g3a_bench.qc.dataset import get_training_columns as get_training_columns
from g3a_bench.qc.dataset import load_dataset as load_dataset
from g3a_bench.qc.env import TASK_PREFIX as TASK_PREFIX
from g3a_bench.qc.env import G3aEnv as G3aEnv
from g3a_bench.qc.env import is_g3a_task as is_g3a_task
from g3a_bench.qc.episode import EpisodeStats as EpisodeStats
from g3a_bench.qc.transitions import QC_KEYS as QC_KEYS
