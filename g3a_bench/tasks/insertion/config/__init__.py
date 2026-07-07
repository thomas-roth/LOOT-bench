"""Task registration for all insertion variants on all supported robots."""

from g3a_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from g3a_bench.tasks.common.registration import register_category
from g3a_bench.tasks.insertion.insertion_env_cfg import (
    REWARD_MODE_TERMS,
    make_insertion_env_cfg,
)
from g3a_bench.tasks.insertion.variants import INSERTION_VARIANTS

register_category(
    category="Insertion",
    variants=INSERTION_VARIANTS.values(),
    make_env_cfg=make_insertion_env_cfg,
    robots=(get_unimanual_dk1_interface(),),
    # The two_sheet variant adds "sheet_rel_grasp"; absent keys are skipped so
    # the same tuple fits every variant.
    grasp_obs_terms=("pegs_rel_grasp", "sheet_rel_grasp"),
    reward_mode_terms=REWARD_MODE_TERMS,
)
