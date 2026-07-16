"""Task registration for all stacking variants on all supported robots."""

from g3a_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from g3a_bench.tasks.common.registration import register_category
from g3a_bench.tasks.stacking.stacking_env_cfg import make_stacking_env_cfg
from g3a_bench.tasks.stacking.variants import STACKING_VARIANTS

register_category(
    category="Stacking",
    variants=STACKING_VARIANTS.values(),
    make_env_cfg=make_stacking_env_cfg,
    robots=(get_unimanual_dk1_interface(),),
    grasp_obs_terms=("blocks_rel_grasp",),
    max_iterations=10_000,
)
