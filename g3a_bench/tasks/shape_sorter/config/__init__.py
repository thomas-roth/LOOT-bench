"""Task registration for all shape-sorter variants on all supported robots."""

from g3a_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from g3a_bench.tasks.common.registration import register_category
from g3a_bench.tasks.shape_sorter.shape_sorter_env_cfg import make_shape_sorter_env_cfg
from g3a_bench.tasks.shape_sorter.variants import SHAPE_SORTER_VARIANTS

register_category(
    category="ShapeSorter",
    variants=SHAPE_SORTER_VARIANTS.values(),
    make_env_cfg=make_shape_sorter_env_cfg,
    robots=(get_unimanual_dk1_interface(),),
    grasp_obs_terms=("pegs_rel_grasp",),
    max_iterations=10_000,
)
