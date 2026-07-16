"""Task registration for all cube variants on all supported robots."""

from g3a_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from g3a_bench.tasks.common.registration import register_category
from g3a_bench.tasks.cube.cube_env_cfg import make_cube_env_cfg
from g3a_bench.tasks.cube.variants import CUBE_VARIANTS

register_category(
    category="Cube",
    variants=CUBE_VARIANTS.values(),
    make_env_cfg=make_cube_env_cfg,
    robots=(get_unimanual_dk1_interface(),),
    grasp_obs_terms=("pieces_rel_grasp",),
    max_iterations=10_000,
)
