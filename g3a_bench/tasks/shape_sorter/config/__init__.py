"""Task registration for all shape-sorter variants on all supported robots."""

from g3a_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from g3a_bench.tasks.common.registration import register_category
from g3a_bench.tasks.shape_sorter.shape_sorter_env_cfg import make_shape_sorter_env_cfg
from g3a_bench.tasks.shape_sorter.variants import (
    SHAPE_SORTER_VARIANTS,
    ShapeSorterVariantCfg,
)

_VARIATION_BY_SPAWN = {
    "fixed": "FixedOrientation",
    "no_rotation": "NoRotation",
    "full": "Full",
}


def _task_id_parts(variant: ShapeSorterVariantCfg) -> tuple[str, str]:
    task = f"{variant.num_pieces}Shapes"
    if variant.wide_positions:
        return task, "RandomPositions"
    return task, _VARIATION_BY_SPAWN[variant.spawn]


register_category(
    category="Sorter",
    variants=SHAPE_SORTER_VARIANTS.values(),
    make_env_cfg=make_shape_sorter_env_cfg,
    robots=(get_unimanual_dk1_interface(),),
    grasp_obs_terms=(),
    task_id_parts=_task_id_parts,
    max_iterations=10_000,
)
