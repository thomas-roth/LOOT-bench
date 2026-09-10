"""Task registration for all shape-sorter variants on all supported robots."""

from loot_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from loot_bench.tasks.common.commands import GRIP_CONTACT_SENSOR_NAME
from loot_bench.tasks.common.registration import register_category
from loot_bench.tasks.shape_sorter.shape_sorter_env_cfg import make_shape_sorter_env_cfg
from loot_bench.tasks.shape_sorter.variants import (
    SHAPE_SORTER_VARIANTS,
    ShapeSorterVariantCfg,
)

_VARIATION_BY_SPAWN = {
    "fixed": "FixedOrientation",
    "no_rotation": "NoRotation",
    "full": "Full",
}


def task_id_parts(variant: ShapeSorterVariantCfg) -> tuple[str, str]:
    """The ``(task, variation)`` id segments; also read by ``split_demos``."""
    task = f"{variant.num_pieces}Shapes"
    return task, _VARIATION_BY_SPAWN[variant.spawn]


register_category(
    category="Sorter",
    variants=SHAPE_SORTER_VARIANTS.values(),
    make_env_cfg=make_shape_sorter_env_cfg,
    robots=(get_unimanual_dk1_interface(),),
    task_id_parts=task_id_parts,
    grip_contact_sensor=GRIP_CONTACT_SENSOR_NAME,
)
