"""Task registration for all stack variants on all supported robots.

The target height is part of the category segment (``Stack3``/``Stack4``/
``Stack5``) and the block shape is the task segment, so the ids read
``LOOT-Stack{n}-{Easy,Medium,Hard}-<robot>``.
"""

from loot_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from loot_bench.tasks.common.commands import GRIP_CONTACT_SENSOR_NAME
from loot_bench.tasks.common.registration import register_category
from loot_bench.tasks.stack.stack_env_cfg import make_stack_env_cfg
from loot_bench.tasks.stack.variants import STACK_VARIANTS, StackVariantCfg


def category_of(variant: StackVariantCfg) -> str:
    """The category segment; also read by ``scripts/split_demos.py``."""
    return f"Stack{variant.target_blocks}"


def task_id_parts(variant: StackVariantCfg) -> tuple[str, str]:
    """The ``(task, variation)`` id segments; also read by ``split_demos``."""
    return variant.difficulty, ""


_BY_CATEGORY: dict[str, list[StackVariantCfg]] = {}
for _variant in STACK_VARIANTS.values():
    _BY_CATEGORY.setdefault(category_of(_variant), []).append(_variant)

for _category, _variants in _BY_CATEGORY.items():
    register_category(
        category=_category,
        variants=_variants,
        make_env_cfg=make_stack_env_cfg,
        robots=(get_unimanual_dk1_interface(),),
        task_id_parts=task_id_parts,
        grip_contact_sensor=GRIP_CONTACT_SENSOR_NAME,
    )
