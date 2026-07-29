"""Task registration for all stack variants on all supported robots.

The target height is part of the category segment (``Stack3``/``Stack4``/
``Stack5``) and the block shape is the task segment, so the ids read
``G3A-Stack{n}-{Easy,Medium,Hard}-<robot>``.
"""

from g3a_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from g3a_bench.tasks.common.commands import GRIP_CONTACT_SENSOR_NAME
from g3a_bench.tasks.common.registration import register_category
from g3a_bench.tasks.stack.stack_env_cfg import make_stack_env_cfg
from g3a_bench.tasks.stack.variants import STACK_VARIANTS, StackVariantCfg


def _category(variant: StackVariantCfg) -> str:
    return f"Stack{variant.target_blocks}"


def _task_id_parts(variant: StackVariantCfg) -> tuple[str, str]:
    return variant.difficulty, ""


_BY_CATEGORY: dict[str, list[StackVariantCfg]] = {}
for _v in STACK_VARIANTS.values():
    _BY_CATEGORY.setdefault(_category(_v), []).append(_v)

for _category_name, _variants in _BY_CATEGORY.items():
    register_category(
        category=_category_name,
        variants=_variants,
        make_env_cfg=make_stack_env_cfg,
        robots=(get_unimanual_dk1_interface(),),
        task_id_parts=_task_id_parts,
        max_iterations=10_000,
        grip_contact_sensor=GRIP_CONTACT_SENSOR_NAME,
    )
