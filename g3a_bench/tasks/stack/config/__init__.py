"""Task registration for all stack variants on all supported robots."""

from g3a_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from g3a_bench.tasks.common.registration import register_category
from g3a_bench.tasks.stack.stack_env_cfg import make_stack_env_cfg
from g3a_bench.tasks.stack.variants import STACK_VARIANTS, StackVariantCfg

_TASK_BY_TARGET_BLOCKS = {3: "Low", 4: "Medium", 5: "High"}


def _task_id_parts(variant: StackVariantCfg) -> tuple[str, str]:
    task = _TASK_BY_TARGET_BLOCKS[variant.target_blocks]
    if variant.two_stacks:
        return task, "TwoStacks"
    if any(b.form != "cube" for b in variant.blocks):
        return task, "DifferentBlocks"
    return task, ""


register_category(
    category="Stack",
    variants=STACK_VARIANTS.values(),
    make_env_cfg=make_stack_env_cfg,
    robots=(get_unimanual_dk1_interface(),),
    grasp_obs_terms=("blocks_rel_grasp",),
    task_id_parts=_task_id_parts,
    max_iterations=10_000,
)
