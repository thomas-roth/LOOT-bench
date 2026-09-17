"""Task registration for all pi variants on all supported robots.

The category segment stays ``Stack`` and the level count is the task
segment, so the ids read ``LOOT-Stack-{1Pi,2Pis,3Pis}-<robot>``.
"""

from loot_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from loot_bench.tasks.common.commands import GRIP_CONTACT_SENSOR_NAME
from loot_bench.tasks.common.registration import register_category
from loot_bench.tasks.stack.stack_env_cfg import make_pi_env_cfg
from loot_bench.tasks.stack.variants import PI_VARIANTS, PiVariantCfg


def task_id_parts(variant: PiVariantCfg) -> tuple[str, str]:
    """The ``(task, variation)`` id segments; also read by ``split_demos``.

    The singular/plural split (``1Pi``, ``2Pis``) lives here and
    nowhere else, so the splitter cannot re-derive it differently.
    """
    n = variant.levels
    return (f"{n}Pi" if n == 1 else f"{n}Pis"), ""


register_category(
    category="Stack",
    variants=PI_VARIANTS.values(),
    make_env_cfg=make_pi_env_cfg,
    robots=(get_unimanual_dk1_interface(),),
    task_id_parts=task_id_parts,
    grip_contact_sensor=GRIP_CONTACT_SENSOR_NAME,
)
