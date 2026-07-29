"""Task registration for all domino variants on all supported robots."""

from g3a_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from g3a_bench.tasks.common.commands import GRIP_CONTACT_SENSOR_NAME
from g3a_bench.tasks.common.registration import register_category
from g3a_bench.tasks.domino.domino_env_cfg import make_domino_env_cfg
from g3a_bench.tasks.domino.variants import DOMINO_VARIANTS, DominoVariantCfg


def _task_id_parts(variant: DominoVariantCfg) -> tuple[str, str]:
    task = f"{variant.num_dominoes}Pieces"
    if variant.random_row:
        return task, "RandomRow"
    if variant.different:
        return task, "DifferentDominoes"
    if variant.knock_over:
        return task, "KnockOver"
    return task, ""


register_category(
    category="Domino",
    variants=DOMINO_VARIANTS.values(),
    make_env_cfg=make_domino_env_cfg,
    robots=(get_unimanual_dk1_interface(),),
    task_id_parts=_task_id_parts,
    max_iterations=10_000,
    grip_contact_sensor=GRIP_CONTACT_SENSOR_NAME,
)
