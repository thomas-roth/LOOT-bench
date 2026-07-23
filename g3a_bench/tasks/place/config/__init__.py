"""Task registration for the place category on all supported robots."""

from g3a_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from g3a_bench.tasks.common.commands import GRIP_CONTACT_SENSOR_NAME
from g3a_bench.tasks.common.registration import register_category
from g3a_bench.tasks.place.place_env_cfg import make_place_env_cfg
from g3a_bench.tasks.place.variants import PLACE_VARIANTS, PlaceVariantCfg


def _task_id_parts(variant: PlaceVariantCfg) -> tuple[str, str]:
    return variant.name, ""


register_category(
    category="Place",
    variants=PLACE_VARIANTS.values(),
    make_env_cfg=make_place_env_cfg,
    robots=(get_unimanual_dk1_interface(),),
    task_id_parts=_task_id_parts,
    grip_contact_sensor=GRIP_CONTACT_SENSOR_NAME,
)
