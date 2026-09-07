"""Task registration for the place category on all supported robots."""

from loot_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from loot_bench.tasks.common.commands import (
    GRIP_CONTACT_SENSOR_NAME,
    OBJECT_CONTACT_SENSOR_NAME,
)
from loot_bench.tasks.common.registration import register_category
from loot_bench.tasks.place.place_env_cfg import make_place_env_cfg
from loot_bench.tasks.place.variants import PLACE_VARIANTS, PlaceVariantCfg


def _task_id_parts(variant: PlaceVariantCfg) -> tuple[str, str]:
    # The two randomness axes fill the <task>-<variation> slots:
    # LOOT-Place-RandS-FixedT-Dk1.
    spawn, target = variant.name.split("-")
    return spawn, target


register_category(
    category="Place",
    variants=PLACE_VARIANTS.values(),
    make_env_cfg=make_place_env_cfg,
    robots=(get_unimanual_dk1_interface(),),
    task_id_parts=_task_id_parts,
    grip_contact_sensor=GRIP_CONTACT_SENSOR_NAME,
    object_contact_sensor=OBJECT_CONTACT_SENSOR_NAME,
)
