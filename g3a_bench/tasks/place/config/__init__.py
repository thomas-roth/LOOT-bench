"""Task registration for the place category on all supported robots."""

from g3a_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from g3a_bench.tasks.common.commands import (
    GRIP_CONTACT_SENSOR_NAME,
    OBJECT_CONTACT_SENSOR_NAME,
)
from g3a_bench.tasks.common.registration import register_category
from g3a_bench.tasks.place.place_env_cfg import make_place_env_cfg
from g3a_bench.tasks.place.variants import PLACE_VARIANTS, PlaceVariantCfg


def _task_id_parts(variant: PlaceVariantCfg) -> tuple[str, str]:
    # The reward mode is the category's only variation axis, so it carries the
    # <Variation> segment: G3A-Place-Easy-Dense-Dk1_uni.
    if variant.reward_mode == "progress":
        return "Easy", "Dense"
    return variant.name, ""


register_category(
    category="Place",
    variants=PLACE_VARIANTS.values(),
    make_env_cfg=make_place_env_cfg,
    robots=(get_unimanual_dk1_interface(),),
    action_spaces=("joint", "ik"),
    task_id_parts=_task_id_parts,
    grip_contact_sensor=(GRIP_CONTACT_SENSOR_NAME, OBJECT_CONTACT_SENSOR_NAME),
)
