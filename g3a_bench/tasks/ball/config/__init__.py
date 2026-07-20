"""Task registration for all ball variants on all supported robots."""

from g3a_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from g3a_bench.tasks.ball.ball_env_cfg import make_ball_env_cfg
from g3a_bench.tasks.ball.variants import BALL_VARIANTS, BallVariantCfg
from g3a_bench.tasks.common.registration import register_category

_MINIGOLF_VARIATION_BY_LAYOUT = {
    "direct": "Direct",
    "using_wall": "UsingWall",
    "obstacle": "Obstacle",
    "ramp": "Ramp",
}
_THROW_VARIATION_BY_TARGET = {"u": "U", "marked": "MarkedArea"}


def _task_id_parts(variant: BallVariantCfg) -> tuple[str, str]:
    if variant.task == "minigolf":
        return "Minigolf", _MINIGOLF_VARIATION_BY_LAYOUT[variant.minigolf_layout]
    if variant.movable_wall:
        return "Throw", "WallVariation"
    return "Throw", _THROW_VARIATION_BY_TARGET[variant.throw_target]


register_category(
    category="Ball",
    variants=BALL_VARIANTS.values(),
    make_env_cfg=make_ball_env_cfg,
    robots=(get_unimanual_dk1_interface(),),
    task_id_parts=_task_id_parts,
)
