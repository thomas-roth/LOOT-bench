"""Task registration for all ball variants on all supported robots."""

from g3a_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from g3a_bench.tasks.ball.ball_env_cfg import REWARD_MODE_TERMS, make_ball_env_cfg
from g3a_bench.tasks.ball.variants import BALL_VARIANTS
from g3a_bench.tasks.common.registration import register_category

register_category(
    category="Ball",
    variants=BALL_VARIANTS.values(),
    make_env_cfg=make_ball_env_cfg,
    robots=(get_unimanual_dk1_interface(),),
    grasp_obs_terms=("ball_rel_grasp",),
    reward_mode_terms=REWARD_MODE_TERMS,
)
