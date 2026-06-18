"""Task registration for all ball variants on all supported robots."""

from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from g3a_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from g3a_bench.tasks.ball.ball_env_cfg import make_ball_env_cfg
from g3a_bench.tasks.ball.rl_cfg import ball_ppo_runner_cfg
from g3a_bench.tasks.ball.variants import BALL_VARIANTS

_robot_interface = get_unimanual_dk1_interface()
for _variant in BALL_VARIANTS.values():
    register_mjlab_task(
        task_id=f"G3a-Ball-{_variant.name}-{_robot_interface.name}",
        env_cfg=make_ball_env_cfg(_variant, _robot_interface),
        play_env_cfg=make_ball_env_cfg(_variant, _robot_interface, play=True),
        rl_cfg=ball_ppo_runner_cfg(
            f"ball_{_variant.name.lower()}_{_robot_interface.name.lower()}"
        ),
        runner_cls=ManipulationOnPolicyRunner,
    )
