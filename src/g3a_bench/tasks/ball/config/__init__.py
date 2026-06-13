"""Task registration for all ball variants on all supported robots."""

from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from g3a_bench.robots.interface import get_dk1_interface, get_yam_interface
from g3a_bench.tasks.ball.ball_env_cfg import make_ball_env_cfg
from g3a_bench.tasks.ball.rl_cfg import ball_ppo_runner_cfg
from g3a_bench.tasks.ball.variants import BALL_VARIANTS

for _get_iface in (get_dk1_interface, get_yam_interface):
    _iface = _get_iface()
    for _variant in BALL_VARIANTS.values():
        register_mjlab_task(
            task_id=f"G3a-Ball-{_variant.name}-{_iface.name}",
            env_cfg=make_ball_env_cfg(_variant, _iface),
            play_env_cfg=make_ball_env_cfg(_variant, _iface, play=True),
            rl_cfg=ball_ppo_runner_cfg(
                f"ball_{_variant.name.lower()}_{_iface.name.lower()}"
            ),
            runner_cls=ManipulationOnPolicyRunner,
        )
