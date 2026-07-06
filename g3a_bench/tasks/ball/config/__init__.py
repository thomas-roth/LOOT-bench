"""Task registration for all ball variants on all supported robots."""

from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from g3a_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from g3a_bench.tasks.ball.ball_env_cfg import make_ball_env_cfg
from g3a_bench.tasks.ball.rl_cfg import ball_ppo_runner_cfg
from g3a_bench.tasks.ball.variants import BALL_VARIANTS
from g3a_bench.tasks.common.manipulator_wiring import wire_manipulator_into_env

_GRASP_OBS_TERMS = ("ball_rel_grasp",)

_robot_interface = get_unimanual_dk1_interface()
for _variant in BALL_VARIANTS.values():
    _env_cfg = make_ball_env_cfg(_variant)
    wire_manipulator_into_env(
        _env_cfg, _robot_interface, grasp_obs_terms=_GRASP_OBS_TERMS
    )
    _play_env_cfg = make_ball_env_cfg(_variant, play=True)
    wire_manipulator_into_env(
        _play_env_cfg, _robot_interface, grasp_obs_terms=_GRASP_OBS_TERMS
    )
    register_mjlab_task(
        task_id=f"G3a-Ball-{_variant.name}-{_robot_interface.name}",
        env_cfg=_env_cfg,
        play_env_cfg=_play_env_cfg,
        rl_cfg=ball_ppo_runner_cfg(
            f"ball_{_variant.name.lower()}_{_robot_interface.name.lower()}"
        ),
        runner_cls=ManipulationOnPolicyRunner,
    )
