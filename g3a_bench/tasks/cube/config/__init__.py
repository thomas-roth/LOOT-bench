"""Task registration for all cube variants on all supported robots."""

from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from g3a_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from g3a_bench.tasks.cube.cube_env_cfg import make_cube_env_cfg
from g3a_bench.tasks.cube.manipulator_wiring import wire_manipulator_into_env
from g3a_bench.tasks.cube.rl_cfg import cube_ppo_runner_cfg
from g3a_bench.tasks.cube.variants import CUBE_VARIANTS

_robot_interface = get_unimanual_dk1_interface()
for _variant in CUBE_VARIANTS.values():
    _env_cfg = make_cube_env_cfg(_variant)
    wire_manipulator_into_env(_env_cfg, _robot_interface)
    _play_env_cfg = make_cube_env_cfg(_variant, play=True)
    wire_manipulator_into_env(_play_env_cfg, _robot_interface)
    register_mjlab_task(
        task_id=f"G3a-Cube-{_variant.name}-{_robot_interface.name}",
        env_cfg=_env_cfg,
        play_env_cfg=_play_env_cfg,
        rl_cfg=cube_ppo_runner_cfg(
            f"cube_{_variant.name.lower()}_{_robot_interface.name.lower()}"
        ),
        runner_cls=ManipulationOnPolicyRunner,
    )
