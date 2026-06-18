"""Task registration for all insertion variants on all supported robots."""

from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from g3a_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from g3a_bench.tasks.insertion.insertion_env_cfg import make_insertion_env_cfg
from g3a_bench.tasks.insertion.rl_cfg import insertion_ppo_runner_cfg
from g3a_bench.tasks.insertion.variants import INSERTION_VARIANTS

_robot_interface = get_unimanual_dk1_interface()
for _variant in INSERTION_VARIANTS.values():
    register_mjlab_task(
        task_id=f"G3a-Insertion-{_variant.name}-{_robot_interface.name}",
        env_cfg=make_insertion_env_cfg(_variant, _robot_interface),
        play_env_cfg=make_insertion_env_cfg(_variant, _robot_interface, play=True),
        rl_cfg=insertion_ppo_runner_cfg(
            f"insertion_{_variant.name.lower()}_{_robot_interface.name.lower()}"
        ),
        runner_cls=ManipulationOnPolicyRunner,
    )
