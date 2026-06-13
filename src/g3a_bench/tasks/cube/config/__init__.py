"""Task registration for all cube variants on all supported robots."""

from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from g3a_bench.robots.interface import get_dk1_interface, get_yam_interface
from g3a_bench.tasks.cube.cube_env_cfg import make_cube_env_cfg
from g3a_bench.tasks.cube.rl_cfg import cube_ppo_runner_cfg
from g3a_bench.tasks.cube.variants import CUBE_VARIANTS

for _get_iface in (get_dk1_interface, get_yam_interface):
    _iface = _get_iface()
    for _variant in CUBE_VARIANTS.values():
        register_mjlab_task(
            task_id=f"G3a-Cube-{_variant.name}-{_iface.name}",
            env_cfg=make_cube_env_cfg(_variant, _iface),
            play_env_cfg=make_cube_env_cfg(_variant, _iface, play=True),
            rl_cfg=cube_ppo_runner_cfg(
                f"cube_{_variant.name.lower()}_{_iface.name.lower()}"
            ),
            runner_cls=ManipulationOnPolicyRunner,
        )
