"""Task registration for all insertion variants on all supported robots."""

from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from g3a_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from g3a_bench.tasks.insertion.insertion_env_cfg import (
    ActionMode,
    make_insertion_env_cfg,
)
from g3a_bench.tasks.insertion.manipulator_wiring import wire_manipulator_into_env
from g3a_bench.tasks.insertion.rl_cfg import insertion_ppo_runner_cfg
from g3a_bench.tasks.insertion.variants import INSERTION_VARIANTS

_robot_interface = get_unimanual_dk1_interface()
# EEF (task-space) control is the standard mode and keeps the canonical task id;
# joint (angle-based) control is offered additionally under a "-Joint" suffix.
_ACTION_MODES: tuple[tuple[ActionMode, str], ...] = (("eef", ""), ("joint", "-Joint"))
for _variant in INSERTION_VARIANTS.values():
    for _mode, _suffix in _ACTION_MODES:
        _env_cfg = make_insertion_env_cfg(_variant, action_mode=_mode)
        wire_manipulator_into_env(_env_cfg, _robot_interface)
        _play_env_cfg = make_insertion_env_cfg(_variant, play=True, action_mode=_mode)
        wire_manipulator_into_env(_play_env_cfg, _robot_interface)
        register_mjlab_task(
            task_id=(f"G3a-Insertion-{_variant.name}-{_robot_interface.name}{_suffix}"),
            env_cfg=_env_cfg,
            play_env_cfg=_play_env_cfg,
            rl_cfg=insertion_ppo_runner_cfg(
                f"insertion_{_variant.name.lower()}_"
                f"{_robot_interface.name.lower()}{_suffix.lower()}"
            ),
            runner_cls=ManipulationOnPolicyRunner,
        )
