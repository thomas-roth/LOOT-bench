"""Task registration for all insertion variants on all supported robots."""

from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from g3a_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from g3a_bench.tasks.insertion.insertion_env_cfg import (
    ActionMode,
    ObsMode,
    make_insertion_env_cfg,
)
from g3a_bench.tasks.insertion.manipulator_wiring import wire_manipulator_into_env
from g3a_bench.tasks.insertion.rl_cfg import insertion_ppo_runner_cfg
from g3a_bench.tasks.insertion.variants import INSERTION_VARIANTS

_robot_interface = get_unimanual_dk1_interface()
# (action_mode, obs_mode, task-id suffix). EEF + state control is the standard
# mode and keeps the canonical task id; joint (angle-based) control and camera
# (vision) observations are offered additionally under explicit suffixes.
_MODES: tuple[tuple[ActionMode, ObsMode, str], ...] = (
    ("eef", "state", ""),
    ("joint", "state", "-Joint"),
    ("eef", "camera", "-Camera"),
)
for _variant in INSERTION_VARIANTS.values():
    for _action_mode, _obs_mode, _suffix in _MODES:
        _vision = _obs_mode == "camera"
        _env_cfg = make_insertion_env_cfg(
            _variant, action_mode=_action_mode, obs_mode=_obs_mode
        )
        wire_manipulator_into_env(_env_cfg, _robot_interface)
        _play_env_cfg = make_insertion_env_cfg(
            _variant, play=True, action_mode=_action_mode, obs_mode=_obs_mode
        )
        wire_manipulator_into_env(_play_env_cfg, _robot_interface)
        register_mjlab_task(
            task_id=(f"G3a-Insertion-{_variant.name}-{_robot_interface.name}{_suffix}"),
            env_cfg=_env_cfg,
            play_env_cfg=_play_env_cfg,
            rl_cfg=insertion_ppo_runner_cfg(
                f"insertion_{_variant.name.lower()}_"
                f"{_robot_interface.name.lower()}{_suffix.lower()}",
                vision=_vision,
            ),
            runner_cls=ManipulationOnPolicyRunner,
        )
