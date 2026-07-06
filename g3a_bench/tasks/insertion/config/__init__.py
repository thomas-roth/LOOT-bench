"""Task registration for all insertion variants on all supported robots."""

from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from g3a_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from g3a_bench.tasks.common.manipulator_wiring import wire_manipulator_into_env
from g3a_bench.tasks.insertion.insertion_env_cfg import make_insertion_env_cfg
from g3a_bench.tasks.insertion.rl_cfg import insertion_ppo_runner_cfg
from g3a_bench.tasks.insertion.variants import INSERTION_VARIANTS

# "sheet_rel_grasp" only exists in two_sheet variants; absent terms are skipped.
_GRASP_OBS_TERMS = ("pegs_rel_grasp", "sheet_rel_grasp")
_EE_CONTACT_SENSORS = ("ee_werkbox_collision",)

_robot_interface = get_unimanual_dk1_interface()
for _variant in INSERTION_VARIANTS.values():
    _env_cfg = make_insertion_env_cfg(_variant)
    wire_manipulator_into_env(
        _env_cfg,
        _robot_interface,
        grasp_obs_terms=_GRASP_OBS_TERMS,
        ee_contact_sensors=_EE_CONTACT_SENSORS,
    )
    _play_env_cfg = make_insertion_env_cfg(_variant, play=True)
    wire_manipulator_into_env(
        _play_env_cfg,
        _robot_interface,
        grasp_obs_terms=_GRASP_OBS_TERMS,
        ee_contact_sensors=_EE_CONTACT_SENSORS,
    )
    register_mjlab_task(
        task_id=f"G3a-Insertion-{_variant.name}-{_robot_interface.name}",
        env_cfg=_env_cfg,
        play_env_cfg=_play_env_cfg,
        rl_cfg=insertion_ppo_runner_cfg(
            f"insertion_{_variant.name.lower()}_{_robot_interface.name.lower()}"
        ),
        runner_cls=ManipulationOnPolicyRunner,
    )
