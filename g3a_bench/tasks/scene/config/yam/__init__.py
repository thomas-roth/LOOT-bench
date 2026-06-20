from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from g3a_bench.tasks.scene.config.yam.env_cfgs import yam_scene_env_cfg
from g3a_bench.tasks.scene.config.yam.rl_config import scene_yam_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Scene-Yam",
    env_cfg=yam_scene_env_cfg(),
    play_env_cfg=yam_scene_env_cfg(play=True),
    rl_cfg=scene_yam_ppo_runner_cfg(),
    runner_cls=ManipulationOnPolicyRunner,
)
