from config.yam.env_cfgs import yam_scene_env_cfg
from mjlab.tasks.manipulation.config.yam.rl_cfg import yam_lift_cube_ppo_runner_cfg
from mjlab.tasks.registry import register_mjlab_task

register_mjlab_task(
    task_id="Mjlab-Scene-Yam",
    env_cfg=yam_scene_env_cfg(),
    play_env_cfg=yam_scene_env_cfg(play=True),
    rl_cfg=yam_lift_cube_ppo_runner_cfg(),
    runner_cls=None,
)
