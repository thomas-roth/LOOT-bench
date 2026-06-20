from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from g3a_bench.tasks.scene.config.yam.env_cfgs import (
    yam_scene_env_cfg,
    yam_scene_vision_env_cfg,
)
from g3a_bench.tasks.scene.rl_cfg import scene_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Scene-Yam",
    env_cfg=yam_scene_env_cfg(),
    play_env_cfg=yam_scene_env_cfg(play=True),
    rl_cfg=scene_ppo_runner_cfg("scene_yam"),
    runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Scene-Yam-Rgb",
    env_cfg=yam_scene_vision_env_cfg("rgb"),
    play_env_cfg=yam_scene_vision_env_cfg("rgb", play=True),
    rl_cfg=scene_ppo_runner_cfg("scene_yam_rgb"),
    runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Scene-Yam-Depth",
    env_cfg=yam_scene_vision_env_cfg("depth"),
    play_env_cfg=yam_scene_vision_env_cfg("depth", play=True),
    rl_cfg=scene_ppo_runner_cfg("scene_yam_depth"),
    runner_cls=ManipulationOnPolicyRunner,
)
