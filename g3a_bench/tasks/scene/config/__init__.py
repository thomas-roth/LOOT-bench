from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from g3a_bench.robots.interface import get_yam_interface
from g3a_bench.tasks.scene.config.yam.env_cfgs import (
    yam_scene_env_cfg,
    yam_scene_vision_env_cfg,
)
from g3a_bench.tasks.scene.rl_cfg import scene_ppo_runner_cfg

_robot = get_yam_interface()

register_mjlab_task(
    task_id=f"G3a-Scene-{_robot.name}",
    env_cfg=yam_scene_env_cfg(_robot),
    play_env_cfg=yam_scene_env_cfg(_robot, play=True),
    rl_cfg=scene_ppo_runner_cfg(f"scene_{_robot.name.lower()}"),
    runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
    task_id=f"G3a-Scene-{_robot.name}-Rgb",
    env_cfg=yam_scene_vision_env_cfg(_robot, "rgb"),
    play_env_cfg=yam_scene_vision_env_cfg(_robot, "rgb", play=True),
    rl_cfg=scene_ppo_runner_cfg(f"scene_{_robot.name.lower()}_rgb"),
    runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
    task_id=f"G3a-Scene-{_robot.name}-Depth",
    env_cfg=yam_scene_vision_env_cfg(_robot, "depth"),
    play_env_cfg=yam_scene_vision_env_cfg(_robot, "depth", play=True),
    rl_cfg=scene_ppo_runner_cfg(f"scene_{_robot.name.lower()}_depth"),
    runner_cls=ManipulationOnPolicyRunner,
)
