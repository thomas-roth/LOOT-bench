from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from g3a_bench.tasks.cleaning_surface.config.yam.env_cfgs import (
    yam_cleaning_env_cfg,
)
from g3a_bench.tasks.cleaning_surface.config.yam.rl_config import (
    cleaning_yam_ppo_runner_cfg,
)

register_mjlab_task(
    task_id="Mjlab-CleaningSurface-Yam",
    env_cfg=yam_cleaning_env_cfg(),
    play_env_cfg=yam_cleaning_env_cfg(play=True),
    rl_cfg=cleaning_yam_ppo_runner_cfg(),
    runner_cls=ManipulationOnPolicyRunner,
)
