"""Register local Peg Insertion tasks in the mjlab task registry."""

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from .env_cfgs import *  # noqa: F403  (re-export task cfgs at package level)
from .env_cfgs import peg_insertion_env_cfg
from .rl_cfg import *  # noqa: F403
from .rl_cfg import peg_insertion_ppo_runner_cfg

# MJlab task registry raises if a task is registered twice, so guard on the
# task id before registering.
if "Mjlab-PegInsertion" not in list_tasks():
    register_mjlab_task(
        task_id="Mjlab-PegInsertion",
        env_cfg=peg_insertion_env_cfg(),
        play_env_cfg=peg_insertion_env_cfg(play=True),
        rl_cfg=peg_insertion_ppo_runner_cfg(),
    )
