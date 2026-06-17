from .env_cfgs import *
from .rl_cfg import *

"""Register local Peg Insertion tasks in the mjlab task registry."""

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from .env_cfgs import (
    peg_insertion_env_cfg
)
from .rl_cfg import (
    peg_insertion_ppo_runner_cfg
)

if "Mjlab-PegInsertion" not in list_tasks():
  register_mjlab_task(
    task_id="Mjlab-PegInsertion",
    env_cfg=peg_insertion_env_cfg(),
    play_env_cfg=peg_insertion_env_cfg(play=True),
    rl_cfg=peg_insertion_ppo_runner_cfg(),
  )
