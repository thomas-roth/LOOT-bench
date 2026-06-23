from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import g3a_bench  # noqa: F401

cfg = load_env_cfg("G3a-Scene-Yam", play=True)
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")

breakpoint()

env.close()
