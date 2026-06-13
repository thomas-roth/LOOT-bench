from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

cfg = load_env_cfg("Mjlab-Scene-Yam", play=True)
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")

breakpoint()

env.close()
