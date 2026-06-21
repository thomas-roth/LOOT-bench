import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg


class Simulator:
    def __init__(
        self,
        env_cfg: ManagerBasedRlEnvCfg,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        render_mode: str | None = None,
    ):
        self.env = ManagerBasedRlEnv(
            cfg=env_cfg, device=device, render_mode=render_mode
        )

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, action: np.ndarray | torch.Tensor):
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.env.device)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        obs, reward, terminated, time_out, extras = self.env.step(action)
        done = (terminated | time_out).bool()
        return obs, float(reward[0].cpu()), bool(done[0].cpu()), extras

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
