"""Single-env gym-style adapter over mjlab's `ManagerBasedRlEnv` (cube tasks, DK1 unimanual).

No JAX/PyTorch reference exists for this file -- it's genuinely new code, not
a port, and can't be verified by `torch.allclose`. Every fact this design
relies on was checked directly against the real, running env and the real
`qc_fql` code, not assumed:

- Observations: `env.reset()`/`env.step()` return `{"actor": Tensor(1,53),
  "critic": Tensor(1,53)}` for the cube category -- confirmed live. This
  adapter uses the `"actor"` group (noisy/corrupted), matching what a
  deployed policy would actually perceive.
- Actions: the real `joint_pos` action term's `target_names` is exactly
  `['joint1',...,'joint6','gripper_left']`, and its `scale`/`offset` tensors
  are bit-identical to `g3a_bench.scripts.live_bridge.mapper`'s
  `ACTION_SCALE`/`HOME_JOINT_POS` -- confirmed live, not assumed. So
  `mapper.PHYSICAL_RAW_ACTION_MIN/MAX` (the fixed DK1 joint range in
  raw_action space -- a robot constant, independent of any specific
  recorded dataset) can be used directly to un-normalize a QC-FQL policy's
  `[-1,1]` output back to raw units before calling `env.step()`.
- `auto_reset=True` (mjlab's default, kept per explicit decision): confirmed
  live that when an episode ends, the observation `step()` returns is
  already the FRESH post-reset observation of the next episode, not the
  true terminal one (`masks=1-terminated` already neutralizes this
  correctly for bootstrapping, matching qc's own convention). Since
  `evaluate.py`/`train_online.py` are written against a standard gym
  contract (call `.reset()` explicitly after `done`), and mjlab already
  auto-reset, an unconditional external `.reset()` call would trigger a
  redundant *second* reset. This adapter caches the auto-reset observation
  and returns it from the next `.reset()` call instead of resetting again --
  contained entirely in this file, so `evaluate.py`/`train_online.py` and
  their existing synthetic-env tests need no changes.
- `terminated`/`time_outs` map directly to gym's `terminated`/`truncated`
  (confirmed: `time_out` is a distinct termination term with `Time Out=True`,
  `success`/`ee_ground_collision` are `Time Out=False`).
- Success signal: no framework-level "success" key exists in `extras`. Cube
  tasks expose it via `env.command_manager.get_term("arrangement")
  .episode_success` (confirmed live) -- included in `info["success"]` so
  `evaluate.py`'s existing generic `flatten(info)` stat aggregation picks it
  up with no changes needed there either.
"""

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from g3a_bench.scripts.live_bridge.mapper import (
    PHYSICAL_RAW_ACTION_MAX,
    PHYSICAL_RAW_ACTION_MIN,
)


class MjlabSingleEnv:
    """Wraps a single-env (`num_envs=1`) mjlab cube task as a gym-style env."""

    def __init__(
        self,
        task_id: str,
        obs_group: str = "actor",
        device: str = "cpu",
        play: bool = True,
    ):
        env_cfg = load_env_cfg(task_id, play=play)
        env_cfg.scene.num_envs = 1
        self.env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
        self.obs_group = obs_group
        self._pending_reset_obs: np.ndarray | None = None

    def _extract_obs(self, obs_dict: dict) -> np.ndarray:
        return obs_dict[self.obs_group][0].detach().cpu().numpy()

    @staticmethod
    def _denormalize_action(action: np.ndarray) -> np.ndarray:
        """[-1, 1] -> raw physical units, using the fixed DK1 joint range."""
        clipped = np.clip(action, -1.0, 1.0)
        return (clipped + 1.0) / 2.0 * (
            PHYSICAL_RAW_ACTION_MAX - PHYSICAL_RAW_ACTION_MIN
        ) + PHYSICAL_RAW_ACTION_MIN

    def _get_success(self) -> float | None:
        try:
            command = self.env.command_manager.get_term("arrangement")
            return float(command.episode_success[0].item())
        except (KeyError, AttributeError):
            return None

    def reset(self) -> tuple[np.ndarray, dict]:
        if self._pending_reset_obs is not None:
            obs = self._pending_reset_obs
            self._pending_reset_obs = None
            return obs, {}
        obs_dict, _info = self.env.reset()
        return self._extract_obs(obs_dict), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        raw_action = self._denormalize_action(action)
        action_t = torch.from_numpy(raw_action).float().unsqueeze(0).to(self.env.device)

        obs_dict, reward, terminated, time_outs, _extras = self.env.step(action_t)

        next_obs = self._extract_obs(obs_dict)
        terminated_flag = bool(terminated[0].item())
        truncated_flag = bool(time_outs[0].item())

        info = {}
        success = self._get_success()
        if success is not None:
            info["success"] = success

        if terminated_flag or truncated_flag:
            # mjlab already auto-reset internally -- `next_obs` IS the fresh
            # start of the next episode. Cache it so a caller's subsequent
            # `.reset()` (expected by the gym-style contract) doesn't trigger
            # a redundant second reset.
            self._pending_reset_obs = next_obs

        return next_obs, float(reward[0].item()), terminated_flag, truncated_flag, info

    def close(self) -> None:
        self.env.close()
