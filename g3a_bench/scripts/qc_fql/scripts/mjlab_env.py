"""Gym-style single-env adapter over a real mjlab task (G3a tasks, DK1 unimanual).

The training/eval loops (ported from a repo whose envs are gym-style) expect
flat numpy observations, scalar rewards, and an explicit `reset()` after
`done`. mjlab's `ManagerBasedRlEnv` is none of that: batched torch tensors,
dict-of-groups observations, and internal auto-reset. This file is the ONLY
place those two worlds meet -- it has no reference-repo counterpart and no
gold data; its verification is running it against the real env (see
tests/test_mjlab_env.py, which constructs the actual task).

Three impedance mismatches this class absorbs, each checked against the
live env rather than assumed:

1. Observations: G3a tasks expose two groups, {"actor": (1,D), "critic":
   (1,D)} -- same terms, but "actor" has observation corruption enabled.
   Default here is "actor": train/eval on what a deployed policy would see.

2. Actions: the policy emits [-1, 1]; mjlab applies `target = raw * scale +
   home` with per-joint scales spanning ~3 orders of magnitude, and nothing
   in mjlab clips or normalizes. Denormalization uses
   `mapper.PHYSICAL_RAW_ACTION_MIN/MAX` -- the DK1's full joint range in
   raw-action space, a robot constant, and the same convention
   `lerobot_to_qc.py` uses (its default "physical" mode) to normalize the
   demos. Policy and dataset therefore agree on what [-1, 1] means. Checked
   live: the env's action term orders joints exactly as mapper's constants
   do (joint1..joint6, gripper_left), with identical scale/offset values.
   DK1-unimanual-specific: other robots need their own constants.

3. Resets: with mjlab's default auto_reset=True, the observation `step()`
   returns on a done step is already the NEXT episode's first observation
   (the true terminal observation is never surfaced -- fine for training,
   since masks=1-terminated already excludes terminal bootstraps). The
   gym-style callers still call `reset()` after done, which would trigger a
   redundant second reset and skip an episode's initial state; instead the
   auto-reset observation is cached and handed back by the next `reset()`.
"""

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from g3a_bench.scripts.live_bridge.mapper import ActionNormalizer


class MjlabSingleEnv:
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
        self._normalizer = ActionNormalizer.physical()

    def _extract_obs(self, obs_dict: dict) -> np.ndarray:
        # [0]: strip the num_envs=1 batch dim mjlab always keeps.
        return obs_dict[self.obs_group][0].detach().cpu().numpy()

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Policy's [-1, 1] -> mjlab raw-action units over the DK1 joint range."""
        return self._normalizer.denormalize(action)

    def _get_success(self) -> float | None:
        # mjlab has no framework-level success flag; every G3a task tracks it
        # on its "task" command term. Returns None (key omitted from info) for
        # tasks without this term rather than failing.
        try:
            command = self.env.command_manager.get_term("task")
            return float(command.episode_success[0].item())
        except (KeyError, AttributeError):
            return None

    def reset(self) -> tuple[np.ndarray, dict]:
        if self._pending_reset_obs is not None:
            # Episode start already produced by mjlab's auto-reset (see
            # module docstring, point 3) -- consume it instead of resetting
            # a second time.
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
        # mjlab separates terminations from time-outs exactly along gym's
        # terminated/truncated line (the task's `time_out` term is the only
        # one flagged Time Out=True).
        terminated_flag = bool(terminated[0].item())
        truncated_flag = bool(time_outs[0].item())

        info = {}
        success = self._get_success()
        if success is not None:
            info["success"] = success

        if terminated_flag or truncated_flag:
            self._pending_reset_obs = next_obs

        return next_obs, float(reward[0].item()), terminated_flag, truncated_flag, info

    def close(self) -> None:
        self.env.close()
