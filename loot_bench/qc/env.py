"""A single-environment gym adapter over one mjlab task.

``qc`` drives a plain gymnasium env: it resets, steps one action at a time,
and reads ``terminated``/``truncated`` itself. mjlab gives a vectorized env
that returns torch tensors and, by default, resets finished envs in place.
This module is the seam between the two, and nothing more: no reward
shaping, no observation rewriting, no action remapping. Anything that changes
the MDP belongs in the task definition, where the recorder sees it too.

Four things here are load-bearing rather than stylistic:

* **Auto-reset is off.** ``qc``'s evaluation and online loops both call
  ``reset()`` themselves and read the final transition of each episode. With
  mjlab's default auto-reset the observation returned alongside ``done`` is
  the *next* episode's first observation, which would corrupt both the last
  transition pushed into the replay buffer and the info dict the success
  metric is read from.
* **Observation corruption is off.** The offline and online observations have
  to come from one distribution, and the demos are recorded through the
  uncorrupted observation (``ADR-015``).
* **Terminated and truncated are kept apart.** ``terminated`` means the task
  reached an absorbing state, and offline RL uses it to decide whether to
  bootstrap past the final transition. A timeout is not absorbing: the
  episode was cut short, and the value of the state it was cut at is not
  zero. Collapsing the two is the single easiest way to teach an agent that
  running out of time is as good as finishing.
* **Contact forces are observed and reported, never shaped.** The
  ``grip_contact`` reading joins the observation as the width-1
  ``pad_contact_force`` proprioceptive term and gates the release half of
  success, and both contact readings go into ``eval.csv``. Their key names
  are a cross-repo contract that ``qc/evaluation.py`` matches on. See
  :meth:`EpisodeStats.info` for the key shape and :meth:`LootEnv._pad_force`
  for what each sensor measures.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium
import numpy as np
import torch
from gymnasium.spaces import Box
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensor
from mjlab.tasks.registry import load_env_cfg

from loot_bench.qc.dataset import CONVENTION_NAME
from loot_bench.qc.episode import EpisodeStats
from loot_bench.tasks.common.actions import (
    EE_DELTA_ACTION_NAME,
    GRIPPER_ACTION_NAME,
)
from loot_bench.tasks.common.commands import (
    GRIP_CONTACT_SENSOR_NAME,
    OBJECT_CONTACT_SENSOR_NAME,
    pad_contact_force,
)

TASK_PREFIX = "LOOT-"
"""Every task this adapter serves is registered under this prefix.

``qc``'s ``make_env_and_datasets`` dispatches on the env name, so its LOOT
branch keys off this.
"""


def is_loot_task(env_name: str) -> bool:
    """Whether ``env_name`` names a task this adapter can build."""
    return env_name.startswith(TASK_PREFIX)


class LootEnv(gymnasium.Env):
    """One mjlab task, presented as a single-env gymnasium environment.

    Observations are the task's ``actor`` group, flattened to float32.
    Actions are the task's own action space, clipped to ``[-1, 1]``.
    """

    metadata = {"render_modes": [None, "rgb_array"]}

    def __init__(
        self,
        task_id: str,
        *,
        device: str = "cuda",
        seed: int | None = None,
        start_pose_noise: float | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Build the task.

        Args:
            task_id: a registered mjlab task id, e.g. ``LOOT-Stack-1Pi-Dk1``.
            device: torch device for the simulation.
            seed: seed for the reset sampling. mjlab seeds a process-global
                RNG, so seeding one env seeds them all. Build training
                and evaluation envs with different seeds to diverge.
            start_pose_noise: override the task's start-pose sampling range
                [rad]. ``0.0`` gives a bit-identical start every episode,
                which replay verification needs and training does not want.
            render_mode: ``"rgb_array"`` to enable :meth:`render`. Leave unset
                unless frames are actually wanted; it costs a render context.
        """
        cfg = load_env_cfg(task_id, play=False)
        cfg.scene.num_envs = 1
        cfg.auto_reset = False
        # Redundant while no task declares noise; kept as the seam's guarantee.
        cfg.observations["actor"].enable_corruption = False
        if start_pose_noise is not None:
            cfg.events["reset_robot_joints"].params["position_range"] = (
                -start_pose_noise,
                start_pose_noise,
            )

        self.task_id = task_id
        self._start_pose_noise = float(
            cfg.events["reset_robot_joints"].params["position_range"][1]
        )
        self._mjlab_env = ManagerBasedRlEnv(
            cfg=cfg, device=device, render_mode=render_mode
        )
        self.render_mode = render_mode
        if seed is not None:
            self._mjlab_env.seed(seed)

        group_dim = self._mjlab_env.observation_manager.group_obs_dim["actor"]
        obs_dim = int(np.prod(group_dim))
        action_dim = int(self._mjlab_env.action_manager.total_action_dim)
        self.observation_space = Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        self.action_space = Box(-1.0, 1.0, (action_dim,), dtype=np.float32)

        # Ball tasks declare no contact sensors; resolve once per task.
        self._grip_sensor = self._contact_sensor(GRIP_CONTACT_SENSOR_NAME)
        self._object_sensor = self._contact_sensor(OBJECT_CONTACT_SENSOR_NAME)

        self._stats = EpisodeStats()

    @property
    def mjlab_env(self) -> ManagerBasedRlEnv:
        """The mjlab env underneath, for tools that need the simulator.

        Read-only: the handle owns a MuJoCo context and must not be swapped.
        """
        return self._mjlab_env

    def action_convention(self) -> dict[str, Any]:
        """What an action means in this env, as the stamped record.

        Published so a dataset can be checked against the MDP this env
        actually runs (``qc.dataset.check_dataset_matches_env``) rather than
        against a copy of it. A method here, not a property on a term: no
        single term owns the record.
        """
        ee_delta = self._mjlab_env.action_manager.get_term(EE_DELTA_ACTION_NAME)
        gripper = self._mjlab_env.action_manager.get_term(GRIPPER_ACTION_NAME)
        return {
            "convention": CONVENTION_NAME,
            "delta_pos_scale": ee_delta.cfg.delta_pos_scale,
            "delta_ori_scale": ee_delta.cfg.delta_ori_scale,
            "gripper_scale": float(gripper.scale.reshape(-1)[0].item()),
            "gripper_offset": float(gripper.offset.reshape(-1)[0].item()),
            "control_step_s": float(self._mjlab_env.step_dt),
            "start_pose_noise": self._start_pose_noise,
            "task_id": self.task_id,
        }

    def _observation(self, obs: Mapping[str, Any]) -> np.ndarray:
        """The actor group, as a fresh float32 row.

        The copy is deliberate. mjlab hands back a view on simulator memory
        that the next step overwrites, and qc keeps the previous observation
        alive across the step to build its transition. Without a copy, every
        transition would record its own next state as its current state.
        """
        actor = obs["actor"]
        assert isinstance(actor, torch.Tensor)
        return actor[0].detach().cpu().numpy().astype(np.float32, copy=True)

    def _contact_sensor(self, name: str) -> ContactSensor | None:
        """The named contact sensor, or ``None`` if the task declares none."""
        sensor = self._mjlab_env.scene.sensors.get(name)
        return sensor if isinstance(sensor, ContactSensor) else None

    def _pad_force(self, sensor: ContactSensor | None) -> float | None:
        """[N] the strongest finger-pad contact force this sensor sees.

        ``None`` when the task declares no such sensor, which keeps the key
        out of ``info`` entirely rather than reporting a 0.0 that was never
        measured.

        The two sensors differ only in their filter. ``grip_contact`` has no
        secondary match, so it counts contact with **anything except the
        other pad**: ``dk1.xml`` excludes the finger-pad pair, so a gripper
        closed on empty air reads zero here. ``object_contact`` is filtered
        to the scored object and answers the question that one cannot.
        """
        if sensor is None:
            return None
        return float(pad_contact_force(sensor)[0].item())

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, float]]:
        del options  # Unused.
        obs, _ = self._mjlab_env.reset(seed=seed)
        self._stats = EpisodeStats(
            pad_contact_force=self._pad_force(self._grip_sensor),
            object_contact_force=self._pad_force(self._object_sensor),
        )
        return self._observation(obs), self._stats.info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        action_clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        action_tensor = torch.as_tensor(
            action_clipped, dtype=torch.float32, device=self._mjlab_env.device
        ).reshape(1, -1)

        obs, reward, terminated, truncated, _ = self._mjlab_env.step(action_tensor)

        reward_f = float(reward[0].item())
        # Pass through as reported; both can be true when success lands on the
        # last step, and that must survive (no forced truncation).
        terminated_f = bool(terminated[0].item())
        truncated_f = bool(truncated[0].item())

        self._stats.record(
            reward_f,
            self._pad_force(self._grip_sensor),
            self._pad_force(self._object_sensor),
        )

        return (
            self._observation(obs),
            reward_f,
            terminated_f,
            truncated_f,
            self._stats.info(),
        )

    def render(self) -> np.ndarray | None:
        return self._mjlab_env.render()

    def close(self) -> None:
        self._mjlab_env.close()
