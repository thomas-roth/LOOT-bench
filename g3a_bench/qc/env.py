"""A single-environment gym adapter over one mjlab task.

``qc`` drives a plain gymnasium env: it resets, steps one action at a time,
and reads ``terminated``/``truncated`` itself. mjlab gives a vectorized env
that returns torch tensors and, by default, resets finished envs in place.
This module is the seam between the two, and nothing more -- no reward
shaping, no observation rewriting, no action remapping. Anything that changes
the MDP belongs in the task definition, where the recorder sees it too.

Use :func:`make_g3a_env`.

Three things here are load-bearing rather than stylistic:

* **Auto-reset is off.** ``qc``'s evaluation and online loops both call
  ``reset()`` themselves and read the final transition of each episode. With
  mjlab's default auto-reset the observation returned alongside ``done`` is
  the *next* episode's first observation, which would corrupt both the last
  transition pushed into the replay buffer and the info dict the success
  metric is read from.
* **Observation corruption is off.** The offline and online observations have
  to come from one distribution, and the demos are recorded through the
  uncorrupted observation. No task declares observation noise any more (see
  ``tasks/common/env_cfg``), so this now costs nothing; the adapter still sets
  the flag, because a cfg that arrives with corruption on must not reach a
  training run.
* **Terminated and truncated are kept apart.** ``terminated`` means the task
  reached an absorbing state, and offline RL uses it to decide whether to
  bootstrap past the final transition. A timeout is not absorbing -- the
  episode was cut short, and the value of the state it was cut at is not
  zero. Collapsing the two is the single easiest way to teach an agent that
  running out of time is as good as finishing.
* **Contact forces are published flat.** ``qc``'s evaluation counts grips out
  of ``info["gripper_contact"]``, and the contact sensors are the only way it
  can see whether a policy ever touched anything. Two are published:
  ``gripper_contact`` (unfiltered, what ``qc`` counts, and a closed gripper
  reports its own pads) and ``object_contact`` (filtered to the scored
  object), with the per-episode totals ``object_contact_steps`` and
  ``object_touched`` beside it. All of it is telemetry, not part of the MDP:
  the numbers go into ``eval.csv``, never into the observation. See
  :meth:`EpisodeStats.info` for why the flat key is the only safe shape, and
  :meth:`G3aEnv._pad_force` for what each one means.
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

from g3a_bench.qc.episode import EpisodeStats
from g3a_bench.tasks.common.commands import (
    GRIP_CONTACT_SENSOR_NAME,
    OBJECT_CONTACT_SENSOR_NAME,
    pad_contact_force,
)

TASK_PREFIX = "G3A-"
"""Every task this adapter serves is registered under this prefix.

``qc``'s ``make_env_and_datasets`` dispatches on the env name, so its G3A
branch keys off this.
"""


def is_g3a_task(env_name: str) -> bool:
    """Whether ``env_name`` names a task this adapter can build."""
    return env_name.startswith(TASK_PREFIX)


class G3aEnv(gymnasium.Env):
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
            task_id: a registered mjlab task id, e.g. ``G3A-Stack-Low-Dk1_uni``.
            device: torch device for the simulation.
            seed: seed for the reset sampling. mjlab seeds a process-global
                RNG, so seeding one env seeds them all -- build the training
                and evaluation envs with different seeds if you want them to
                diverge.
            start_pose_noise: override the task's start-pose sampling range
                [rad]. ``0.0`` gives a bit-identical start every episode,
                which replay verification needs and training does not want.
            render_mode: ``"rgb_array"`` to enable :meth:`render`. Leave unset
                unless frames are actually wanted; it costs a render context.
        """
        cfg = load_env_cfg(task_id, play=False)
        cfg.scene.num_envs = 1
        cfg.auto_reset = False
        # Redundant while no task declares observation noise, and kept as the
        # seam's own guarantee -- see the module docstring.
        cfg.observations["actor"].enable_corruption = False
        if start_pose_noise is not None:
            cfg.events["reset_robot_joints"].params["position_range"] = (
                -start_pose_noise,
                start_pose_noise,
            )

        self.task_id = task_id
        self._env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=render_mode)
        self.render_mode = render_mode
        if seed is not None:
            self._env.seed(seed)

        group_dim = self._env.observation_manager.group_obs_dim["actor"]
        obs_dim = int(np.prod(group_dim))
        action_dim = int(self._env.action_manager.total_action_dim)
        self.observation_space = Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        self.action_space = Box(-1.0, 1.0, (action_dim,), dtype=np.float32)

        # Resolved once: the categories that grasp declare these sensors, ball
        # does not, and whether a key appears in ``info`` has to be a property
        # of the task rather than of the step.
        self._grip_sensor = self._contact_sensor(GRIP_CONTACT_SENSOR_NAME)
        self._object_sensor = self._contact_sensor(OBJECT_CONTACT_SENSOR_NAME)

        self._stats = EpisodeStats()

    @property
    def unwrapped_env(self) -> ManagerBasedRlEnv:
        """The mjlab env underneath, for tools that need the simulator."""
        return self._env

    def action_term(self):
        """The joint action term, which owns the scale and offset.

        Published so a dataset can be checked against the convention this env
        actually runs (``qc.convert.check_dataset_matches_env``) rather than
        against a copy of it.
        """
        return self._env.action_manager.get_term("joint_pos")

    def _observation(self, obs: Mapping[str, Any]) -> np.ndarray:
        """The actor group, as a fresh float32 row.

        The copy is deliberate. mjlab hands back a view on simulator memory
        that the next step overwrites, and qc keeps the previous observation
        alive across the step to build its transition -- without a copy, every
        transition would record its own next state as its current state.
        """
        actor = obs["actor"]
        assert isinstance(actor, torch.Tensor)
        return actor[0].detach().cpu().numpy().astype(np.float32, copy=True)

    def _contact_sensor(self, name: str) -> ContactSensor | None:
        """The named contact sensor, or ``None`` if the task declares none."""
        sensor = self._env.scene.sensors.get(name)
        return sensor if isinstance(sensor, ContactSensor) else None

    def _pad_force(self, sensor: ContactSensor | None) -> float | None:
        """[N] the strongest finger-pad contact force this sensor sees.

        ``None`` when the task declares no such sensor, which keeps the key
        out of ``info`` entirely rather than reporting a 0.0 that was never
        measured.

        The two sensors differ only in their filter, and the difference
        matters. ``grip_contact`` has no secondary match, so it counts contact
        with **anything, the other pad included**: on the DK1 the pad faces
        meet at gripper joint -0.038 m while the joint travels to -0.045 m, so
        a gripper closed on empty air reads 1.6 to 2.5 N. ``object_contact``
        is filtered to the scored object and answers the question that one
        cannot.
        """
        if sensor is None:
            return None
        return float(pad_contact_force(sensor)[0].item())

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, float]]:
        del options  # Unused.
        obs, _ = self._env.reset(seed=seed)
        self._stats = EpisodeStats(
            gripper_contact=self._pad_force(self._grip_sensor),
            object_contact=self._pad_force(self._object_sensor),
        )
        return self._observation(obs), self._stats.info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        tensor = torch.as_tensor(
            clipped, dtype=torch.float32, device=self._env.device
        ).reshape(1, -1)

        obs, reward, terminated, truncated, _ = self._env.step(tensor)

        reward_f = float(reward[0].item())
        # Passed through as mjlab reports them. Its termination manager already
        # routes each term by its own ``time_out`` flag, so the two buffers are
        # disjoint by construction and need no untangling here. Both can be
        # true on one step -- a task that succeeds on its very last step -- and
        # that has to survive: forcing truncation in that case would tell the
        # learner to bootstrap past a genuine absorbing state.
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
        return self._env.render()

    def close(self) -> None:
        self._env.close()


def make_g3a_env(
    task_id: str,
    *,
    device: str = "cuda",
    seed: int | None = None,
    start_pose_noise: float | None = None,
    render_mode: str | None = None,
) -> G3aEnv:
    """Build a :class:`G3aEnv`. See its constructor for the arguments."""
    return G3aEnv(
        task_id,
        device=device,
        seed=seed,
        start_pose_noise=start_pose_noise,
        render_mode=render_mode,
    )
