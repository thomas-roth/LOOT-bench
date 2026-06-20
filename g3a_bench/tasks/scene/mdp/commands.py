"""Scene task command: multi-task sampling, object placement, goal tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from g3a_bench.tasks.scene.mdp.rewards import scene_success
from g3a_bench.tasks.scene.mdp.utils import (
    detect_new_button_presses,
    get_button_positions,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

# ---- OGBench task definitions (from SceneEnv.set_tasks()) ----
# Cube positions are relative to env_origin (robot base frame).
# Drawer range: [-0.16, 0], window range: [0, 0.2].
_SCENE_TASKS = [
    {
        "name": "task1_open",
        "init": {
            "cube_pos": [0.35, 0.05, 0.02],
            "button_states": [1.0, 1.0],
            "drawer_pos": 0.0,
            "window_pos": 0.0,
        },
        "goal": {
            "cube_pos": [0.35, 0.05, 0.02],
            "button_states": [1.0, 1.0],
            "drawer_pos": -0.16,
            "window_pos": 0.2,
        },
    },
    {
        "name": "task2_unlock_and_lock",
        "init": {
            "cube_pos": [0.35, -0.05, 0.02],
            "button_states": [0.0, 0.0],
            "drawer_pos": -0.16,
            "window_pos": 0.2,
        },
        "goal": {
            "cube_pos": [0.35, -0.05, 0.02],
            "button_states": [0.0, 0.0],
            "drawer_pos": 0.0,
            "window_pos": 0.0,
        },
    },
    {
        "name": "task3_rearrange_medium",
        "init": {
            "cube_pos": [0.4, -0.05, 0.02],
            "button_states": [1.0, 0.0],
            "drawer_pos": 0.0,
            "window_pos": 0.2,
        },
        "goal": {
            "cube_pos": [0.4, 0.15, 0.02],
            "button_states": [1.0, 1.0],
            "drawer_pos": -0.16,
            "window_pos": 0.0,
        },
    },
    {
        "name": "task4_put_in_drawer",
        "init": {
            "cube_pos": [0.35, 0.05, 0.02],
            "button_states": [0.0, 0.0],
            "drawer_pos": 0.0,
            "window_pos": 0.0,
        },
        "goal": {
            "cube_pos": [0.33, -0.356, 0.065986],
            "button_states": [1.0, 0.0],
            "drawer_pos": 0.0,
            "window_pos": 0.0,
        },
    },
    {
        "name": "task5_rearrange_hard",
        "init": {
            "cube_pos": [0.35, 0.15, 0.02],
            "button_states": [0.0, 0.0],
            "drawer_pos": 0.0,
            "window_pos": 0.0,
        },
        "goal": {
            "cube_pos": [0.33, -0.356, 0.065986],
            "button_states": [0.0, 0.0],
            "drawer_pos": 0.0,
            "window_pos": 0.2,
        },
    },
]

_NUM_TASKS = len(_SCENE_TASKS)


@dataclass(kw_only=True)
class SceneTaskCommandCfg(CommandTermCfg):
    def build(self, env: ManagerBasedRlEnv) -> SceneTaskCommand:
        return SceneTaskCommand(self, env)


class SceneTaskCommand(CommandTerm):
    cfg: SceneTaskCommandCfg

    def __init__(self, cfg: SceneTaskCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        dev = self.device
        E = self.num_envs

        # Pre-build task tensors for vectorized lookup.
        self._init_cube_pos = torch.tensor(
            [t["init"]["cube_pos"] for t in _SCENE_TASKS],
            dtype=torch.float32,
            device=dev,
        )  # (T, 3)
        self._init_button_states = torch.tensor(
            [t["init"]["button_states"] for t in _SCENE_TASKS],
            dtype=torch.float32,
            device=dev,
        )  # (T, 2)
        self._init_drawer_pos = torch.tensor(
            [t["init"]["drawer_pos"] for t in _SCENE_TASKS],
            dtype=torch.float32,
            device=dev,
        )  # (T,)
        self._init_window_pos = torch.tensor(
            [t["init"]["window_pos"] for t in _SCENE_TASKS],
            dtype=torch.float32,
            device=dev,
        )  # (T,)

        self._goal_cube_pos = torch.tensor(
            [t["goal"]["cube_pos"] for t in _SCENE_TASKS],
            dtype=torch.float32,
            device=dev,
        )  # (T, 3)
        self._goal_button_states = torch.tensor(
            [t["goal"]["button_states"] for t in _SCENE_TASKS],
            dtype=torch.float32,
            device=dev,
        )  # (T, 2)
        self._goal_drawer_pos = torch.tensor(
            [t["goal"]["drawer_pos"] for t in _SCENE_TASKS],
            dtype=torch.float32,
            device=dev,
        )  # (T,)
        self._goal_window_pos = torch.tensor(
            [t["goal"]["window_pos"] for t in _SCENE_TASKS],
            dtype=torch.float32,
            device=dev,
        )  # (T,)

        # Entity references (available after env construction).
        self._cube = env.scene["cube"]
        self._drawer = env.scene["drawer"]
        self._window = env.scene["window"]

        # Local joint slot indices within each entity.
        self._drawer_joint_slot = torch.tensor(
            [list(self._drawer.joint_names).index("drawer_slide")],
            dtype=torch.long,
            device=dev,
        )
        self._window_joint_slot = torch.tensor(
            [list(self._window.joint_names).index("window_slide")],
            dtype=torch.long,
            device=dev,
        )

        # Target tensors — stored on the command, accessed via get_term("scene_task").
        self.target_cube_pos = torch.zeros(E, 3, device=dev)
        self.target_button_states = torch.zeros(E, 2, device=dev)
        self.target_drawer_pos = torch.zeros(E, device=dev)
        self.target_window_pos = torch.zeros(E, device=dev)

        # Task state — owned by the command (not monkey-patched onto env).
        self.scene_button_states = torch.zeros(E, 2, device=dev)
        self.prev_button_positions = torch.zeros(E, 2, device=dev)
        self.drawer_lock_target = torch.zeros(E, device=dev)
        self.window_lock_target = torch.zeros(E, device=dev)

        self.episode_success = torch.zeros(E, device=dev)
        self.metrics["episode_success"] = torch.zeros(E, device=dev)

    @property
    def command(self) -> torch.Tensor:
        """Current goal vector. Shape: (num_envs, 7)."""
        return torch.cat(
            [
                self.target_cube_pos,  # (E, 3)
                self.target_button_states,  # (E, 2)
                self.target_drawer_pos.unsqueeze(-1),  # (E, 1)
                self.target_window_pos.unsqueeze(-1),  # (E, 1)
            ],
            dim=-1,
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        dev = self.device
        env = self._env

        task_ids = torch.randint(0, _NUM_TASKS, (n,), device=dev)
        origins = env.scene.env_origins[env_ids]  # (n, 3)

        # ---- Goal targets ----
        self.target_cube_pos[env_ids] = self._goal_cube_pos[task_ids] + origins
        self.target_button_states[env_ids] = self._goal_button_states[task_ids]
        self.target_drawer_pos[env_ids] = self._goal_drawer_pos[task_ids]
        self.target_window_pos[env_ids] = self._goal_window_pos[task_ids]

        # ---- Place cube at init position with ±0.01 XY noise, fixed z ----
        cube_local = self._init_cube_pos[task_ids].clone()  # (n, 3)
        cube_local[:, :2] += torch.zeros(n, 2, device=dev).uniform_(-0.01, 0.01)
        self._cube.write_root_link_pose_to_sim(
            torch.cat([cube_local + origins, self._identity_quat(n, dev)], dim=-1),
            env_ids=env_ids,
        )
        self._cube.write_root_link_velocity_to_sim(
            torch.zeros(n, 6, device=dev), env_ids=env_ids
        )

        # ---- Drawer: init joint position and zero velocity ----
        self._drawer.write_joint_position_to_sim(
            self._init_drawer_pos[task_ids].unsqueeze(-1),
            joint_ids=self._drawer_joint_slot,
            env_ids=env_ids,
        )
        self._drawer.write_joint_velocity_to_sim(
            torch.zeros(n, 1, device=dev),
            joint_ids=self._drawer_joint_slot,
            env_ids=env_ids,
        )

        # ---- Window: init joint position and zero velocity ----
        self._window.write_joint_position_to_sim(
            self._init_window_pos[task_ids].unsqueeze(-1),
            joint_ids=self._window_joint_slot,
            env_ids=env_ids,
        )
        self._window.write_joint_velocity_to_sim(
            torch.zeros(n, 1, device=dev),
            joint_ids=self._window_joint_slot,
            env_ids=env_ids,
        )

        # ---- Initial button logical states and lock targets ----
        self.scene_button_states[env_ids] = self._init_button_states[task_ids]
        self.prev_button_positions[env_ids] = get_button_positions(env)[env_ids]
        self.drawer_lock_target[env_ids] = self._init_drawer_pos[task_ids]
        self.window_lock_target[env_ids] = self._init_window_pos[task_ids]

        self.episode_success[env_ids] = 0.0

    def _update_command(self) -> None:
        env = self._env
        cur_button_pos = get_button_positions(env)
        new_presses = detect_new_button_presses(
            self.prev_button_positions, cur_button_pos
        )
        self.scene_button_states[new_presses] = (
            1.0 - self.scene_button_states[new_presses]
        )
        self.prev_button_positions = cur_button_pos.clone()
        self._apply_locks()

    def _apply_locks(self) -> None:
        """Clamp locked drawer/window joints per-env via data writes (no model mutation)."""
        cur_drawer = self._drawer.data.joint_pos[:, self._drawer_joint_slot[0]]
        cur_window = self._window.data.joint_pos[:, self._window_joint_slot[0]]

        drawer_unlocked = self.scene_button_states[:, 0] > 0.5
        window_unlocked = self.scene_button_states[:, 1] > 0.5

        # Track current position while unlocked; hold stored target while locked.
        self.drawer_lock_target = torch.where(
            drawer_unlocked, cur_drawer, self.drawer_lock_target
        )
        self.window_lock_target = torch.where(
            window_unlocked, cur_window, self.window_lock_target
        )

        drawer_locked = (~drawer_unlocked).nonzero(as_tuple=False).squeeze(-1)
        window_locked = (~window_unlocked).nonzero(as_tuple=False).squeeze(-1)

        if drawer_locked.numel() > 0:
            self._drawer.write_joint_position_to_sim(
                self.drawer_lock_target[drawer_locked].unsqueeze(-1),
                joint_ids=self._drawer_joint_slot,
                env_ids=drawer_locked,
            )
            self._drawer.write_joint_velocity_to_sim(
                torch.zeros(drawer_locked.numel(), 1, device=self.device),
                joint_ids=self._drawer_joint_slot,
                env_ids=drawer_locked,
            )

        if window_locked.numel() > 0:
            self._window.write_joint_position_to_sim(
                self.window_lock_target[window_locked].unsqueeze(-1),
                joint_ids=self._window_joint_slot,
                env_ids=window_locked,
            )
            self._window.write_joint_velocity_to_sim(
                torch.zeros(window_locked.numel(), 1, device=self.device),
                joint_ids=self._window_joint_slot,
                env_ids=window_locked,
            )

    def _update_metrics(self) -> None:
        success = scene_success(self._env).float()
        self.episode_success = torch.maximum(self.episode_success, success)
        self.metrics["episode_success"] = self.episode_success

    @staticmethod
    def _identity_quat(n: int, device: str) -> torch.Tensor:
        q = torch.zeros(n, 4, device=device)
        q[:, 0] = 1.0  # w=1, x=y=z=0
        return q
