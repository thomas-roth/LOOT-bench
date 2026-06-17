from __future__ import annotations

import dataclasses

import mujoco
import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg


@dataclasses.dataclass
class MocapDeltaActionCfg(ActionTermCfg):
    """4-DOF delta action: [dx, dy, dz, gripper]."""

    entity_name: str = "sawyer"
    mocap_name: str = "mocap"
    pos_scale: float = 0.02
    x_limits: tuple = (-0.5, 0.5)
    y_limits: tuple = (0.3, 0.9)
    z_limits: tuple = (0.02, 0.5)

    def build(self, env) -> "MocapDeltaAction":
        return MocapDeltaAction(self, env)


class MocapDeltaAction(ActionTerm):
    """Moves the mocap body by a delta xyz and sets gripper ctrl."""

    cfg: MocapDeltaActionCfg

    def __init__(self, cfg: MocapDeltaActionCfg, env):
        super().__init__(cfg, env)
        mj_model = env.unwrapped.sim.mj_model
        mocap_bid = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"{cfg.entity_name}/{cfg.mocap_name}",
        )
        self._mocap_idx = int(mj_model.body_mocapid[mocap_bid])
        self._actions = torch.zeros(env.num_envs, 4, device=env.unwrapped.device)

        limits = torch.tensor(
            [cfg.x_limits, cfg.y_limits, cfg.z_limits], device=env.unwrapped.device
        )
        self._pos_lo = limits[:, 0]
        self._pos_hi = limits[:, 1]

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_action(self) -> torch.Tensor:
        return self._actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._actions = actions.clamp(-1.0, 1.0)

    def apply_actions(self) -> None:
        sim = self._env.unwrapped.sim
        scale = self.cfg.pos_scale

        delta = self._actions[:, :3] * scale
        new_pos = sim.data.mocap_pos[:, self._mocap_idx] + delta
        new_pos = torch.clamp(new_pos, self._pos_lo, self._pos_hi)
        sim.data.mocap_pos[:, self._mocap_idx] = new_pos

        grip = self._actions[:, 3]
        sim.data.ctrl[:, 0] = grip
        sim.data.ctrl[:, 1] = -grip


__all__ = ["MocapDeltaActionCfg", "MocapDeltaAction"]
