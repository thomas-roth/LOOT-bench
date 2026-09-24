"""Joint-position actions with [-1, 1] spanning each joint travel, at 20 Hz.

Limits come from the compiled model. This is the pre-ADR-016 joint-absolute
convention (last used at 50 Hz before the eef-delta merge) moved to the
current 20 Hz control rate: only the rate changed, the meaning of an action
did not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_Array = TypeVar("_Array", np.ndarray, torch.Tensor)

JOINT_POS_ACTION_NAME = "joint_pos"
"""Key of the single joint-position term in every task's action manager."""


def normalized_scale_offset(lower: _Array, upper: _Array) -> tuple[_Array, _Array]:
    """Map [-1, 1] onto [lower, upper]. Shared with the teleop mapper."""
    return (upper - lower) / 2.0, (upper + lower) / 2.0


class NormalizedJointPositionAction(JointPositionAction):
    """Joint-position control whose action box is the joint limits."""

    def __init__(self, cfg: JointPositionActionCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg=cfg, env=env)
        limits = self._entity.data.joint_pos_limits
        assert limits is not None
        # Use hard limits so the full teleoperated range stays commandable.
        joint_limits = limits[:, self._target_ids]
        # Unlimited joints would give inf scale and NaN offset. Fail at build.
        if not bool(torch.isfinite(joint_limits).all()):
            unlimited = [
                name
                for index, name in enumerate(self._target_names)
                if not bool(torch.isfinite(joint_limits[:, index]).all())
            ]
            raise ValueError(
                f"joints {unlimited} have no position limit, so [-1, 1] cannot "
                "be mapped onto their travel; give them a range in the MJCF"
            )
        self._scale, self._offset = normalized_scale_offset(
            joint_limits[..., 0], joint_limits[..., 1]
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        """Clamp to [-1, 1], then apply scale and offset. Clamping needed as
        DK1 gripper's mechanical range wider than controller command range.
        """
        super().process_actions(actions.clamp(-1.0, 1.0))


@dataclass(kw_only=True)
class NormalizedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`NormalizedJointPositionAction`."""

    use_default_offset: bool = False

    def build(self, env: ManagerBasedRlEnv) -> NormalizedJointPositionAction:
        return NormalizedJointPositionAction(self, env)
