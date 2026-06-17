from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from .mdp import observations as obs
from .mdp.commands import MocapDeltaActionCfg
from .mdp.rewards import peg_insertion_reward
from .mdp.terminations import time_out
from .mdp.events import reset_peg_and_box

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_PEG_XML = (
    Path(__file__).parent
    / "assets"
    / "sawyer_xyz"
    / "sawyer_peg_insertion_side.xml"
)

def _get_spec() -> mujoco.MjSpec:
    print("LOADING XML:", _PEG_XML.resolve())
    return mujoco.MjSpec.from_file(str(_PEG_XML))


_SAWYER_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(XmlActuatorCfg(target_names_expr=(".*",)),),
)

_DEFAULT_INIT = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={"r_close": 0.0, "l_close": 0.0},
    joint_vel={".*": 0.0},
)


def _get_sawyer_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_spec,
        articulation=_SAWYER_ARTICULATION,
        init_state=_DEFAULT_INIT,
    )


def make_peg_insertion_env_cfg() -> ManagerBasedRlEnvCfg:
    actor_terms = {
        "eef_pos": ObservationTermCfg(
            func=obs.eef_pos_cartesian,
            params={"eef_cfg": obs._EEF_SITE_CFG},
        ),
        "gripper_open": ObservationTermCfg(
            func=obs.gripper_openness,
            params={"eef_cfg": obs._EEF_CFG},
        ),
        "peg_pos": ObservationTermCfg(
            func=obs.site_pos_abs,
            params={"asset_cfg": obs._PEG_GRASP_CFG},
        ),
        "eef_to_peg": ObservationTermCfg(
            func=obs.eef_to_peg_vec,
            params={"peg_cfg": obs._PEG_GRASP_CFG, "eef_cfg": obs._EEF_SITE_CFG},
        ),
        "peg_quat": ObservationTermCfg(
            func=obs.peg_quat,
        ),
        "eef_vel": ObservationTermCfg(
            func=obs.eef_vel_cartesian,
            params={"eef_cfg": obs._EEF_SITE_CFG},
        ),
        "goal_pos": ObservationTermCfg(
            func=obs.goal_pos_abs,
        ),
    }

    critic_terms = {**actor_terms}

    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
    }

    actions: dict[str, ActionTermCfg] = {
        "mocap_delta": MocapDeltaActionCfg(
            entity_name="sawyer",
            mocap_name="mocap",
            pos_scale=0.02,
            x_limits=(-0.5, 0.5),
            y_limits=(0.3, 0.9),
            z_limits=(0.0, 0.4),
        )
    }

    commands: dict[str, CommandTermCfg] = {}

    events = {
        "reset_peg_and_box": EventTermCfg(
            func=reset_peg_and_box,
            mode="reset",
            params={
                # randomize=False -> fixed, reproducible scene (peg in front of
                # the hole) for visual comparison against the native viewer.
                # Set True to sample peg/box from MetaWorld's ranges per reset
                # (domain randomization), which is what you want for RL training.
                "randomize": False,
                "asset_cfg": SceneEntityCfg("sawyer"),
            },
        ),
    }

    rewards = {
        "peg_insertion": RewardTermCfg(
            func=peg_insertion_reward,
            weight=1.0,
            params={
                "peg_cfg": obs._PEG_GRASP_CFG,
                "peg_head_cfg": obs._PEG_HEAD_CFG,
                "eef_cfg": obs._EEF_CFG,
            },
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
    }

    curriculum = {}

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=None,
            entities={"sawyer": _get_sawyer_cfg()},
            num_envs=1,
            env_spacing=4.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="sawyer",
            body_name="base",
            distance=2.0,
            elevation=-20.0,
            azimuth=45.0,
        ),
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=0.002,
                disableflags=("multiccd", "nativeccd"),
            ),
            nconmax=256,
            njmax=256,
        ),
        decimation=5,
        episode_length_s=50.0,
    )


def peg_insertion_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_peg_insertion_env_cfg()
    if play:
        cfg.episode_length_s = 1e10
        cfg.observations["actor"].enable_corruption = False
    return cfg


__all__ = [
    "make_peg_insertion_env_cfg",
    "peg_insertion_env_cfg",
    "_get_sawyer_cfg",
    "reset_peg_and_box",
]



'''"""Sawyer peg insertion side environment configuration."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import time_out
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_PEG_XML = (
    Path(__file__).parent
    / "assets"
    / "sawyer_xyz"
    / "sawyer_peg_insertion_side.xml"
)

# ---------------------------------------------------------------------------
# SceneEntityCfg constants
# ---------------------------------------------------------------------------

_PEG_GRASP_CFG = SceneEntityCfg("sawyer", site_names=("pegGrasp",))
_PEG_HEAD_CFG  = SceneEntityCfg("sawyer", site_names=("pegHead",))
_GOAL_CFG      = SceneEntityCfg("sawyer", site_names=("goal",))
_EEF_CFG       = SceneEntityCfg("sawyer", joint_names=("r_close", "l_close"))
_EEF_RIGHT_CFG = SceneEntityCfg("sawyer", site_names=("rightEndEffector",))
_EEF_LEFT_CFG  = SceneEntityCfg("sawyer", site_names=("leftEndEffector",))
_EEF_SITE_CFG  = SceneEntityCfg("sawyer", site_names=("endEffector",))
_EEF_BRC1_CFG  = SceneEntityCfg("sawyer", site_names=("bottom_right_corner_collision_box_1",))
_EEF_TLC1_CFG  = SceneEntityCfg("sawyer", site_names=("top_left_corner_collision_box_1",))
_EEF_BRC2_CFG  = SceneEntityCfg("sawyer", site_names=("bottom_right_corner_collision_box_2",))
_EEF_TLC2_CFG  = SceneEntityCfg("sawyer", site_names=("top_left_corner_collision_box_2",))

TARGET_RADIUS: float = 0.07

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SITE_ID_CACHE: dict[str, int] = {}


def _site_pos(asset: Entity, cfg: SceneEntityCfg) -> torch.Tensor:
    """World position of a single named site. Shape: [N, 3]."""
    name = cfg.site_names[0]
    if name not in _SITE_ID_CACHE:
        _SITE_ID_CACHE[name] = asset.site_names.index(name)
    return asset.data.site_pos_w[:, _SITE_ID_CACHE[name], :]


def _site_lin_vel(asset: Entity, cfg: SceneEntityCfg) -> torch.Tensor:
    """World linear velocity of a single named site. Shape: [N, 3]."""
    name = cfg.site_names[0]
    if name not in _SITE_ID_CACHE:
        _SITE_ID_CACHE[name] = asset.site_names.index(name)
    return asset.data.site_lin_vel_w[:, _SITE_ID_CACHE[name], :]


def _get_mj(env: ManagerBasedRlEnv) -> tuple[mujoco.MjModel, mujoco.MjData]:
    return env.unwrapped.sim.mj_model, env.unwrapped.sim.mj_data


# ---------------------------------------------------------------------------
# Mocap Delta Action
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MocapDeltaActionCfg(ActionTermCfg):
    """4-DOF delta action: [dx, dy, dz, gripper]."""
    entity_name: str = "sawyer"
    mocap_name: str = "mocap"
    pos_scale: float = 0.02
    x_limits: tuple = (-0.5,  0.5)
    y_limits: tuple = ( 0.3,  0.9)
    z_limits: tuple = ( 0.02, 0.5)

    def build(self, env: "ManagerBasedRlEnv") -> "MocapDeltaAction":
        return MocapDeltaAction(self, env)


class MocapDeltaAction(ActionTerm):
    """Moves the mocap body by a delta xyz and sets gripper ctrl."""

    cfg: MocapDeltaActionCfg

    def __init__(self, cfg: MocapDeltaActionCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        mj_model = env.unwrapped.sim.mj_model
        self._mocap_body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, cfg.mocap_name
        )
        self._actions = torch.zeros(env.num_envs, 4,
                                    device=env.unwrapped.device)

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_action(self) -> torch.Tensor:   # singular — matches abstract property
        return self._actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._actions = actions.clamp(-1.0, 1.0)

    def apply_actions(self) -> None:
        mj_model, mj_data = _get_mj(self._env)
        mocap_idx = int(mj_model.body_mocapid[self._mocap_body_id])
        scale = self.cfg.pos_scale

        for i in range(self._env.num_envs):
            delta = self._actions[i, :3].cpu().numpy() * scale
            mj_data.mocap_pos[mocap_idx] += delta

            mj_data.mocap_pos[mocap_idx, 0] = float(np.clip(
                mj_data.mocap_pos[mocap_idx, 0], *self.cfg.x_limits))
            mj_data.mocap_pos[mocap_idx, 1] = float(np.clip(
                mj_data.mocap_pos[mocap_idx, 1], *self.cfg.y_limits))
            mj_data.mocap_pos[mocap_idx, 2] = float(np.clip(
                mj_data.mocap_pos[mocap_idx, 2], *self.cfg.z_limits))

            g = float(self._actions[i, 3])
            mj_data.ctrl[0] =  g
            mj_data.ctrl[1] = -g


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

def _get_spec() -> mujoco.MjSpec:
    print("LOADING XML:", _PEG_XML.resolve())
    return mujoco.MjSpec.from_file(str(_PEG_XML))


_SAWYER_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(XmlActuatorCfg(target_names_expr=(".*",)),),
)

_DEFAULT_INIT = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={"r_close": 0.0, "l_close": 0.0},
    joint_vel={".*": 0.0},
)


def _get_sawyer_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_spec,
        articulation=_SAWYER_ARTICULATION,
        init_state=_DEFAULT_INIT,
    )


# ---------------------------------------------------------------------------
# Reset event
# ---------------------------------------------------------------------------

def reset_peg_and_box(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    peg_pos_range: tuple[float, float] = (-0.1, 0.1),
    box_pos_range: tuple[float, float] = (-0.1, 0.1),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("sawyer"),
) -> None:
    """Randomise peg (freejoint) and box (body) positions at reset.
    
    Base positions match MetaWorld peg-insert-side-v3 exactly.
    """
    mj_model, mj_data = _get_mj(env)

    peg_bid    = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "peg")
    peg_jnt_id = mj_model.body_jntadr[peg_bid]
    box_bid      = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY,  "box")
    goal_sid     = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE,  "goal")
    peg_head_sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE,  "pegHead")
    mocap_bid    = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY,  "mocap")

    peg_qpos_adr = int(mj_model.jnt_qposadr[peg_jnt_id])
    peg_dof_adr  = int(mj_model.jnt_dofadr[peg_jnt_id])
    mocap_idx    = int(mj_model.body_mocapid[mocap_bid])

    device   = env_ids.device
    num_envs = env.num_envs
    lo_p, hi_p = peg_pos_range
    lo_b, hi_b = box_pos_range

    if not hasattr(env.unwrapped, "target_pos"):
        env.unwrapped.target_pos      = torch.zeros(num_envs, 3, device=device)
        env.unwrapped.in_place_margin = torch.ones(num_envs,    device=device)
        env.unwrapped.obj_init_pos    = torch.zeros(num_envs, 3, device=device)

    for env_id in env_ids:
        i = int(env_id)

        # --- Peg: freejoint (x, y, z, qw, qx, qy, qz) ---------------------
        # Base position aus MetaWorld: [0.070, 0.663, 0.02]
        peg_x = 0.070 + float(torch.empty(1).uniform_(lo_p, hi_p))
        peg_y = 0.663 + float(torch.empty(1).uniform_(lo_p, hi_p))
        peg_z = 0.02

        mj_data.qpos[peg_qpos_adr + 0] = peg_x
        mj_data.qpos[peg_qpos_adr + 1] = peg_y
        mj_data.qpos[peg_qpos_adr + 2] = peg_z
        mj_data.qpos[peg_qpos_adr + 3] = 1.0   # qw
        mj_data.qpos[peg_qpos_adr + 4] = 0.0   # qx
        mj_data.qpos[peg_qpos_adr + 5] = 0.0   # qy
        mj_data.qpos[peg_qpos_adr + 6] = 0.0   # qz
        mj_data.qvel[peg_dof_adr : peg_dof_adr + 6] = 0.0

        env.unwrapped.obj_init_pos[i] = torch.tensor(
            [peg_x, peg_y, peg_z], device=device
        )

        # --- Box: body_pos in model -----------------------------------------
        # Base position aus MetaWorld: [-0.324, 0.437, 0.0005]
        box_x = -0.324 + float(torch.empty(1).uniform_(lo_b, hi_b))
        box_y =  0.437 + float(torch.empty(1).uniform_(lo_b, hi_b))
        box_z =  0.0005

        mj_model.body_pos[box_bid, 0] = box_x
        mj_model.body_pos[box_bid, 1] = box_y
        mj_model.body_pos[box_bid, 2] = box_z

        # --- Goal: box_pos + offset aus MetaWorld: [0.030, 0.0, 0.130] -----
        target = np.array([box_x + 0.030, box_y, box_z + 0.130])
        mj_model.site_pos[goal_sid] = target
        env.unwrapped.target_pos[i] = torch.tensor(target, device=device)

        # --- Mocap: init_tcp aus MetaWorld: [0.004, 0.601, 0.150] ----------
        mj_data.mocap_pos[mocap_idx, 0] = 0.004
        mj_data.mocap_pos[mocap_idx, 1] = 0.601
        mj_data.mocap_pos[mocap_idx, 2] = 0.150

        # --- Forward kinematics aktualisieren -------------------------------
        mujoco.mj_forward(mj_model, mj_data)

        # --- in_place_margin: Abstand pegHead → goal bei Reset --------------
        peg_head_pos = torch.tensor(
            mj_data.site_xpos[peg_head_sid].copy(), device=device
        )
        env.unwrapped.in_place_margin[i] = torch.norm(
            peg_head_pos - env.unwrapped.target_pos[i]
        ).clamp(min=1e-3)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def eef_pos_cartesian(
    env: ManagerBasedRlEnv,
    eef_cfg: SceneEntityCfg = _EEF_SITE_CFG,
) -> torch.Tensor:
    """End-effector world position. Shape: [N, 3]."""
    asset: Entity = env.scene[eef_cfg.name]
    return _site_pos(asset, eef_cfg)


def eef_vel_cartesian(
    env: ManagerBasedRlEnv,
    eef_cfg: SceneEntityCfg = _EEF_SITE_CFG,
) -> torch.Tensor:
    """End-effector linear velocity. Shape: [N, 3]."""
    asset: Entity = env.scene[eef_cfg.name]
    return _site_lin_vel(asset, eef_cfg)


def site_pos_abs(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Absolute world position of a single named site. Shape: [N, 3]."""
    asset: Entity = env.scene[asset_cfg.name]
    return _site_pos(asset, asset_cfg)


def eef_to_peg_vec(
    env: ManagerBasedRlEnv,
    peg_cfg: SceneEntityCfg = _PEG_GRASP_CFG,
    eef_cfg: SceneEntityCfg = _EEF_SITE_CFG,
) -> torch.Tensor:
    """Vector from end-effector to peg grasp site. Shape: [N, 3]."""
    asset: Entity = env.scene[peg_cfg.name]
    return _site_pos(asset, peg_cfg) - _site_pos(asset, eef_cfg)


def peg_to_goal_vec(
    env: ManagerBasedRlEnv,
    peg_cfg: SceneEntityCfg = _PEG_HEAD_CFG,
    goal_cfg: SceneEntityCfg = _GOAL_CFG,
) -> torch.Tensor:
    """Vector from peg head to goal site. Shape: [N, 3]."""
    asset: Entity = env.scene[peg_cfg.name]
    return _site_pos(asset, goal_cfg) - _site_pos(asset, peg_cfg)


def gripper_openness(
    env: ManagerBasedRlEnv,
    eef_cfg: SceneEntityCfg = _EEF_CFG,
) -> torch.Tensor:
    """Sum of finger joint positions. Shape: [N, 1]."""
    asset: Entity = env.scene[eef_cfg.name]
    q = asset.data.joint_pos[:, eef_cfg.joint_ids]
    return q.sum(dim=-1, keepdim=True)

def peg_quat(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Quaternion des Peg-Freejoints [w, x, y, z]. Shape: [N, 4]."""
    mj_model, mj_data = _get_mj(env)
    jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "peg_free")
    adr = int(mj_model.jnt_qposadr[jid])
    q = torch.tensor(
        mj_data.qpos[adr + 3 : adr + 7].copy(),
        dtype=torch.float32, device=env.unwrapped.device,
    ).unsqueeze(0).expand(env.num_envs, -1)
    return q  # [N, 4]


def goal_pos_abs(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Absolute Zielposition. Shape: [N, 3]."""
    if not hasattr(env.unwrapped, "target_pos"):
        return torch.zeros(env.num_envs, 3, device=env.unwrapped.device)
    return env.unwrapped.target_pos.float()

# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------

_GAUSSIAN_SCALE = math.sqrt(-2 * math.log(0.1))


def _gaussian_tolerance(x: torch.Tensor, margin: float) -> torch.Tensor:
    if margin == 0:
        return (x == 0).float()
    return torch.exp(-0.5 * (x / margin * _GAUSSIAN_SCALE) ** 2)


def _long_tail_tolerance(x: torch.Tensor, margin: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + (x / (margin + 1e-6)) ** 2)


def _hamacher_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    denom = a + b - a * b
    return torch.where(denom > 0, (a * b) / denom, torch.zeros_like(a))


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def peg_insertion_reward(
    env: ManagerBasedRlEnv,
    peg_cfg: SceneEntityCfg = _PEG_GRASP_CFG,
    peg_head_cfg: SceneEntityCfg = _PEG_HEAD_CFG,
    eef_cfg: SceneEntityCfg = _EEF_CFG,
) -> torch.Tensor:
    """Smooth peg-insertion reward following MetaWorld v2. Returns shape [N]."""
    if not hasattr(env.unwrapped, "target_pos"):
        asset: Entity = env.scene["sawyer"]
        return torch.zeros(env.num_envs, device=asset.data.joint_pos.device)
    
    asset: Entity = env.scene[peg_cfg.name]

    peg_pos      = _site_pos(asset, peg_cfg)        # [N, 3]
    peg_head     = _site_pos(asset, peg_head_cfg)   # [N, 3]
    right_finger = _site_pos(asset, _EEF_RIGHT_CFG)
    left_finger  = _site_pos(asset, _EEF_LEFT_CFG)
    brc1         = _site_pos(asset, _EEF_BRC1_CFG)
    tlc1         = _site_pos(asset, _EEF_TLC1_CFG)
    brc2         = _site_pos(asset, _EEF_BRC2_CFG)
    tlc2         = _site_pos(asset, _EEF_TLC2_CFG)

    goal     = env.unwrapped.target_pos                          # [N, 3]
    tcp      = (right_finger + left_finger) / 2.0               # [N, 3]
    tcp_open = gripper_openness(env, eef_cfg).squeeze(-1)        # [N]

    scale         = peg_pos.new_tensor([1.0, 2.0, 2.0])
    tcp_to_obj    = torch.norm(peg_pos - tcp, dim=-1)            # [N]
    obj_to_target = torch.norm((peg_head - goal) * scale, dim=-1)# [N]

    in_place = _long_tail_tolerance(obj_to_target, env.unwrapped.in_place_margin)

    def _rect_prism(curr, one, zero):
        lo = torch.min(one, zero)
        hi = torch.max(one, zero)
        return ((curr >= lo) & (curr <= hi)).all(dim=-1).float()

    collision_boxes = _hamacher_product(
        _rect_prism(peg_head, tlc1, brc1),
        _rect_prism(peg_head, tlc2, brc2),
    )
    in_place = _hamacher_product(in_place, collision_boxes)

    # Genauer (zwei Toleranzen kombiniert wie MetaWorld):
    reach_xy = torch.norm(peg_pos[:, :2] - tcp[:, :2], dim=-1)
    reach_z  = torch.abs(peg_pos[:, 2] - tcp[:, 2])
    object_grasped = _gaussian_tolerance(reach_xy, margin=0.005) * \
                    _gaussian_tolerance(reach_z,  margin=0.08)
                    
    lifted = (peg_pos[:, 2] - 0.01 > env.unwrapped.obj_init_pos[:, 2]).float()
    grasped_and_lifted = (
        (tcp_to_obj < 0.08).float() * (tcp_open > 0).float() * lifted
    )
    object_grasped = torch.where(
        grasped_and_lifted > 0,
        torch.ones_like(object_grasped),
        object_grasped,
    )

    reward = _hamacher_product(object_grasped, in_place)
    reward = reward + grasped_and_lifted * (1.0 + 5.0 * in_place)
    success = (obj_to_target <= TARGET_RADIUS).float()
    reward  = torch.where(success > 0, torch.full_like(reward, 10.0), reward)

    return reward   # [N]


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------

def _make_env_cfg() -> ManagerBasedRlEnvCfg:
    # Obs shape: 3 + 1 + 3 + 3 + 3 + 3 = 16
    actor_terms = {
    "eef_pos": ObservationTermCfg(          # [3]
        func=eef_pos_cartesian,
        params={"eef_cfg": _EEF_SITE_CFG},
    ),
    "gripper_open": ObservationTermCfg(     # [1]
        func=gripper_openness,
        params={"eef_cfg": _EEF_CFG},
    ),
    "peg_pos": ObservationTermCfg(          # [3]
        func=site_pos_abs,
        params={"asset_cfg": _PEG_GRASP_CFG},
    ),
    "eef_to_peg": ObservationTermCfg(       # [3]
        func=eef_to_peg_vec,
        params={"peg_cfg": _PEG_GRASP_CFG, "eef_cfg": _EEF_SITE_CFG},
    ),
    "peg_quat": ObservationTermCfg(         # [4]  ← NEU statt peg_to_goal
        func=peg_quat,
    ),
    "eef_vel": ObservationTermCfg(          # [3]
        func=eef_vel_cartesian,
        params={"eef_cfg": _EEF_SITE_CFG},
    ),
    "goal_pos": ObservationTermCfg(         # [3]  ← NEU
        func=goal_pos_abs,
    ),
    # Frame-Stacking macht mjlab automatisch wenn history_length=2 gesetzt
}
# Gesamt: 3+1+3+3+4+3+3 = 20D (ohne Frame-Stacking)

    observations = {
        "actor":  ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg({**actor_terms}),
    }

    actions = {
        "mocap_delta": MocapDeltaActionCfg(
        entity_name="sawyer",
        mocap_name="mocap",
        pos_scale=0.02,
        x_limits=(-0.5, 0.5),
        y_limits=( 0.3, 0.9),
        z_limits=( 0.0, 0.4),
    ),
    }

    events = {
        "reset_peg_and_box": EventTermCfg(
            func=reset_peg_and_box,
            mode="reset",
            params={
                "peg_pos_range": (-0.1, 0.1),
                "box_pos_range": (-0.1, 0.1),
                "asset_cfg": SceneEntityCfg("sawyer"),
            },
        ),
    }

    rewards = {
        "peg_insertion": RewardTermCfg(
            func=peg_insertion_reward,
            weight=1.0,
            params={
                "peg_cfg": _PEG_GRASP_CFG,
                "peg_head_cfg": _PEG_HEAD_CFG,
                "eef_cfg": _EEF_CFG,
            },
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=None,
            entities={"sawyer": _get_sawyer_cfg()},
            num_envs=1,
            env_spacing=4.0,
        ),
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="sawyer",
            body_name="base",
            distance=2.0,
            elevation=-20.0,
            azimuth=45.0,
        ),
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=0.002,
                disableflags=("multiccd", "nativeccd"),
            ),
        ),
        decimation=5,
        episode_length_s=50.0,
    )


def peg_insertion_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = _make_env_cfg()
    if play:
        cfg.episode_length_s = 1e10
        cfg.observations["actor"].enable_corruption = False
    return cfg


# ---------------------------------------------------------------------------
# RL config
# ---------------------------------------------------------------------------

def peg_insertion_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="peg_insertion_side",
        save_interval=50,
        num_steps_per_env=64,
        max_iterations=1000,
    )'''