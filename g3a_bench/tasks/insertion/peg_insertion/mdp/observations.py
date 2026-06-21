from __future__ import annotations

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

_SITE_ID_CACHE: dict[str, int] = {}

# Peg sites live on the "peg" entity, end-effector/gripper on "sawyer",
# the goal marker and the box-mouth collision corners on "box"/"world".
_PEG_GRASP_CFG = SceneEntityCfg("peg", site_names=("pegGrasp",))
_PEG_HEAD_CFG = SceneEntityCfg("peg", site_names=("pegHead",))
_GOAL_CFG = SceneEntityCfg("world", site_names=("goal",))
_EEF_CFG = SceneEntityCfg("sawyer", joint_names=("r_close", "l_close"))
_EEF_RIGHT_CFG = SceneEntityCfg("sawyer", site_names=("rightEndEffector",))
_EEF_LEFT_CFG = SceneEntityCfg("sawyer", site_names=("leftEndEffector",))
_EEF_SITE_CFG = SceneEntityCfg("sawyer", site_names=("endEffector",))
_EEF_BRC1_CFG = SceneEntityCfg(
    "box", site_names=("bottom_right_corner_collision_box_1",)
)
_EEF_TLC1_CFG = SceneEntityCfg("box", site_names=("top_left_corner_collision_box_1",))
_EEF_BRC2_CFG = SceneEntityCfg(
    "box", site_names=("bottom_right_corner_collision_box_2",)
)
_EEF_TLC2_CFG = SceneEntityCfg("box", site_names=("top_left_corner_collision_box_2",))


def _site_pos(asset: Entity, cfg: SceneEntityCfg) -> torch.Tensor:
    name = cfg.site_names[0]
    if name not in _SITE_ID_CACHE:
        _SITE_ID_CACHE[name] = asset.site_names.index(name)
    return asset.data.site_pos_w[:, _SITE_ID_CACHE[name], :]


def _site_lin_vel(asset: Entity, cfg: SceneEntityCfg) -> torch.Tensor:
    name = cfg.site_names[0]
    if name not in _SITE_ID_CACHE:
        _SITE_ID_CACHE[name] = asset.site_names.index(name)
    return asset.data.site_lin_vel_w[:, _SITE_ID_CACHE[name], :]


def eef_pos_cartesian(env, eef_cfg: SceneEntityCfg = _EEF_SITE_CFG) -> torch.Tensor:
    asset: Entity = env.scene[eef_cfg.name]
    return _site_pos(asset, eef_cfg)


def eef_vel_cartesian(env, eef_cfg: SceneEntityCfg = _EEF_SITE_CFG) -> torch.Tensor:
    asset: Entity = env.scene[eef_cfg.name]
    return _site_lin_vel(asset, eef_cfg)


def site_pos_abs(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return _site_pos(asset, asset_cfg)


def eef_to_peg_vec(
    env,
    peg_cfg: SceneEntityCfg = _PEG_GRASP_CFG,
    eef_cfg: SceneEntityCfg = _EEF_SITE_CFG,
) -> torch.Tensor:
    return _site_pos(env.scene[peg_cfg.name], peg_cfg) - _site_pos(
        env.scene[eef_cfg.name], eef_cfg
    )


def peg_to_goal_vec(
    env,
    peg_cfg: SceneEntityCfg = _PEG_HEAD_CFG,
    goal_cfg: SceneEntityCfg = _GOAL_CFG,
) -> torch.Tensor:
    return _site_pos(env.scene[goal_cfg.name], goal_cfg) - _site_pos(
        env.scene[peg_cfg.name], peg_cfg
    )


def gripper_openness(env, eef_cfg: SceneEntityCfg = _EEF_CFG) -> torch.Tensor:
    asset: Entity = env.scene[eef_cfg.name]
    q = asset.data.joint_pos[:, eef_cfg.joint_ids]
    return q.sum(dim=-1, keepdim=True)


_PEG_CFG = SceneEntityCfg("peg")


def peg_quat(env, asset_cfg: SceneEntityCfg = _PEG_CFG) -> torch.Tensor:
    sim = env.unwrapped.sim
    mj_model = sim.mj_model
    jid = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"{asset_cfg.name}/peg_free"
    )
    adr = int(mj_model.jnt_qposadr[jid])
    return sim.data.qpos[:, adr + 3 : adr + 7].clone().to(torch.float32)


def goal_pos_abs(env) -> torch.Tensor:
    if not hasattr(env.unwrapped, "target_pos"):
        return torch.zeros(env.num_envs, 3, device=env.unwrapped.device)
    return env.unwrapped.target_pos.float()


__all__ = [
    "_PEG_GRASP_CFG",
    "_PEG_HEAD_CFG",
    "_GOAL_CFG",
    "_EEF_CFG",
    "_EEF_RIGHT_CFG",
    "_EEF_LEFT_CFG",
    "_EEF_SITE_CFG",
    "_EEF_BRC1_CFG",
    "_EEF_TLC1_CFG",
    "_EEF_BRC2_CFG",
    "_EEF_TLC2_CFG",
    "_site_pos",
    "eef_pos_cartesian",
    "eef_vel_cartesian",
    "site_pos_abs",
    "eef_to_peg_vec",
    "peg_to_goal_vec",
    "gripper_openness",
    "peg_quat",
    "goal_pos_abs",
]
