# pyright: reportAttributeAccessIssue=false
from pathlib import Path

import mujoco
from mjlab.asset_zoo.robots import (
    YAM_ACTION_SCALE,
    get_yam_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g3a_bench.tasks.cleaning_surface.cleaning_env_cfg import make_cleaning_env_cfg

_ASSETS = Path(__file__).parents[2] / "assets"
_SPONGE_XML = _ASSETS / "sponge.xml"
_BIN_XML = _ASSETS / "bin.xml"


def get_sponge_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_SPONGE_XML))


def get_bin_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_BIN_XML))


def get_debris_spec(name: str) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=name)
    body.add_freejoint(name=f"{name}_joint")
    body.add_geom(
        name=f"{name}_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.015, 0.015, 0.015],
        mass=0.01,
        rgba=[0.55, 0.35, 0.15, 1.0],
        friction=[0.5, 0.05, 0.01],
        contype=3,
        conaffinity=3,
        group=1,
    )
    return spec


def yam_cleaning_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_cleaning_env_cfg()

    cfg.scene.entities = {
        "robot": get_yam_robot_cfg(),
        "sponge": EntityCfg(spec_fn=get_sponge_spec),
        "debris_0": EntityCfg(spec_fn=lambda: get_debris_spec("debris_0")),
        "debris_1": EntityCfg(spec_fn=lambda: get_debris_spec("debris_1")),
        "bin": EntityCfg(
            spec_fn=get_bin_spec,
            init_state=EntityCfg.InitialStateCfg(pos=(0.30, 0.25, 0.0)),
        ),
    }

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = YAM_ACTION_SCALE

    cfg.viewer.body_name = "arm"

    # Pin the bin marker to env_origins + init_state.pos on each reset
    cfg.events["reset_bin"] = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bin"),
        },
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False

    return cfg
