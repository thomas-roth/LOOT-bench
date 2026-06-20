from pathlib import Path

import mujoco
from mjlab.asset_zoo.robots import (
    YAM_ACTION_SCALE,
    get_yam_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg

from g3a_bench.tasks.scene.scene_env_cfg import make_scene_env_cfg

_ASSETS = Path(__file__).parents[2] / "assets"
_CUBE_XML = _ASSETS / "cube.xml"
_BUTTONS_XML = _ASSETS / "buttons.xml"
_DRAWER_XML = _ASSETS / "drawer.xml"
_WINDOW_XML = _ASSETS / "window.xml"


def get_cube_spec(
    cube_size: float = 0.04,
    mass: float = 0.05,
    rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
) -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_CUBE_XML))


def get_window_spec(
    size: tuple[float, float] = (0.2, 0.2),
    mass: float = 0.1,
) -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_WINDOW_XML))


def get_buttons_spec(
    size: tuple[float, float] = (0.1, 0.1),
    mass: float = 0.1,
) -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_BUTTONS_XML))


def get_drawer_spec(
    size: tuple[float, float] = (0.2, 0.1),
    mass: float = 0.1,
) -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_DRAWER_XML))


def yam_scene_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_scene_env_cfg()

    cfg.scene.entities = {
        "robot": get_yam_robot_cfg(),
        "cube": EntityCfg(spec_fn=get_cube_spec),
        "buttons": EntityCfg(
            spec_fn=get_buttons_spec,
            init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        ),
        "drawer": EntityCfg(
            spec_fn=get_drawer_spec,
            init_state=EntityCfg.InitialStateCfg(
                pos=(0.33, -0.42, 0.084),
                rot=(0.0, 0.0, 0.0, 1.0),  # wxyz: 180° around Z
            ),
        ),
        "window": EntityCfg(
            spec_fn=get_window_spec,
            init_state=EntityCfg.InitialStateCfg(
                pos=(0.30, 0.30, 0.202),
            ),
        ),
    }

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = YAM_ACTION_SCALE

    cfg.viewer.body_name = "arm"

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False

    return cfg
