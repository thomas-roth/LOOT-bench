# pyright: reportAttributeAccessIssue=false
from pathlib import Path
from typing import Literal

import mujoco
from mjlab.asset_zoo.robots import (
    YAM_ACTION_SCALE,
    get_yam_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp

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
        "buttons": EntityCfg(spec_fn=get_buttons_spec),
        "drawer": EntityCfg(spec_fn=get_drawer_spec),
        "window": EntityCfg(spec_fn=get_window_spec),
    }

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = YAM_ACTION_SCALE

    cfg.viewer.body_name = "arm"

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False

    return cfg


def yam_scene_vision_env_cfg(
    cam_type: Literal["rgb", "depth"],
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = yam_scene_env_cfg(play=play)

    cam_sensor = CameraSensorCfg(
        name="camera_d405",
        camera_name="robot/camera_d405",
        height=32,
        width=32,
        data_types=(cam_type,),
        enabled_geom_groups=(0, 3),
        use_shadows=False,
        use_textures=True,
    )
    cfg.scene.sensors = (cam_sensor,)

    if cam_type == "depth":
        cam_term = ObservationTermCfg(
            func=manipulation_mdp.camera_depth,
            params={"sensor_name": "camera_d405", "cutoff_distance": 0.5},
        )
    else:
        cam_term = ObservationTermCfg(
            func=manipulation_mdp.camera_rgb,
            params={"sensor_name": "camera_d405"},
        )
    cfg.observations["camera"] = ObservationGroupCfg(
        {f"camera_d405_{cam_type}": cam_term},
        enable_corruption=False,
        concatenate_terms=True,
    )

    return cfg
