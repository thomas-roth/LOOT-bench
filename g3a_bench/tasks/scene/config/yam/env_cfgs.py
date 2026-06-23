# pyright: reportAttributeAccessIssue=false
from pathlib import Path
from typing import Literal

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp

from g3a_bench.robots.interface import ManipulatorInterface
from g3a_bench.tasks.scene.scene_env_cfg import make_scene_env_cfg

_ASSETS = Path(__file__).parents[2] / "assets"
_CUBE_XML = _ASSETS / "cube.xml"
_BUTTONS_XML = _ASSETS / "buttons.xml"
_DRAWER_XML = _ASSETS / "drawer.xml"
_WINDOW_XML = _ASSETS / "window.xml"


def get_xml_spec(_XML_PATH_: Path) -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_XML_PATH_))


def yam_scene_env_cfg(
    robot: ManipulatorInterface,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_scene_env_cfg(robot)

    cfg.scene.entities = {
        "robot": robot.entity_cfg_fn(),
        "cube": EntityCfg(spec_fn=lambda: get_xml_spec(_CUBE_XML)),
        "buttons": EntityCfg(spec_fn=lambda: get_xml_spec(_BUTTONS_XML)),
        "drawer": EntityCfg(spec_fn=lambda: get_xml_spec(_DRAWER_XML)),
        "window": EntityCfg(spec_fn=lambda: get_xml_spec(_WINDOW_XML)),
    }

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False

    return cfg


def yam_scene_vision_env_cfg(
    robot: ManipulatorInterface,
    cam_type: Literal["rgb", "depth"],
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = yam_scene_env_cfg(robot, play=play)

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
