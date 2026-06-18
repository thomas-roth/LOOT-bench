import colorsys
from typing import Any, Literal

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.mdp import MultiCubeLiftingCommandCfg

from g3a_bench.robots.yam import get_yam_interface

from ...manipulator_wiring import wire_manipulator_into_env


def get_cube_spec(
    cube_size: float = 0.02,
    mass: float = 0.05,
    rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="cube")
    body.add_freejoint(name="cube_joint")
    body.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(cube_size,) * 3,
        mass=mass,
        rgba=rgba,
    )
    return spec


def yam_lift_cube_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_lift_cube_env_cfg()

    cfg.scene.entities = {"cube": EntityCfg(spec_fn=get_cube_spec)}
    wire_manipulator_into_env(cfg, get_yam_interface())

    # Apply play mode overrides.
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}

        # Higher command resampling frequency for more dynamic play.
        assert cfg.commands is not None
        cfg.commands["lift_height"].resampling_time_range = (4.0, 4.0)

    return cfg


def yam_lift_cube_vision_env_cfg(
    cam_type: Literal["rgb", "depth"],
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = yam_lift_cube_env_cfg(play=play)

    camera_names = ["robot/camera_d405"]
    cam_kwargs = {
        "robot/camera_d405": {
            "height": 32,
            "width": 32,
        },
    }
    shared_cam_kwargs = {
        "data_types": (cam_type,),
        "enabled_geom_groups": (0, 3),
        "use_shadows": False,
        "use_textures": True,
    }

    cam_terms = {}
    for cam_name in camera_names:
        cam_cfg = CameraSensorCfg(
            name=cam_name.split("/")[-1],
            camera_name=cam_name,
            **cam_kwargs[cam_name],  # type: ignore[invalid-argument-type]
            **shared_cam_kwargs,
        )
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)
        param_kwargs: dict[str, Any] = {"sensor_name": cam_cfg.name}
        if cam_type == "depth":
            param_kwargs["cutoff_distance"] = 0.5
            func = manipulation_mdp.camera_depth
        else:
            func = manipulation_mdp.camera_rgb
        cam_terms[f"{cam_name.split('/')[-1]}_{cam_type}"] = ObservationTermCfg(
            func=func, params=param_kwargs
        )

    camera_obs = ObservationGroupCfg(
        terms=cam_terms, enable_corruption=False, concatenate_terms=True
    )
    cfg.observations["camera"] = camera_obs

    if cam_type == "rgb":
        cfg.events["cube_color"] = EventTermCfg(
            func=dr.geom_rgba,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("cube", geom_names=(".*",)),
                "operation": "abs",
                "distribution": "uniform",
                "axes": [0, 1, 2],
                "ranges": (0.0, 1.0),
            },
        )

    # Pop privileged info from actor observations.
    actor_obs = cfg.observations["actor"]
    actor_obs.terms.pop("ee_to_cube")
    actor_obs.terms.pop("cube_to_goal")

    # Add goal_position to actor observations.
    actor_obs.terms["goal_position"] = ObservationTermCfg(
        func=manipulation_mdp.target_position,
        params={
            "command_name": "lift_height",
            "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
        },
        # NOTE: No noise for goal position.
    )

    return cfg


def _cube_color(i: int, n: int) -> tuple[float, float, float, float]:
    """Generate a distinct color for cube i of n using HSV hue rotation."""
    h = i / max(n, 1)
    r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.9)
    return (r, g, b, 1.0)


def yam_multi_cube_seg_env_cfg(
    num_cubes: int = 3,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Multi-cube task: depth + segmentation mask for goal conditioning."""
    cfg = make_lift_cube_env_cfg()

    cube_names = [f"cube_{i}" for i in range(num_cubes)]
    entities: dict[str, EntityCfg] = {}
    for i, name in enumerate(cube_names):
        color = _cube_color(i, num_cubes)
        entities[name] = EntityCfg(
            spec_fn=lambda c=color: get_cube_spec(rgba=c),
        )
    cfg.scene.entities = entities

    cfg.commands = {
        "lift_height": MultiCubeLiftingCommandCfg(
            entity_names=tuple(cube_names),
            resampling_time_range=(8.0, 12.0),
            debug_vis=True,
            difficulty="dynamic",
        ),
    }

    cfg.rewards["lift"] = RewardTermCfg(
        func=manipulation_mdp.multi_cube_staged_position_reward,
        weight=1.0,
        params={
            "command_name": "lift_height",
            "reaching_std": 0.2,
            "bringing_std": 0.3,
            "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
        },
    )
    cfg.rewards["lift_precise"] = RewardTermCfg(
        func=manipulation_mdp.multi_cube_bring_object_reward,
        weight=1.0,
        params={
            "command_name": "lift_height",
            "std": 0.05,
        },
    )

    cfg.sim.nconmax = max(cfg.sim.nconmax or 55, 55 + num_cubes * 120)

    wire_manipulator_into_env(cfg, get_yam_interface())

    cam_cfg = CameraSensorCfg(
        name="camera_d405",
        camera_name="robot/camera_d405",
        height=32,
        width=32,
        data_types=("depth", "segmentation"),
        enabled_geom_groups=(0, 3),
        use_shadows=False,
        use_textures=True,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)

    cam_terms = {
        "depth": ObservationTermCfg(
            func=manipulation_mdp.camera_depth,
            params={
                "sensor_name": "camera_d405",
                "cutoff_distance": 0.5,
            },
        ),
        "target_mask": ObservationTermCfg(
            func=manipulation_mdp.camera_target_cube_mask,
            params={
                "sensor_name": "camera_d405",
                "command_name": "lift_height",
            },
        ),
    }
    cfg.observations["camera"] = ObservationGroupCfg(
        terms=cam_terms,
        enable_corruption=False,
        concatenate_terms=True,
        concatenate_dim=0,
    )

    for group_name in ("actor", "critic"):
        obs = cfg.observations[group_name]
        obs.terms.pop("ee_to_cube", None)
        obs.terms.pop("cube_to_goal", None)
        obs.terms["goal_position"] = ObservationTermCfg(
            func=manipulation_mdp.target_position,
            params={
                "command_name": "lift_height",
                "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
            },
        )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}
        assert cfg.commands is not None
        cfg.commands["lift_height"].resampling_time_range = (
            4.0,
            4.0,
        )

    return cfg
