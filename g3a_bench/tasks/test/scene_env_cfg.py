import mdp as scene_mdp
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_lift_cube_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create base cube lifting task configuration."""

    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        # End-effector / gripper site.
        "ee_pos": ObservationTermCfg(
            func=scene_mdp.site_pos_w,
            params={
                "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
            },
        ),
        "ee_yaw": ObservationTermCfg(
            func=scene_mdp.site_yaw_cos_sin,
            params={
                "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
            },
        ),
        # Approximate gripper opening.
        "gripper_opening": ObservationTermCfg(
            func=scene_mdp.gripper_opening,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(
                        "left_finger",
                        "right_finger",
                    ),
                ),
            },
        ),
        # Placeholder for now.
        "gripper_contact": ObservationTermCfg(
            func=scene_mdp.gripper_contact_zero,
        ),
        # Cube.
        "cube_pos": ObservationTermCfg(
            func=scene_mdp.root_pos_w,
            params={"entity_name": "cube"},
        ),
        "cube_quat": ObservationTermCfg(
            func=scene_mdp.root_quat_w,
            params={"entity_name": "cube"},
        ),
        "cube_yaw": ObservationTermCfg(
            func=scene_mdp.root_yaw_cos_sin,
            params={"entity_name": "cube"},
        ),
        # Buttons.
        "button_0_pos": ObservationTermCfg(
            func=scene_mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "buttons", joint_names=("buttonbox_joint_0",)
                ),
            },
        ),
        "button_0_vel": ObservationTermCfg(
            func=scene_mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "buttons", joint_names=("buttonbox_joint_0",)
                ),
            },
        ),
        "button_0_pressed": ObservationTermCfg(
            func=scene_mdp.button_state_from_joint,
            params={
                "asset_cfg": SceneEntityCfg(
                    "buttons", joint_names=("buttonbox_joint_0",)
                ),
                "threshold": -0.02,
            },
        ),
        "button_1_pos": ObservationTermCfg(
            func=scene_mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "buttons", joint_names=("buttonbox_joint_1",)
                ),
            },
        ),
        "button_1_vel": ObservationTermCfg(
            func=scene_mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "buttons", joint_names=("buttonbox_joint_1",)
                ),
            },
        ),
        "button_1_pressed": ObservationTermCfg(
            func=scene_mdp.button_state_from_joint,
            params={
                "asset_cfg": SceneEntityCfg(
                    "buttons", joint_names=("buttonbox_joint_1",)
                ),
                "threshold": -0.02,
            },
        ),
        "logical_button_states": ObservationTermCfg(
            func=scene_mdp.logical_button_states,
        ),
        # Drawer.
        "drawer_pos": ObservationTermCfg(
            func=scene_mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg("drawer", joint_names=("drawer_slide",)),
            },
        ),
        "drawer_vel": ObservationTermCfg(
            func=scene_mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg("drawer", joint_names=("drawer_slide",)),
            },
        ),
        "drawer_handle_pos": ObservationTermCfg(
            func=scene_mdp.site_pos_w,
            params={
                "asset_cfg": SceneEntityCfg(
                    "drawer",
                    site_names=("drawer_handle_center",),
                ),
            },
        ),
        "drawer_handle_yaw": ObservationTermCfg(
            func=scene_mdp.site_yaw_cos_sin,
            params={
                "asset_cfg": SceneEntityCfg(
                    "drawer",
                    site_names=("drawer_handle_center",),
                ),
            },
        ),
        # Window.
        "window_pos": ObservationTermCfg(
            func=scene_mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg("window", joint_names=("window_slide",)),
            },
        ),
        "window_vel": ObservationTermCfg(
            func=scene_mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg("window", joint_names=("window_slide",)),
            },
        ),
        "window_handle_pos": ObservationTermCfg(
            func=scene_mdp.site_pos_w,
            params={
                "asset_cfg": SceneEntityCfg(
                    "window",
                    site_names=("window_handle_center",),
                ),
            },
        ),
        "window_handle_yaw": ObservationTermCfg(
            func=scene_mdp.site_yaw_cos_sin,
            params={
                "asset_cfg": SceneEntityCfg(
                    "window",
                    site_names=("window_handle_center",),
                ),
            },
        ),
    }

    critic_terms = {**actor_terms}

    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
    }

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,  # Override per-robot.
            use_default_offset=True,
        )
    }

    events = {
        # For positioning the base of the robot at env_origins.
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "reset_scene_logic": EventTermCfg(
            func=scene_mdp.reset_scene_logic,
            mode="reset",
        ),
        "update_scene_logic": EventTermCfg(
            func=scene_mdp.update_scene_logic,
            mode="step",
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1,
            env_spacing=1.0,
        ),
        observations=observations,
        actions=actions,
        commands={},
        events=events,
        rewards={},
        terminations={},
        curriculum={},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="",  # Set per-robot.
            distance=1.5,
            elevation=-5.0,
            azimuth=120.0,
        ),
        sim=SimulationCfg(
            nconmax=55,
            njmax=600,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=4,
        episode_length_s=20.0,
    )
