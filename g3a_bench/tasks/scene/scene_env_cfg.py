from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from g3a_bench.tasks.scene import mdp as scene_mdp

_ROBOT_GRASP_CFG = SceneEntityCfg("robot", site_names=("grasp_site",))


def make_scene_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create base scene task configuration."""

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
        # End-effector site.
        "ee_pos": ObservationTermCfg(
            func=scene_mdp.ee_pos_b,
            params={
                "asset_cfg": _ROBOT_GRASP_CFG,
            },
        ),
        "ee_yaw": ObservationTermCfg(
            func=scene_mdp.site_yaw_cos_sin,
            params={
                "asset_cfg": _ROBOT_GRASP_CFG,
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
        # Cube — relative vectors.
        "ee_to_cube": ObservationTermCfg(
            func=scene_mdp.ee_to_object_distance,
            params={"object_name": "cube", "asset_cfg": _ROBOT_GRASP_CFG},
        ),
        "cube_to_goal": ObservationTermCfg(
            func=scene_mdp.object_to_goal_distance,
            params={"object_name": "cube", "asset_cfg": _ROBOT_GRASP_CFG},
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
        # Drawer — joint state + relative handle + goal delta.
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
        "ee_to_drawer_handle": ObservationTermCfg(
            func=scene_mdp.ee_to_site_distance,
            params={
                "robot_cfg": _ROBOT_GRASP_CFG,
                "site_cfg": SceneEntityCfg(
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
        "drawer_to_goal": ObservationTermCfg(func=scene_mdp.drawer_to_goal),
        # Window — joint state + relative handle + goal delta.
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
        "ee_to_window_handle": ObservationTermCfg(
            func=scene_mdp.ee_to_site_distance,
            params={
                "robot_cfg": _ROBOT_GRASP_CFG,
                "site_cfg": SceneEntityCfg(
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
        "window_to_goal": ObservationTermCfg(func=scene_mdp.window_to_goal),
        # Goal targets (needed for multi-task: policy must know what to achieve).
        "target_button_states": ObservationTermCfg(
            func=scene_mdp.scene_target_button_states,
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

    commands: dict[str, CommandTermCfg] = {
        "scene_task": scene_mdp.SceneTaskCommandCfg(
            resampling_time_range=(1.0e9, 1.0e9),
        ),
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
        "reset_buttons": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("buttons"),
            },
        ),
        "reset_drawer": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("drawer"),
            },
        ),
        "reset_window": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("window"),
            },
        ),
    }

    rewards = {
        "scene": RewardTermCfg(func=scene_mdp.scene_ogbench_reward, weight=1.0),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=scene_mdp.time_out, time_out=True),
        "cube_out_of_bounds": TerminationTermCfg(func=scene_mdp.cube_out_of_bounds),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1,
            env_spacing=1.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
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
