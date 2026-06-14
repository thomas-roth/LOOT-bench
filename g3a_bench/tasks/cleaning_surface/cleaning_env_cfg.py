from mjlab.envs import ManagerBasedRlEnvCfg, mdp
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
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from g3a_bench.tasks.cleaning_surface import mdp as cleaning_mdp


def make_cleaning_env_cfg() -> ManagerBasedRlEnvCfg:
    """Base cleaning-surface task configuration (robot-agnostic)."""

    grasp_cfg = SceneEntityCfg("robot", site_names=("grasp_site",))
    finger_cfg = SceneEntityCfg("robot", joint_names=("left_finger", "right_finger"))

    actor_terms = {
        # Robot state (21 dims)
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        # End-effector (6 dims)
        "ee_pos": ObservationTermCfg(
            func=cleaning_mdp.site_pos_w,
            params={"asset_cfg": grasp_cfg},
        ),
        "ee_yaw": ObservationTermCfg(
            func=cleaning_mdp.site_yaw_cos_sin,
            params={"asset_cfg": grasp_cfg},
        ),
        "gripper_opening": ObservationTermCfg(
            func=cleaning_mdp.gripper_opening,
            params={"asset_cfg": finger_cfg},
        ),
        # Sponge (7 dims)
        "sponge_pos": ObservationTermCfg(
            func=cleaning_mdp.root_pos_w,
            params={"entity_name": "sponge"},
        ),
        "sponge_quat": ObservationTermCfg(
            func=cleaning_mdp.root_quat_w,
            params={"entity_name": "sponge"},
        ),
        # Dirt state (5 dims)
        "dirt_cleaned_mask": ObservationTermCfg(func=cleaning_mdp.dirt_cleaned_mask),
        "cleaned_ratio": ObservationTermCfg(func=cleaning_mdp.cleaned_ratio),
        # Debris (6 dims)
        "debris_0_pos": ObservationTermCfg(
            func=cleaning_mdp.root_pos_w,
            params={"entity_name": "debris_0"},
        ),
        "debris_1_pos": ObservationTermCfg(
            func=cleaning_mdp.root_pos_w,
            params={"entity_name": "debris_1"},
        ),
        # Goal context (18 dims)
        "bin_pos": ObservationTermCfg(func=cleaning_mdp.bin_pos),
        "task_variant": ObservationTermCfg(func=cleaning_mdp.task_variant_onehot),
        "target_dirt_positions": ObservationTermCfg(
            func=cleaning_mdp.target_dirt_positions
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
            scale=0.5,
            use_default_offset=True,
        )
    }

    commands: dict[str, CommandTermCfg] = {
        "cleaning_task": cleaning_mdp.CleaningSurfaceCommandCfg(
            resampling_time_range=(1.0e9, 1.0e9),
        ),
    }

    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {}},
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
        "update_cleaning_logic": EventTermCfg(
            func=cleaning_mdp.update_cleaning_logic,
            mode="step",
        ),
    }

    rewards = {
        "cleaning": RewardTermCfg(func=cleaning_mdp.cleaning_reward, weight=1.0),
        "reach_sponge": RewardTermCfg(
            func=cleaning_mdp.reach_sponge_reward,
            weight=0.1,
            params={"asset_cfg": grasp_cfg},
        ),
        "wipe_progress": RewardTermCfg(
            func=cleaning_mdp.wipe_progress_reward, weight=0.3
        ),
        "debris_approach": RewardTermCfg(
            func=cleaning_mdp.debris_approach_reward, weight=0.1
        ),
        "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=cleaning_mdp.time_out, time_out=True),
        "success": TerminationTermCfg(
            func=cleaning_mdp.cleaning_success,
            params={"command_name": "cleaning_task"},
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1,
            env_spacing=1.5,
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
            body_name="",
            distance=1.5,
            elevation=-5.0,
            azimuth=120.0,
        ),
        sim=SimulationCfg(
            nconmax=120,
            njmax=800,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=4,
        episode_length_s=30.0,
    )
