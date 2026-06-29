"""Environment configuration for the ball task category."""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from g3a_bench.tasks.ball import mdp as ball_mdp
from g3a_bench.tasks.ball.assets import get_ball_spec, get_field_spec
from g3a_bench.tasks.ball.variants import BallVariantCfg

COMMAND_NAME = "ball_target"


def _action_rewards(
    variant: BallVariantCfg, grasp_cfg: SceneEntityCfg
) -> dict[str, RewardTermCfg]:
    """Action-specific dense shaping (golf strike / push guide / throw).

    Not used in the default sparse reward setup; kept defined so re-enabling it
    is a one-line ``rewards.update(_action_rewards(...))`` in ``make_ball_env_cfg``.
    """
    if variant.action == "golf":
        return {
            "strike": RewardTermCfg(
                func=ball_mdp.strike_reward,
                weight=0.5,
                params={"command_name": COMMAND_NAME},
            ),
            "reach": RewardTermCfg(
                func=ball_mdp.gripper_ball_reach_reward,
                weight=0.3,
                params={
                    "command_name": COMMAND_NAME,
                    "std": 0.15,
                    "asset_cfg": grasp_cfg,
                },
            ),
            "contact_time": RewardTermCfg(
                func=ball_mdp.contact_time_penalty,
                weight=-0.2,
                params={
                    "command_name": COMMAND_NAME,
                    "contact_dist": 0.06,
                    "asset_cfg": grasp_cfg,
                },
            ),
        }
    if variant.action == "push":
        return {
            "push_guide": RewardTermCfg(
                func=ball_mdp.push_guide_reward,
                weight=1.5,
                params={
                    "command_name": COMMAND_NAME,
                    "reaching_std": 0.15,
                    "bringing_std": 0.2,
                    "asset_cfg": grasp_cfg,
                },
            ),
        }
    # throw
    return {
        "reach": RewardTermCfg(
            func=ball_mdp.gripper_ball_reach_reward,
            weight=0.5,
            params={"command_name": COMMAND_NAME, "std": 0.15, "asset_cfg": grasp_cfg},
        ),
        "throw_height": RewardTermCfg(
            func=ball_mdp.throw_height_reward,
            weight=0.5,
            params={"command_name": COMMAND_NAME},
        ),
    }


def make_ball_env_cfg(
    variant: BallVariantCfg,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the robot-agnostic ball cfg.

    Robot-specific knobs are left as placeholders here (empty grasp site, action
    scale, friction geoms, viewer body, and no ``"robot"`` scene entity). Fill
    them by calling ``wire_manipulator_into_env`` with a concrete
    ``ManipulatorInterface`` before registering the task.
    """
    entities: dict[str, EntityCfg] = {
        "field": EntityCfg(spec_fn=lambda v=variant: get_field_spec(v)),
        "ball": EntityCfg(spec_fn=get_ball_spec),
    }

    grasp_cfg = SceneEntityCfg("robot", site_names=())  # Set per-robot.

    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
        ),
        "ball_rel_grasp": ObservationTermCfg(
            func=ball_mdp.ball_rel_grasp,
            params={"command_name": COMMAND_NAME, "asset_cfg": grasp_cfg},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "ball_to_target": ObservationTermCfg(
            func=ball_mdp.ball_to_target,
            params={"command_name": COMMAND_NAME},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "ball_velocity": ObservationTermCfg(
            func=ball_mdp.ball_velocity,
            params={"command_name": COMMAND_NAME},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "command": ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": COMMAND_NAME}
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }

    observations = {
        "actor": ObservationGroupCfg(dict(actor_terms), enable_corruption=True),
        "critic": ObservationGroupCfg(dict(actor_terms), enable_corruption=False),
    }

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,  # Set per-robot.
            use_default_offset=True,
        )
    }

    commands: dict[str, CommandTermCfg] = {
        COMMAND_NAME: ball_mdp.BallCommandCfg(
            ball_name="ball",
            variant=variant,
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
            grasp_cfg=grasp_cfg,
        )
    }

    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {}},
        ),
        "reset_field": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("field"),
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
        "fingertip_friction_slide": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
                "operation": "abs",
                "distribution": "uniform",
                "axes": [0],
                "ranges": (0.6, 1.5),
            },
        ),
    }

    # Sparse reward design: the task signal is only the (near-)binary outcome of
    # the ball reaching the target -- no dense distance/strike/reach/push shaping.
    # This is meant to be paired with a demonstration pipeline (GELLO teleop ->
    # offline-to-online RL), which supplies the exploration that dense shaping
    # previously provided. The dense terms (``ball_distance_reward`` and the
    # action-specific shaping assembled by ``_action_rewards``) are left defined so
    # this is a one-line switch back if needed (re-add the ``rewards.update`` below).
    rewards = {
        # Sparse task signal: only fires while the ball is at the target.
        "success": RewardTermCfg(
            func=ball_mdp.success_reward,
            weight=5.0,
            params={"command_name": COMMAND_NAME},
        ),
        # Regularization (not task shaping): keep motions smooth and feasible.
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        "joint_vel_hinge": RewardTermCfg(
            func=manipulation_mdp.joint_velocity_hinge_penalty,
            weight=-0.01,
            params={
                "max_vel": 0.5,
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }
    # Dense shaping, disabled for the sparse + demonstration setup. Re-enable with:
    # rewards.update(_action_rewards(variant, grasp_cfg))

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "success": TerminationTermCfg(
            func=ball_mdp.ball_at_target,
            params={"command_name": COMMAND_NAME, "hold_time": 0.0},
        ),
        "out_of_bounds": TerminationTermCfg(
            func=ball_mdp.ball_out_of_bounds,
            params={"command_name": COMMAND_NAME, "max_radius": 1.2},
        ),
        "ground_collision": TerminationTermCfg(
            func=ball_mdp.ball_hits_ground,
            params={"command_name": COMMAND_NAME},
        ),
        "wall_collision": TerminationTermCfg(
            func=ball_mdp.ball_hits_wall,
            params={"command_name": COMMAND_NAME},
        ),
    }

    curriculum = {
        "joint_vel_hinge_weight": CurriculumTermCfg(
            func=manipulation_mdp.reward_curriculum,
            params={
                "reward_name": "joint_vel_hinge",
                "stages": [
                    {"step": 0, "weight": -0.01},
                    {"step": 500 * 24, "weight": -0.1},
                    {"step": 1000 * 24, "weight": -1.0},
                ],
            },
        ),
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities=entities,
            num_envs=1,
            env_spacing=2.5,
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
            entity_name="robot",
            body_name="",  # Set per-robot.
            distance=1.6,
            elevation=-20.0,
            azimuth=150.0,
        ),
        sim=SimulationCfg(
            nconmax=150,
            njmax=800,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=12,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=4,
        episode_length_s=12.0,
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}
        cfg.terminations.pop("success")

    return cfg
