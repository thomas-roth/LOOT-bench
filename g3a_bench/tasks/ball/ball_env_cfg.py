"""Environment configuration for the ball task category."""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from g3a_bench.tasks.ball import mdp as ball_mdp
from g3a_bench.tasks.ball.assets import get_ball_spec, get_field_spec
from g3a_bench.tasks.ball.variants import BallVariantCfg
from g3a_bench.tasks.common.env_cfg import (
    apply_play_overrides,
    fingertip_friction_event,
    fixed_prop_reset_event,
    joint_vel_hinge_curriculum,
    regularization_rewards,
    robot_reset_events,
)
from g3a_bench.tasks.common.table import TABLE_POS, get_table_spec

COMMAND_NAME = "ball_target"

# The ball task has no dense progress term; both reward modes are identical.
REWARD_MODE_TERMS: dict[str, frozenset[str]] = {
    "success": frozenset(["success"]),
    "progress": frozenset(["success"]),
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
        # Shared visual-only table, same asset reused by every task category.
        "table": EntityCfg(
            spec_fn=get_table_spec,
            init_state=EntityCfg.InitialStateCfg(pos=TABLE_POS),
        ),
    }

    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
        ),
        "ball_rel_grasp": ObservationTermCfg(
            func=ball_mdp.ball_rel_grasp,
            params={
                "command_name": COMMAND_NAME,
                # site_names set per-robot by manipulator wiring.
                "asset_cfg": SceneEntityCfg("robot", site_names=()),
            },
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
        # No generated_commands term: BallCommand.command is the same
        # target-minus-ball vector as ball_to_target above.
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
            # site_names set per-robot by manipulator wiring.
            grasp_cfg=SceneEntityCfg("robot", site_names=()),
        )
    }

    events = {
        **robot_reset_events(),
        "reset_field": fixed_prop_reset_event("field"),
        "reset_table": fixed_prop_reset_event("table"),
        "fingertip_friction_slide": fingertip_friction_event(),
    }

    # Sparse reward design: the task signal is only the (near-)binary outcome of
    # the ball reaching the target -- no dense distance/strike/reach/push shaping.
    # This is meant to be paired with a demonstration pipeline (GELLO teleop ->
    # offline-to-online RL), which supplies the exploration that dense shaping
    # previously provided.
    rewards = {
        # Sparse task signal: only fires while the ball is at the target.
        "success": RewardTermCfg(
            func=ball_mdp.success_reward,
            weight=5.0,
            params={"command_name": COMMAND_NAME},
        ),
        **regularization_rewards(),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "success": TerminationTermCfg(
            func=ball_mdp.ball_at_target,
            params={"command_name": COMMAND_NAME},
        ),
        "out_of_bounds": TerminationTermCfg(
            func=ball_mdp.ball_out_of_bounds,
            params={"command_name": COMMAND_NAME, "max_radius": 1.2},
        ),
    }
    if variant.has_platform:
        terminations["off_platform"] = TerminationTermCfg(
            func=ball_mdp.ball_off_platform,
            params={"command_name": COMMAND_NAME},
        )
    if variant.has_wall:
        terminations["wall_collision"] = TerminationTermCfg(
            func=ball_mdp.ball_hits_wall,
            params={"command_name": COMMAND_NAME},
        )

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
        curriculum=joint_vel_hinge_curriculum(),
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
        apply_play_overrides(cfg)

    return cfg
