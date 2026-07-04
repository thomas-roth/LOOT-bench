"""Environment configuration for the cube (piece arrangement) task category."""

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

from g3a_bench.tasks.common.table import TABLE_POS, get_table_spec
from g3a_bench.tasks.cube import mdp as cube_mdp
from g3a_bench.tasks.cube.assets import get_piece_spec
from g3a_bench.tasks.cube.variants import CubeVariantCfg

COMMAND_NAME = "arrangement"

REWARD_MODE_TERMS: dict[str, frozenset[str]] = {
    "success": frozenset(["success"]),
    "progress": frozenset(["placed_slots", "success"]),
}


def make_cube_env_cfg(
    variant: CubeVariantCfg,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the robot-agnostic cube cfg.

    Robot-specific knobs are left as placeholders (empty grasp site, action
    scale 0.5, empty friction geoms, empty ee subtree, empty viewer body, no
    ``"robot"`` scene entity). Fill them by calling
    ``wire_manipulator_into_env`` with a concrete ``ManipulatorInterface``
    before registering the task.
    """
    piece_names = tuple(f"piece_{i}" for i in range(len(variant.pieces)))

    entities: dict[str, EntityCfg] = {
        # Shared visual-only table, same asset reused by every task category.
        "table": EntityCfg(
            spec_fn=get_table_spec,
            init_state=EntityCfg.InitialStateCfg(pos=TABLE_POS),
        ),
    }  # Robot added by wire_manipulator_into_env.
    for name, piece in zip(piece_names, variant.pieces, strict=True):
        entities[name] = EntityCfg(spec_fn=lambda p=piece, n=name: get_piece_spec(p, n))

    grasp_cfg = SceneEntityCfg("robot", site_names=())  # Set per-robot.

    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "pieces_rel_grasp": ObservationTermCfg(
            func=cube_mdp.pieces_rel_grasp,
            params={"command_name": COMMAND_NAME, "asset_cfg": grasp_cfg},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "slot_goal_deltas": ObservationTermCfg(
            func=cube_mdp.slot_goal_deltas,
            params={"command_name": COMMAND_NAME},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "slot_placed": ObservationTermCfg(
            func=cube_mdp.slot_placed_flags,
            params={"command_name": COMMAND_NAME},
        ),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": COMMAND_NAME},
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
            scale=0.5,  # Set per-robot by wire_manipulator_into_env.
            use_default_offset=True,
        )
    }

    commands: dict[str, CommandTermCfg] = {
        COMMAND_NAME: cube_mdp.PieceArrangementCommandCfg(
            entity_names=piece_names,
            variant=variant,
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
        )
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
        "reset_table": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("table"),
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

    rewards = {
        "placed_slots": RewardTermCfg(
            func=cube_mdp.placed_slots_reward,
            weight=2.0,
            params={"command_name": COMMAND_NAME},
        ),
        "success": RewardTermCfg(
            func=cube_mdp.success_reward,
            weight=2.0,
            params={"command_name": COMMAND_NAME},
        ),
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

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "success": TerminationTermCfg(
            func=cube_mdp.arrangement_complete,
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

    num_pieces = len(variant.pieces)
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities=entities,
            num_envs=1,
            env_spacing=1.5,
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
            body_name="",  # Set per-robot by wire_manipulator_into_env.
            distance=1.5,
            elevation=-15.0,
            azimuth=150.0,
        ),
        sim=SimulationCfg(
            nconmax=55 + 80 * num_pieces,
            njmax=600 + 60 * num_pieces,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=4,
        episode_length_s=10.0 * max(1, variant.num_goal_pieces),
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}
        cfg.terminations.pop("success")

    return cfg
