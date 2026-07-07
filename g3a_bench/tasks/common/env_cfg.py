"""Shared tabletop-manipulation env-cfg skeleton for all task categories.

Every G3A task is the same workbench — a manipulator at a table with
joint-position control, proprioceptive + task observations, regularization
rewards with the standard curriculum, and a success/time-out episode — plus a
category-shaped core (entities, command, task observations/rewards/
terminations). :func:`make_manipulation_env_cfg` builds the skeleton and takes
the core as parameters; the per-category ``make_*_env_cfg`` are thin caps.

Robot-specific knobs are left as placeholders (empty grasp site, action scale
0.5, empty friction geoms, empty viewer body, no ``"robot"`` scene entity).
Fill them by calling :func:`~g3a_bench.tasks.common.manipulator_wiring.
wire_manipulator_into_env` with a concrete ``ManipulatorInterface`` before
registering the task.
"""

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
from mjlab.sensor import SensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_manipulation_env_cfg(
    *,
    entities: dict[str, EntityCfg],
    fixed_props: tuple[str, ...],
    command_name: str,
    command_cfg: CommandTermCfg,
    task_obs_terms: dict[str, ObservationTermCfg],
    command_obs: bool = True,
    trailing_obs_terms: dict[str, ObservationTermCfg] | None = None,
    task_rewards: dict[str, RewardTermCfg],
    task_terminations: dict[str, TerminationTermCfg],
    sensors: tuple[SensorCfg, ...] = (),
    env_spacing: float,
    episode_length_s: float,
    solver_iterations: int,
    solver_ls_iterations: int,
    nconmax: int,
    njmax: int,
    viewer_distance: float = 1.5,
    viewer_elevation: float = -15.0,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Assemble the shared workbench around a category core.

    Args:
        entities: all scene entities except the robot, in composition order.
        fixed_props: names of fixed-base (mocap) entities in ``entities`` to
            pin to ``env_origins + init_state.pos`` each reset; one
            ``reset_<name>`` event is generated per prop.
        command_name: key of the command term (also drives the generic
            ``command`` observation).
        command_cfg: the category's goal command.
        task_obs_terms: category observation terms, placed between the
            proprioceptive terms and ``command``/``actions``.
        command_obs: whether to expose the raw command as an observation.
            Disable when the task observations already encode the goal
            relative to the objects (the raw command would be redundant or
            constant).
        trailing_obs_terms: observation terms appended *after* ``actions``
            (layout-compatible late additions such as the two-sheet grasp
            term).
        task_rewards: category reward terms, placed before the shared
            regularization terms.
        task_terminations: category terminations, placed after ``time_out``
            (``success`` must be one of them for the play block to strip).
        sensors: scene sensors (e.g. the insertion contact sensor).
        play: strip corruption/curriculum/success-termination and make the
            episode endless, for interactive play.
    """
    actor_terms: dict[str, ObservationTermCfg] = {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
        ),
        **task_obs_terms,
        **(
            {
                "command": ObservationTermCfg(
                    func=mdp.generated_commands, params={"command_name": command_name}
                )
            }
            if command_obs
            else {}
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        **(trailing_obs_terms or {}),
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
        # Pin each fixed-base prop to env_origins + init_state.pos (mocap
        # prop convention).
        **{
            f"reset_{name}": EventTermCfg(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {},
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg(name),
                },
            )
            for name in fixed_props
        },
        "fingertip_friction_slide": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Per-robot.
                "operation": "abs",
                "distribution": "uniform",
                "axes": [0],
                "ranges": (0.6, 1.5),
            },
        ),
    }

    # Regularization (not task shaping): keep motions smooth and feasible.
    # Term names are mirrored in registration.REGULARIZATION_TERMS — keep in sync.
    rewards = {
        **task_rewards,
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
        **task_terminations,
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
            env_spacing=env_spacing,
            sensors=sensors,
        ),
        observations=observations,
        actions=actions,
        commands={command_name: command_cfg},
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="",  # Set per-robot.
            distance=viewer_distance,
            elevation=viewer_elevation,
            azimuth=150.0,
        ),
        sim=SimulationCfg(
            nconmax=nconmax,
            njmax=njmax,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=solver_iterations,
                ls_iterations=solver_ls_iterations,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=4,
        episode_length_s=episode_length_s,
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}
        cfg.terminations.pop("success")

    return cfg
