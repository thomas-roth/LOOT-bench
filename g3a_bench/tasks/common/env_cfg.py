"""Shared tabletop-manipulation env-cfg skeleton for all task categories.

Every G3A task is the same workbench — a manipulator at a table with
joint-position control, proprioceptive + task observations, and a
success/time-out episode — plus a category-shaped core (entities, command,
task observations/rewards/terminations). :func:`make_manipulation_env_cfg`
builds the skeleton and takes the core as parameters; the per-category
``make_*_env_cfg`` are thin caps.

Robot-specific knobs are left as placeholders (empty grasp site, action scale
0.5, empty viewer body, no ``"robot"`` scene entity).
Fill them by calling :func:`~g3a_bench.tasks.common.manipulator_wiring.
wire_manipulator_into_env` with a concrete ``ManipulatorInterface`` before
registering the task.
"""

from mjlab.entity import EntityCfg
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
from mjlab.sensor import SensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from g3a_bench.tasks.common.observations import ee_pose_w


def make_manipulation_env_cfg(
    *,
    entities: dict[str, EntityCfg],
    fixed_props: tuple[str, ...],
    command_name: str,
    command_cfg: CommandTermCfg,
    task_obs_terms: dict[str, ObservationTermCfg],
    command_obs: bool = True,
    task_rewards: dict[str, RewardTermCfg],
    task_terminations: dict[str, TerminationTermCfg],
    sensors: tuple[SensorCfg, ...] = (),
    env_spacing: float,
    episode_length_s: float,
    nconmax: int,
    njmax: int,
    viewer_distance: float = 1.5,
    viewer_elevation: float = -15.0,
    play: bool = False,
    play_timestep: float = 0.01,
    play_decimation: int = 1,
    play_solver_iterations: int = 20,
    play_solver_ls_iterations: int = 10,
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
        task_rewards: category reward terms.
        task_terminations: category terminations, placed after ``time_out``.
        sensors: scene sensors (e.g. an end-effector contact sensor).
        play: strip observation corruption, make the episode endless, and
            retune physics for real-time responsiveness (see ``play_*``
            args), for interactive play. The ``success`` termination is
            kept — live_bridge demo playback needs episodes to terminate
            on success.
        play_timestep, play_decimation, play_solver_iterations,
            play_solver_ls_iterations: physics knobs applied only when
            ``play=True``. A bigger timestep with fewer substeps
            (``decimation``) cuts wall-clock cost per ``env.step()`` for
            live/teleop use; more solver iterations with a shorter
            line-search partially offsets the coarser timestep for
            constraint stability. Override per category if the default
            (tuned against the cube task) causes instability, e.g. a fast
            projectile tunneling through thin geometry.
    """
    actor_terms: dict[str, ObservationTermCfg] = {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
        ),
        "ee_pose": ObservationTermCfg(
            func=ee_pose_w,
            params={"asset_cfg": SceneEntityCfg("robot", site_names=())},
            noise=Unoise(n_min=-0.01, n_max=0.01),
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
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        **task_terminations,
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
        rewards=dict(task_rewards),
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="",  # Set per-robot.
            distance=viewer_distance,
            elevation=viewer_elevation,
            azimuth=150.0,
            enable_shadows=False,
            enable_reflections=False,
        ),
        # Same settings as the mjlab manipulation example, except task-specific
        # values and nativeccd disabled.
        sim=SimulationCfg(
            nconmax=nconmax,
            njmax=njmax,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
                # Disabling nativeccd makes MuJoCo Warp use the analytic
                # box-box collider (up to 8 contacts with per-contact depth)
                # instead of GJK/EPA (<=4 contacts sharing one depth/frame),
                # which jitters. Affects all box-box pairs: objects on the
                # table top and in the box finger pads. No-op for other pairs.
                disableflags=("nativeccd",),
            ),
        ),
        decimation=4,
        episode_length_s=episode_length_s,
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.sim.mujoco.timestep = play_timestep
        cfg.decimation = play_decimation
        cfg.sim.mujoco.iterations = play_solver_iterations
        cfg.sim.mujoco.ls_iterations = play_solver_ls_iterations

    return cfg
