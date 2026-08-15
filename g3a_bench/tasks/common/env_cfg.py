"""Shared tabletop-manipulation env-cfg skeleton for all task categories.

Every G3A task is the same workbench — a manipulator at a table with
joint-position control, proprioceptive + task observations, and a
success/time-out episode — plus a category-shaped core (entities, command,
task observations/rewards/terminations). :func:`make_manipulation_env_cfg`
builds the skeleton and takes the core as parameters; the per-category
``make_*_env_cfg`` are thin caps.

Robot-specific knobs are left as placeholders (empty grasp site, empty viewer
body, no ``"robot"`` scene entity).
Fill them by calling :func:`~g3a_bench.tasks.common.manipulator_wiring.
wire_manipulator_into_env` with a concrete ``ManipulatorInterface`` before
registering the task.

**No observation carries noise, and no group corrupts.** Observation noise is a
robustness device for an on-policy run that collects its own data; this repo
trains QC-FQL offline from recorded demonstrations, and there the noise only
splits the two halves apart. The demonstrations were recorded through the
uncorrupted observation, so a corrupted training observation would describe a
state the operator never saw. Both paths already forced it off -- the adapter
in ``g3a_bench/qc/env.py`` and the recorder's ``play`` cfg -- and the terms are
now free of it, so the config says what it does instead of relying on two
callers to undo it. See ``docs/adr/015-observations-carry-no-noise.md``.
"""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
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
from mjlab.viewer import ViewerConfig

from g3a_bench.tasks.common.actions import NormalizedJointPositionActionCfg
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
    start_pose_noise: float = 0.0,
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
        task_rewards: category reward terms.
        task_terminations: category terminations, placed after ``time_out``.
        sensors: scene sensors (e.g. an end-effector contact sensor).
        start_pose_noise: half-width [rad] of the uniform sample added to
            every robot joint at reset. ``0.0`` (the default) gives a
            bit-identical start every episode, which every demo replay check
            needs -- and which offers a policy exactly one start state to
            generalize from. Raise it for training. Read back off
            ``cfg.events["reset_robot_joints"].params["position_range"]``,
            which is also where to override it on an already-built cfg.
        play: make the episode endless, for interactive play. Physics are
            unchanged, so demos and eval run under the same dynamics as
            training. The ``success`` termination is kept — live_bridge demo
            playback needs episodes to terminate on success.
    """
    actor_terms: dict[str, ObservationTermCfg] = {
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
        "ee_pose": ObservationTermCfg(
            func=ee_pose_w, params={"asset_cfg": SceneEntityCfg("robot", site_names=())}
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
        # Named, so this reads the action term's own buffer rather than the
        # manager's. The manager stores the policy output before any term
        # touches it, and the term clamps to the action box (see
        # tasks/common/actions.py) -- so the unnamed form would report an
        # action the simulator did not apply. Teleoperation is where the two
        # differ: gravity compensation can leave the box near a joint limit.
        "actions": ObservationTermCfg(
            func=mdp.last_action, params={"action_name": "joint_pos"}
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(dict(actor_terms)),
        "critic": ObservationGroupCfg(dict(actor_terms)),
    }

    actions: dict[str, ActionTermCfg] = {
        # Scale and offset are derived from the compiled model, not set here
        # and not set per-robot: [-1, 1] spans each joint's own travel. See
        # tasks/common/actions.py for why a residual scale cannot work with a
        # bounded policy.
        "joint_pos": NormalizedJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
        )
    }

    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {}},
        ),
        # Robots declaring a reset pose get the absolute-pose reset swapped in
        # by wire_manipulator_into_env; the ranges below survive the swap.
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-start_pose_noise, start_pose_noise),
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
                iterations=100,
                ls_iterations=50,
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
        # Keep the reward exactly what the term returns. mjlab's default
        # multiplies every reward by step_dt, which would tie the numbers
        # baked into a recorded dataset to the decimation above. See the
        # convention note in tasks/common/rewards.py.
        scale_rewards_by_dt=False,
    )

    if play:
        cfg.episode_length_s = int(1e9)

    return cfg
