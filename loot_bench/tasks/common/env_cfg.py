"""Shared workbench for all tasks. Categories pass core parts as parameters.

The control rate and the action term live here and nowhere else: a
per-category fork would put two MDP conventions in one workbench. Categories
choose their own ``episode_length_s``, not their rate or their action space.

Joint-absolute normalized control at 20 Hz: one 7-wide term, ``[-1, 1]``
spanning each joint's travel (see ``tasks/common/actions.py``).

Robot placeholders are filled by wire_manipulator_into_env. No observation
carries noise (ADR-015).
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

from loot_bench.tasks.common.actions import (
    JOINT_POS_ACTION_NAME,
    NormalizedJointPositionActionCfg,
)
from loot_bench.tasks.common.commands import GRIP_CONTACT_SENSOR_NAME
from loot_bench.tasks.common.observations import ee_pose_w, pad_contact_force_obs

SIM_TIMESTEP = 0.005
"""[s] physics timestep, shared by every task."""

CONTROL_DECIMATION = 10
"""Physics steps per control step. 10 x 0.005 s is 20 Hz.

Both QC-validated reference domains run at 20 Hz, and ``horizon_length=5``
then spans 0.25 s -- the middle of the band QC's own Fig. 6 sweeps as working.
At the previous 50 Hz the same chunk spanned 0.10 s, nearer the failing end.
See ADR-016."""

CONTROL_STEP_S = SIM_TIMESTEP * CONTROL_DECIMATION
"""[s] one control step, 0.05 s. Stamped into every dataset, so changing the
rate invalidates recordings loudly instead of silently (ADR-011)."""


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
        # The release gate's own number, so an episode never ends for a
        # reason the policy cannot see. Only when the task declares the
        # sensor: ball has no release gate and stays untouched.
        **(
            {
                "pad_contact_force": ObservationTermCfg(
                    func=pad_contact_force_obs,
                    params={"sensor_name": GRIP_CONTACT_SENSOR_NAME},
                )
            }
            if any(s.name == GRIP_CONTACT_SENSOR_NAME for s in sensors)
            else {}
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
        # No `actions` term: it made "repeat the last action" a trivially
        # strong regressor (0.0242 against the BC gate bar of 0.0766 then in
        # force; the bar has since been withdrawn, ADR-012), so a
        # policy could score on the demos without reading the scene (ADR-016).
    }

    observations = {
        "actor": ObservationGroupCfg(dict(actor_terms)),
        "critic": ObservationGroupCfg(dict(actor_terms)),
    }

    # One joint-absolute term, 7 wide. Limits come from the compiled model;
    # see tasks/common/actions.py. No per-robot wiring needed.
    actions: dict[str, ActionTermCfg] = {
        JOINT_POS_ACTION_NAME: NormalizedJointPositionActionCfg(
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
        # Same as the mjlab manipulation example, with nativeccd disabled.
        sim=SimulationCfg(
            nconmax=nconmax,
            njmax=njmax,
            mujoco=MujocoCfg(
                timestep=SIM_TIMESTEP,
                iterations=100,
                ls_iterations=50,
                impratio=10,
                cone="elliptic",
                # Analytic box-box collider. Stable contacts for boxes.
                disableflags=("nativeccd",),
            ),
        ),
        decimation=CONTROL_DECIMATION,
        episode_length_s=episode_length_s,
        # Keep raw reward values, unscaled by step dt.
        scale_rewards_by_dt=False,
    )

    if play:
        cfg.episode_length_s = int(1e9)

    return cfg
