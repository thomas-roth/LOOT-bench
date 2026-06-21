"""Environment configuration for the object-insertion task category."""

from typing import Literal

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import DifferentialIKActionCfg, JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from g3a_bench.tasks.insertion import mdp as ins_mdp
from g3a_bench.tasks.insertion.assets import (
    BOARD_POS,
    DECOR_POS,
    get_board_spec,
    get_decor_spec,
    get_peg_spec,
    get_sheet_board_spec,
    get_sheet_spec,
)
from g3a_bench.tasks.insertion.variants import InsertionVariantCfg

COMMAND_NAME = "insertion"

ActionMode = Literal["eef", "joint"]
"""``"eef"``: task-space end-effector control (differential IK) for the arm plus
a direct gripper joint action -- the standard mode. ``"joint"``: the original
joint-position (angle-based) control over all actuators, kept as an option."""


def _make_actions(action_mode: ActionMode) -> dict[str, ActionTermCfg]:
    """Build the action block for the chosen control mode.

    Robot-specific fields (actuator names, IK frame, scales) are left as
    placeholders and filled by ``wire_manipulator_into_env``.
    """
    if action_mode == "joint":
        return {
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=0.5,  # Set per-robot.
                use_default_offset=True,
            )
        }
    # EEF mode: differential-IK on the arm + a direct gripper joint action.
    return {
        "arm": DifferentialIKActionCfg(
            entity_name="robot",
            actuator_names=(),  # Arm joints set per-robot.
            frame_type="site",
            frame_name="",  # Grasp site set per-robot.
            use_relative_mode=True,
            delta_pos_scale=0.05,  # Action [-1,1] -> up to 5 cm EEF move/step.
            delta_ori_scale=0.1,  # Action [-1,1] -> up to 0.1 rad/step.
            orientation_weight=1.0,  # 6D pose tracking (peg axis alignment).
            joint_limit_weight=1.0,  # Steer away from joint limits.
        ),
        "gripper": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(),  # Gripper joints set per-robot.
            scale=1.0,  # Set per-robot.
            use_default_offset=True,
        ),
    }


def make_insertion_env_cfg(
    variant: InsertionVariantCfg,
    play: bool = False,
    action_mode: ActionMode = "eef",
) -> ManagerBasedRlEnvCfg:
    """Build the robot-agnostic insertion cfg.

    Robot-specific knobs are left as placeholders here (empty grasp site, action
    scale, friction geoms, end-effector subtree, viewer body, and no ``"robot"``
    scene entity). Fill them by calling ``wire_manipulator_into_env`` with a
    concrete ``ManipulatorInterface`` before registering the task.

    ``action_mode`` selects task-space EEF control (``"eef"``, the standard) or
    joint-position control (``"joint"``); see :data:`ActionMode`.
    """
    two_sheet = variant.kind == "two_sheet"
    peg_names = tuple(f"peg_{i}" for i in range(len(variant.peg_shapes)))
    sheet_name = "sheet" if two_sheet else None

    board_init = EntityCfg.InitialStateCfg(pos=BOARD_POS)
    entities: dict[str, EntityCfg] = {
        # Shared visual-only workstation dressing (same for every variant).
        "decor": EntityCfg(
            spec_fn=get_decor_spec,
            init_state=EntityCfg.InitialStateCfg(pos=DECOR_POS),
        ),
    }
    if two_sheet:
        entities["board"] = EntityCfg(
            spec_fn=lambda v=variant: get_sheet_board_spec(v), init_state=board_init
        )
        entities["sheet"] = EntityCfg(spec_fn=lambda v=variant: get_sheet_spec(v))
    else:
        entities["board"] = EntityCfg(
            spec_fn=lambda v=variant: get_board_spec(v), init_state=board_init
        )
    for name, shape, color in zip(
        peg_names, variant.peg_shapes, variant.peg_colors, strict=True
    ):
        entities[name] = EntityCfg(
            spec_fn=lambda s=shape, c=color, n=name: get_peg_spec(s, c, n)
        )

    grasp_cfg = SceneEntityCfg("robot", site_names=())  # Set per-robot.

    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
        ),
        "pegs_rel_grasp": ObservationTermCfg(
            func=ins_mdp.pegs_rel_grasp,
            params={"command_name": COMMAND_NAME, "asset_cfg": grasp_cfg},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "socket_deltas": ObservationTermCfg(
            func=ins_mdp.socket_deltas,
            params={"command_name": COMMAND_NAME},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "socket_inserted": ObservationTermCfg(
            func=ins_mdp.socket_inserted_flags,
            params={"command_name": COMMAND_NAME},
        ),
        "command": ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": COMMAND_NAME}
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }
    if two_sheet:
        actor_terms["sheet_rel_grasp"] = ObservationTermCfg(
            func=ins_mdp.sheet_rel_grasp,
            params={"command_name": COMMAND_NAME, "asset_cfg": grasp_cfg},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

    observations = {
        "actor": ObservationGroupCfg(dict(actor_terms), enable_corruption=True),
        "critic": ObservationGroupCfg(dict(actor_terms), enable_corruption=False),
    }

    actions = _make_actions(action_mode)

    commands: dict[str, CommandTermCfg] = {
        COMMAND_NAME: ins_mdp.InsertionCommandCfg(
            board_name="board",
            peg_names=peg_names,
            sheet_name=sheet_name,
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
        "reset_board": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("board"),
            },
        ),
        "reset_decor": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("decor"),
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

    ee_ground_collision_cfg = ContactSensorCfg(
        name="ee_ground_collision",
        # pattern set per-robot.
        primary=ContactMatch(mode="subtree", pattern="", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    rewards = {
        "staged_next_socket": RewardTermCfg(
            func=ins_mdp.staged_next_socket_reward,
            weight=1.0,
            params={
                "command_name": COMMAND_NAME,
                "reaching_std": 0.2,
                "bringing_std": 0.3,
                "asset_cfg": grasp_cfg,
            },
        ),
        "insertion_depth": RewardTermCfg(
            func=ins_mdp.insertion_depth_reward,
            weight=2.0,
            params={"command_name": COMMAND_NAME},
        ),
        "inserted_count": RewardTermCfg(
            func=ins_mdp.inserted_count_reward,
            weight=2.0,
            params={"command_name": COMMAND_NAME},
        ),
        "success": RewardTermCfg(
            func=ins_mdp.success_reward,
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
    if two_sheet:
        rewards["sheet_alignment"] = RewardTermCfg(
            func=ins_mdp.sheet_alignment_reward,
            weight=1.5,
            params={
                "command_name": COMMAND_NAME,
                "reaching_std": 0.2,
                "bringing_std": 0.05,
                "asset_cfg": grasp_cfg,
            },
        )

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "ee_ground_collision": TerminationTermCfg(
            func=manipulation_mdp.illegal_contact,
            params={"sensor_name": "ee_ground_collision", "force_threshold": 10.0},
        ),
        "success": TerminationTermCfg(
            func=ins_mdp.insertion_complete,
            params={"command_name": COMMAND_NAME, "hold_time": 0.5},
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

    num_pegs = len(peg_names)
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities=entities,
            num_envs=1,
            env_spacing=2.0,
            sensors=(ee_ground_collision_cfg,),
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
            distance=1.5,
            elevation=-15.0,
            azimuth=150.0,
        ),
        sim=SimulationCfg(
            nconmax=120 + 120 * num_pegs,
            njmax=900 + 90 * num_pegs,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=15,
                ls_iterations=25,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=4,
        episode_length_s=10.0 + 8.0 * variant.num_sockets,
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}
        cfg.terminations.pop("success")

    return cfg
