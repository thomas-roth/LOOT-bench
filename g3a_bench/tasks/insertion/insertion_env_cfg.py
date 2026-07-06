"""Environment configuration for the object-insertion task category."""

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
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from g3a_bench.tasks.common.env_cfg import (
    apply_play_overrides,
    fingertip_friction_event,
    fixed_prop_reset_event,
    joint_vel_hinge_curriculum,
    regularization_rewards,
    robot_reset_events,
)
from g3a_bench.tasks.common.table import TABLE_POS, get_table_spec
from g3a_bench.tasks.insertion import mdp as ins_mdp
from g3a_bench.tasks.insertion.assets import (
    BOARD_POS,
    get_board_spec,
    get_decor_spec,
    get_peg_spec,
    get_sheet_board_spec,
    get_sheet_spec,
)
from g3a_bench.tasks.insertion.variants import InsertionVariantCfg

COMMAND_NAME = "insertion"

REWARD_MODE_TERMS: dict[str, frozenset[str]] = {
    "success": frozenset(["success"]),
    "progress": frozenset(["inserted_count", "success"]),
}


def make_insertion_env_cfg(
    variant: InsertionVariantCfg,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the robot-agnostic insertion cfg.

    Robot-specific knobs are left as placeholders here (empty grasp site, action
    scale, friction geoms, end-effector subtree, viewer body, and no ``"robot"``
    scene entity). Fill them by calling ``wire_manipulator_into_env`` with a
    concrete ``ManipulatorInterface`` before registering the task.
    """
    two_sheet = variant.kind == "two_sheet"
    peg_names = tuple(f"peg_{i}" for i in range(len(variant.peg_shapes)))
    sheet_name = "sheet" if two_sheet else None

    board_init = EntityCfg.InitialStateCfg(pos=BOARD_POS)
    entities: dict[str, EntityCfg] = {
        # Shared visual-only workstation dressing (same for every variant);
        # its geom poses are already env-local, so it sits at the env origin.
        "decor": EntityCfg(spec_fn=get_decor_spec),
        # Shared visual-only table, same asset reused by every task category.
        "table": EntityCfg(
            spec_fn=get_table_spec,
            init_state=EntityCfg.InitialStateCfg(pos=TABLE_POS),
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

    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
        ),
        "pegs_rel_grasp": ObservationTermCfg(
            func=ins_mdp.pegs_rel_grasp,
            params={
                "command_name": COMMAND_NAME,
                # site_names set per-robot by manipulator wiring.
                "asset_cfg": SceneEntityCfg("robot", site_names=()),
            },
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
        # No generated_commands term: the board is pinned to the env origin,
        # so InsertionCommand.command is constant per variant; the actual
        # signal is socket_deltas above.
        "actions": ObservationTermCfg(func=mdp.last_action),
    }
    if two_sheet:
        actor_terms["sheet_rel_grasp"] = ObservationTermCfg(
            func=ins_mdp.sheet_rel_grasp,
            params={
                "command_name": COMMAND_NAME,
                # site_names set per-robot by manipulator wiring.
                "asset_cfg": SceneEntityCfg("robot", site_names=()),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

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
        **robot_reset_events(),
        "reset_board": fixed_prop_reset_event("board"),
        "reset_decor": fixed_prop_reset_event("decor"),
        "reset_table": fixed_prop_reset_event("table"),
        "fingertip_friction_slide": fingertip_friction_event(),
    }

    # Robot end-effector colliding with the parts tray ("Werkbox"). The tray is
    # a child body of the decor entity with real collision geometry (see
    # decor/workstation_decor.xml); the rest of the decor stays visual-only.
    ee_werkbox_collision_cfg = ContactSensorCfg(
        name="ee_werkbox_collision",
        # pattern set per-robot.
        primary=ContactMatch(mode="subtree", pattern="", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="werkbox", entity="decor"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    sensors = (ee_werkbox_collision_cfg,)

    # Sparse reward design: the task signal is only the (near-)binary outcome of
    # actually inserting pegs -- no dense distance/depth shaping. This is meant to
    # be paired with a demonstration pipeline (GELLO teleop -> offline-to-online
    # RL), which supplies the exploration that dense shaping previously provided.
    rewards = {
        "inserted_count": RewardTermCfg(
            func=ins_mdp.inserted_count_reward,
            weight=2.0,
            params={"command_name": COMMAND_NAME},
        ),
        "success": RewardTermCfg(
            func=ins_mdp.success_reward,
            weight=5.0,
            params={"command_name": COMMAND_NAME},
        ),
        **regularization_rewards(),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "ee_werkbox_collision": TerminationTermCfg(
            func=manipulation_mdp.illegal_contact,
            params={"sensor_name": "ee_werkbox_collision", "force_threshold": 10.0},
        ),
        "success": TerminationTermCfg(
            func=ins_mdp.insertion_complete,
            params={"command_name": COMMAND_NAME},
        ),
    }

    num_pegs = len(peg_names)
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities=entities,
            num_envs=1,
            env_spacing=2.0,
            sensors=sensors,
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
            distance=1.5,
            elevation=-15.0,
            azimuth=150.0,
        ),
        sim=SimulationCfg(
            # Two-sheet: two 16-box rings stacked on top of each other need
            # extra contact/constraint headroom.
            nconmax=120 + 120 * num_pegs + (200 if two_sheet else 0),
            njmax=900 + 90 * num_pegs + (400 if two_sheet else 0),
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
        apply_play_overrides(cfg)

    return cfg
