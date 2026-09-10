"""Archived peg-insertion env config. Unregistered, see README."""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from loot_bench.tasks.common.commands import GRIP_CONTACT_SENSOR_NAME
from loot_bench.tasks.common.env_cfg import make_manipulation_env_cfg
from loot_bench.tasks.common.table import table_entity_cfg
from loot_bench.tasks.peg_insertion import mdp as ins_mdp
from loot_bench.tasks.peg_insertion.assets import (
    BOARD_POS,
    get_board_spec,
    get_peg_spec,
    get_sheet_board_spec,
    get_sheet_spec,
)
from loot_bench.tasks.peg_insertion.variants import InsertionVariantCfg

COMMAND_NAME = "task"


def make_insertion_env_cfg(
    variant: InsertionVariantCfg,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the robot-agnostic insertion cfg (see ``tasks/common/env_cfg.py``
    for the placeholder/wiring convention)."""
    two_sheet = variant.kind == "two_sheet"
    peg_names = tuple(f"peg_{i}" for i in range(len(variant.peg_shapes)))
    sheet_name = "sheet" if two_sheet else None

    board_init = EntityCfg.InitialStateCfg(pos=BOARD_POS)
    entities: dict[str, EntityCfg] = {
        "table": table_entity_cfg(),
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

    task_obs_terms = {
        "board_pose": ObservationTermCfg(
            func=ins_mdp.board_pose,
            params={"command_name": COMMAND_NAME},
        ),
        "peg_poses": ObservationTermCfg(
            func=ins_mdp.peg_poses,
            params={"command_name": COMMAND_NAME},
        ),
        "socket_inserted": ObservationTermCfg(
            func=ins_mdp.socket_inserted_flags,
            params={"command_name": COMMAND_NAME},
        ),
    }
    if two_sheet:
        task_obs_terms["sheet_pose"] = ObservationTermCfg(
            func=ins_mdp.sheet_pose,
            params={"command_name": COMMAND_NAME},
        )

    task_rewards = {
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
    }

    task_terminations = {
        "success": TerminationTermCfg(
            func=ins_mdp.insertion_complete,
            params={"command_name": COMMAND_NAME},
        ),
    }

    num_pegs = len(peg_names)
    sensors = (
        ContactSensorCfg(
            name=GRIP_CONTACT_SENSOR_NAME,
            primary=ContactMatch(mode="geom", pattern="", entity="robot"),
            fields=("force",),
        ),
    )
    return make_manipulation_env_cfg(
        entities=entities,
        fixed_props=("table", "board"),
        command_name=COMMAND_NAME,
        command_cfg=ins_mdp.InsertionCommandCfg(
            board_name="board",
            peg_names=peg_names,
            sheet_name=sheet_name,
            variant=variant,
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
        ),
        task_obs_terms=task_obs_terms,
        command_obs=False,
        task_rewards=task_rewards,
        task_terminations=task_terminations,
        sensors=sensors,
        env_spacing=2.0,
        # Two-sheet gets an extra time budget for the alignment sub-task
        episode_length_s=10.0 + 8.0 * variant.num_sockets + (8.0 if two_sheet else 0.0),
        # Two-sheet: two 16-box rings stacked on top of each other need extra
        # contact/constraint headroom.
        nconmax=120 + 120 * num_pegs + (200 if two_sheet else 0),
        njmax=900 + 90 * num_pegs + (400 if two_sheet else 0),
        play=play,
    )
