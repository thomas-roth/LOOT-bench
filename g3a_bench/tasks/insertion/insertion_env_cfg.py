"""Environment configuration for the object-insertion task category."""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from g3a_bench.tasks.common.env_cfg import make_manipulation_env_cfg
from g3a_bench.tasks.common.table import table_entity_cfg
from g3a_bench.tasks.insertion import mdp as ins_mdp
from g3a_bench.tasks.insertion.assets import (
    BOARD_POS,
    get_board_spec,
    get_decor_spec,
    get_peg_spec,
    get_sheet_board_spec,
    get_sheet_spec,
    get_sorter_piece_spec,
    get_sorter_spec,
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
    """Build the robot-agnostic insertion cfg (see ``tasks/common/env_cfg.py``
    for the placeholder/wiring convention)."""
    two_sheet = variant.kind == "two_sheet"
    sorter = variant.kind == "sorter"
    peg_names = tuple(f"peg_{i}" for i in range(len(variant.peg_shapes)))
    sheet_name = "sheet" if two_sheet else None

    board_init = EntityCfg.InitialStateCfg(pos=BOARD_POS)
    entities: dict[str, EntityCfg] = {
        # Shared visual-only workstation dressing (same for every variant).
        "decor": EntityCfg(spec_fn=get_decor_spec),
        "table": table_entity_cfg(),
    }
    if two_sheet:
        entities["board"] = EntityCfg(
            spec_fn=lambda v=variant: get_sheet_board_spec(v), init_state=board_init
        )
        entities["sheet"] = EntityCfg(spec_fn=lambda v=variant: get_sheet_spec(v))
    elif sorter:
        entities["board"] = EntityCfg(spec_fn=get_sorter_spec)
    else:
        entities["board"] = EntityCfg(
            spec_fn=lambda v=variant: get_board_spec(v), init_state=board_init
        )
    for name, shape, color in zip(
        peg_names, variant.peg_shapes, variant.peg_colors, strict=True
    ):
        if sorter:
            entities[name] = EntityCfg(
                spec_fn=lambda s=shape, c=color: get_sorter_piece_spec(s, c)
            )
        else:
            entities[name] = EntityCfg(
                spec_fn=lambda s=shape, c=color, n=name: get_peg_spec(s, c, n)
            )

    grasp_cfg = SceneEntityCfg("robot", site_names=())  # Set per-robot.

    task_obs_terms = {
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
    }
    # Appended after "actions" to keep the two-sheet observation layout
    # backward compatible.
    trailing_obs_terms = (
        {
            "sheet_rel_grasp": ObservationTermCfg(
                func=ins_mdp.sheet_rel_grasp,
                params={"command_name": COMMAND_NAME, "asset_cfg": grasp_cfg},
                noise=Unoise(n_min=-0.01, n_max=0.01),
            )
        }
        if two_sheet
        else None
    )

    # Sparse reward design: the task signal is only the (near-)binary outcome of
    # actually inserting pegs -- no dense distance/depth shaping. This is meant to
    # be paired with a demonstration pipeline (GELLO teleop -> offline-to-online
    # RL), which supplies the exploration that dense shaping previously provided.
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
    fixed_props = ("decor", "table") if sorter else ("decor", "table", "board")
    return make_manipulation_env_cfg(
        entities=entities,
        fixed_props=fixed_props,
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
        # Fixed boards make the raw command constant (mouths pinned to the env
        # origin); only the sorter's moving mouths + target one-hot carry info.
        command_obs=sorter,
        trailing_obs_terms=trailing_obs_terms,
        task_rewards=task_rewards,
        task_terminations=task_terminations,
        env_spacing=2.0,
        # Sorter: one commanded insert but the cube may need reorienting first.
        episode_length_s=30.0 if sorter else 10.0 + 8.0 * variant.num_sockets,
        # Two-sheet: two 16-box rings stacked on top of each other need
        # extra contact/constraint headroom. Sorter: ~90 convex mesh parts on
        # the cube colliding with pieces, table and gripper -- 500/2000 has
        # 1.7-8x margin over observed peak usage (~289 contacts at a fresh
        # reset, comfortably clean through the piece-tumbling latch check
        # down to 60/240) and is ~2x faster to step than the original
        # 2000/8000, which was far larger than the geometry ever needs.
        solver_iterations=15,
        solver_ls_iterations=25,
        nconmax=500 if sorter else 120 + 120 * num_pegs + (200 if two_sheet else 0),
        njmax=2000 if sorter else 900 + 90 * num_pegs + (400 if two_sheet else 0),
        play=play,
    )
