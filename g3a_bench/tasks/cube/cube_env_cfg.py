"""Environment configuration for the cube (piece arrangement) task category."""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from g3a_bench.tasks.common.env_cfg import make_manipulation_env_cfg
from g3a_bench.tasks.common.table import table_entity_cfg
from g3a_bench.tasks.cube import mdp as cube_mdp
from g3a_bench.tasks.cube.assets import get_maze_spec, get_piece_spec
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
    """Build the robot-agnostic cube cfg (see ``tasks/common/env_cfg.py``
    for the placeholder/wiring convention)."""
    piece_names = tuple(f"piece_{i}" for i in range(len(variant.pieces)))

    entities: dict[str, EntityCfg] = {
        "table": table_entity_cfg(),
    }  # Robot added by wire_manipulator_into_env.
    if variant.arrangement == "push":
        # Fixed-base barrier wall the pieces must be pushed through.
        entities["maze"] = EntityCfg(spec_fn=lambda v=variant: get_maze_spec(v))
    for name, piece in zip(piece_names, variant.pieces, strict=True):
        entities[name] = EntityCfg(spec_fn=lambda p=piece, n=name: get_piece_spec(p, n))

    grasp_cfg = SceneEntityCfg("robot", site_names=())  # Set per-robot.

    task_obs_terms = {
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
    }

    task_rewards = {
        "placed_slots": RewardTermCfg(
            func=cube_mdp.placed_slots_reward,
            weight=2.0,
            params={"command_name": COMMAND_NAME},
        ),
        "success": RewardTermCfg(
            func=cube_mdp.success_reward,
            weight=5.0,
            params={"command_name": COMMAND_NAME},
        ),
    }

    task_terminations = {
        "success": TerminationTermCfg(
            func=cube_mdp.arrangement_complete,
            params={"command_name": COMMAND_NAME},
        ),
    }

    num_pieces = len(variant.pieces)
    # Extra contact/constraint headroom for the maze wall segments ("push").
    maze_ncon = 60 if variant.arrangement == "push" else 0
    return make_manipulation_env_cfg(
        entities=entities,
        fixed_props=("table", "maze") if variant.arrangement == "push" else ("table",),
        command_name=COMMAND_NAME,
        command_cfg=cube_mdp.PieceArrangementCommandCfg(
            piece_names=piece_names,
            variant=variant,
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
        ),
        task_obs_terms=task_obs_terms,
        task_rewards=task_rewards,
        task_terminations=task_terminations,
        env_spacing=1.5,
        episode_length_s=10.0 * max(1, variant.num_goal_pieces),
        solver_iterations=10,
        solver_ls_iterations=20,
        nconmax=55 + 80 * num_pieces + maze_ncon,
        njmax=600 + 60 * num_pieces + 2 * maze_ncon,
        play=play,
    )
