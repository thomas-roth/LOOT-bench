"""Environment configuration for the shape-sorter task category."""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from g3a_bench.tasks.common.env_cfg import make_manipulation_env_cfg
from g3a_bench.tasks.common.table import table_entity_cfg
from g3a_bench.tasks.shape_sorter import mdp as sorter_mdp
from g3a_bench.tasks.shape_sorter.assets import get_sorter_piece_spec, get_sorter_spec
from g3a_bench.tasks.shape_sorter.variants import ShapeSorterVariantCfg

COMMAND_NAME = "task"


def make_shape_sorter_env_cfg(
    variant: ShapeSorterVariantCfg,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the robot-agnostic shape-sorter cfg (see ``tasks/common/env_cfg.py``
    for the placeholder/wiring convention)."""
    piece_names = tuple(f"piece_{i}" for i in range(variant.num_pieces))

    entities: dict[str, EntityCfg] = {
        "table": table_entity_cfg(),
        "cube": EntityCfg(spec_fn=get_sorter_spec),
    }  # Robot added by wire_manipulator_into_env.
    for name, shape, color in zip(
        piece_names, variant.shapes, variant.colors, strict=True
    ):
        entities[name] = EntityCfg(
            spec_fn=lambda s=shape, c=color: get_sorter_piece_spec(s, c)
        )

    grasp_cfg = SceneEntityCfg("robot", site_names=())  # Set per-robot.

    task_obs_terms = {
        "pegs_rel_grasp": ObservationTermCfg(
            func=sorter_mdp.pegs_rel_grasp,
            params={"command_name": COMMAND_NAME, "asset_cfg": grasp_cfg},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "socket_deltas": ObservationTermCfg(
            func=sorter_mdp.socket_deltas,
            params={"command_name": COMMAND_NAME},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "socket_inserted": ObservationTermCfg(
            func=sorter_mdp.socket_inserted,
            params={"command_name": COMMAND_NAME},
        ),
    }

    task_rewards = {
        "progress": RewardTermCfg(
            func=sorter_mdp.progress_reward,
            weight=1.0,
            params={"command_name": COMMAND_NAME},
        ),
        "success": RewardTermCfg(
            func=sorter_mdp.success_reward,
            weight=1.0,
            params={"command_name": COMMAND_NAME},
        ),
    }

    task_terminations = {
        "success": TerminationTermCfg(
            func=sorter_mdp.sorter_complete,
            params={"command_name": COMMAND_NAME},
        ),
    }

    cfg = make_manipulation_env_cfg(
        entities=entities,
        fixed_props=("table",),
        command_name=COMMAND_NAME,
        command_cfg=sorter_mdp.ShapeSorterCommandCfg(
            cube_name="cube",
            piece_names=piece_names,
            variant=variant,
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
        ),
        task_obs_terms=task_obs_terms,
        command_obs=True,
        task_rewards=task_rewards,
        task_terminations=task_terminations,
        env_spacing=2.0,
        episode_length_s=10.0 * variant.num_pieces,
        # Sorter physics knobs as the legacy insertion sorter kind used:
        # ~90 convex mesh parts on the cube colliding with pieces, table and
        # gripper need real contact/constraint headroom.
        solver_iterations=15,
        solver_ls_iterations=25,
        nconmax=500,
        njmax=2000,
        play=play,
    )

    if play:
        # live_bridge demo playback needs episodes to actually terminate on
        # success.
        cfg.terminations["success"] = task_terminations["success"]

    return cfg
