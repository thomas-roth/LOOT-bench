"""Environment configuration for the stacking task category."""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from g3a_bench.tasks.common.env_cfg import make_manipulation_env_cfg
from g3a_bench.tasks.common.table import table_entity_cfg
from g3a_bench.tasks.stacking import mdp as stacking_mdp
from g3a_bench.tasks.stacking.assets import get_block_spec
from g3a_bench.tasks.stacking.variants import StackingVariantCfg

COMMAND_NAME = "task"


def make_stacking_env_cfg(
    variant: StackingVariantCfg,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the robot-agnostic stacking cfg (see ``tasks/common/env_cfg.py``
    for the placeholder/wiring convention)."""
    block_names = tuple(f"block_{i}" for i in range(len(variant.blocks)))

    entities: dict[str, EntityCfg] = {
        "table": table_entity_cfg(),
    }  # Robot added by wire_manipulator_into_env.
    for name, block in zip(block_names, variant.blocks, strict=True):
        entities[name] = EntityCfg(spec_fn=lambda b=block, n=name: get_block_spec(b, n))

    grasp_cfg = SceneEntityCfg("robot", site_names=())  # Set per-robot.

    task_obs_terms = {
        "blocks_rel_grasp": ObservationTermCfg(
            func=stacking_mdp.blocks_rel_grasp,
            params={"command_name": COMMAND_NAME, "asset_cfg": grasp_cfg},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
    }

    task_rewards = {
        "progress": RewardTermCfg(
            func=stacking_mdp.progress_reward,
            weight=1.0,
            params={"command_name": COMMAND_NAME},
        ),
        "success": RewardTermCfg(
            func=stacking_mdp.success_reward,
            weight=1.0,
            params={"command_name": COMMAND_NAME},
        ),
    }

    task_terminations = {
        "success": TerminationTermCfg(
            func=stacking_mdp.stack_complete,
            params={"command_name": COMMAND_NAME},
        ),
    }

    num_blocks = len(variant.blocks)
    cfg = make_manipulation_env_cfg(
        entities=entities,
        fixed_props=("table",),
        command_name=COMMAND_NAME,
        command_cfg=stacking_mdp.StackingCommandCfg(
            block_names=block_names,
            variant=variant,
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
        ),
        task_obs_terms=task_obs_terms,
        # The target height is a per-task constant; the raw command would be
        # a constant observation.
        command_obs=False,
        task_rewards=task_rewards,
        task_terminations=task_terminations,
        env_spacing=1.5,
        episode_length_s=10.0 * variant.target_blocks,
        solver_iterations=10,
        solver_ls_iterations=20,
        nconmax=55 + 80 * num_blocks,
        # nativeccd is disabled for better box-to-box collisions, which raises
        # the constraint count for multi-block contacts; give njmax headroom.
        njmax=600 + 100 * num_blocks,
        play=play,
    )

    if play:
        # live_bridge demo playback needs episodes to actually terminate on
        # success.
        cfg.terminations["success"] = task_terminations["success"]

    return cfg
