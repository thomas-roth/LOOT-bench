"""Environment configuration for the stack task category."""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from g3a_bench.tasks.common.env_cfg import make_manipulation_env_cfg
from g3a_bench.tasks.common.table import table_entity_cfg
from g3a_bench.tasks.stack import mdp as stack_mdp
from g3a_bench.tasks.stack.assets import get_block_spec
from g3a_bench.tasks.stack.variants import StackVariantCfg

COMMAND_NAME = "task"


def make_stack_env_cfg(
    variant: StackVariantCfg,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the robot-agnostic stack cfg (see ``tasks/common/env_cfg.py``
    for the placeholder/wiring convention)."""
    block_names = tuple(f"block_{i}" for i in range(len(variant.blocks)))

    entities: dict[str, EntityCfg] = {
        "table": table_entity_cfg(),
    }  # Robot added by wire_manipulator_into_env.
    for name, block in zip(block_names, variant.blocks, strict=True):
        entities[name] = EntityCfg(spec_fn=lambda b=block, n=name: get_block_spec(b, n))

    task_obs_terms = {
        "block_poses": ObservationTermCfg(
            func=stack_mdp.block_poses,
            params={"command_name": COMMAND_NAME},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
    }

    task_rewards = {
        "progress": RewardTermCfg(
            func=stack_mdp.progress_reward,
            weight=1.0,
            params={"command_name": COMMAND_NAME},
        ),
        "success": RewardTermCfg(
            func=stack_mdp.success_reward,
            weight=1.0,
            params={"command_name": COMMAND_NAME},
        ),
    }

    task_terminations = {
        "success": TerminationTermCfg(
            func=stack_mdp.stack_complete,
            params={"command_name": COMMAND_NAME},
        ),
    }

    num_blocks = len(variant.blocks)
    return make_manipulation_env_cfg(
        entities=entities,
        fixed_props=("table",),
        command_name=COMMAND_NAME,
        command_cfg=stack_mdp.StackCommandCfg(
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
        episode_length_s=10.0 * variant.target_blocks * (variant.two_stacks + 1),
        nconmax=100 + 120 * num_blocks,
        # nativeccd is disabled for better box-to-box collisions, which raises
        # the constraint count for multi-block contacts; give njmax headroom.
        # Block sets now carry more material (2n / 2n+2, see variants.py), so
        # this needs more headroom than the old n-block formula provided.
        njmax=1200 + (400 if variant.two_stacks else 300) * num_blocks,
        play=play,
    )
