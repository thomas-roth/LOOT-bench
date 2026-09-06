"""Environment configuration for the stack task category."""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from g3a_bench.tasks.common.commands import GRIP_CONTACT_SENSOR_NAME
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
    grasp_cfg = SceneEntityCfg("robot", site_names=())  # Set per-robot by wiring.

    entities: dict[str, EntityCfg] = {
        "table": table_entity_cfg(),
    }  # Robot added by wire_manipulator_into_env.
    for name, block in zip(block_names, variant.blocks, strict=True):
        entities[name] = EntityCfg(spec_fn=lambda b=block, n=name: get_block_spec(b, n))

    task_obs_terms = {
        "block_poses": ObservationTermCfg(
            func=stack_mdp.block_poses,
            params={"command_name": COMMAND_NAME},
        ),
    }

    task_rewards = {
        "task": RewardTermCfg(
            func=stack_mdp.task_reward,
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
    sensors = (
        ContactSensorCfg(
            name=GRIP_CONTACT_SENSOR_NAME,
            primary=ContactMatch(mode="geom", pattern="", entity="robot"),
            fields=("force",),
        ),
    )

    return make_manipulation_env_cfg(
        entities=entities,
        fixed_props=("table",),
        command_name=COMMAND_NAME,
        command_cfg=stack_mdp.StackCommandCfg(
            block_names=block_names,
            variant=variant,
            grasp_cfg=grasp_cfg,
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
        ),
        task_obs_terms=task_obs_terms,
        # Target height is constant, so hide the command observation.
        command_obs=False,
        task_rewards=task_rewards,
        task_terminations=task_terminations,
        sensors=sensors,
        env_spacing=1.5,
        episode_length_s=10.0 * variant.target_blocks,
        nconmax=100 + 120 * num_blocks,
        # Extra headroom for multi-block contacts with nativeccd disabled.
        njmax=1200 + 300 * num_blocks,
        play=play,
    )
