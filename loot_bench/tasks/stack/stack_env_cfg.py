"""Environment configuration for the pi (stack) task category."""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from loot_bench.tasks.common.commands import GRIP_CONTACT_SENSOR_NAME
from loot_bench.tasks.common.env_cfg import make_manipulation_env_cfg
from loot_bench.tasks.common.table import table_entity_cfg
from loot_bench.tasks.stack import mdp as pi_mdp
from loot_bench.tasks.stack.assets import get_block_spec
from loot_bench.tasks.stack.variants import BLOCKS_PER_LEVEL, PiVariantCfg

COMMAND_NAME = "task"

SECONDS_PER_BLOCK = 25.0
"""Episode budget per block, so 75 / 150 / 225 s for 1 / 2 / 3 levels.

ADR-016 has the ``Place`` demos (one carry-and-place) at median 5.9 s,
p90 11.8 s, max 17.5 s. A pi block costs that *plus*, for a lintel, a
knock-over and a re-grasp: ``joint4`` (``dk1.xml:173``) has only 22 deg of
pitch from home, so a standing block cannot be rolled flat in the wrist and
has to be tipped over and picked up again. Nine placements at p90 alone are
106 s. Erring short is not symmetric with erring long: the recorder
truncates at the episode length (ADR-016), so a short budget produces
silently *truncated* demos, while a long one only wastes steps on takes
that were failing anyway.

The consequence, accepted knowingly (ADR-020): at 225 s = 4500 steps QC's
100-step action horizon covers 2.2 % of an episode, against the 33-50 %
band ADR-016 validates. Do not "fix" it by shortening the episode."""


def make_pi_env_cfg(
    variant: PiVariantCfg,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the robot-agnostic pi cfg (see ``tasks/common/env_cfg.py``
    for the placeholder/wiring convention)."""
    block_names = tuple(f"block_{i}" for i in range(variant.num_blocks))
    grasp_cfg = SceneEntityCfg("robot", site_names=())  # Set per-robot by wiring.

    entities: dict[str, EntityCfg] = {
        "table": table_entity_cfg(),
    }  # Robot added by wire_manipulator_into_env.
    for name, color in zip(block_names, variant.block_colors, strict=True):
        entities[name] = EntityCfg(spec_fn=lambda c=color, n=name: get_block_spec(c, n))

    # No new observation term: level state is a function of the block poses.
    task_obs_terms = {
        "block_poses": ObservationTermCfg(
            func=pi_mdp.block_poses,
            params={"command_name": COMMAND_NAME},
        ),
    }

    task_rewards = {
        "task": RewardTermCfg(
            func=pi_mdp.task_reward,
            weight=1.0,
            params={"command_name": COMMAND_NAME},
        ),
    }

    task_terminations = {
        "success": TerminationTermCfg(
            func=pi_mdp.pi_complete,
            params={"command_name": COMMAND_NAME},
        ),
    }

    num_blocks = variant.num_blocks
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
        command_cfg=pi_mdp.PiCommandCfg(
            block_names=block_names,
            variant=variant,
            grasp_cfg=grasp_cfg,
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
        ),
        task_obs_terms=task_obs_terms,
        # The level heights are constant, so hide the command observation.
        command_obs=False,
        task_rewards=task_rewards,
        task_terminations=task_terminations,
        sensors=sensors,
        env_spacing=1.5,
        episode_length_s=SECONDS_PER_BLOCK * BLOCKS_PER_LEVEL * variant.levels,
        nconmax=100 + 120 * num_blocks,
        # Extra headroom for multi-block contacts with nativeccd disabled.
        njmax=1200 + 300 * num_blocks,
        play=play,
    )
