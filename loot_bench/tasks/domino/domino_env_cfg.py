"""Environment configuration for the domino task category."""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from loot_bench.tasks.common.colors import COLOR_NAMES
from loot_bench.tasks.common.commands import GRIP_CONTACT_SENSOR_NAME
from loot_bench.tasks.common.env_cfg import make_manipulation_env_cfg
from loot_bench.tasks.common.table import table_entity_cfg
from loot_bench.tasks.domino import mdp as domino_mdp
from loot_bench.tasks.domino.assets import get_domino_spec
from loot_bench.tasks.domino.variants import DominoVariantCfg

COMMAND_NAME = "task"


def make_domino_env_cfg(
    variant: DominoVariantCfg,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the robot-agnostic domino cfg (see ``tasks/common/env_cfg.py``
    for the placeholder/wiring convention)."""
    domino_names = tuple(f"domino_{i}" for i in range(variant.num_dominoes))

    entities: dict[str, EntityCfg] = {
        "table": table_entity_cfg(),
    }  # Robot added by wire_manipulator_into_env.
    for i, name in enumerate(domino_names):
        entities[name] = EntityCfg(
            spec_fn=lambda i=i, n=name: get_domino_spec(
                i, variant.different, color=COLOR_NAMES[i], name=n
            )
        )

    task_obs_terms = {
        "domino_poses": ObservationTermCfg(
            func=domino_mdp.domino_poses,
            params={"command_name": COMMAND_NAME},
        ),
        "slot_placed": ObservationTermCfg(
            func=domino_mdp.slot_placed,
            params={"command_name": COMMAND_NAME},
        ),
    }
    if variant.knock_over:
        task_obs_terms["phase"] = ObservationTermCfg(
            func=domino_mdp.phase_obs,
            params={"command_name": COMMAND_NAME},
        )

    task_rewards = {
        "task": RewardTermCfg(
            func=domino_mdp.task_reward,
            weight=1.0,
            params={"command_name": COMMAND_NAME},
        ),
    }

    task_terminations = {
        "success": TerminationTermCfg(
            func=domino_mdp.domino_task_complete,
            params={"command_name": COMMAND_NAME},
        ),
    }

    num_dominoes = variant.num_dominoes
    episode_length_s = 22.0 * num_dominoes + (30.0 if variant.knock_over else 0.0)
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
        command_cfg=domino_mdp.DominoCommandCfg(
            domino_names=domino_names,
            variant=variant,
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
        ),
        task_obs_terms=task_obs_terms,
        command_obs=True,
        task_rewards=task_rewards,
        task_terminations=task_terminations,
        sensors=sensors,
        env_spacing=1.5,
        episode_length_s=episode_length_s,
        nconmax=55 + 80 * num_dominoes,
        # nativeccd is disabled for better box-to-box collisions, which raises
        # the constraint count for multi-domino contacts; give njmax headroom.
        njmax=600 + 100 * num_dominoes,
        play=play,
    )
