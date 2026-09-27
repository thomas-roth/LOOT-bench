"""Environment configuration for the place task category.

The QC pipeline's sanity check: 36 observations, a 7-wide action, a 15 s
episode. The budget is its only departure from the shared workbench.
"""

import os

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from loot_bench.tasks.common.commands import (
    GRIP_CONTACT_SENSOR_NAME,
    OBJECT_CONTACT_SENSOR_NAME,
)
from loot_bench.tasks.common.env_cfg import make_manipulation_env_cfg
from loot_bench.tasks.common.table import table_entity_cfg
from loot_bench.tasks.place import mdp as place_mdp
from loot_bench.tasks.place.assets import get_cube_spec
from loot_bench.tasks.place.variants import CUBE_COLOR, PlaceVariantCfg

COMMAND_NAME = "task"

# Half-width [rad] of the start-pose sample (ADR-008). On the task, not on a
# launcher flag, so recorder and training run read one number. 0.0 is a
# bit-identical start; raise it for start-state variety, but stay inside the
# box mapper.leader_is_aligned gates on.
START_POSE_NOISE = 0.0

DENSE_REWARD_ENV_VAR = "LOOT_PLACE_DENSE_REWARD"
"""Opt back into the shaped place reward. Read when ``make_place_env_cfg``
gets no explicit ``dense_reward``: ``1``/``true``/``yes``/``on`` selects it,
anything else (or unset) keeps the sparse default.

On the env var, not on a second task id, so one registration covers both
densities and recorder and training run read one switch (``train_qc.sh
--dense`` exports it). Record and train under the same setting: the
dataset carries the reward its session was flown with, and nothing checks the
density at load time.
"""


def _dense_reward_from_env() -> bool:
    return os.environ.get(DENSE_REWARD_ENV_VAR, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def make_place_env_cfg(
    variant: PlaceVariantCfg,
    play: bool = False,
    dense_reward: bool | None = None,
) -> ManagerBasedRlEnvCfg:
    """Build the robot-agnostic place cfg (see ``tasks/common/env_cfg.py``
    for the placeholder/wiring convention).

    Args:
        variant: which spawn/target randomness to build.
        play: strip the cfg down for the teleop player.
        dense_reward: the shaped travel-fraction reward when true, the
            sparse -1-until-placed reward when false. ``None`` (the
            registration default) reads ``LOOT_PLACE_DENSE_REWARD``.
    """
    if dense_reward is None:
        dense_reward = _dense_reward_from_env()
    reward_func = place_mdp.dense_task_reward if dense_reward else place_mdp.task_reward
    entities: dict[str, EntityCfg] = {
        "table": table_entity_cfg(),
        "cube": EntityCfg(spec_fn=lambda: get_cube_spec(CUBE_COLOR, name="cube")),
    }  # Robot added by wire_manipulator_into_env.

    task_obs_terms = {
        "cube_pose": ObservationTermCfg(
            func=place_mdp.cube_pose,
            params={"command_name": COMMAND_NAME},
        ),
    }

    task_rewards = {
        "task": RewardTermCfg(
            func=reward_func,
            weight=1.0,
            params={"command_name": COMMAND_NAME},
        ),
    }

    task_terminations = {
        "success": TerminationTermCfg(
            func=place_mdp.place_task_complete,
            params={"command_name": COMMAND_NAME},
        ),
    }

    sensors = (
        ContactSensorCfg(
            name=GRIP_CONTACT_SENSOR_NAME,
            primary=ContactMatch(mode="geom", pattern="", entity="robot"),
            fields=("force",),
        ),
        # Filtered to the cube. Telemetry only, no gate reads it.
        ContactSensorCfg(
            name=OBJECT_CONTACT_SENSOR_NAME,
            primary=ContactMatch(mode="geom", pattern="", entity="robot"),
            secondary=ContactMatch(mode="geom", pattern="cube_geom", entity="cube"),
            secondary_policy="error",
            fields=("force",),
        ),
    )

    return make_manipulation_env_cfg(
        entities=entities,
        fixed_props=("table",),
        command_name=COMMAND_NAME,
        command_cfg=place_mdp.PlaceCommandCfg(
            cube_name="cube",
            variant=variant,
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
        ),
        task_obs_terms=task_obs_terms,
        command_obs=True,  # Target varies per episode, unlike stack.
        task_rewards=task_rewards,
        task_terminations=task_terminations,
        sensors=sensors,
        env_spacing=1.5,
        # 300 steps at 20 Hz, matching robomimic `can`. The old 10 s cut the
        # ending off 20 of the 99 demos; 15 s keeps 98 % whole (ADR-016).
        episode_length_s=15.0,
        start_pose_noise=START_POSE_NOISE,
        nconmax=120,
        njmax=700,
        play=play,
    )
