"""Environment configuration for the place task category."""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from g3a_bench.tasks.common.commands import (
    GRIP_CONTACT_SENSOR_NAME,
    OBJECT_CONTACT_SENSOR_NAME,
)
from g3a_bench.tasks.common.env_cfg import make_manipulation_env_cfg
from g3a_bench.tasks.common.table import table_entity_cfg
from g3a_bench.tasks.place import mdp as place_mdp
from g3a_bench.tasks.place.assets import get_cube_spec
from g3a_bench.tasks.place.variants import CUBE_COLOR, PlaceVariantCfg

COMMAND_NAME = "task"

# Start-pose sample half-width. Shared by recorder and training.
START_POSE_NOISE = 0.0


def make_place_env_cfg(
    variant: PlaceVariantCfg,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the robot-agnostic place cfg (see ``tasks/common/env_cfg.py``
    for the placeholder/wiring convention)."""
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
            func=place_mdp.task_reward,
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
        episode_length_s=10.0,
        start_pose_noise=START_POSE_NOISE,
        nconmax=120,
        njmax=700,
        play=play,
    )
