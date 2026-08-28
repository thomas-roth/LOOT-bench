"""Environment configuration for the place task category.

The one-cube sanity check the QC pipeline is brought up on: 35 observation
values, a 7-wide end-effector delta action, and a 15 s episode. The budget is
this category's only departure from the shared workbench's defaults, and
ADR-016 holds the demo durations behind the number.
"""

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

# Half-width [rad] of the start-pose sample, applied to every arm joint at
# reset. It lives on the task, not on a launcher flag, so the demo recorder and
# the training run read the same number without either passing an override.
# 0.0 gives a bit-identical start every episode: the arm begins at the home
# pose alone, which is the state the demonstrations were aligned to. Raise it
# to give a policy more than one start state to generalize from, but keep any
# value well under the recorder's 0.2 rad alignment gate
# (live_bridge/recording_mixin.RESET_THRESHOLD_RAD), which is measured against
# the nominal pose: a wider sample and the gate fight each other.
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
        # Same pads, filtered to the cube. Telemetry only -- no gate reads it.
        # The unfiltered sensor above cannot answer "did the policy touch the
        # cube": a gripper closed on empty air reports its own two pads
        # pressing together, which is what a failing policy does all episode.
        # `secondary_policy="error"` rather than "first": the cube has exactly
        # one geom today, and a future cube with two should fail at build time
        # instead of silently measuring one of them.
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
        command_obs=True,  # Target varies every episode, unlike stack's constant height.
        task_rewards=task_rewards,
        task_terminations=task_terminations,
        sensors=sensors,
        env_spacing=1.5,
        # 15 s is 300 steps at 20 Hz -- exactly robomimic `can`, whose
        # gamma-horizon fraction it also matches. The old 10 s truncated 20 of
        # the 99 recorded demos, severing each one's successful ending; 15 s
        # keeps 98 % of episodes whole. 18 s would keep the last two and is
        # rejected for leaving the reference envelope to buy 100 % over 98 %.
        # See ADR-016's durations table.
        episode_length_s=15.0,
        start_pose_noise=START_POSE_NOISE,
        nconmax=120,
        njmax=700,
        play=play,
    )
