"""Environment configuration for the place task category."""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

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
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
    }

    task_rewards = {
        "success": RewardTermCfg(
            func=place_mdp.success_reward,
            weight=1.0,
            params={"command_name": COMMAND_NAME},
        ),
    }
    if variant.reward_mode == "progress":
        # Shaping on top of the sparse indicator, not instead of it: reward_max
        # becomes 2.0, so this variant needs its own re-labelled dataset.
        task_rewards["progress"] = RewardTermCfg(
            func=place_mdp.progress_reward,
            weight=1.0,
            params={"command_name": COMMAND_NAME},
        )

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
        # Same pads, filtered to the cube. Unlike the gate sensor above this
        # one must NOT see pad-on-pad contact: it exists to answer "did the
        # policy reach the cube?" for eval logging, and an empty gripper
        # closing on air registers ~2.4 N on the unfiltered sensor.
        ContactSensorCfg(
            name=OBJECT_CONTACT_SENSOR_NAME,
            primary=ContactMatch(mode="geom", pattern="", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="cube", entity="cube"),
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
        # Copycat fix: diagnostics on a trained QC-FQL policy showed it keyed on
        # the previous action (momentum) ~2x more than on the cube position,
        # yielding a canned reach that ignores where the cube is (gripper closes
        # in the air -> zero grasp contacts at eval). Dropping last_action forces
        # the policy to read the cube. The offline dataset must be sliced to match
        # (scripts/live_bridge/drop_last_action_obs.py). Restore by removing this.
        include_last_action=False,
        task_rewards=task_rewards,
        task_terminations=task_terminations,
        sensors=sensors,
        env_spacing=1.5,
        episode_length_s=10.0,
        nconmax=120,
        njmax=700,
        play=play,
    )
