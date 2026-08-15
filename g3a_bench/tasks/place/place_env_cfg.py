"""Environment configuration for the place task category.

The actor observation carries the last action, as every category's does. This
category once dropped it: diagnostics on a QC-FQL policy trained under the old
pipeline showed it keyed on the previous action about twice as strongly as on
the cube position, and reached on momentum with an empty gripper. That policy
was trained in the action space step 5 retired, where no action could carry the
gripper to the cube and momentum was the only signal left, so the finding does
not carry over unexamined. Re-read it on a fresh dataset instead of assuming
either way -- fit ``tests/qc/bc_gate.py`` twice, once on the full observation
and once with the last-action columns dropped.
"""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from g3a_bench.tasks.common.commands import GRIP_CONTACT_SENSOR_NAME
from g3a_bench.tasks.common.env_cfg import make_manipulation_env_cfg
from g3a_bench.tasks.common.table import table_entity_cfg
from g3a_bench.tasks.place import mdp as place_mdp
from g3a_bench.tasks.place.assets import get_cube_spec
from g3a_bench.tasks.place.variants import CUBE_COLOR, PlaceVariantCfg

COMMAND_NAME = "task"

# Half-width [rad] of the start-pose sample, applied to every arm joint at
# reset. It lives on the task, not on a launcher flag, so the demo recorder and
# the training run read the same number without either passing an override.
# A constant start gives the policy exactly one state to generalize from, and
# that is the best remaining explanation for the flat-zero curve of the first
# run. Keep the value well under the recorder's 0.2 rad alignment gate
# (live_bridge/recording_mixin.RESET_THRESHOLD_RAD), which is measured against
# the nominal pose: a wider sample and the gate fight each other.
START_POSE_NOISE = 0.05


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
        episode_length_s=10.0,
        start_pose_noise=START_POSE_NOISE,
        nconmax=120,
        njmax=700,
        play=play,
    )
