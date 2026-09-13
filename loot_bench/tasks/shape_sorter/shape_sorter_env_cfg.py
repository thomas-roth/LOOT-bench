"""Environment configuration for the shape-sorter task category."""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from loot_bench.tasks.common.commands import GRIP_CONTACT_SENSOR_NAME
from loot_bench.tasks.common.env_cfg import make_manipulation_env_cfg
from loot_bench.tasks.common.table import table_entity_cfg
from loot_bench.tasks.shape_sorter import mdp as sorter_mdp
from loot_bench.tasks.shape_sorter.assets import get_sorter_piece_spec, get_sorter_spec
from loot_bench.tasks.shape_sorter.variants import ShapeSorterVariantCfg

COMMAND_NAME = "task"


def make_shape_sorter_env_cfg(
    variant: ShapeSorterVariantCfg,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the robot-agnostic shape-sorter cfg (see ``tasks/common/env_cfg.py``
    for the placeholder/wiring convention)."""
    piece_names = tuple(f"piece_{i}" for i in range(variant.num_pieces))

    entities: dict[str, EntityCfg] = {
        "table": table_entity_cfg(),
        "cube": EntityCfg(spec_fn=get_sorter_spec),
    }  # Robot added by wire_manipulator_into_env.
    for name, shape, color in zip(
        piece_names, variant.shapes, variant.colors, strict=True
    ):
        entities[name] = EntityCfg(
            spec_fn=lambda s=shape, c=color: get_sorter_piece_spec(s, c)
        )

    task_obs_terms = {
        "cube_pose": ObservationTermCfg(
            func=sorter_mdp.cube_pose,
            params={"command_name": COMMAND_NAME},
        ),
        "piece_poses": ObservationTermCfg(
            func=sorter_mdp.piece_poses,
            params={"command_name": COMMAND_NAME},
        ),
        "socket_inserted": ObservationTermCfg(
            func=sorter_mdp.socket_inserted,
            params={"command_name": COMMAND_NAME},
        ),
    }

    task_rewards = {
        "task": RewardTermCfg(
            func=sorter_mdp.task_reward,
            weight=1.0,
            params={"command_name": COMMAND_NAME},
        ),
    }

    task_terminations = {
        "success": TerminationTermCfg(
            func=sorter_mdp.sorter_complete,
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

    # Teleop budget: ~10 s per insertion, matching the domino category's
    # ~20 s per placement (a sorter insertion is one grasp-place cycle with
    # no travel between slots, so half the domino rate).
    episode_length_s = 10.0 * variant.num_pieces

    return make_manipulation_env_cfg(
        entities=entities,
        fixed_props=("table",),
        command_name=COMMAND_NAME,
        command_cfg=sorter_mdp.ShapeSorterCommandCfg(
            cube_name="cube",
            piece_names=piece_names,
            variant=variant,
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
        ),
        task_obs_terms=task_obs_terms,
        # The cube's pose is already observed directly via cube_pose above;
        # the raw command (mouth positions + axis vectors) would be redundant.
        command_obs=False,
        task_rewards=task_rewards,
        task_terminations=task_terminations,
        sensors=sensors,
        env_spacing=2.0,
        episode_length_s=episode_length_s,
        # ~90 convex mesh parts on the cube colliding with pieces, table and
        # gripper need real contact/constraint headroom: a fixed base for the
        # cube assembly plus per-piece contacts, mirroring the domino
        # category's scaled budgets. Sized so the hardest variant (5 pieces)
        # caps at 500/2000.
        nconmax=100 + 80 * variant.num_pieces,
        njmax=1500 + 100 * variant.num_pieces,
        play=play,
    )
