"""Environment configuration for the domino task category."""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from g3a_bench.tasks.common.env_cfg import make_manipulation_env_cfg
from g3a_bench.tasks.common.table import table_entity_cfg
from g3a_bench.tasks.domino import mdp as domino_mdp
from g3a_bench.tasks.domino.assets import get_domino_spec
from g3a_bench.tasks.domino.variants import DominoVariantCfg

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
            spec_fn=lambda i=i, n=name: get_domino_spec(i, variant.different, name=n)
        )

    grasp_cfg = SceneEntityCfg("robot", site_names=())  # Set per-robot.

    task_obs_terms = {
        "dominoes_rel_grasp": ObservationTermCfg(
            func=domino_mdp.dominoes_rel_grasp,
            params={"command_name": COMMAND_NAME, "asset_cfg": grasp_cfg},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "slot_deltas": ObservationTermCfg(
            func=domino_mdp.slot_deltas,
            params={"command_name": COMMAND_NAME},
            noise=Unoise(n_min=-0.01, n_max=0.01),
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
        "progress": RewardTermCfg(
            func=domino_mdp.progress_reward,
            weight=1.0,
            params={"command_name": COMMAND_NAME},
        ),
        "success": RewardTermCfg(
            func=domino_mdp.success_reward,
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
    episode_length_s = 8.0 * num_dominoes + (8.0 if variant.knock_over else 0.0)
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
        env_spacing=1.5,
        episode_length_s=episode_length_s,
        solver_iterations=10,
        solver_ls_iterations=20,
        nconmax=55 + 80 * num_dominoes,
        # nativeccd is disabled for better box-to-box collisions, which raises
        # the constraint count for multi-domino contacts; give njmax headroom.
        njmax=600 + 100 * num_dominoes,
        play=play,
    )
