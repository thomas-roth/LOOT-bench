"""Environment configuration for the ball task category."""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from g3a_bench.tasks.ball import mdp as ball_mdp
from g3a_bench.tasks.ball.assets import get_ball_spec, get_field_spec
from g3a_bench.tasks.ball.variants import BallVariantCfg
from g3a_bench.tasks.common.env_cfg import make_manipulation_env_cfg
from g3a_bench.tasks.common.table import table_entity_cfg

COMMAND_NAME = "ball_target"

REWARD_MODE_TERMS: dict[str, frozenset[str]] = {
    "success": frozenset(["success"]),
    "progress": frozenset(["success"]),
}


def make_ball_env_cfg(
    variant: BallVariantCfg,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the robot-agnostic ball cfg (see ``tasks/common/env_cfg.py``
    for the placeholder/wiring convention)."""
    entities: dict[str, EntityCfg] = {
        "field": EntityCfg(spec_fn=lambda v=variant: get_field_spec(v)),
        "ball": EntityCfg(spec_fn=get_ball_spec),
        "table": table_entity_cfg(),
    }

    grasp_cfg = SceneEntityCfg("robot", site_names=())  # Set per-robot.

    task_obs_terms = {
        "ball_rel_grasp": ObservationTermCfg(
            func=ball_mdp.ball_rel_grasp,
            params={"command_name": COMMAND_NAME, "asset_cfg": grasp_cfg},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "ball_to_target": ObservationTermCfg(
            func=ball_mdp.ball_to_target,
            params={"command_name": COMMAND_NAME},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "ball_velocity": ObservationTermCfg(
            func=ball_mdp.ball_velocity,
            params={"command_name": COMMAND_NAME},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
    }

    # Sparse reward design: the task signal is only the (near-)binary outcome of
    # the ball reaching the target -- no dense distance/strike/reach/push shaping.
    # This is meant to be paired with a demonstration pipeline (GELLO teleop ->
    # offline-to-online RL), which supplies the exploration that dense shaping
    # previously provided.
    task_rewards = {
        # Sparse task signal: only fires while the ball is at the target.
        "success": RewardTermCfg(
            func=ball_mdp.success_reward,
            weight=5.0,
            params={"command_name": COMMAND_NAME},
        ),
    }

    task_terminations = {
        "success": TerminationTermCfg(
            func=ball_mdp.ball_at_target,
            params={"command_name": COMMAND_NAME},
        ),
        "out_of_bounds": TerminationTermCfg(
            func=ball_mdp.ball_out_of_bounds,
            params={"command_name": COMMAND_NAME, "max_radius": 1.2},
        ),
    }
    if variant.has_platform:
        task_terminations["off_platform"] = TerminationTermCfg(
            func=ball_mdp.ball_off_platform,
            params={"command_name": COMMAND_NAME},
        )
    if variant.has_wall:
        task_terminations["wall_collision"] = TerminationTermCfg(
            func=ball_mdp.ball_hits_wall,
            params={"command_name": COMMAND_NAME},
        )
    # No EE-ground termination: the ball rests on the terrain plane (table
    # is visual-only, see tasks/common/table.py), so near-terrain EE height
    # is routine here, not a collision to guard against.

    # push must guide the ball the full distance (target_x up to 0.68m, near
    # the ~0.74m reach) and retreat -- longer than golf/throw's one strike.
    # Estimate, not training-validated.
    episode_length_s = 16.0 if variant.action == "push" else 12.0

    return make_manipulation_env_cfg(
        entities=entities,
        fixed_props=("field", "table"),
        command_name=COMMAND_NAME,
        command_cfg=ball_mdp.BallCommandCfg(
            ball_name="ball",
            variant=variant,
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
            grasp_cfg=grasp_cfg,
        ),
        task_obs_terms=task_obs_terms,
        # ball_to_target already encodes the goal; the raw command is redundant.
        command_obs=False,
        task_rewards=task_rewards,
        task_terminations=task_terminations,
        env_spacing=2.5,
        episode_length_s=episode_length_s,
        solver_iterations=12,
        solver_ls_iterations=20,
        nconmax=150,
        njmax=800,
        viewer_distance=1.6,
        viewer_elevation=-20.0,
        play=play,
    )
