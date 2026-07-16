"""Environment configuration for the ball task category."""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from g3a_bench.tasks.ball import mdp as ball_mdp
from g3a_bench.tasks.ball.assets import get_ball_spec, get_field_spec, get_wall_spec
from g3a_bench.tasks.ball.variants import WALL_X, BallVariantCfg
from g3a_bench.tasks.common.env_cfg import make_manipulation_env_cfg
from g3a_bench.tasks.common.table import table_entity_cfg

COMMAND_NAME = "task"

# Goal randomization: the field (and, for ThrowWallVariation, the wall) is
# perturbed around its init pose every reset instead of pinned exactly.
_FIELD_POSE_RANGE = {"x": (-0.02, 0.02), "y": (-0.03, 0.03)}
_WALL_POSE_RANGE = {"x": (-0.06, 0.06), "z": (-0.08, 0.0)}


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
    fixed_props = ["field", "table"]
    wall_name = None
    if variant.movable_wall:
        entities["wall"] = EntityCfg(
            spec_fn=get_wall_spec,
            init_state=EntityCfg.InitialStateCfg(pos=(WALL_X, 0.0, 0.0)),
        )
        fixed_props.append("wall")
        wall_name = "wall"

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

    task_rewards = {
        "success": RewardTermCfg(
            func=ball_mdp.success_reward,
            weight=1.0,
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
    if variant.task == "minigolf":
        task_terminations["off_platform"] = TerminationTermCfg(
            func=ball_mdp.ball_off_platform,
            params={"command_name": COMMAND_NAME},
        )
    if variant.has_wall:
        task_terminations["wall_collision"] = TerminationTermCfg(
            func=ball_mdp.ball_hits_wall,
            params={"command_name": COMMAND_NAME},
        )

    cfg = make_manipulation_env_cfg(
        entities=entities,
        fixed_props=tuple(fixed_props),
        command_name=COMMAND_NAME,
        command_cfg=ball_mdp.BallCommandCfg(
            ball_name="ball",
            field_name="field",
            wall_name=wall_name,
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
        episode_length_s=12.0,
        solver_iterations=12,
        solver_ls_iterations=20,
        nconmax=150,
        njmax=800,
        viewer_distance=1.6,
        viewer_elevation=-20.0,
        play=play,
    )

    cfg.events["reset_field"].params["pose_range"] = _FIELD_POSE_RANGE
    if wall_name is not None:
        cfg.events["reset_wall"].params["pose_range"] = _WALL_POSE_RANGE

    return cfg
