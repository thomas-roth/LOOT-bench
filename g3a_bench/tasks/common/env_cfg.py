"""Manager-term boilerplate shared by every task category's env cfg.

Only the parts that are byte-identical across cube/ball/insertion live here
(reset events, regularization rewards, curriculum, play overrides); everything
with per-category values (sim sizes, viewer pose, episode length, task terms)
stays in the category's ``make_<cat>_env_cfg``.
"""

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp


def robot_reset_events() -> dict[str, EventTermCfg]:
    """Pin the robot base and joints back to the home keyframe each reset."""
    return {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {}},
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }


def fixed_prop_reset_event(entity_name: str) -> EventTermCfg:
    """Pin a fixed-base prop to ``env_origins + init_state.pos`` each reset."""
    return EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg(entity_name),
        },
    )


def fingertip_friction_event() -> EventTermCfg:
    """Startup randomization of the finger-pad sliding friction."""
    return EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
            "operation": "abs",
            "distribution": "uniform",
            "axes": [0],
            "ranges": (0.6, 1.5),
        },
    )


def regularization_rewards() -> dict[str, RewardTermCfg]:
    """Regularization (not task shaping): keep motions smooth and feasible.

    Always active regardless of ``--reward-mode`` (see
    ``scripts/train.py::_REGULARIZATION``, which lists these keys).
    """
    return {
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        "joint_vel_hinge": RewardTermCfg(
            func=manipulation_mdp.joint_velocity_hinge_penalty,
            weight=-0.01,
            params={
                "max_vel": 0.5,
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }


def joint_vel_hinge_curriculum() -> dict[str, CurriculumTermCfg]:
    """Ramp the joint-velocity hinge penalty up over training."""
    return {
        "joint_vel_hinge_weight": CurriculumTermCfg(
            func=manipulation_mdp.reward_curriculum,
            params={
                "reward_name": "joint_vel_hinge",
                "stages": [
                    {"step": 0, "weight": -0.01},
                    {"step": 500 * 24, "weight": -0.1},
                    {"step": 1000 * 24, "weight": -1.0},
                ],
            },
        ),
    }


def apply_play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
    """Play mode: run forever, no obs noise, no curriculum, no success reset."""
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    cfg.terminations.pop("success")
