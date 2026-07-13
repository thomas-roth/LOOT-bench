"""One registration seam for all task categories.

:func:`register_category` registers every robot x variant combination of a
category into mjlab's registry *and* records each task in :data:`TASK_TABLE`,
so tooling (``g3a-train --reward-mode``) can find a task's env cfg and its
reward-mode tiers without reflective module lookups.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from g3a_bench.robots.interface import ManipulatorInterface
from g3a_bench.tasks.common.manipulator_wiring import wire_manipulator_into_env
from g3a_bench.tasks.common.rl_cfg import manipulation_ppo_runner_cfg


@dataclass(frozen=True)
class TaskRecord:
    """What the tooling needs to know about one registered task."""

    env_cfg: ManagerBasedRlEnvCfg
    """The registered *training* cfg (shared object: mutating its reward
    weights affects the run mjlab builds from the registry)."""
    reward_mode_terms: dict[str, frozenset[str]]
    """Reward-mode name -> task reward terms active in that mode."""


TASK_TABLE: dict[str, TaskRecord] = {}
"""All G3a tasks by task id, filled by :func:`register_category`."""


def register_category(
    *,
    category: str,
    variants: Iterable,
    make_env_cfg: Callable[..., ManagerBasedRlEnvCfg],
    robots: tuple[ManipulatorInterface, ...],
    grasp_obs_terms: tuple[str, ...],
    ee_contact_sensors: tuple[str, ...] = (),
    reward_mode_terms: dict[str, frozenset[str]],
    max_iterations: int = 15_000,
) -> None:
    """Register all robot x variant tasks of one category.

    Args:
        category: capitalized category name; task ids become
            ``G3a-<category>-<variant.name>-<robot.name>``.
        variants: the category's ``*_VARIANTS`` values; each needs a ``name``.
        make_env_cfg: ``make_<cat>_env_cfg(variant, play=...)``.
        robots: manipulator interfaces to register against.
        grasp_obs_terms / ee_contact_sensors: forwarded to
            ``wire_manipulator_into_env``.
        reward_mode_terms: the category's ``REWARD_MODE_TERMS``.
        max_iterations: PPO iteration budget for the category.
    """
    for robot in robots:
        for variant in variants:
            env_cfg = make_env_cfg(variant)
            wire_manipulator_into_env(
                env_cfg,
                robot,
                grasp_obs_terms=grasp_obs_terms,
                ee_contact_sensors=ee_contact_sensors,
            )

            play_env_cfg = make_env_cfg(variant, play=True)
            wire_manipulator_into_env(
                play_env_cfg,
                robot,
                grasp_obs_terms=grasp_obs_terms,
                ee_contact_sensors=ee_contact_sensors,
            )

            task_id = f"G3a-{category}-{variant.name}-{robot.name}"
            register_mjlab_task(
                task_id=task_id,
                env_cfg=env_cfg,
                play_env_cfg=play_env_cfg,
                rl_cfg=manipulation_ppo_runner_cfg(
                    f"{category.lower()}_{variant.name.lower()}_{robot.name.lower()}",
                    max_iterations=max_iterations,
                ),
                runner_cls=ManipulationOnPolicyRunner,
            )
            TASK_TABLE[task_id] = TaskRecord(env_cfg, reward_mode_terms)
