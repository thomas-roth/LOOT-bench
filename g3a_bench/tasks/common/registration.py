"""One registration seam for all task categories.

:func:`register_category` registers every robot x variant combination of a
category into mjlab's registry *and* records each task in :data:`TASK_TABLE`,
so tooling can find a task's env cfg (and switch its reward mode via
:func:`apply_reward_mode`) without reflective module lookups.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from g3a_bench.robots.interface import ManipulatorInterface
from g3a_bench.tasks.common.manipulator_wiring import wire_manipulator_into_env
from g3a_bench.tasks.common.rl_cfg import manipulation_ppo_runner_cfg

REWARD_MODES: dict[str, frozenset[str]] = {
    "success": frozenset({"success"}),
    "progress": frozenset({"progress", "success"}),
}
"""Reward-mode name -> active task reward terms. Every category names its
terms ``progress``/``success``; a term a task doesn't define is skipped."""


@dataclass(frozen=True)
class TaskRecord:
    """What the tooling needs to know about one registered task."""

    env_cfg: ManagerBasedRlEnvCfg
    """The registered *training* cfg (shared object: mutating its reward
    weights affects the run mjlab builds from the registry)."""


TASK_TABLE: dict[str, TaskRecord] = {}
"""All G3a tasks by task id, filled by :func:`register_category`."""


def apply_reward_mode(task_id: str, mode: str) -> None:
    """Zero the reward weights outside *mode*'s active set for *task_id*.

    Loud by design: an unknown task id or mode aborts instead of silently
    training with untouched rewards.
    """
    record = TASK_TABLE.get(task_id)
    if record is None:
        raise SystemExit(
            f"reward mode: {task_id!r} is not a registered G3a task "
            f"(known: {', '.join(sorted(TASK_TABLE))})"
        )
    active = REWARD_MODES.get(mode)
    if active is None:
        raise SystemExit(
            f"reward mode: unknown mode {mode!r} "
            f"(available: {', '.join(sorted(REWARD_MODES))})"
        )
    for name, term in record.env_cfg.rewards.items():
        if name not in active:
            term.weight = 0.0


def register_category(
    *,
    category: str,
    variants: Iterable,
    make_env_cfg: Callable[..., ManagerBasedRlEnvCfg],
    robots: tuple[ManipulatorInterface, ...],
    grasp_obs_terms: tuple[str, ...],
    ee_contact_sensors: tuple[str, ...] = (),
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
        max_iterations: PPO iteration budget (kept so ``g3a-play`` and the
            mjlab registry tooling work; QC has its own trainer).
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
            TASK_TABLE[task_id] = TaskRecord(env_cfg)
