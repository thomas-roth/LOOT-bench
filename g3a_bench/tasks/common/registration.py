"""One registration seam for all task categories.

:func:`register_category` registers every robot x variant combination of a
category into mjlab's registry; tooling enumerates tasks via
``mjlab.tasks.registry`` (``list_tasks`` / ``load_env_cfg``).
"""

from collections.abc import Callable, Iterable

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from g3a_bench.robots.interface import ManipulatorInterface
from g3a_bench.tasks.common.manipulator_wiring import wire_manipulator_into_env
from g3a_bench.tasks.common.rl_cfg import manipulation_ppo_runner_cfg


def register_category(
    *,
    category: str,
    variants: Iterable,
    make_env_cfg: Callable[..., ManagerBasedRlEnvCfg],
    robots: tuple[ManipulatorInterface, ...],
    task_id_parts: Callable[..., tuple[str, str]],
    ee_contact_sensors: tuple[str, ...] = (),
    grip_contact_sensor: str | None = None,
    object_contact_sensor: str | None = None,
) -> None:
    """Register all robot x variant tasks of one category.

    Args:
        category: capitalized category name; task ids become
            ``G3A-<category>-<task>-<variation>-<robot.name>`` (the
            ``<variation>`` segment is omitted when ``task_id_parts`` returns
            an empty string for it).
        variants: the category's ``*_VARIANTS`` values; each needs a ``name``
            (used only for the internal experiment-name string, not the
            task id).
        make_env_cfg: ``make_<cat>_env_cfg(variant, play=...)``.
        robots: manipulator interfaces to register against.
        ee_contact_sensors: forwarded to ``wire_manipulator_into_env``.
        grip_contact_sensor: forwarded to ``wire_manipulator_into_env``.
        object_contact_sensor: forwarded to ``wire_manipulator_into_env``.
        task_id_parts: ``(task, variation)`` for a variant, e.g.
            ``("Low", "DifferentBlocks")`` or ``("Low", "")`` for the base
            case.
    """
    seen_task_ids: dict[str, str] = {}
    for robot in robots:
        for variant in variants:
            task, variation = task_id_parts(variant)
            task_id = (
                f"G3A-{category}-{task}-{variation}-{robot.name}"
                if variation
                else f"G3A-{category}-{task}-{robot.name}"
            )
            colliding_variant = seen_task_ids.get(task_id)
            assert colliding_variant is None, (
                f"task_id_parts collision: variants {colliding_variant!r} and "
                f"{variant.name!r} both map to task id {task_id!r}"
            )
            seen_task_ids[task_id] = variant.name

            env_cfg = make_env_cfg(variant)
            wire_manipulator_into_env(
                env_cfg,
                robot,
                ee_contact_sensors=ee_contact_sensors,
                grip_contact_sensor=grip_contact_sensor,
                object_contact_sensor=object_contact_sensor,
            )

            play_env_cfg = make_env_cfg(variant, play=True)
            wire_manipulator_into_env(
                play_env_cfg,
                robot,
                ee_contact_sensors=ee_contact_sensors,
                grip_contact_sensor=grip_contact_sensor,
                object_contact_sensor=object_contact_sensor,
            )

            register_mjlab_task(
                task_id=task_id,
                env_cfg=env_cfg,
                play_env_cfg=play_env_cfg,
                rl_cfg=manipulation_ppo_runner_cfg(
                    f"{category.lower()}_{variant.name.lower()}_{robot.name.lower()}"
                ),
                runner_cls=ManipulationOnPolicyRunner,
            )
