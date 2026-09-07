"""Build, reset, and step every registered LOOT task. See tests/tasks/README.md."""

import argparse
import sys
import traceback
from pathlib import Path

import torch
from _render import domain_of, save_env_render
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg

import loot_bench  # noqa: F401  (registers all tasks)

_NUM_ENVS = 4
_STEPS = 2
_RENDER_DIR = Path("out/render")


def _run(
    task_id: str, device: str, render_dir: Path | None
) -> tuple[str | None, str | None]:
    """Build, reset, and step *task_id*.

    Returns ``(build_error, render_warning)`` where each is ``None`` on success.
    A render failure is reported separately so it never flips a building task to
    a failure.
    """
    try:
        cfg = load_env_cfg(task_id)
        cfg.scene.num_envs = _NUM_ENVS
        env = ManagerBasedRlEnv(cfg=cfg, device=device)
        zeros = torch.zeros(
            (_NUM_ENVS, env.action_manager.action.shape[1]), device=device
        )

        for _ in range(_STEPS):
            env.step(zeros)
        env.reset()

        for _ in range(_STEPS):
            env.step(zeros)

        render_warn = None
        if render_dir is not None:
            path = render_dir / domain_of(task_id) / f"{task_id}.png"
            render_warn = save_env_render(env, path)
        env.close()
        return None, render_warn

    except Exception as exc:
        return (
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            None,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test all LOOT tasks.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU device.")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help=f"Skip saving a scene picture per task to {_RENDER_DIR}/.",
    )
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    tasks = sorted(t for t in list_tasks() if t.startswith("LOOT"))
    render_dir = None if args.no_render else _RENDER_DIR

    print(f"Device: {device}   Tasks: {len(tasks)}")
    if render_dir is not None:
        print(f"Scene pictures: {render_dir}/")
    print()

    failures: dict[str, str] = {}
    for task_id in tasks:
        err, render_warn = _run(task_id, device, render_dir)
        if err is None:
            print(f"  PASS  {task_id}")
        else:
            print(f"  FAIL  {task_id}")
            failures[task_id] = err
        if render_warn is not None:
            print(f"        (render skipped: {render_warn})")

    passed = len(tasks) - len(failures)
    print()
    print(f"Result: {passed}/{len(tasks)} passed, {len(failures)} failed")

    if failures:
        print()
        for task_id, tb in failures.items():
            print(f"--- {task_id} ---")
            print(tb)
        sys.exit(1)


if __name__ == "__main__":
    main()
