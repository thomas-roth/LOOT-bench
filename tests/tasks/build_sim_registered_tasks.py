"""Build, reset, and step every registered G3A task. See tests/tasks/README.md."""

import argparse
import sys
import traceback

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg

import g3a_bench  # noqa: F401  (registers all tasks)

_NUM_ENVS = 4
_STEPS = 2


def _run(task_id: str, device: str) -> str | None:
    """Build, reset, and step *task_id*.  Returns an error string or None."""
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
        env.close()
        return None

    except Exception as exc:
        return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test all G3a tasks.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU device.")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    tasks = sorted(t for t in list_tasks() if t.startswith("G3a"))

    print(f"Device: {device}   Tasks: {len(tasks)}")
    print()

    failures: dict[str, str] = {}
    for task_id in tasks:
        err = _run(task_id, device)
        if err is None:
            print(f"  PASS  {task_id}")
        else:
            print(f"  FAIL  {task_id}")
            failures[task_id] = err

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
