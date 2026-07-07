"""Train entry point: mjlab's trainer with g3a-bench tasks registered.

Adds a ``--reward-mode success|progress`` pre-parse step that zeros reward
weights outside the chosen tier before forwarding to mjlab. Default is
``progress`` (sparse task signal plus discrete progress shaping).
"""

import argparse
import sys


def _apply_reward_mode(task_id: str | None, mode: str) -> None:
    """Zero reward weights not in the active set for *mode*.

    Loud by design: an unknown task id or an unrecorded mode aborts instead of
    silently training with untouched rewards.
    """
    from g3a_bench.tasks.common.registration import REGULARIZATION_TERMS, TASK_TABLE

    if task_id is None:
        raise SystemExit("--reward-mode given but no task id on the command line")
    record = TASK_TABLE.get(task_id)
    if record is None:
        raise SystemExit(
            f"--reward-mode: {task_id!r} is not a registered G3a task "
            f"(known: {', '.join(sorted(TASK_TABLE))})"
        )
    terms = record.reward_mode_terms.get(mode)
    if terms is None:
        raise SystemExit(
            f"--reward-mode: task {task_id!r} records no {mode!r} mode "
            f"(available: {', '.join(sorted(record.reward_mode_terms))})"
        )
    active = terms | REGULARIZATION_TERMS
    for name, term in record.env_cfg.rewards.items():
        if name not in active:
            term.weight = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--reward-mode",
        default="progress",
        choices=["success", "progress"],
    )
    args, remaining = parser.parse_known_args()
    # Strip --reward-mode and its value from sys.argv before mjlab parses it.
    sys.argv = [sys.argv[0]] + remaining

    import g3a_bench  # noqa: F401  (registers benchmark tasks)

    if args.reward_mode != "progress":
        task_id = next((a for a in remaining if not a.startswith("-")), None)
        _apply_reward_mode(task_id, args.reward_mode)

    from mjlab.scripts.train import main as mjlab_main

    mjlab_main()


if __name__ == "__main__":
    main()
