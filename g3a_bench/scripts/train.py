"""Train entry point: mjlab's trainer with g3a-bench tasks registered.

Adds a ``--reward-mode success|progress`` pre-parse step that zeros reward
weights outside the chosen tier before forwarding to mjlab. Default is
``progress`` (sparse task signal plus discrete progress shaping).
"""

import argparse
import sys

_DOMAIN_MAP = {
    "Cube": "g3a_bench.tasks.cube.cube_env_cfg",
    "Ball": "g3a_bench.tasks.ball.ball_env_cfg",
    "Insertion": "g3a_bench.tasks.insertion.insertion_env_cfg",
}
_REGULARIZATION = frozenset(["action_rate_l2", "joint_pos_limits", "joint_vel_hinge"])


def _apply_reward_mode(task_id: str, mode: str) -> None:
    """Zero reward weights not in the active set for *mode*."""
    import importlib

    from mjlab.tasks.registry import _REGISTRY  # type: ignore[attr-defined]

    domain = next((d for d in _DOMAIN_MAP if f"-{d}-" in task_id), None)
    if domain is None:
        return  # Unknown domain — leave rewards unchanged.

    module = importlib.import_module(_DOMAIN_MAP[domain])
    active: frozenset[str] = module.REWARD_MODE_TERMS[mode] | _REGULARIZATION
    env_cfg = _REGISTRY[task_id].env_cfg
    for name, term in env_cfg.rewards.items():
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
        if task_id is not None:
            _apply_reward_mode(task_id, args.reward_mode)

    from mjlab.scripts.train import main as mjlab_main

    mjlab_main()


if __name__ == "__main__":
    main()
