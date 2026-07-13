"""Episode-rollout evaluation with the action-chunk queue.

Port of `qc/evaluation.py::evaluate()`. The one algorithm-specific thing
here is the queue: the policy emits H actions at once, and the env consumes
them one per step -- so the agent is queried only every H steps, and each
chunk is executed to completion (the reference does the same during eval;
re-planning mid-chunk would be a different algorithm).

Env contract is gym-style single-env: `obs, info = env.reset()` and
`next_obs, reward, terminated, truncated, info = env.step(action)`, the same
assumption the reference makes. `scripts/mjlab_env.py::MjlabSingleEnv`
adapts a real mjlab task to exactly this contract.

Scope vs. the reference (each drop verified against the reference's actual
call sites, all of which do `eval_info, _, _ = evaluate(...)`): trajectory
and video recording are always discarded, `eval_temperature` is dead even in
the reference body, `eval_gaussian`/`action_shape`/`observation_shape` are
never passed, and gripper-contact tracking is paper-robot-specific -- none
ported. Added beyond the reference: `return`/`length` aggregation, since the
reference reports only whatever the env's terminal info dict contains.
"""

from collections import defaultdict

import numpy as np
import torch

from g3a_bench.scripts.qc_fql.agent.chunked_policy import ChunkedPolicy


def _flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """{'a': {'b': 1}} -> {'a.b': 1}; lets nested env info average per-key."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if hasattr(v, "items"):
            items.extend(_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _add_to(dict_of_lists: dict, single_dict: dict) -> None:
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


@torch.no_grad()
def evaluate(
    agent, env, action_dim: int, num_eval_episodes: int = 50
) -> dict[str, float]:
    """Roll out full episodes, return per-key means over episodes.

    Whatever the env reports in its final step's `info` (e.g. `success`) is
    averaged in alongside `return`/`length` -- no key names are hardcoded.
    """
    stats: dict[str, list] = defaultdict(list)
    device = next(agent.parameters()).device
    policy = ChunkedPolicy(agent, action_dim, device)

    for _ in range(num_eval_episodes):
        observation, _info = env.reset()
        policy.reset()  # fresh chunk per episode
        done = False
        step = 0
        episode_return = 0.0

        while not done:
            action = policy.next_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(
                np.clip(action, -1.0, 1.0)
            )
            done = terminated or truncated
            step += 1
            episode_return += reward
            observation = next_observation

        stats["return"].append(episode_return)
        stats["length"].append(step)
        _add_to(stats, _flatten(info))

    return {k: float(np.mean(v)) for k, v in stats.items()}
