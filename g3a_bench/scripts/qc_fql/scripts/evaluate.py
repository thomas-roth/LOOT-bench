from collections import defaultdict

import numpy as np
import torch


def _flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dict, e.g. {'a': {'b': 1}} -> {'a.b': 1}."""
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
    """Port of `qc/evaluation.py::evaluate()`.

    Assumes a gym-style single env (`obs, info = env.reset()`; `next_obs,
    reward, terminated, truncated, info = env.step(action)`), matching the
    JAX original's own assumption -- not mjlab's vectorized API. Wiring the
    real mjlab env to satisfy this interface (or adapting this loop to
    mjlab's actual API) is step 11's job, not this one's.

    Dropped vs. the source (confirmed unused by every real caller, or dead in
    the source itself): explicit `rng` threading (`supply_rng`), full
    trajectory/video recording (`trajs`/`renders` -- always discarded via
    `_, _` at every call site in `main.py`/`main_online.py`), `eval_temperature`
    (never referenced in the source's own function body), `eval_gaussian`/
    `action_shape`/`observation_shape` (real code paths, but never passed a
    non-default value anywhere), and gripper-contact-length tracking
    (tied to the original paper's specific robot/sensor conventions).

    Added, NOT from source: `return`/`length` tracking. The source relies
    entirely on whatever the env's terminal `info` dict happens to report
    (via `flatten(info)`) and never sums reward into an episode return or
    counts steps itself -- a real gap for a general eval loop. This is a
    deliberate addition, not a port.
    """
    stats: dict[str, list] = defaultdict(list)

    for _ in range(num_eval_episodes):
        observation, _info = env.reset()
        done = False
        step = 0
        episode_return = 0.0
        action_queue: list[np.ndarray] = []

        while not done:
            if len(action_queue) == 0:
                obs_t = torch.from_numpy(np.asarray(observation)).float().unsqueeze(0)
                action = agent.sample_actions(obs_t).squeeze(0).cpu().numpy()
                action = action.reshape(-1, action_dim)
                action_queue.extend(action)

            action = action_queue.pop(0)

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
