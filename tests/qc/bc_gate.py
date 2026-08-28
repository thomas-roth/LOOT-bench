# pyright: reportMissingImports=false
"""Behaviour-cloning gate: do these demonstrations generalize?

A training run costs hours. This costs about two minutes and answers the one
question that decides whether the run is worth starting: **fit the recorded
actions with a plain regressor, and see how well the fit transfers to episodes
it never saw.** Nothing here is pipeline code, and nothing here trains a
policy. It is a measurement instrument.

The regressor gets exactly what the flow actor gets -- the same observation,
the same ``horizon_length`` action chunk, the same ``valid`` mask, the same MLP
body (imported from qc, so the two cannot drift apart) -- and nothing else. No
flow-matching noise, no critic, no sampling. So a poor read here is a statement
about the data, not about the agent.

Read the two numbers together:

* **held-out error near the trained error** -- the mapping generalizes. The
  data is fine and a run is worth starting.
* **trained error low, held-out error high** -- the regressor memorized the
  demonstrations. More of the same demonstrations will not help.
* **both high** -- the observation does not determine the action. Look at the
  observation layout or at how the demonstrations were recorded.

**The bar is 0.0766**, and it is measured, not inherited. It is what this gate
reads on the old 100-demonstration set -- the pre-rebuild pipeline's export,
``G3A-Place-Easy-Dk1_uni_qc_physical.npz``, ten episodes held out -- which
fitted to 0.0075 on the episodes it saw. That is a single legacy file, *not*
the committed sessions under ``dataset/demos/G3A-Place-RandS-RandT-Dk1_uni``:
that directory has since grown past the set the number was read on, so
pointing this gate at it will not reproduce 0.0766.

Two numbers survive from the old pipeline for that same file -- 0.166 from the
lost repository tool, 0.118 from ``probes/probe17_regressor.py``, which skipped
one long episode -- and **neither is the bar here**. They were read differently:
the holdout taken from the front of the session, the read stopped at frame 500,
a constant learning rate, 25k steps. This gate reproduces none of them, so its
number is not theirs and cannot be compared to them. It holds out at random,
because the first attempts of a session are the clumsy ones, and it reads the
whole body of every held-out episode.

Beat 0.0766. Do not read a pass as a promise: **the set that reads 0.0766 scored
0 successes in 20 evaluation episodes.** A dataset the regressor cannot transfer
is certainly not worth a run. A dataset it can transfer has cleared the cheapest
of the reasons a run fails, and no more.

Runs from the qc venv (it needs JAX), with both checkouts on the path::

    PYTHONPATH=/disk/users/id310/code/g3a-bench:/disk/users/id310/code/qc \\
    /disk/users/id310/code/qc/.venv/bin/python tests/qc/bc_gate.py \\
        out/qc/stack.npz

Exit status is the verdict: 0 when the held-out interior error beats
``--target``, 1 when it does not.
"""

from __future__ import annotations

import os

# Must precede ``import jax``, which is why it is here and not in ``main``:
# JAX otherwise takes about 75 % of the card at import and the number this
# gate produces is worth nothing if it cannot share a machine with a run.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

TARGET = 0.0766
"""Held-out interior error of the old 100-demonstration set, read by this gate.

Measured on the legacy ``G3A-Place-Easy-Dk1_uni_qc_physical.npz`` with
``--holdout 10`` and every other setting left at its default -- see this
module's docstring for why that file, and not today's demo directory, is the
one the number belongs to. The old pipeline's recorded 0.166 came from a
different read of the same file.
"""

INTERIOR_SKIP = 5
"""First frame of an episode the interior read starts from.

The opening frames of every episode look alike, because the arm starts from
one pose. Reading them measures the start pose, not the policy.
"""


def build_chunks(
    actions: NDArray[np.floating], terminals: NDArray[np.floating], horizon: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """The chunk target and ``valid`` mask qc's ``acfql`` regresses on.

    Mirrors ``qc/utils/datasets.py::sample_sequence``: a window may cross an
    episode boundary, and the steps at or after the first terminal inside it
    are masked out rather than dropped. Reproducing the mask matters -- a gate
    that trained on windows the agent never sees would answer a different
    question.

    Returns:
        ``(chunk, valid)`` of shape ``(n, horizon, action_dim)`` and
        ``(n, horizon)``, where ``n = len(actions) - horizon + 1``.

    Raises:
        ValueError: if the recording is shorter than one chunk.
    """
    n = len(actions) - horizon + 1
    if n < 1:
        raise ValueError(
            f"{len(actions)} transitions cannot fill a chunk of {horizon}; "
            "record more, or lower --horizon"
        )
    idx = np.arange(n)[:, None] + np.arange(horizon)[None, :]
    chunk = np.asarray(actions, dtype=np.float32)[idx]
    term = np.maximum.accumulate(np.asarray(terminals)[idx], axis=1)
    valid = np.ones((n, horizon), dtype=np.float32)
    valid[:, 1:] = 1.0 - term[:, :-1]
    return chunk, valid


def episode_bounds(
    episode_index: NDArray[np.integer],
) -> dict[int, tuple[int, int]]:
    """Half-open ``[start, stop)`` row bounds of each episode, keyed by its id.

    Raises:
        ValueError: if an episode's rows are not contiguous, which would make
            every bound a lie without anything else noticing.
    """
    episode_index = np.asarray(episode_index)
    changes = np.flatnonzero(episode_index[1:] != episode_index[:-1]) + 1
    starts = np.concatenate(([0], changes))
    stops = np.concatenate((changes, [len(episode_index)]))
    bounds = {
        int(episode_index[s]): (int(s), int(e))
        for s, e in zip(starts, stops, strict=True)
    }
    if len(bounds) != len(starts):
        raise ValueError(
            "each episode's rows must be contiguous, but at least one episode "
            "index appears in more than one run"
        )
    return bounds


def split_episodes(
    episode_ids: list[int], holdout: int, seed: int
) -> tuple[list[int], list[int]]:
    """Split episode ids into the ones fitted and the ones held out.

    The split is random, not the first ``holdout`` episodes. Recording order
    carries operator skill: the first attempts of a session are the clumsy
    ones, so holding out the head measures a harder set than it fits.

    Raises:
        ValueError: if the split would leave no episode on either side.
    """
    if holdout < 1:
        raise ValueError(f"--holdout must be at least 1, got {holdout}")
    if holdout >= len(episode_ids):
        raise ValueError(
            f"cannot hold out {holdout} of {len(episode_ids)} episodes and "
            "still fit anything"
        )
    shuffled = list(episode_ids)
    np.random.default_rng(seed).shuffle(shuffled)
    held = sorted(shuffled[:holdout])
    train = sorted(shuffled[holdout:])
    return train, held


def training_starts(
    episode_index: NDArray[np.integer], horizon: int, held: list[int]
) -> NDArray[np.intp]:
    """Window start rows that touch no held-out episode.

    A window may cross an episode boundary, so it is not enough to drop the
    windows that *start* in a held-out episode: the whole window is checked.
    """
    n = len(episode_index) - horizon + 1
    window = np.asarray(episode_index)[
        np.arange(n)[:, None] + np.arange(horizon)[None, :]
    ]
    return np.flatnonzero(~np.isin(window, held).any(axis=1))


def first_frames(
    bounds: dict[int, tuple[int, int]], ids: list[int]
) -> NDArray[np.intp]:
    """The opening row of each episode: the read at ``t = 0``."""
    return np.array([bounds[e][0] for e in ids], dtype=np.intp)


def interior_frames(
    bounds: dict[int, tuple[int, int]],
    ids: list[int],
    horizon: int,
    per_episode: int,
) -> NDArray[np.intp]:
    """Rows sampled evenly through the body of each episode.

    Every episode in ``ids`` contributes, and the whole body of each one is in
    range. The old probe skipped the first episode of its pool and capped the
    read at frame 500, which is the whole difference between its 0.118 and the
    0.166 this gate compares against.

    Raises:
        ValueError: if no episode is long enough to read from.
    """
    selected = []
    for episode in ids:
        start, stop = bounds[episode]
        low, high = start + INTERIOR_SKIP, stop - horizon
        if high <= low:
            continue
        rows = np.arange(low, high)
        stride = max(1, -(-len(rows) // per_episode))
        selected.append(rows[::stride])
    if not selected:
        raise ValueError(
            f"no episode is longer than {INTERIOR_SKIP + horizon} rows, so "
            "there is no interior to read"
        )
    return np.concatenate(selected)


def standardizer(
    observations: NDArray[np.floating], rows: NDArray[np.intp]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Mean and standard deviation, over the fitted rows only.

    Statistics from the whole dataset would let the held-out episodes shift the
    scale the regressor is fitted on. The leak is small, and a gate whose
    number is only nearly honest is not worth having.

    A constant observation dimension gets a standard deviation of 1: the DK1
    start pose is deterministic, so around 23 of them are constant, and
    dividing by their true 0.0 would put infinities into every input.
    """
    fitted = np.asarray(observations, dtype=np.float32)[rows]
    mean = fitted.mean(axis=0)
    std = fitted.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def load_columns(path: Path) -> dict[str, NDArray]:
    """Read the columns this gate needs out of a converted ``.npz``.

    Raises:
        ValueError: if the file is missing one, naming it.
    """
    needed = ("observations", "actions", "terminals", "episode_index")
    with np.load(path) as loaded:
        missing = [key for key in needed if key not in loaded.files]
        if missing:
            raise ValueError(
                f"{path} has no {', '.join(missing)}; it did not come from "
                "g3a-convert-demos"
            )
        return {key: np.asarray(loaded[key]) for key in needed}


def fit(
    observations: NDArray[np.float32],
    chunk: NDArray[np.float32],
    valid: NDArray[np.float32],
    starts: NDArray[np.intp],
    reads: dict[str, tuple[NDArray[np.intp], NDArray[np.float32]]],
    *,
    hidden: list[int],
    horizon: int,
    action_dim: int,
    steps: int,
    batch_size: int,
    lr: float,
    lr_schedule: str,
    layer_norm: bool,
    seed: int,
    eval_every: int,
    report: Any,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Fit observation -> action chunk, reading the error as it goes.

    Args:
        observations: standardized, one row per transition.
        chunk: the regression target from :func:`build_chunks`.
        valid: its mask.
        starts: window start rows the fit may draw from.
        reads: named query sets, each ``(rows, reference first action)``.
        report: called with the history row after each evaluation.

    Returns:
        The history, and the final read of each query set.
    """
    import jax
    import jax.numpy as jnp
    import optax

    try:
        from utils.networks import MLP
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the gate uses qc's actor body, so the qc checkout has to be on "
            "PYTHONPATH: PYTHONPATH=<g3a-bench>:<qc>"
        ) from error

    model = MLP(
        (*hidden, horizon * action_dim), activate_final=False, layer_norm=layer_norm
    )
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = model.init(init_key, jnp.asarray(observations[:1]))["params"]
    schedule = optax.cosine_decay_schedule(lr, steps) if lr_schedule == "cosine" else lr
    tx = optax.adam(schedule)
    opt_state = tx.init(params)

    obs_d = jnp.asarray(observations)
    chunk_d = jnp.asarray(chunk)
    valid_d = jnp.asarray(valid)
    starts_d = jnp.asarray(starts)

    @jax.jit
    def train_step(params, opt_state, key):
        key, sub = jax.random.split(key)
        rows = starts_d[jax.random.randint(sub, (batch_size,), 0, len(starts_d))]

        def loss_fn(p):
            pred = model.apply({"params": p}, obs_d[rows]).reshape(
                batch_size, horizon, action_dim
            )
            return jnp.mean((pred - chunk_d[rows]) ** 2 * valid_d[rows][..., None])

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, key, loss

    @jax.jit
    def first_action(params, rows):
        return model.apply({"params": params}, obs_d[rows]).reshape(
            -1, horizon, action_dim
        )[:, 0]

    def read(params) -> dict[str, float]:
        """Teacher-forced error of the first action, per query set.

        The largest deviation over the joints, averaged over the states. The
        regressor is deterministic, so this is the whole story: no draw, no
        spread.
        """
        out = {}
        for name, (rows, reference) in reads.items():
            pred = np.asarray(first_action(params, jnp.asarray(rows)))
            out[name] = float(np.abs(pred - reference).max(axis=-1).mean())
        return out

    history: list[dict[str, float]] = []
    final: dict[str, float] = {}
    loss = jnp.asarray(0.0)
    for step in range(1, steps + 1):
        params, opt_state, key, loss = train_step(params, opt_state, key)
        if step % eval_every == 0 or step == steps:
            final = read(params)
            row = {"step": step, "mse": float(loss), **final}
            history.append(row)
            report(row)
    return history, final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", help=".npz from g3a-convert-demos")
    parser.add_argument(
        "--target",
        type=float,
        default=TARGET,
        help="held-out interior error to beat (default: the old set's, read here)",
    )
    parser.add_argument(
        "--holdout",
        type=int,
        default=0,
        help="episodes held out (default: a fifth, at least one)",
    )
    parser.add_argument("--holdout-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine"),
        default="cosine",
        help="cosine settles the tail, which oscillates once the loss is small",
    )
    parser.add_argument("--horizon", type=int, default=5, help="action chunk length")
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 512, 512])
    parser.add_argument(
        "--layer-norm",
        action="store_true",
        help="match an agent trained with actor_layer_norm=true",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reads-per-episode",
        type=int,
        default=40,
        help="interior states read from each episode",
    )
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--out", default="", help="write history.json here")
    args = parser.parse_args()

    data = load_columns(Path(args.dataset))
    observations = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    action_dim = actions.shape[1]
    bounds = episode_bounds(data["episode_index"])
    episode_ids = sorted(bounds)

    holdout = args.holdout or max(1, round(0.2 * len(episode_ids)))
    train_ids, held_ids = split_episodes(episode_ids, holdout, args.holdout_seed)

    chunk, valid = build_chunks(actions, data["terminals"], args.horizon)
    starts = training_starts(data["episode_index"], args.horizon, held_ids)
    if len(starts) == 0:
        raise ValueError("the holdout leaves no window to fit; hold out fewer episodes")

    mean, std = standardizer(observations, starts)
    standardized = (observations - mean) / std

    reads = {}
    for label, ids in (("train", train_ids), ("held", held_ids)):
        rows0 = first_frames(bounds, ids)
        rows_i = interior_frames(bounds, ids, args.horizon, args.reads_per_episode)
        reads[f"{label} t=0"] = (rows0, actions[rows0])
        reads[f"{label} t>={INTERIOR_SKIP}"] = (rows_i, actions[rows_i])

    step = np.abs(np.diff(actions, axis=0)).max(axis=-1)
    print(f"dataset    : {args.dataset}")
    print(
        f"             {len(actions)} transitions, {len(episode_ids)} episodes, "
        f"{action_dim}-d actions"
    )
    print(
        f"split      : fit {len(train_ids)} episodes ({len(starts)} of "
        f"{len(chunk)} windows), hold out {len(held_ids)}: {held_ids}"
    )
    print(f"model      : MLP {args.hidden} -> {args.horizon * action_dim} (qc's body)")
    print(
        f"optimizer  : adam lr={args.lr} ({args.lr_schedule}), "
        f"batch {args.batch_size}, {args.steps} steps"
    )
    print("observations: standardized on the fitted episodes only")
    print(
        f"scale      : the demos' own |a[t+1] - a[t]| is "
        f"{np.median(step):.4f} median, {step.mean():.4f} mean"
    )
    print(f"bar        : held-out interior error under {args.target}\n")

    columns = list(reads)
    header = f"{'step':>8}{'train mse':>12}" + "".join(f"{c:>13}" for c in columns)
    print(header)
    print("-" * len(header))

    started = time.time()
    history, final = fit(
        standardized,
        chunk,
        valid,
        starts,
        reads,
        hidden=list(args.hidden),
        horizon=args.horizon,
        action_dim=action_dim,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        layer_norm=args.layer_norm,
        seed=args.seed,
        eval_every=args.eval_every,
        report=lambda row: print(
            f"{row['step']:8d}{row['mse']:12.6f}"
            + "".join(f"{row[c]:13.4f}" for c in columns)
        ),
    )
    print(f"\n{args.steps} steps in {time.time() - started:.0f} s")

    held_interior = final[f"held t>={INTERIOR_SKIP}"]
    train_interior = final[f"train t>={INTERIOR_SKIP}"]
    passed = held_interior < args.target

    print(f"\nfitted   interior error : {train_interior:.4f}")
    print(f"held out interior error : {held_interior:.4f}   (bar {args.target})")
    if passed:
        print("\nPASS. The fit transfers to unseen episodes. A run is worth starting.")
    elif train_interior < 0.5 * held_interior:
        print(
            "\nFAIL. The regressor fits what it saw and does not transfer, so "
            "these demonstrations are memorized, not learned. More of the same "
            "will not help."
        )
    else:
        print(
            "\nFAIL, and the fitted episodes read badly too: the observation "
            "does not determine the action. Look at the observation layout "
            "before recording more."
        )

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(f"wrote {out / 'history.json'}")

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
