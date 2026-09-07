# pyright: reportMissingImports=false
"""Behaviour-cloning gate: do these demonstrations generalize?

A training run costs hours. This costs about two minutes and answers the one
question that decides whether the run is worth starting: fit the recorded
actions with a plain regressor, and see how well the fit transfers to episodes
it never saw. It trains no policy; it is a measurement instrument.

The regressor gets exactly what the flow actor gets: the same observation,
action chunk, ``valid`` mask, and MLP body. No flow noise, critic, or
sampling, so a poor read reflects the data, not the agent.

Read the held-out error against the fitted one. The verdict printed at the end
names which of the three outcomes the run hit. Do not read a pass as a
promise: the set that set the default ``--target`` scored 0 successes in 20
evaluation episodes. A dataset that clears this gate has cleared the cheapest
of the reasons a run fails, and no more.

Provenance of the bar, and why the old 0.166 and 0.118 readings cannot be
compared with it: ``docs/adr/012-gate-a-demo-set-with-a-bc-fit.md``.

Runs from the qc venv (it needs JAX), with both checkouts on the path::

    PYTHONPATH=<loot-bench>:<qc> <qc>/.venv/bin/python tests/qc/bc_gate.py \\
        out/qc/demos.npz

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

TARGET = 0.0766
"""Held-out interior error of the old 100-demonstration set, read by this gate.

Measured on the legacy ``LOOT-Place-Easy-Dk1_qc_physical.npz`` with
``--holdout 10`` and defaults elsewhere. See
``docs/adr/012-gate-a-demo-set-with-a-bc-fit.md`` for why that file, not
today's demo directory, owns the number.
"""

INTERIOR_SKIP = 5
"""First frame of an episode the interior read starts from.

The opening frames of every episode look alike, because the arm starts from
one pose. Reading them measures the start pose, not the policy.
"""


@dataclass(frozen=True)
class FitCfg:
    """Training hyper-parameters that always travel together (Data Clump)."""

    hidden: tuple[int, ...]
    steps: int
    batch_size: int
    lr: float
    lr_schedule: str
    layer_norm: bool
    seed: int
    eval_every: int


@dataclass(frozen=True)
class FitData:
    """Everything the gate reads from the dataset, as one object (Data Clump).

    ``horizon`` and ``action_dim`` live here, not in :class:`FitCfg`, because
    they shape the data, not the fit.
    """

    standardized: NDArray[np.float32]
    chunk: NDArray[np.float32]
    valid: NDArray[np.float32]
    starts: NDArray[np.intp]
    reads: dict[str, tuple[NDArray[np.intp], NDArray[np.float32]]]
    action_dim: int
    horizon: int
    train_ids: list[int]
    held_ids: list[int]
    actions: NDArray[np.float32]


def build_chunks(
    actions: NDArray[np.floating], terminals: NDArray[np.floating], horizon: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """The chunk target and ``valid`` mask qc's ``acfql`` regresses on.

    Mirrors ``qc/utils/datasets.py::sample_sequence``: a window may cross an
    episode boundary, and the steps at or after the first terminal inside it
    are masked out rather than dropped. Reproducing the mask matters: a gate
    trained on windows the agent never sees would answer another question.

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
                "loot-convert-demos"
            )
        return {key: np.asarray(loaded[key]) for key in needed}


def fit(
    data: FitData, cfg: FitCfg, report: Any
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Fit observation -> action chunk, reading the error as it goes.

    Args:
        data: the prepared problem: standardized observations, the chunk
            target from :func:`build_chunks` and its mask, the window starts
            the fit may draw from, and the named read sets, each
            ``(rows, reference first action)``.
        cfg: training hyper-parameters.
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
            "PYTHONPATH: PYTHONPATH=<loot-bench>:<qc>"
        ) from error

    model = MLP(
        (*cfg.hidden, data.horizon * data.action_dim),
        activate_final=False,
        layer_norm=cfg.layer_norm,
    )
    key = jax.random.PRNGKey(cfg.seed)
    key, init_key = jax.random.split(key)
    params = model.init(init_key, jnp.asarray(data.standardized[:1]))["params"]
    schedule = (
        optax.cosine_decay_schedule(cfg.lr, cfg.steps)
        if cfg.lr_schedule == "cosine"
        else cfg.lr
    )
    tx = optax.adam(schedule)
    opt_state = tx.init(params)

    obs_d = jnp.asarray(data.standardized)
    chunk_d = jnp.asarray(data.chunk)
    valid_d = jnp.asarray(data.valid)
    starts_d = jnp.asarray(data.starts)

    @jax.jit
    def train_step(params, opt_state, key):
        key, sub = jax.random.split(key)
        rows = starts_d[jax.random.randint(sub, (cfg.batch_size,), 0, len(starts_d))]

        def loss_fn(p):
            pred = model.apply({"params": p}, obs_d[rows]).reshape(
                cfg.batch_size, data.horizon, data.action_dim
            )
            return jnp.mean((pred - chunk_d[rows]) ** 2 * valid_d[rows][..., None])

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, key, loss

    @jax.jit
    def first_action(params, rows):
        return model.apply({"params": params}, obs_d[rows]).reshape(
            -1, data.horizon, data.action_dim
        )[:, 0]

    def read(params) -> dict[str, float]:
        """Teacher-forced error of the first action, per query set.

        The largest deviation over the joints, averaged over the states. The
        regressor is deterministic, so this is the whole story: no draw, no
        spread.
        """
        out = {}
        for name, (rows, reference) in data.reads.items():
            pred = np.asarray(first_action(params, jnp.asarray(rows)))
            out[name] = float(np.abs(pred - reference).max(axis=-1).mean())
        return out

    history: list[dict[str, float]] = []
    final: dict[str, float] = {}
    loss = jnp.asarray(0.0)
    for step in range(1, cfg.steps + 1):
        params, opt_state, key, loss = train_step(params, opt_state, key)
        if step % cfg.eval_every == 0 or step == cfg.steps:
            final = read(params)
            row = {"step": step, "mse": float(loss), **final}
            history.append(row)
            report(row)
    return history, final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", help=".npz from loot-convert-demos")
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
    return parser.parse_args()


def prepare_data(args: argparse.Namespace) -> FitData:
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

    return FitData(
        standardized=standardized,
        chunk=chunk,
        valid=valid,
        starts=starts,
        reads=reads,
        action_dim=action_dim,
        horizon=args.horizon,
        train_ids=train_ids,
        held_ids=held_ids,
        actions=actions,
    )


def print_context(args: argparse.Namespace, data: FitData) -> list[str]:
    step = np.abs(np.diff(data.actions, axis=0)).max(axis=-1)
    print(f"dataset    : {args.dataset}")
    print(
        f"             {len(data.actions)} transitions, "
        f"{len(data.train_ids) + len(data.held_ids)} episodes, "
        f"{data.action_dim}-d actions"
    )
    print(
        f"split      : fit {len(data.train_ids)} episodes "
        f"({len(data.starts)} of {len(data.chunk)} windows), "
        f"hold out {len(data.held_ids)}: {data.held_ids}"
    )
    print(
        f"model      : MLP {args.hidden} -> "
        f"{data.horizon * data.action_dim} (qc's body)"
    )
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

    columns = [
        "train t=0",
        f"train t>={INTERIOR_SKIP}",
        "held t=0",
        f"held t>={INTERIOR_SKIP}",
    ]
    header = f"{'step':>8}{'train mse':>12}" + "".join(f"{c:>13}" for c in columns)
    print(header)
    print("-" * len(header))
    return columns


def run_fit(
    data: FitData, cfg: FitCfg, columns: list[str]
) -> tuple[list[dict[str, float]], dict[str, float]]:
    started = time.time()
    history, final = fit(
        data,
        cfg,
        report=lambda row: print(
            f"{row['step']:8d}{row['mse']:12.6f}"
            + "".join(f"{row.get(c, 0):13.4f}" for c in columns)
        ),
    )
    print(f"\n{cfg.steps} steps in {time.time() - started:.0f} s")
    return history, final


def print_verdict(
    train_interior: float,
    held_interior: float,
    target: float,
    out: str,
    history: list[dict[str, float]],
) -> None:
    passed = held_interior < target

    print(f"\nfitted   interior error : {train_interior:.4f}")
    print(f"held out interior error : {held_interior:.4f}   (bar {target})")
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

    if out:
        path = Path(out)
        path.mkdir(parents=True, exist_ok=True)
        (path / "history.json").write_text(json.dumps(history, indent=2))
        print(f"wrote {path / 'history.json'}")

    raise SystemExit(0 if passed else 1)


def main() -> None:
    args = parse_args()
    data = prepare_data(args)
    cfg = FitCfg(
        hidden=tuple(args.hidden),
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        layer_norm=args.layer_norm,
        seed=args.seed,
        eval_every=args.eval_every,
    )
    columns = print_context(args, data)
    history, final = run_fit(data, cfg, columns)
    held_interior = final[f"held t>={INTERIOR_SKIP}"]
    train_interior = final[f"train t>={INTERIOR_SKIP}"]
    print_verdict(train_interior, held_interior, args.target, args.out, history)


if __name__ == "__main__":
    main()
