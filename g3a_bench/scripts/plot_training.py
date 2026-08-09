"""Plot reward and learning curves from a QC-FQL run directory.

QC writes its metrics as plain CSVs next to the checkpoints, so a run can be
inspected while it is still training -- the CSVs are flushed on every log
interval.

    uv run python g3a_bench/scripts/plot_training.py <run_dir>
    uv run python g3a_bench/scripts/plot_training.py <run_dir> --watch 60

<run_dir> is either the directory holding eval.csv (the save_dir QC creates,
``exp/qc/<group>/<env>/<exp_name>/``) or any parent of it -- the newest
matching directory below it is used.

Panels:
    1. reward -- eval episode return and success rate over training steps
    2. episode -- final reward and episode length (did it end early?)
    3. critic -- Q value range, i.e. what the agent believes it can achieve
    4. losses -- critic / BC-flow / distillation loss

The vertical line marks the offline -> online transition.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

Series = dict[str, list[float]]


def _read_csv(path: Path) -> Series:
    """Read a QC metrics CSV into {column: [values]}, skipping unparsable cells.

    A run killed mid-write can leave a truncated final line; that must not
    stop us from plotting everything before it.
    """
    if not path.exists():
        return {}
    out: Series = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return {}
        for name in reader.fieldnames:
            out[name] = []
        for row in reader:
            if row.get("step") in (None, ""):
                continue
            for name in reader.fieldnames:
                try:
                    out[name].append(float(row[name]))
                except (TypeError, ValueError):
                    out[name].append(float("nan"))
    return {k: v for k, v in out.items() if v}


def find_run_dir(root: Path) -> Path:
    """Return `root` if it holds eval.csv, else the newest such dir below it."""
    if (root / "eval.csv").exists() or (root / "offline_agent.csv").exists():
        return root
    candidates = [p.parent for p in root.rglob("eval.csv")]
    candidates += [p.parent for p in root.rglob("offline_agent.csv")]
    if not candidates:
        raise SystemExit(f"no QC run (eval.csv / offline_agent.csv) found under {root}")
    return max(set(candidates), key=lambda p: p.stat().st_mtime)


def _legend(ax, loc: str) -> None:
    """Add a legend only if something on the axis is labelled."""
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc=loc, fontsize=9)


def _plot(run_dir: Path, out_path: Path) -> None:
    ev = _read_csv(run_dir / "eval.csv")
    off = _read_csv(run_dir / "offline_agent.csv")
    on = _read_csv(run_dir / "online_agent.csv")

    # The offline->online boundary is the last offline step logged.
    boundary = max(off.get("step", [0])) if off.get("step") else None

    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    fig.suptitle(f"QC-FQL: {run_dir.name}", fontsize=13)

    def mark_boundary(ax):
        if boundary:
            ax.axvline(boundary, color="0.4", ls="--", lw=1)

    # --- 1. reward -------------------------------------------------------
    ax = axes[0]
    if ev.get("episode.return"):
        ax.plot(ev["step"], ev["episode.return"], "o-", color="C0", label="eval return")
        ax.set_ylabel("episode return")
    ax.grid(alpha=0.3)
    mark_boundary(ax)
    if ev.get("success"):
        ax2 = ax.twinx()
        ax2.plot(ev["step"], ev["success"], "s-", color="C3", label="success rate")
        ax2.set_ylabel("success rate", color="C3")
        ax2.set_ylim(-0.02, 1.02)
        ax2.tick_params(axis="y", labelcolor="C3")
    ax.set_title("reward: what the policy actually achieves")
    _legend(ax, "upper left")

    # --- 2. episode shape ------------------------------------------------
    ax = axes[1]
    if ev.get("episode.final_reward"):
        ax.plot(ev["step"], ev["episode.final_reward"], "o-", color="C2",
                label="final reward (0 = success)")
    if ev.get("episode.length"):
        ax2 = ax.twinx()
        ax2.plot(ev["step"], ev["episode.length"], "^-", color="C1", label="episode length")
        ax2.set_ylabel("steps", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
    ax.set_ylabel("final reward")
    ax.grid(alpha=0.3)
    mark_boundary(ax)
    ax.set_title("episode: did it terminate early (success) or run to the limit?")
    _legend(ax, "upper left")

    # --- 3. critic -------------------------------------------------------
    ax = axes[2]
    for src, style in ((off, "-"), (on, "-")):
        if src.get("critic/q_mean"):
            ax.plot(src["step"], src["critic/q_mean"], style, color="C0", lw=1.2)
            if src.get("critic/q_min") and src.get("critic/q_max"):
                ax.fill_between(src["step"], src["critic/q_min"], src["critic/q_max"],
                                color="C0", alpha=0.15)
    ax.set_ylabel("Q value")
    ax.grid(alpha=0.3)
    mark_boundary(ax)
    ax.set_title("critic: expected return (band = min/max over the batch)")

    # --- 4. losses -------------------------------------------------------
    ax = axes[3]
    for key, color in (("critic/critic_loss", "C0"),
                       ("actor/bc_flow_loss", "C1"),
                       ("actor/distill_loss", "C2")):
        for src in (off, on):
            if src.get(key):
                ax.plot(src["step"], src[key], color=color, lw=1.2,
                        label=key if src is off else None)
    ax.set_yscale("log")
    ax.set_ylabel("loss (log)")
    ax.set_xlabel("training step")
    ax.grid(alpha=0.3)
    mark_boundary(ax)
    ax.set_title("losses: is the BC flow actually fitting the demos?")
    _legend(ax, "upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    # A plot nobody reads is useless; print the headline numbers too.
    if ev.get("success"):
        last = ev["success"][-1]
        best = max(ev["success"])
        print(f"  eval success: last={last:.2f}  best={best:.2f}  "
              f"({len(ev['success'])} evaluations)")
    if ev.get("episode.return"):
        print(f"  eval return:  last={ev['episode.return'][-1]:.1f}  "
              f"best={max(ev['episode.return']):.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="QC run dir (or a parent of it)")
    ap.add_argument("--out", type=Path, default=None, help="output PNG (default: <run_dir>/training.png)")
    ap.add_argument("--watch", type=float, default=0.0,
                    help="re-render every N seconds (for a running job)")
    args = ap.parse_args()

    run_dir = find_run_dir(args.run_dir)
    out_path = args.out or run_dir / "training.png"
    print(f"run: {run_dir}")

    while True:
        _plot(run_dir, out_path)
        print(f"  wrote {out_path}")
        if not args.watch:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
