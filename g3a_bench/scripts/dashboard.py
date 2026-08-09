"""Generate an interactive HTML dashboard from QC run metrics.

Reads eval.csv, offline_agent.csv, online_agent.csv from a QC run directory
and renders a Plotly dashboard with zoom, pan, and hover-value readout.

Usage:
    uv run python g3a_bench/scripts/dashboard.py <run_dir>
    uv run python g3a_bench/scripts/dashboard.py <run_dir> --watch 30

Opens in the default browser. Re-run to update the dashboard while training.
"""

from __future__ import annotations

import argparse
import csv
import time
import webbrowser
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    raise SystemExit(
        "plotly not installed. Install: pip install plotly\n"
        "or: cd ../g3a-bench && uv sync"
    )

Series = dict[str, list[float]]


def _read_csv(path: Path) -> Series:
    """Read a CSV into {column: [values]}, skipping unparsable cells."""
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
    """Return root if it holds eval.csv, else the newest such dir below it."""
    if (root / "eval.csv").exists() or (root / "offline_agent.csv").exists():
        return root
    candidates = [p.parent for p in root.rglob("eval.csv")]
    candidates += [p.parent for p in root.rglob("offline_agent.csv")]
    if not candidates:
        raise SystemExit(f"no QC run found under {root}")
    return max(set(candidates), key=lambda p: p.stat().st_mtime)


def _dashboard(run_dir: Path, out_path: Path) -> None:
    """Generate dashboard HTML."""
    ev = _read_csv(run_dir / "eval.csv")
    off = _read_csv(run_dir / "offline_agent.csv")
    on = _read_csv(run_dir / "online_agent.csv")

    boundary = max(off.get("step", [0])) if off.get("step") else None

    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "Eval: Return & Success Rate",
            "Eval: Episode Length & Final Reward",
            "Training: Critic Q-values",
            "Training: Losses (log scale)",
        ),
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
        vertical_spacing=0.12,
    )

    # --- Panel 1: eval return + success -------
    if ev.get("step") and ev.get("episode.return"):
        fig.add_trace(
            go.Scatter(
                x=ev["step"],
                y=ev["episode.return"],
                mode="lines+markers",
                name="eval return",
                line=dict(color="blue", width=2),
                marker=dict(size=6),
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
    if ev.get("step") and ev.get("success"):
        fig.add_trace(
            go.Scatter(
                x=ev["step"],
                y=ev["success"],
                mode="lines+markers",
                name="success rate",
                line=dict(color="red", width=2),
                marker=dict(size=6),
            ),
            row=1,
            col=1,
            secondary_y=True,
        )
    fig.update_yaxes(title_text="episode return", row=1, col=1, secondary_y=False)
    fig.update_yaxes(
        title_text="success rate", row=1, col=1, secondary_y=True, range=[-0.05, 1.05]
    )

    # --- Panel 2: episode length + final reward -------
    if ev.get("step") and ev.get("episode.length"):
        fig.add_trace(
            go.Scatter(
                x=ev["step"],
                y=ev["episode.length"],
                mode="lines+markers",
                name="episode length",
                line=dict(color="orange", width=2),
                marker=dict(size=6),
            ),
            row=2,
            col=1,
            secondary_y=True,
        )
    if ev.get("step") and ev.get("episode.final_reward"):
        fig.add_trace(
            go.Scatter(
                x=ev["step"],
                y=ev["episode.final_reward"],
                mode="lines+markers",
                name="final reward",
                line=dict(color="green", width=2),
                marker=dict(size=6),
            ),
            row=2,
            col=1,
            secondary_y=False,
        )
    fig.update_yaxes(title_text="final reward", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="episode length", row=2, col=1, secondary_y=True)

    # --- Panel 3: critic Q values -------
    for src, style in ((off, "solid"), (on, "solid")):
        if src.get("step") and src.get("critic/q_mean"):
            fig.add_trace(
                go.Scatter(
                    x=src["step"],
                    y=src["critic/q_mean"],
                    mode="lines",
                    name="q_mean",
                    line=dict(color="blue", width=2, dash=style),
                ),
                row=3,
                col=1,
            )
            if src.get("critic/q_min") and src.get("critic/q_max"):
                fig.add_trace(
                    go.Scatter(
                        x=src["step"],
                        y=src["critic/q_min"],
                        mode="lines",
                        name="q_min",
                        line=dict(width=0),
                        showlegend=False,
                    ),
                    row=3,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=src["step"],
                        y=src["critic/q_max"],
                        mode="lines",
                        name="q_max",
                        line=dict(width=0),
                        fillcolor="rgba(0,100,200,0.2)",
                        fill="tonexty",
                        showlegend=False,
                    ),
                    row=3,
                    col=1,
                )
            break  # only once
    fig.update_yaxes(title_text="Q value", row=3, col=1, type="linear")

    # --- Panel 4: losses (log) -------
    for key, color in (
        ("critic/critic_loss", "blue"),
        ("actor/bc_flow_loss", "orange"),
        ("actor/distill_loss", "green"),
    ):
        for src in (off, on):
            if src.get("step") and src.get(key):
                fig.add_trace(
                    go.Scatter(
                        x=src["step"],
                        y=src[key],
                        mode="lines",
                        name=key,
                        line=dict(color=color, width=1.5),
                    ),
                    row=4,
                    col=1,
                )
                break  # only from offline
    fig.update_yaxes(title_text="loss (log)", row=4, col=1, type="log")
    fig.update_xaxes(title_text="training step", row=4, col=1)

    # Mark offline/online boundary on all axes
    if boundary:
        for i in range(1, 5):
            fig.add_vline(x=boundary, line_dash="dash", line_color="gray", row=i, col=1)

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    fig.update_layout(
        title_text=f"QC-FQL Dashboard: {run_dir.name}",
        title_font_size=16,
        hovermode="x unified",
        height=1200,
        font=dict(size=11),
        showlegend=True,
    )

    fig.write_html(str(out_path))
    print(f"wrote {out_path}")
    if out_path.exists():
        print(f"open with: open {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dir", type=Path, help="QC run dir (or parent)")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output HTML (default: <run_dir>/dashboard.html)",
    )
    ap.add_argument(
        "--watch",
        type=float,
        default=0.0,
        help="re-render every N seconds (for a running job)",
    )
    ap.add_argument(
        "--open", action="store_true", help="open in browser after first render"
    )
    args = ap.parse_args()

    run_dir = find_run_dir(args.run_dir)
    out_path = args.out or run_dir / "dashboard.html"
    print(f"run: {run_dir}")

    first = True
    while True:
        _dashboard(run_dir, out_path)
        if first and args.open:
            webbrowser.open(f"file://{out_path.resolve()}")
            first = False
        if not args.watch:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
