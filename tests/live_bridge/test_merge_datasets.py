"""Merging must preserve the reward convention, not just the arrays.

``g3a_bench.qc.validate_reward_convention`` refuses to train on a dataset whose
``reward_max`` it cannot read, so a merge that drops that metadata silently
turns valid inputs into an untrainable output.
"""

import sys

import numpy as np
import pytest

from g3a_bench.scripts.live_bridge import merge_datasets


def _write(
    path,
    *,
    reward_max: float | None = 2.0,
    task_id: str = "G3A-Stack-Low-Dk1_uni",
    size: int = 3,
):
    arrays = {
        "observations": np.ones((size, 4), dtype=np.float32),
        "next_observations": np.ones((size, 4), dtype=np.float32),
        "actions": np.zeros((size, 7), dtype=np.float32),
        "rewards": np.full(size, -1.0, dtype=np.float32),
        "terminals": np.array([0.0] * (size - 1) + [1.0], dtype=np.float32),
        "masks": np.array([1.0] * (size - 1) + [0.0], dtype=np.float32),
        "episode_index": np.zeros(size, dtype=np.int64),
        "frame_index": np.arange(size, dtype=np.int64),
        "timestamp": np.arange(size, dtype=np.float64),
        "success": np.ones(size, dtype=bool),
        "sim_done": np.ones(size, dtype=bool),
        "action_min": np.full((7,), -1.0, dtype=np.float32),
        "action_max": np.full((7,), 1.0, dtype=np.float32),
        "action_norm_method": np.array("physical"),
    }
    if reward_max is not None:
        arrays["reward_max"] = np.array(reward_max, dtype=np.float32)
        arrays["task_id"] = np.array(task_id)
    np.savez(path, **arrays)
    return path


def _run(monkeypatch, inputs, output):
    argv = ["merge_datasets.py", *[str(p) for p in inputs], "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    merge_datasets.main()


def test_merged_output_keeps_the_reward_convention(tmp_path, monkeypatch):
    a = _write(tmp_path / "a.npz")
    b = _write(tmp_path / "b.npz")
    out = tmp_path / "merged.npz"

    _run(monkeypatch, [a, b], out)

    merged = np.load(out)
    assert float(np.asarray(merged["reward_max"]).reshape(-1)[0]) == pytest.approx(2.0)
    assert str(np.atleast_1d(merged["task_id"])[0]) == "G3A-Stack-Low-Dk1_uni"
    assert len(merged["observations"]) == 6


def test_rejects_mixed_reward_max(tmp_path, monkeypatch):
    """A 2.0 file and a 1.0 file put two reward scales in one dataset."""
    a = _write(tmp_path / "a.npz", reward_max=2.0)
    b = _write(tmp_path / "b.npz", reward_max=1.0, task_id="G3A-Place-Easy-Dk1_uni")

    with pytest.raises(ValueError, match="Mixed reward_max"):
        _run(monkeypatch, [a, b], tmp_path / "merged.npz")


def test_rejects_mixed_task_ids(tmp_path, monkeypatch):
    a = _write(tmp_path / "a.npz", task_id="G3A-Stack-Low-Dk1_uni")
    b = _write(tmp_path / "b.npz", task_id="G3A-Stack-High-Dk1_uni")

    with pytest.raises(ValueError, match="Mixed task ids"):
        _run(monkeypatch, [a, b], tmp_path / "merged.npz")


def test_rejects_pre_convention_input(tmp_path, monkeypatch):
    """Merging cannot invent reward_max; say so here rather than at train time."""
    a = _write(tmp_path / "a.npz")
    b = _write(tmp_path / "b.npz", reward_max=None)

    with pytest.raises(ValueError, match="no reward_max metadata"):
        _run(monkeypatch, [a, b], tmp_path / "merged.npz")
