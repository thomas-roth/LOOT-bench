"""Tests for the LeRobot-level session fuse."""

from typing import Any

import numpy as np
import pytest

from loot_bench.scripts.fuse_demos import fuse_datasets, fuse_extra_info
from loot_bench.scripts.live_bridge.dataset_extra_info import (
    load_extra_info,
    write_extra_info,
)

OBS_DIM = 5
ACTION_DIM = 7
TASK_ID = "LOOT-Stack-1Pi-Dk1"


def _info(noise: float = 0.0, seeds: dict[int, int] | None = None, **extra: Any):
    info: dict[str, Any] = {"start_pose_noise": noise}
    if seeds is not None:
        info["episode_seeds"] = {str(k): v for k, v in seeds.items()}
    info.update(extra)
    return info


class TestFuseExtraInfo:
    def test_empty_sidecars_stay_empty(self):
        assert fuse_extra_info([{}, {}], [2, 3]) == {}

    def test_a_missing_sidecar_counts_as_zero_noise(self):
        """Mirrors convert: synthetic roots without a sidecar are 0.0."""
        out = fuse_extra_info([{}, _info(0.0)], [1, 1])
        assert out["start_pose_noise"] == 0.0

    def test_mixed_noise_is_refused(self):
        """One fused root stamps one noise; two start distributions must not
        train as one (the same rule convert enforces)."""
        with pytest.raises(ValueError, match="mixed start_pose_noise"):
            fuse_extra_info([_info(0.0), _info(0.05)], [1, 1])

    def test_a_sidecar_without_the_key_names_it(self):
        with pytest.raises(ValueError, match="no start_pose_noise"):
            fuse_extra_info([{"operator_id": 1}, _info(0.0)], [1, 1])

    def test_seeds_are_reindexed_with_an_episode_offset(self):
        """Every source numbers from zero; without the offset two episodes
        would claim one seed and replay would restore the wrong RNG."""
        out = fuse_extra_info([_info(0.0, {0: 11, 1: 12}), _info(0.0, {0: 21})], [2, 1])
        assert out["episode_seeds"] == {"0": 11, "1": 12, "2": 21}

    def test_unanimous_session_keys_carry_over(self):
        out = fuse_extra_info(
            [
                _info(0.0, operator_id=4, seed_device="cuda"),
                _info(0.0, operator_id=4, seed_device="cuda"),
            ],
            [1, 1],
        )
        assert out["operator_id"] == 4
        assert out["seed_device"] == "cuda"

    def test_a_disagreeing_session_key_is_dropped(self):
        """Stamping one session's operator on another's episodes would be a
        lie; omitting it is the honest fuse."""
        out = fuse_extra_info(
            [_info(0.0, operator_id=4), _info(0.0, operator_id=7)], [1, 1]
        )
        assert "operator_id" not in out

    def test_a_key_missing_in_one_source_is_dropped(self):
        out = fuse_extra_info([_info(0.0, operator_id=4), _info(0.0)], [1, 1])
        assert "operator_id" not in out

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="sidecars"):
            fuse_extra_info([_info(0.0)], [1, 1])


def _write_session(root, n_episodes, task=TASK_ID, start_value=0.0):
    """One tiny LeRobot session: ``n_episodes`` of two frames each."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {"dtype": "float32", "shape": (OBS_DIM,), "names": None},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }
    dataset = LeRobotDataset.create(
        "live_record", fps=20, features=features, root=str(root)
    )
    for episode in range(n_episodes):
        for frame in range(2):
            value = start_value + episode * 10 + frame
            dataset.add_frame(
                {
                    "observation.state": np.full(OBS_DIM, value, dtype=np.float32),
                    "action": np.full(ACTION_DIM, 0.1 * value, dtype=np.float32),
                    "next.reward": np.array([-0.5], dtype=np.float32),
                    "next.done": np.array([frame == 1]),
                    "task": task,
                }
            )
        dataset.save_episode()
    dataset.finalize()
    return root


class TestFuseDatasets:
    def test_fuses_episodes_in_order_with_a_merged_sidecar(self, tmp_path):
        first = _write_session(tmp_path / "a", 2)
        write_extra_info(
            first, _info(0.0, {0: 11, 1: 12}, operator_id=4, seed_device="cuda")
        )
        second = _write_session(tmp_path / "b", 1, start_value=100.0)
        write_extra_info(second, _info(0.0, {0: 21}, seed_device="cuda"))

        merged = fuse_datasets([first, second], tmp_path / "fused")

        assert merged.meta.total_episodes == 3
        assert merged.meta.total_frames == 6
        episodes = list(merged.hf_dataset["episode_index"])
        assert episodes == [0, 0, 1, 1, 2, 2]
        first_values = np.stack(list(merged.hf_dataset["observation.state"])).reshape(
            6, OBS_DIM
        )[:, 0]
        assert first_values[0] == 0.0
        assert first_values[-1] == 100.0 + 1

        extra = load_extra_info(tmp_path / "fused")
        assert extra["start_pose_noise"] == 0.0
        assert extra["episode_seeds"] == {"0": 11, "1": 12, "2": 21}
        # seed_device agrees, operator_id is only in the first: dropped.
        assert extra["seed_device"] == "cuda"
        assert "operator_id" not in extra

    def test_the_fused_root_converts(self, tmp_path):
        """The point of the fuse: one root in, one npz out."""
        from loot_bench.qc.convert import convert

        first = _write_session(tmp_path / "a", 1)
        write_extra_info(first, _info(0.0, {0: 11}))
        second = _write_session(tmp_path / "b", 1, start_value=1.0)
        write_extra_info(second, _info(0.0, {0: 22}))
        fuse_datasets([first, second], tmp_path / "fused")

        data = convert([tmp_path / "fused"], tmp_path / "demos.npz")
        assert data["task_id"] == TASK_ID
        assert data["start_pose_noise"] == 0.0
        # Four fused rows, minus each timed-out episode's unusable last row.
        assert len(data["observations"]) == 2

    def test_refuses_to_overwrite(self, tmp_path):
        first = _write_session(tmp_path / "a", 1)
        out = tmp_path / "fused"
        out.mkdir()
        with pytest.raises(FileExistsError, match="already exists"):
            fuse_datasets([first], out)

    def test_a_root_without_meta_says_so(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not a LeRobot dataset root"):
            fuse_datasets([tmp_path / "nowhere"], tmp_path / "fused")

    def test_no_sources_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="no source roots"):
            fuse_datasets([], tmp_path / "fused")


class TestMain:
    def test_repeated_src_flags_all_reach_fuse(self, tmp_path, monkeypatch):
        """Regression: ``nargs='+'`` silently kept only the last ``--src``,
        fusing one session while reporting success."""
        import sys
        from unittest.mock import patch

        import loot_bench.scripts.fuse_demos as fuse_mod

        argv = [
            "loot-fuse-demos",
            "--src",
            "a",
            "--src",
            "b",
            "--src",
            "c",
            "--out",
            str(tmp_path / "fused"),
        ]
        monkeypatch.setattr(sys, "argv", argv)
        with patch.object(fuse_mod, "fuse_datasets") as mock_fuse:
            fuse_mod.main()
        (srcs, out), _ = mock_fuse.call_args
        assert srcs == ["a", "b", "c"]
        assert out == str(tmp_path / "fused")
