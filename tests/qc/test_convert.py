"""Tests for the recording -> training-format converter.

The episode-structure rules live in ``test_transitions.py``. What is left here
is everything the converter adds around them: the action column, the row order,
and the convention record that lets a mismatched dataset fail loudly instead of
training to zero.
"""

import numpy as np
import pytest

from loot_bench.qc.convert import (
    NUMERIC_STAMP_FIELDS,
    STAMP_KEYS,
    action_convention,
    build_dataset,
    check_dataset_matches_env,
    convert,
    load_dataset,
)
from loot_bench.qc.dataset import CONVENTION_NAME
from loot_bench.qc.transitions import QC_KEYS

OBS_DIM = 5
ACTION_DIM = 7
TASK_ID = "LOOT-Stack-1Henge-Dk1"
START_POSE_NOISE = 0.0


def _action(value: float) -> list[float]:
    """One recorded action: a 6-D end-effector delta, then the gripper.

    Inside the box by construction, because a recorded action is one the
    mapping already clamped -- see ``tests/live_bridge/test_mapper.py``.
    """
    return [value, -value, 0.5 * value, 0.0, 0.0, 0.1 * value, 1.0]


def _columns(
    episodes: dict[int, list[float]],
) -> dict[str, np.ndarray]:
    """Build recording columns from ``{episode: [rewards...]}``."""
    episode_index, frame_index, observations, actions, rewards = [], [], [], [], []
    for episode, episode_rewards in episodes.items():
        for frame, reward in enumerate(episode_rewards):
            episode_index.append(episode)
            frame_index.append(frame)
            observations.append(np.full(OBS_DIM, float(frame), dtype=np.float32))
            actions.append(_action(0.1 * frame))
            rewards.append(reward)
    return {
        "episode_index": np.array(episode_index, dtype=np.int64),
        "frame_index": np.array(frame_index, dtype=np.int64),
        "observations": np.array(observations, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
    }


SUCCESS_THEN_TIMEOUT = _columns({0: [-0.7, -0.4, 0.0], 1: [-0.9, -0.8, -0.6]})


def _build(columns: dict[str, np.ndarray]):
    """Call the converter with named columns, not a splat.

    The splat reads better but hides the signature from the type checker,
    which then cannot tell that a column was renamed out from under it.
    """
    return build_dataset(
        episode_index=columns["episode_index"],
        frame_index=columns["frame_index"],
        observations=columns["observations"],
        actions=columns["actions"],
        rewards=columns["rewards"],
        task_id=TASK_ID,
        start_pose_noise=START_POSE_NOISE,
    )


class TestStartPoseNoiseStamp:
    def test_stamps_the_noise_the_episodes_were_recorded_under(self):
        """Recording and evaluation must begin at the same place (ADR-008),
        and nothing else in the record would catch a difference."""
        assert _build(SUCCESS_THEN_TIMEOUT)["start_pose_noise"] == pytest.approx(
            START_POSE_NOISE
        )


class TestActionColumn:
    def test_the_recorded_action_is_copied_through(self):
        """Not re-derived. A delta means "move this far from where you are",
        and the pose it was measured against is not in the file -- so the only
        action that is still true is the one ``env.step()`` was given."""
        out = _build(SUCCESS_THEN_TIMEOUT)
        # The last row of the timed-out episode is dropped, the successful
        # episode's rows all survive; row order is (episode, frame).
        np.testing.assert_array_equal(
            out["actions"][:3], SUCCESS_THEN_TIMEOUT["actions"][:3]
        )

    def test_actions_are_inside_the_policy_box(self):
        """A bounded policy can only emit [-1, 1]. An action outside it is one
        the policy can never reproduce, however well it fits the data."""
        out = _build(SUCCESS_THEN_TIMEOUT)
        assert np.all(np.abs(out["actions"]) <= 1.0), out["actions"].max()

    def test_an_out_of_box_action_is_rejected(self):
        """The recorder's mapping clamps and the action term clamps, so an
        out-of-box row means the two disagreed -- which is the one thing this
        seam exists to prevent. Loud, not silently clipped."""
        columns = {k: v.copy() for k, v in SUCCESS_THEN_TIMEOUT.items()}
        columns["actions"][1, 2] = 1.5
        with pytest.raises(ValueError, match="outside the action box"):
            _build(columns)

    def test_action_width_is_the_delta_plus_the_gripper(self):
        out = _build(SUCCESS_THEN_TIMEOUT)
        assert out["actions"].shape[1] == ACTION_DIM


class TestRowOrder:
    def test_shuffled_rows_are_sorted_before_assembly(self):
        """Episode structure is read off row order, so an out-of-order file
        would silently produce transitions that cross frames."""
        shuffled = {k: v.copy() for k, v in SUCCESS_THEN_TIMEOUT.items()}
        order = np.array([3, 0, 4, 1, 5, 2])
        shuffled = {k: v[order] for k, v in shuffled.items()}
        assert np.array_equal(
            _build(shuffled)["observations"],
            _build(SUCCESS_THEN_TIMEOUT)["observations"],
        )

    def test_the_action_column_is_sorted_with_the_rest(self):
        """A sort applied to the observations but not the actions would pair
        every state with another state's action, and nothing would raise."""
        shuffled = {k: v[::-1].copy() for k, v in SUCCESS_THEN_TIMEOUT.items()}
        np.testing.assert_array_equal(
            _build(shuffled)["actions"], _build(SUCCESS_THEN_TIMEOUT)["actions"]
        )

    def test_interleaved_episodes_do_not_raise_after_sorting(self):
        """The sort is what makes transitions.py's contiguity rule satisfiable."""
        _build({k: v[::-1] for k, v in SUCCESS_THEN_TIMEOUT.items()})


class TestStamp:
    def test_publishes_every_stamp_key(self):
        out = _build(SUCCESS_THEN_TIMEOUT)
        assert set(out) >= STAMP_KEYS

    def test_stamps_the_current_convention(self):
        out = _build(SUCCESS_THEN_TIMEOUT)
        assert out["convention"] == CONVENTION_NAME

    def test_stamps_the_task_the_episodes_were_flown_on(self):
        """Every ``place`` variant shares its widths, so without this a
        RandS-RandT recording trains against a FixedS-FixedT env in silence."""
        assert _build(SUCCESS_THEN_TIMEOUT)["task_id"] == TASK_ID

    def test_stamps_the_scales_the_action_terms_apply(self):
        out = _build(SUCCESS_THEN_TIMEOUT)
        expected = action_convention(TASK_ID, START_POSE_NOISE)
        for field in NUMERIC_STAMP_FIELDS:
            assert out[field] == pytest.approx(expected[field])

    def test_still_publishes_the_training_keys(self):
        out = _build(SUCCESS_THEN_TIMEOUT)
        assert set(out) >= QC_KEYS


class _FakeEnv:
    """Just enough of the adapter to answer the stamp check."""

    def __init__(self, convention, obs_dim=OBS_DIM, action_dim=ACTION_DIM):
        self._convention = dict(convention)
        self.observation_space = type("S", (), {"shape": (obs_dim,)})()
        self.action_space = type("S", (), {"shape": (action_dim,)})()

    def action_convention(self):
        return self._convention


def _env_from(
    data,
    obs_dim: int = OBS_DIM,
    action_dim: int = ACTION_DIM,
    overrides: dict | None = None,
    **field_overrides,
):
    """An env agreeing with ``data`` except where told otherwise.

    ``overrides`` takes a built dict; the keyword form is for the literal
    cases. Both exist because a ``**{field: value}`` splat also matches the
    two width parameters, which the type checker reads as a possible float
    width.
    """
    convention = {key: data[key] for key in STAMP_KEYS}
    convention.update(overrides or {})
    convention.update(field_overrides)
    return _FakeEnv(convention, obs_dim=obs_dim, action_dim=action_dim)


class TestStampCheck:
    """The failure this exists for: a dataset recorded under a different MDP
    trains without complaint, the loss falls, and evaluation scores zero."""

    def _dataset(self):
        return _build(SUCCESS_THEN_TIMEOUT)

    def test_matching_dataset_passes(self):
        data = self._dataset()
        check_dataset_matches_env(data, _env_from(data))

    @pytest.mark.parametrize("field", NUMERIC_STAMP_FIELDS)
    def test_a_differing_number_is_rejected_by_name(self, field):
        data = self._dataset()
        env = _env_from(data, overrides={field: float(data[field]) * 2.0 + 1.0})
        with pytest.raises(ValueError, match=field):
            check_dataset_matches_env(data, env)

    def test_a_different_task_id_is_rejected(self):
        data = self._dataset()
        env = _env_from(data, task_id="LOOT-Place-RandS-RandT-Dk1")
        with pytest.raises(ValueError, match="task_id"):
            check_dataset_matches_env(data, env)

    def test_a_different_convention_name_is_rejected(self):
        data = self._dataset()
        env = _env_from(data, convention="joint-pos-50hz")
        with pytest.raises(ValueError, match="convention"):
            check_dataset_matches_env(data, env)

    def test_a_dataset_from_the_old_convention_says_so(self):
        """A pre-ADR-016 file stamps ``action_scale``/``action_offset`` and no
        ``convention``. It has to name the convention rather than report
        whichever number happens to differ."""
        data = self._dataset()
        stale = {k: v for k, v in data.items() if k != "convention"}
        stale["action_scale"] = np.full(ACTION_DIM, 0.5, dtype=np.float32)
        stale["action_offset"] = np.zeros(ACTION_DIM, dtype=np.float32)
        with pytest.raises(ValueError, match="convention"):
            check_dataset_matches_env(stale, _env_from(data))

    def test_a_different_observation_width_is_rejected(self):
        data = self._dataset()
        with pytest.raises(ValueError, match="observation"):
            check_dataset_matches_env(data, _env_from(data, obs_dim=OBS_DIM + 1))

    def test_a_different_action_width_is_rejected(self):
        data = self._dataset()
        with pytest.raises(ValueError, match="action width"):
            check_dataset_matches_env(data, _env_from(data, action_dim=3))

    def test_reads_a_torch_convention(self):
        """An env builds its stamp out of the action terms' buffers, which are
        torch tensors on the sim device -- and numpy refuses a CUDA tensor
        outright, so a check that only ever saw floats would raise TypeError on
        every GPU run, which is every real run."""
        import torch

        data = self._dataset()
        env = _env_from(
            data,
            overrides={
                field: torch.as_tensor(float(data[field]))
                for field in NUMERIC_STAMP_FIELDS
            },
        )
        check_dataset_matches_env(data, env)


class TestSessionOffsetting:
    """Two sessions are each numbered from episode zero."""

    def test_colliding_indices_join_two_episodes_silently(self):
        """Why _load_many offsets. Two failed sessions, each numbered from zero,
        concatenate into one long episode. Nothing raises, and the transition
        across the join is built from two unrelated frames."""
        session = _columns({0: [-0.5, -0.4]})
        joined = {k: np.concatenate([v, v]) for k, v in session.items()}
        assert int(_build(joined)["terminals"].sum()) == 1

        offset = {**session, "episode_index": session["episode_index"] + 1}
        separate = {k: np.concatenate([session[k], offset[k]]) for k in session}
        assert int(_build(separate)["terminals"].sum()) == 2

    def test_a_collision_after_a_success_is_caught(self):
        """The one case that is loud rather than silent: session A ended in
        success, so after the join its success reward sits mid-episode."""
        session = _columns({0: [-0.5, 0.0]})
        joined = {k: np.concatenate([v, v]) for k, v in session.items()}
        with pytest.raises(ValueError, match="mid-episode"):
            _build(joined)


class TestRewardValidation:
    def test_a_stale_reward_convention_is_rejected(self):
        """Delegated to transitions.py, but it has to reach the caller."""
        columns = _columns({0: [1.0, 0.0]})
        with pytest.raises(ValueError, match="above zero"):
            _build(columns)


class TestReadingARecording:
    """The file-handling path, against a real LeRobot dataset.

    Everything above works on columns already in memory. This is the only
    coverage of the column names the recorder writes and this module reads --
    a rename on either side is otherwise found by the first real conversion.
    """

    @staticmethod
    def _write(root, episodes, task=TASK_ID):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (OBS_DIM,),
                "names": None,
            },
            "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
            "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
            "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        }
        # 20 Hz: the control rate every task now runs at (ADR-016). The
        # recorder writes the same number, so a stale fps here would be the
        # one place the test suite disagrees with the MDP.
        dataset = LeRobotDataset.create(
            "probe", fps=20, features=features, root=str(root)
        )
        for rewards in episodes:
            for frame, reward in enumerate(rewards):
                dataset.add_frame(
                    {
                        "observation.state": np.full(
                            OBS_DIM, float(frame), dtype=np.float32
                        ),
                        "action": np.array(_action(0.1 * frame), np.float32),
                        "next.reward": np.array([reward], dtype=np.float32),
                        "next.done": np.array([frame == len(rewards) - 1]),
                        "task": task,
                    }
                )
            dataset.save_episode()
        dataset.finalize()
        return root

    def test_converts_a_session_to_an_npz(self, tmp_path):
        root = self._write(tmp_path / "session", [[-0.7, -0.4, 0.0], [-0.9, -0.8]])
        output = tmp_path / "demos.npz"
        convert([root], output)

        loaded = np.load(output)
        assert set(loaded.files) >= QC_KEYS | STAMP_KEYS
        # Six recorded rows, minus the timed-out episode's unusable last row.
        assert len(loaded["observations"]) == 4
        np.testing.assert_array_equal(loaded["terminals"], [0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(loaded["masks"], [1.0, 1.0, 0.0, 1.0])

    def test_stamps_the_task_read_from_the_recording(self, tmp_path):
        """The task id comes back through lerobot's ``task_index`` -> ``tasks``
        reverse map, a frame indexed *by the task string*. Indexing it
        positionally happens to work for one task and silently mislabels the
        moment there are two."""
        root = self._write(tmp_path / "session", [[-0.5, 0.0]])
        data = convert([root], tmp_path / "demos.npz")
        assert data["task_id"] == TASK_ID

    def test_a_session_spanning_two_tasks_is_rejected(self, tmp_path):
        """A dataset stamps one task id, and every ``place`` variant shares its
        widths -- so a mixed file would load against any of them without a
        word."""
        first = self._write(tmp_path / "a", [[-0.5, 0.0]])
        second = self._write(
            tmp_path / "b", [[-0.5, 0.0]], task="LOOT-Place-RandS-RandT-Dk1"
        )
        with pytest.raises(ValueError, match="span"):
            convert([first, second], tmp_path / "demos.npz")

    def test_two_sessions_keep_their_episodes_apart(self, tmp_path):
        first = self._write(tmp_path / "a", [[-0.5, 0.0]])
        second = self._write(tmp_path / "b", [[-0.5, 0.0]])
        data = convert([first, second], tmp_path / "demos.npz")
        assert int(data["terminals"].sum()) == 2

    def test_a_root_that_is_not_a_dataset_says_so(self, tmp_path):
        """Without this, lerobot falls back to the Hugging Face Hub and
        reports a 401 against a repo id the caller never mentioned."""
        with pytest.raises(FileNotFoundError, match="not a LeRobot dataset root"):
            convert([tmp_path / "nowhere"], tmp_path / "demos.npz")


class TestLoadDataset:
    def test_round_trips_through_the_npz(self, tmp_path):
        """Including the two string fields. ``np.savez`` stores a ``str`` as a
        0-d unicode array, so the check has to survive the dtype it gets back
        rather than the one it was handed."""
        data = _build(SUCCESS_THEN_TIMEOUT)
        path = tmp_path / "demos.npz"
        np.savez(path, **data)
        loaded = load_dataset(path)
        for key in QC_KEYS | STAMP_KEYS:
            np.testing.assert_array_equal(loaded[key], data[key])

    def test_checks_a_reloaded_dataset_against_an_env(self, tmp_path):
        """The path a training run actually takes: the stamp arrives from the
        file, not from the dict the converter returned."""
        data = _build(SUCCESS_THEN_TIMEOUT)
        path = tmp_path / "demos.npz"
        np.savez(path, **data)
        load_dataset(path, _env_from(data))

    def test_a_reloaded_mismatch_still_raises(self, tmp_path):
        data = _build(SUCCESS_THEN_TIMEOUT)
        path = tmp_path / "demos.npz"
        np.savez(path, **data)
        env = _env_from(data, delta_pos_scale=float(data["delta_pos_scale"]) * 2.0)
        with pytest.raises(ValueError, match="delta_pos_scale"):
            load_dataset(path, env)

    def test_a_reloaded_task_id_mismatch_still_raises(self, tmp_path):
        """The string field, through the npz round trip."""
        data = _build(SUCCESS_THEN_TIMEOUT)
        path = tmp_path / "demos.npz"
        np.savez(path, **data)
        env = _env_from(data, task_id="LOOT-Place-RandS-RandT-Dk1")
        with pytest.raises(ValueError, match="task_id"):
            load_dataset(path, env)
