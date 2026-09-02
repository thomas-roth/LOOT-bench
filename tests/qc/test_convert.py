"""Tests for the recording -> training-format converter.

The episode-structure rules live in ``test_transitions.py``. What is left here
is everything the converter adds around them: the action conversion, the row
order, and the stamp that lets a mismatched dataset fail loudly instead of
training to zero.
"""

import numpy as np
import pytest

from g3a_bench.qc.convert import build_dataset, convert
from g3a_bench.qc.dataset import (
    STAMP_KEYS,
    _env_start_pose_noise,
    check_dataset_matches_env,
    load_dataset,
)
from g3a_bench.qc.transitions import QC_KEYS

OBS_DIM = 5


def _leader_pose(value: float) -> list[float]:
    """A reachable leader-arm pose. Joints in rad, gripper normalized [0, 1]."""
    return [value, 1.7 + value, 1.7, -1.5, 0.0, 0.0, 0.0]


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
            actions.append(_leader_pose(0.1 * frame))
            rewards.append(reward)
    return {
        "episode_index": np.array(episode_index, dtype=np.int64),
        "frame_index": np.array(frame_index, dtype=np.int64),
        "observations": np.array(observations, dtype=np.float32),
        "leader_actions": np.array(actions, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
    }


SUCCESS_THEN_TIMEOUT = _columns({0: [-0.7, -0.4, 0.0], 1: [-0.9, -0.8, -0.6]})


def _build(
    columns: dict[str, np.ndarray], task: str | None = None
) -> dict[str, np.ndarray]:
    """Call the converter with named columns, not a splat.

    The splat reads better but hides the signature from the type checker,
    which then cannot tell that ``gravity_compensate`` was never passed.
    """
    return build_dataset(
        episode_index=columns["episode_index"],
        frame_index=columns["frame_index"],
        observations=columns["observations"],
        leader_actions=columns["leader_actions"],
        rewards=columns["rewards"],
        task=task,
    )


class TestActionConversion:
    def test_actions_are_inside_the_policy_box(self):
        """A bounded policy can only emit [-1, 1]. An action outside it is one
        the policy can never reproduce, however well it fits the data."""
        out = _build(SUCCESS_THEN_TIMEOUT)
        assert np.all(np.abs(out["actions"]) <= 1.0), out["actions"].max()

    def test_actions_invert_back_to_the_recorded_pose(self):
        """The converter and the simulator must agree about what an action
        means, and the stamp is what carries that agreement."""
        from g3a_bench.scripts.live_bridge.mapper import lerobot_to_mjlab_ctrl

        out = _build(SUCCESS_THEN_TIMEOUT)
        target = out["actions"][0] * out["action_scale"] + out["action_offset"]
        expected = lerobot_to_mjlab_ctrl(SUCCESS_THEN_TIMEOUT["leader_actions"][0])
        # Gravity compensation is applied on the way in, so the recovered
        # target leads the commanded pose on the arm joints. The gripper has
        # no compensation, so it round-trips exactly.
        assert target[6] == pytest.approx(expected[6], abs=1e-5)

    def test_a_pose_at_a_joint_limit_stays_in_the_box(self):
        """Gravity compensation pushes the target past the sag, which at a
        joint limit lands outside the box. The simulator clamps it, so the
        dataset has to record the clamped value."""
        from g3a_bench.scripts.live_bridge.mapper import joint_pos_limits

        lower, _ = joint_pos_limits()
        pose = lower.copy()
        pose[6] = 0.0  # The gripper arrives normalized, not in metres.
        columns = _columns({0: [-0.5, 0.0]})
        columns["leader_actions"] = np.stack([pose, pose]).astype(np.float32)
        assert np.all(np.abs(_build(columns)["actions"]) <= 1.0)

    def test_action_width_matches_the_leader(self):
        out = _build(SUCCESS_THEN_TIMEOUT)
        assert out["actions"].shape[1] == 7


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

    def test_interleaved_episodes_do_not_raise_after_sorting(self):
        """The sort is what makes transitions.py's contiguity rule satisfiable."""
        _build({k: v[::-1] for k, v in SUCCESS_THEN_TIMEOUT.items()})


class TestStamp:
    def test_publishes_every_stamp_key(self):
        out = _build(SUCCESS_THEN_TIMEOUT)
        # The task stamp is conditional: synthetic builds carry no task.
        assert set(out) >= STAMP_KEYS - {"task"}
        assert "task" not in out

    def test_task_is_stamped_when_given(self):
        task = "G3A-Stack-Low-Dk1_uni"
        out = _build(SUCCESS_THEN_TIMEOUT, task=task)
        assert str(np.asarray(out["task"]).reshape(-1)[0]) == task

    def test_stamp_matches_the_action_term(self):
        # Stamp must be read off the built term, not a second DK1-only compile.
        # For DK1 they agree, but the term is the authority.
        from g3a_bench.qc.env import G3aEnv

        task = "G3A-Stack-Low-Dk1_uni"
        env = G3aEnv(task, device="cpu")
        try:
            term = env.action_term()
            scale = term.scale
            offset = term.offset
            if hasattr(scale, "detach"):
                scale = scale.detach().cpu().numpy()  # type: ignore[union-attr]
            if hasattr(offset, "detach"):
                offset = offset.detach().cpu().numpy()  # type: ignore[union-attr]
            scale = np.asarray(scale, dtype=np.float32).reshape(-1)
            offset = np.asarray(offset, dtype=np.float32).reshape(-1)
        finally:
            env.close()
        # Legacy path (no task) still matches DK1 term
        out_legacy = _build(SUCCESS_THEN_TIMEOUT)
        np.testing.assert_array_almost_equal(out_legacy["action_scale"], scale)
        np.testing.assert_array_almost_equal(out_legacy["action_offset"], offset)
        # Robot-agnostic path (explicit task) must also match
        out_task = _build(SUCCESS_THEN_TIMEOUT, task=task)
        np.testing.assert_array_almost_equal(out_task["action_scale"], scale)
        np.testing.assert_array_almost_equal(out_task["action_offset"], offset)

    def test_stamp_is_robot_agnostic(self):
        # Different tasks can have different action dims/scales; the helper
        # must resolve per-task, not via the DK1-only mapper.
        from g3a_bench.qc.convert import _action_scale_offset_from_task

        task_dk1 = "G3A-Stack-Low-Dk1_uni"
        scale, offset = _action_scale_offset_from_task(task_dk1, device="cpu")
        assert scale.shape[0] == 7
        # The task-resolved stamp must equal the term's stamp
        from g3a_bench.qc.env import G3aEnv

        env = G3aEnv(task_dk1, device="cpu")
        try:
            term = env.action_term()
            env_scale = np.asarray(
                term.scale.detach().cpu().numpy()
                if hasattr(term.scale, "detach")
                else term.scale,
                dtype=np.float32,
            ).reshape(-1)
            env_offset = np.asarray(
                term.offset.detach().cpu().numpy()
                if hasattr(term.offset, "detach")
                else term.offset,
                dtype=np.float32,
            ).reshape(-1)
        finally:
            env.close()
        np.testing.assert_array_almost_equal(scale, env_scale)
        np.testing.assert_array_almost_equal(offset, env_offset)

    def test_still_publishes_the_training_keys(self):
        out = _build(SUCCESS_THEN_TIMEOUT)
        assert set(out) >= QC_KEYS


class _FakeTerm:
    def __init__(self, scale, offset):
        self.scale = np.asarray(scale, dtype=np.float32)[None, :]
        self.offset = np.asarray(offset, dtype=np.float32)[None, :]


class _FakeEnv:
    """Just enough of the adapter to answer the stamp check."""

    def __init__(
        self,
        scale,
        offset,
        obs_dim,
        action_dim,
        start_pose_noise=0.0,
        task_id=None,
    ):
        self._term = _FakeTerm(scale, offset)
        self.observation_space = type("S", (), {"shape": (obs_dim,)})()
        self.action_space = type("S", (), {"shape": (action_dim,)})()
        self._start_pose_noise = float(start_pose_noise)
        self.task_id = task_id

    def action_term(self):
        return self._term

    def start_pose_noise(self):
        return self._start_pose_noise


class TestStampCheck:
    def _dataset(self):
        return _build(SUCCESS_THEN_TIMEOUT)

    def _env_from(self, data, **overrides):
        kwargs = {
            "scale": data["action_scale"],
            "offset": data["action_offset"],
            "obs_dim": data["observations"].shape[1],
            "action_dim": data["actions"].shape[1],
            "start_pose_noise": float(
                np.asarray(data["start_pose_noise"]).reshape(-1)[0]
            ),
        }
        if "task" in data:
            kwargs["task_id"] = str(np.asarray(data["task"]).reshape(-1)[0])
        kwargs.update(overrides)
        return _FakeEnv(**kwargs)

    def test_matching_dataset_passes(self):
        data = self._dataset()
        check_dataset_matches_env(data, self._env_from(data))

    def test_a_different_action_scale_is_rejected(self):
        """The failure this exists for: a dataset recorded before the action
        convention changed trains without complaint and scores zero."""
        data = self._dataset()
        env = self._env_from(data, scale=data["action_scale"] * 2.0)
        with pytest.raises(ValueError, match="action scale"):
            check_dataset_matches_env(data, env)

    def test_a_different_action_offset_is_rejected(self):
        data = self._dataset()
        env = self._env_from(data, offset=data["action_offset"] + 1.0)
        with pytest.raises(ValueError, match="action offset"):
            check_dataset_matches_env(data, env)

    def test_a_different_observation_width_is_rejected(self):
        data = self._dataset()
        env = self._env_from(data, obs_dim=OBS_DIM + 1)
        with pytest.raises(ValueError, match="observation"):
            check_dataset_matches_env(data, env)

    def test_a_different_action_width_is_rejected(self):
        data = self._dataset()
        env = self._env_from(data, action_dim=3)
        with pytest.raises(ValueError, match="action width"):
            check_dataset_matches_env(data, env)

    def test_a_different_start_pose_noise_is_rejected(self):
        data = self._dataset()
        env = self._env_from(
            data, start_pose_noise=float(data["start_pose_noise"].reshape(-1)[0]) + 0.05
        )
        with pytest.raises(ValueError, match="start_pose_noise"):
            check_dataset_matches_env(data, env)

    def test_a_missing_start_pose_noise_stamp_is_rejected(self):
        data = self._dataset()
        data.pop("start_pose_noise")
        env = self._env_from(_build(SUCCESS_THEN_TIMEOUT))
        with pytest.raises(ValueError, match="no start_pose_noise"):
            check_dataset_matches_env(data, env)

    def test_a_different_task_is_rejected(self):
        """The failure the width checks cannot see: same robot, same obs and
        action widths, different variant (e.g. Place FixedS-FixedT vs
        RandS-RandT) trains silently on the wrong spawn/target distribution."""
        data = self._dataset()
        data["task"] = np.array("G3A-Place-FixedS-FixedT-Dk1_uni", dtype=np.str_)
        env = self._env_from(data, task_id="G3A-Place-RandS-RandT-Dk1_uni")
        with pytest.raises(ValueError, match="dataset task"):
            check_dataset_matches_env(data, env)

    def test_the_matching_task_passes(self):
        data = self._dataset()
        task = "G3A-Place-RandS-RandT-Dk1_uni"
        data["task"] = np.array(task, dtype=np.str_)
        env = self._env_from(data, task_id=task)
        check_dataset_matches_env(data, env)

    def test_a_missing_task_stamp_warns_but_passes(self):
        """Minimal breakage: a .npz predating the stamp still loads, so an
        existing dataset is not refused by this change."""
        data = self._dataset()
        env = self._env_from(data, task_id="G3A-Place-RandS-RandT-Dk1_uni")
        with pytest.warns(UserWarning, match="no task stamp"):
            check_dataset_matches_env(data, env)


class TestEnvStartPoseNoise:
    """Each source ``_env_start_pose_noise`` reads, in priority order.

    A silent miss is how a wrong env would compare its noise to the stamp,
    so a broken source must say which source it tried and what is missing.
    """

    @staticmethod
    def _env(**attrs):
        return type("NoiseEnv", (), attrs)()

    @staticmethod
    def _cfg_env(events):
        cfg = type("Cfg", (), {"events": events})()
        return type("Unwrapped", (), {"cfg": cfg})()

    def test_the_seam_method_is_the_first_source(self):
        def noise(self):
            return 0.05

        env = self._env(start_pose_noise=noise)
        assert _env_start_pose_noise(env) == pytest.approx(0.05)

    def test_a_seam_attribute_works_too(self):
        env = self._env(start_pose_noise=0.05)
        assert _env_start_pose_noise(env) == pytest.approx(0.05)

    def test_the_private_attribute_is_the_second_source(self):
        env = self._env(_start_pose_noise=0.05)
        assert _env_start_pose_noise(env) == pytest.approx(0.05)

    def test_the_cfg_is_the_third_source(self):
        events = {
            "reset_robot_joints": type(
                "Term", (), {"params": {"position_range": (-0.05, 0.05)}}
            )()
        }
        env = self._env(unwrapped_env=self._cfg_env(events))
        assert _env_start_pose_noise(env) == pytest.approx(0.05)

    def test_a_higher_source_beats_the_cfg(self):
        events = {
            "reset_robot_joints": type(
                "Term", (), {"params": {"position_range": (-0.05, 0.05)}}
            )()
        }
        env = self._env(start_pose_noise=0.1, unwrapped_env=self._cfg_env(events))
        assert _env_start_pose_noise(env) == pytest.approx(0.1)

    def test_a_cfg_without_the_event_names_the_missing_term(self):
        env = self._env(unwrapped_env=self._cfg_env({}))
        with pytest.raises(ValueError, match="reset_robot_joints"):
            _env_start_pose_noise(env)

    def test_a_cfg_without_the_param_names_the_missing_param(self):
        events = {"reset_robot_joints": type("Term", (), {"params": {}})()}
        env = self._env(unwrapped_env=self._cfg_env(events))
        with pytest.raises(ValueError, match="position_range"):
            _env_start_pose_noise(env)

    def test_a_broken_seam_value_names_the_attribute(self):
        env = self._env(start_pose_noise="not-a-number")
        with pytest.raises(ValueError, match="env.start_pose_noise"):
            _env_start_pose_noise(env)

    def test_no_source_at_all_says_none_were_found(self):
        with pytest.raises(ValueError, match="cannot read start_pose_noise"):
            _env_start_pose_noise(self._env())


class TestSessionOffsetting:
    """Two sessions are each numbered from episode zero."""

    def test_colliding_indices_join_two_episodes_silently(self):
        """Why load_many offsets. Two failed sessions, each numbered from zero,
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
    def _write(root, episodes, task: str | list[str] = "G3A-Stack-Low-Dk1_uni"):
        """Write one session. ``task`` may be a per-episode sequence, which is
        how a root with two task ids is made for the mixed-task tests."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (OBS_DIM,),
                "names": None,
            },
            "action": {"dtype": "float32", "shape": (7,), "names": None},
            "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
            "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        }
        dataset = LeRobotDataset.create(
            "probe", fps=50, features=features, root=str(root)
        )
        for episode, rewards in enumerate(episodes):
            episode_task = task[episode] if isinstance(task, (list, tuple)) else task
            for frame, reward in enumerate(rewards):
                dataset.add_frame(
                    {
                        "observation.state": np.full(
                            OBS_DIM, float(frame), dtype=np.float32
                        ),
                        "action": np.array(_leader_pose(0.1 * frame), np.float32),
                        "next.reward": np.array([reward], dtype=np.float32),
                        "next.done": np.array([frame == len(rewards) - 1]),
                        "task": episode_task,
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
        # The task stamp carries the id recorded per frame, so a dataset
        # from one task cannot pass the check against another.
        assert str(np.asarray(loaded["task"]).reshape(-1)[0]) == (
            "G3A-Stack-Low-Dk1_uni"
        )
        # Six recorded rows, minus the timed-out episode's unusable last row.
        assert len(loaded["observations"]) == 4
        np.testing.assert_array_equal(loaded["terminals"], [0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(loaded["masks"], [1.0, 1.0, 0.0, 1.0])

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

    def test_mixed_tasks_across_roots_are_rejected(self, tmp_path):
        """Two sessions of two tasks in one call would stamp one of them and
        train silently on the other."""
        first = self._write(
            tmp_path / "a",
            [[-0.5, 0.0]],
            task="G3A-Place-FixedS-FixedT-Dk1_uni",
        )
        second = self._write(tmp_path / "b", [[-0.5, 0.0]])
        with pytest.raises(ValueError, match="mixed tasks across roots"):
            convert([first, second], tmp_path / "demos.npz")

    def test_mixed_tasks_inside_one_root_are_rejected(self, tmp_path):
        root = self._write(
            tmp_path / "a",
            [[-0.5, 0.0], [-0.5, 0.0]],
            task=["G3A-Place-FixedS-FixedT-Dk1_uni", "G3A-Stack-Low-Dk1_uni"],
        )
        with pytest.raises(ValueError, match="mixed tasks inside"):
            convert([root], tmp_path / "demos.npz")

    def test_a_dataset_without_a_task_uses_the_legacy_stamp(self, tmp_path):
        """An empty task id counts as none: the legacy DK1 stamp is used and
        the output carries no task for the check to match against."""
        root = self._write(tmp_path / "session", [[-0.5, 0.0]], task="")
        data = convert([root], tmp_path / "demos.npz")
        assert "task" not in data
        assert set(data) >= STAMP_KEYS - {"task"}


class TestLoadDataset:
    def test_round_trips_through_the_npz(self, tmp_path):
        data = _build(SUCCESS_THEN_TIMEOUT)
        path = tmp_path / "demos.npz"
        np.savez(path, **data)  # type: ignore[arg-type]
        loaded = load_dataset(path)
        # ``task`` is absent on synthetic builds, so only check the keys
        # that exist.
        for key in QC_KEYS | STAMP_KEYS:
            if key in data:
                np.testing.assert_array_equal(loaded[key], data[key])

    def test_checks_against_an_env_when_given_one(self, tmp_path):
        data = _build(SUCCESS_THEN_TIMEOUT)
        path = tmp_path / "demos.npz"
        np.savez(path, **data)  # type: ignore[arg-type]
        env = _FakeEnv(
            scale=data["action_scale"] * 2.0,
            offset=data["action_offset"],
            obs_dim=data["observations"].shape[1],
            action_dim=data["actions"].shape[1],
            start_pose_noise=float(np.asarray(data["start_pose_noise"]).reshape(-1)[0]),
        )
        with pytest.raises(ValueError, match="action scale"):
            load_dataset(path, env)

    def test_reads_a_torch_action_term(self):
        """The action term's buffers are torch tensors on the sim device, and
        numpy refuses a CUDA tensor outright -- so a check that only ever saw
        numpy would raise TypeError on every GPU run."""
        import torch

        data = _build(SUCCESS_THEN_TIMEOUT)

        class _TorchTerm:
            scale = torch.as_tensor(data["action_scale"])[None, :]
            offset = torch.as_tensor(data["action_offset"])[None, :]

        class _TorchEnv:
            observation_space = type(
                "S", (), {"shape": (data["observations"].shape[1],)}
            )()
            action_space = type("S", (), {"shape": (data["actions"].shape[1],)})()
            _start_pose_noise = float(
                np.asarray(data["start_pose_noise"]).reshape(-1)[0]
            )

            def action_term(self):
                return _TorchTerm()

            def start_pose_noise(self):
                return self._start_pose_noise

        check_dataset_matches_env(data, _TorchEnv())
