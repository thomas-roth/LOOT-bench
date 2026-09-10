"""Tests for the demo splitter (offline relabel to easier same-family tasks).

Everything runs on synthetic episodes: no recorded demo of a splittable
category is committed, and a fabricated one pins the contract more sharply
anyway. Each category's fixture builds an observation in the real layout --
``[joint_pos 8 | joint_vel 8 | ee_pose 9 | pad_contact_force 1 |
object-major sections]``, nothing
after (ADR-016) -- and a reward column obeying the env's one-step lag, so the
source self-check runs on every relabel.
"""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from loot_bench.scripts.live_bridge.dataset_extra_info import load_extra_info
from loot_bench.scripts.split_demos import (
    SplitPlan,
    adapter_for,
    easier_target_task_ids,
    frame_state,
    lagged_rewards,
    parse_task_id,
    proprio_width,
    relabel_episode,
    split_to_task,
    variant_from_task_id,
)
from loot_bench.tasks.common.rewards import shaped_task_reward

PROPRIO = 26
"""joint_pos 8 + joint_vel 8 + ee_pose 9 + pad_contact_force 1, the DK1's
proprioceptive head."""

ROBOT_QPOS_OPEN = np.array([0, 1.7, 1.7, -1.5708, 0, 0, 0.04, 0.04], np.float32)
ROBOT_QPOS_CLOSED = np.array([0, 1.7, 1.7, -1.5708, 0, 0, 0.0, 0.0], np.float32)

FAR_AWAY = np.array([9.0, 9.0, 9.0], np.float32)
"""Grasp-site position with no object near it, so nothing counts as held."""


def _head(grasp_pos: np.ndarray, pad_force: float = 0.0) -> np.ndarray:
    """Proprioceptive head with ``ee_pose`` then ``pad_contact_force`` last."""
    ee_pose = np.concatenate([grasp_pos, [1, 0, 0, 0, 1, 0]]).astype(np.float32)
    return np.concatenate(
        [np.zeros(PROPRIO - 10, np.float32), ee_pose, [pad_force]]
    ).astype(np.float32)


def _source_rewards(progress: torch.Tensor, gate: torch.Tensor) -> np.ndarray:
    """The reward column the env would have recorded for this episode.

    Frame ``t`` is paid what the state at ``t - 1`` earned (ADR-006); frame 0
    is not checked, so it repeats frame 1's value.
    """
    paid = shaped_task_reward(progress[:-1], gate[:-1])
    return torch.cat([paid[:1], paid]).numpy().astype(np.float32)


def _episode_frame(rel, task: str) -> pd.DataFrame:
    """One relabeled episode as the DataFrame ``split_to_task`` consumes."""
    return pd.DataFrame(
        {
            "episode_index": np.zeros(len(rel["obs"]), int),
            "frame_index": np.arange(len(rel["obs"])),
            "observation.state": list(rel["obs"]),
            "observation.robot_state": list(rel["robot_state"]),
            "observation.qpos": list(rel["qpos"]),
            "observation.qvel": list(rel["qvel"]),
            "action": list(rel["action"]),
            "next.reward": list(rel["reward"]),
            "task": task,
        }
    )


# --- task-id resolution ---------------------------------------------------- #


class TestTaskIds:
    def test_stack_family_is_one_difficulty(self):
        assert parse_task_id("LOOT-Stack5-Easy-Dk1") == (
            "Stack5",
            "Easy",
            "",
            "Dk1",
        )
        # The target height rides in the category segment and the block shape
        # is the task, so a split family is one difficulty.
        assert easier_target_task_ids("LOOT-Stack5-Easy-Dk1") == [
            "LOOT-Stack4-Easy-Dk1",
            "LOOT-Stack3-Easy-Dk1",
        ]
        assert easier_target_task_ids("LOOT-Stack5-Hard-Dk1") == [
            "LOOT-Stack4-Hard-Dk1",
            "LOOT-Stack3-Hard-Dk1",
        ]
        assert easier_target_task_ids("LOOT-Stack3-Easy-Dk1") == []

    def test_domino_variations_are_separate_families(self):
        assert parse_task_id("LOOT-Domino-4Pieces-Dk1") == (
            "Domino",
            "4Pieces",
            "",
            "Dk1",
        )
        assert easier_target_task_ids("LOOT-Domino-4Pieces-Dk1") == [
            "LOOT-Domino-3Pieces-Dk1",
            "LOOT-Domino-2Pieces-Dk1",
        ]
        assert easier_target_task_ids("LOOT-Domino-4Pieces-DifferentDominoes-Dk1") == [
            "LOOT-Domino-3Pieces-DifferentDominoes-Dk1",
            "LOOT-Domino-2Pieces-DifferentDominoes-Dk1",
        ]
        # The base row is fixed and RandomRow's is sampled, so neither is a
        # prefix of the other.
        assert "LOOT-Domino-3Pieces-RandomRow-Dk1" not in easier_target_task_ids(
            "LOOT-Domino-4Pieces-Dk1"
        )
        # KnockOver topples as one chain over all pieces: no prefix to cut.
        assert easier_target_task_ids("LOOT-Domino-4Pieces-KnockOver-Dk1") == []

    def test_sorter_splits_within_a_spawn_variation(self):
        assert parse_task_id("LOOT-Sorter-5Shapes-Full-Dk1") == (
            "Sorter",
            "5Shapes",
            "Full",
            "Dk1",
        )
        for variation in ("FixedOrientation", "NoRotation", "Full"):
            assert easier_target_task_ids(f"LOOT-Sorter-5Shapes-{variation}-Dk1") == [
                f"LOOT-Sorter-4Shapes-{variation}-Dk1",
                f"LOOT-Sorter-3Shapes-{variation}-Dk1",
            ]

    def test_unsplittable_category_and_malformed_id(self):
        with pytest.raises(ValueError, match="not splittable"):
            adapter_for("LOOT-Place-FixedS-FixedT-Dk1")
        with pytest.raises(ValueError, match="not a LOOT task id"):
            parse_task_id("Stack5-Easy")

    def test_obs_section_widths(self):
        assert adapter_for("LOOT-Stack3-Easy-Dk1").obs_section_widths == (7,)
        assert adapter_for("LOOT-Domino-3Pieces-Dk1").obs_section_widths == (
            7,
            1,
            7,
        )
        assert adapter_for("LOOT-Sorter-5Shapes-Full-Dk1").obs_section_widths == (
            7,
            1,
        )


class TestPlanValidation:
    def test_rejects_harder_target_and_other_family(self):
        adapter = adapter_for("LOOT-Stack5-Easy-Dk1")
        stack5 = variant_from_task_id("LOOT-Stack5-Easy-Dk1")
        stack3 = variant_from_task_id("LOOT-Stack3-Easy-Dk1")
        stack3_hard = variant_from_task_id("LOOT-Stack3-Hard-Dk1")
        SplitPlan(adapter, stack5, stack3).validate()
        with pytest.raises(AssertionError, match="not easier"):
            SplitPlan(adapter, stack3, stack5).validate()
        with pytest.raises(AssertionError, match="same family"):
            SplitPlan(adapter, stack5, stack3_hard).validate()

    def test_rejects_knock_over_and_crossing_row_families(self):
        adapter = adapter_for("LOOT-Domino-4Pieces-Dk1")
        four = variant_from_task_id("LOOT-Domino-4Pieces-Dk1")
        three = variant_from_task_id("LOOT-Domino-3Pieces-Dk1")
        knock = variant_from_task_id("LOOT-Domino-4Pieces-KnockOver-Dk1")
        three_random = variant_from_task_id("LOOT-Domino-3Pieces-RandomRow-Dk1")
        SplitPlan(adapter, four, three).validate()
        with pytest.raises(AssertionError):
            SplitPlan(adapter, knock, three).validate()
        with pytest.raises(AssertionError, match="same family"):
            SplitPlan(adapter, four, three_random).validate()


# --- Stack ----------------------------------------------------------------- #


def _stack_episode(
    heights_per_frame,
    grasp_pos_per_frame=None,
    gripper_open=True,
    pad_force: float | list[float] | None = None,
):
    """Synthetic Stack5-Easy episode: 10 cubes, only their z varying.

    ``heights_per_frame`` gives each frame's block-centre z, one list per
    frame. Blocks rest at their spawn xy; only height decides Stack success.
    ``pad_force`` is the recorded finger-pad force, a scalar or one value per
    frame; None derives it from ``gripper_open`` (open reads force-free).
    """
    n = 10
    frames = len(heights_per_frame)
    if pad_force is None:
        forces = [0.0 if gripper_open else 5.0] * frames
    elif isinstance(pad_force, list):
        forces = [float(f) for f in pad_force]
    else:
        forces = [pad_force] * frames
    xy = np.array([[0.25 + 0.02 * i, -0.1 + 0.02 * i] for i in range(n)], np.float32)
    robot = ROBOT_QPOS_OPEN if gripper_open else ROBOT_QPOS_CLOSED
    grasp = grasp_pos_per_frame or [FAR_AWAY] * frames
    obs, qpos, qvel = [], [], []
    for heights, grasp_pos, force in zip(heights_per_frame, grasp, forces, strict=True):
        poses = np.concatenate(
            [xy, np.asarray(heights, np.float32).reshape(-1, 1)], axis=1
        )
        poses = np.concatenate(
            [poses, np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32)], axis=1
        )
        obs.append(
            np.concatenate(
                [
                    _head(np.asarray(grasp_pos, np.float32), float(force)),
                    poses.reshape(-1),
                ]
            )
        )
        qpos.append(np.concatenate([poses.reshape(-1), robot]))
        qvel.append(np.zeros(n * 6 + 8, np.float32))
    return (
        np.array(obs, np.float32),
        np.array(qpos, np.float32),
        np.array(qvel, np.float32),
    )


def _stack_state(obs, qvel, take):
    adapter = adapter_for("LOOT-Stack5-Easy-Dk1")
    return frame_state(
        torch.from_numpy(obs.astype(np.float64)),
        torch.from_numpy(qvel.astype(np.float64)),
        proprio=PROPRIO,
        adapter=adapter,
        n_layout=10,
        take=take,
    )


class TestStackRecompute:
    def test_progress_is_the_tallest_block_fraction(self):
        # Easy blocks are 0.04 m cubes; a centre at 0.10 tops out at 0.12,
        # which is Stack3's target height.
        rest, tower = 0.02, 0.10
        obs, _qpos, qvel = _stack_episode([[rest] * 10, [tower] + [rest] * 9])
        adapter = adapter_for("LOOT-Stack3-Easy-Dk1")
        stack3 = variant_from_task_id("LOOT-Stack3-Easy-Dk1")
        progress, gate = adapter.progress_and_gate(
            stack3, _stack_state(obs, qvel, take=6)
        )
        assert progress.tolist() == pytest.approx([1 / 3, 1.0])
        assert gate.tolist() == [False, True]

    def test_a_held_block_earns_no_progress(self):
        rest, tower = 0.02, 0.10
        # The same raised block in both frames; only the grasp site moves onto
        # it, so the drop in progress is the exclusion and nothing else.
        frames = [[tower] + [rest] * 9] * 2
        grasps = [FAR_AWAY, np.array([0.25, -0.1, tower], np.float32)]
        obs, _qpos, qvel = _stack_episode(frames, grasps, gripper_open=False)
        adapter = adapter_for("LOOT-Stack3-Easy-Dk1")
        stack3 = variant_from_task_id("LOOT-Stack3-Easy-Dk1")
        progress, _gate = adapter.progress_and_gate(
            stack3, _stack_state(obs, qvel, take=6)
        )
        # Out of reach it counts; in the closed gripper it falls back to the
        # tallest resting block.
        assert progress.tolist() == pytest.approx([1.0, 1 / 3])

    def test_relabel_cuts_one_frame_after_the_target_success(self):
        rest, tower = 0.02, 0.10
        # Stack3's target is met from frame 1; Stack5's is never met here.
        obs, qpos, qvel = _stack_episode(
            [[rest] * 10, [tower] + [rest] * 9, [tower] + [rest] * 9]
        )
        plan = SplitPlan(
            adapter_for("LOOT-Stack5-Easy-Dk1"),
            variant_from_task_id("LOOT-Stack5-Easy-Dk1"),
            variant_from_task_id("LOOT-Stack3-Easy-Dk1"),
        )
        source_progress, source_gate = plan.adapter.progress_and_gate(
            plan.source, _stack_state(obs, qvel, take=10)
        )
        rel = relabel_episode(
            obs=obs,
            qpos=qpos,
            qvel=qvel,
            action=np.zeros((3, 7), np.float32),
            robot_state=np.zeros((3, 7), np.float32),
            reward=_source_rewards(source_progress, source_gate),
            plan=plan,
        )
        assert rel is not None
        assert rel.cut_frame == 2  # success at frame 1, paid at frame 2
        assert len(rel.obs) == 3
        # The last frame is absorbing: reward exactly 0 and done set there only.
        assert rel.reward[-1, 0] == 0.0
        assert rel.done[-1, 0] and not rel.done[:-1, 0].any()
        assert (rel.reward[:-1] < 0.0).all()
        # Truncation keeps the head and the first-6 blocks of every slice.
        assert rel.obs.shape[1] == PROPRIO + 6 * 7
        assert rel.qpos.shape[1] == 6 * 7 + 8
        assert rel.qvel.shape[1] == 6 * 6 + 8
        assert np.array_equal(rel.obs[:, :PROPRIO], obs[:3, :PROPRIO])
        assert np.array_equal(rel.qpos[:, 6 * 7 :], qpos[:3, 10 * 7 :])

    def test_pads_loaded_to_the_end_credit_no_success(self):
        """Pads loaded to the end: no success, even with the qpos wide open.

        The old qpos proxy read an open gripper as released and would have cut
        a success here; the contact gate holds the whole episode shut.
        """
        rest, tower = 0.02, 0.10
        obs, qpos, qvel = _stack_episode(
            [[rest] * 10, [tower] + [rest] * 9, [tower] + [rest] * 9],
            gripper_open=True,  # qpos open throughout: the proxy's blind spot
            pad_force=5.0,  # ... while the pads stay loaded
        )
        plan = SplitPlan(
            adapter_for("LOOT-Stack5-Easy-Dk1"),
            variant_from_task_id("LOOT-Stack5-Easy-Dk1"),
            variant_from_task_id("LOOT-Stack3-Easy-Dk1"),
        )
        progress, gate = plan.adapter.progress_and_gate(
            plan.source, _stack_state(obs, qvel, take=10)
        )
        # The tower stands, but release never holds.
        assert progress.tolist() == pytest.approx([0.2, 0.6, 0.6])
        assert gate.tolist() == [False, False, False]
        assert (
            relabel_episode(
                obs=obs,
                qpos=qpos,
                qvel=qvel,
                action=np.zeros((3, 7), np.float32),
                robot_state=np.zeros((3, 7), np.float32),
                reward=_source_rewards(progress, gate),
                plan=plan,
            )
            is None
        )

    def test_no_target_success_returns_none(self):
        obs, qpos, qvel = _stack_episode([[0.02] * 10] * 3)
        plan = SplitPlan(
            adapter_for("LOOT-Stack5-Easy-Dk1"),
            variant_from_task_id("LOOT-Stack5-Easy-Dk1"),
            variant_from_task_id("LOOT-Stack3-Easy-Dk1"),
        )
        progress, gate = plan.adapter.progress_and_gate(
            plan.source, _stack_state(obs, qvel, take=10)
        )
        assert (
            relabel_episode(
                obs=obs,
                qpos=qpos,
                qvel=qvel,
                action=np.zeros((3, 7), np.float32),
                robot_state=np.zeros((3, 7), np.float32),
                reward=_source_rewards(progress, gate),
                plan=plan,
            )
            is None
        )

    def test_self_check_rejects_a_mismatched_reward_column(self):
        obs, qpos, qvel = _stack_episode([[0.02] * 10, [0.10] + [0.02] * 9])
        plan = SplitPlan(
            adapter_for("LOOT-Stack5-Easy-Dk1"),
            variant_from_task_id("LOOT-Stack5-Easy-Dk1"),
            variant_from_task_id("LOOT-Stack3-Easy-Dk1"),
        )
        with pytest.raises(AssertionError, match="self-check failed"):
            relabel_episode(
                obs=obs,
                qpos=qpos,
                qvel=qvel,
                action=np.zeros((2, 7), np.float32),
                robot_state=np.zeros((2, 7), np.float32),
                reward=np.zeros(2, np.float32),
                plan=plan,
            )


# --- Domino ---------------------------------------------------------------- #


def _domino_episode(displaced_per_frame, row_yaw=math.pi / 2):
    """Synthetic Domino-4 episode.

    ``displaced_per_frame`` names the pieces sitting off their slot on each
    frame; the rest are exactly on theirs. ``row_yaw`` is the row direction:
    pi/2 is the base family's fixed row, RandomRow's is arbitrary.
    """
    n = 4
    quat = np.array(
        [math.cos(row_yaw / 2), 0, 0, math.sin(row_yaw / 2)], np.float32
    )  # quat_from_euler_xyz(0, 0, row_yaw)
    along = np.array([math.cos(row_yaw), math.sin(row_yaw)])
    slots = np.array(
        [
            [*(np.array([0.3, -0.075]) + i * 0.05 * along), 0.035, *quat]
            for i in range(n)
        ],
        np.float32,
    )
    obs, qpos, qvel = [], [], []
    for displaced in displaced_per_frame:
        poses = slots.copy()
        for piece in displaced:
            poses[piece, 0] += 0.1  # well past the position threshold
        obs.append(
            np.concatenate(
                [
                    _head(FAR_AWAY),
                    poses.reshape(-1),
                    np.zeros(n, np.float32),  # slot_placed
                    slots.reshape(-1),  # command
                ]
            )
        )
        qpos.append(np.concatenate([poses.reshape(-1), ROBOT_QPOS_OPEN]))
        qvel.append(np.zeros(n * 6 + 8, np.float32))
    return (
        np.array(obs, np.float32),
        np.array(qpos, np.float32),
        np.array(qvel, np.float32),
    )


# RandomRow rows are tilted and often negative; the fixed row is exactly pi/2.
# The recompute reads the angle back off the slot quaternion, so it has to hold
# across the whole sampled range.
@pytest.mark.parametrize("row_yaw", [math.pi / 2, -math.pi / 2, -1.3388, 2.1, 0.0])
def test_domino_recompute_and_truncation(row_yaw):
    # Piece 2 lands on frame 1 (completing the first three), piece 3 on frame 2.
    obs, qpos, qvel = _domino_episode([(2, 3), (3,), (), ()], row_yaw=row_yaw)
    adapter = adapter_for("LOOT-Domino-4Pieces-Dk1")
    four = variant_from_task_id("LOOT-Domino-4Pieces-Dk1")
    three = variant_from_task_id("LOOT-Domino-3Pieces-Dk1")
    state = frame_state(
        torch.from_numpy(obs.astype(np.float64)),
        torch.from_numpy(qvel.astype(np.float64)),
        proprio=PROPRIO,
        adapter=adapter,
        n_layout=4,
        take=4,
    )
    progress, gate = adapter.progress_and_gate(four, state)
    assert progress.tolist() == pytest.approx([0.5, 0.75, 1.0, 1.0])
    assert gate.tolist() == [False, False, True, True]

    rel = relabel_episode(
        obs=obs,
        qpos=qpos,
        qvel=qvel,
        action=np.zeros((4, 7), np.float32),
        robot_state=np.zeros((4, 7), np.float32),
        reward=_source_rewards(progress, gate),
        plan=SplitPlan(adapter, four, three),
    )
    assert rel is not None
    # The first 3 are complete at frame 1, so their success is paid at frame 2.
    assert rel.cut_frame == 2
    assert rel.obs.shape[1] == PROPRIO + 3 * 7 + 3 + 3 * 7
    assert rel.qpos.shape[1] == 3 * 7 + 8
    assert rel.qvel.shape[1] == 3 * 6 + 8


def test_domino_dropped_piece_movement_is_reported():
    obs, qpos, qvel = _domino_episode([(), (), ()])
    # Move the dropped 4th piece well past the disturbance tolerance.
    obs[1:, PROPRIO + 3 * 7] += 0.1
    adapter = adapter_for("LOOT-Domino-4Pieces-Dk1")
    four = variant_from_task_id("LOOT-Domino-4Pieces-Dk1")
    three = variant_from_task_id("LOOT-Domino-3Pieces-Dk1")
    state = frame_state(
        torch.from_numpy(obs.astype(np.float64)),
        torch.from_numpy(qvel.astype(np.float64)),
        proprio=PROPRIO,
        adapter=adapter,
        n_layout=4,
        take=4,
    )
    progress, gate = adapter.progress_and_gate(four, state)
    rel = relabel_episode(
        obs=obs,
        qpos=qpos,
        qvel=qvel,
        action=np.zeros((3, 7), np.float32),
        robot_state=np.zeros((3, 7), np.float32),
        reward=_source_rewards(progress, gate),
        plan=SplitPlan(adapter, four, three),
    )
    assert rel is not None and rel.n_dropped_disturbed == 1


# --- Sorter ---------------------------------------------------------------- #


def _sorter_episode(latched_per_frame):
    """Synthetic Sorter-5 episode: the fixed cube body, then 5 pieces.

    The cube's pose is a fixed-width obs term, so it rides in the head rather
    than in a truncatable section, but it *is* a leading free body in qpos.
    """
    n = 5
    cube = np.array([0.3, 0, 0.1, 1, 0, 0, 0], np.float32)
    pieces = np.array(
        [[0.25 + 0.03 * i, 0.1, 0.02, 1, 0, 0, 0] for i in range(n)], np.float32
    )
    obs, qpos, qvel = [], [], []
    for latched in latched_per_frame:
        obs.append(
            np.concatenate(
                [
                    _head(FAR_AWAY),
                    cube,
                    pieces.reshape(-1),
                    np.array(latched, np.float32),
                ]
            )
        )
        qpos.append(np.concatenate([cube, pieces.reshape(-1), ROBOT_QPOS_OPEN]))
        qvel.append(np.zeros(6 + n * 6 + 8, np.float32))
    return (
        np.array(obs, np.float32),
        np.array(qpos, np.float32),
        np.array(qvel, np.float32),
    )


def test_sorter_recompute_and_truncation():
    adapter = adapter_for("LOOT-Sorter-5Shapes-Full-Dk1")
    five = variant_from_task_id("LOOT-Sorter-5Shapes-Full-Dk1")
    three = variant_from_task_id("LOOT-Sorter-3Shapes-Full-Dk1")
    obs, qpos, qvel = _sorter_episode(
        [[0, 0, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    )
    # The cube's fixed-width pose lands in the head, not in a section.
    assert proprio_width(obs.shape[1], adapter, 5) == PROPRIO + 7

    def state(take):
        return frame_state(
            torch.from_numpy(obs.astype(np.float64)),
            torch.from_numpy(qvel.astype(np.float64)),
            proprio=PROPRIO + 7,
            adapter=adapter,
            n_layout=5,
            take=take,
        )

    progress, gate = adapter.progress_and_gate(five, state(5))
    assert progress.tolist() == pytest.approx([0.0, 0.6, 1.0, 1.0])
    assert gate.tolist() == [False, False, True, True]
    progress3, gate3 = adapter.progress_and_gate(three, state(3))
    assert progress3.tolist() == pytest.approx([0.0, 1.0, 1.0, 1.0])
    assert gate3.tolist() == [False, True, True, True]

    rel = relabel_episode(
        obs=obs,
        qpos=qpos,
        qvel=qvel,
        action=np.zeros((4, 7), np.float32),
        robot_state=np.zeros((4, 7), np.float32),
        reward=_source_rewards(progress, gate),
        plan=SplitPlan(adapter, five, three),
    )
    assert rel is not None
    assert rel.cut_frame == 2  # first-3 latched at frame 1, paid at frame 2
    assert rel.obs.shape[1] == PROPRIO + 7 + 3 * 7 + 3
    # The cube stays: qpos keeps [cube | first-3 pieces | robot].
    assert rel.qpos.shape[1] == 7 + 3 * 7 + 8
    assert rel.qvel.shape[1] == 6 + 3 * 6 + 8


# --- reward lag ------------------------------------------------------------ #


def test_lagged_rewards_pay_success_one_frame_late():
    progress = torch.tensor([0.5, 1.0, 1.0])
    gate = torch.tensor([False, True, True])
    rewards = lagged_rewards(progress, gate, last_frame=2)
    assert len(rewards) == 3
    assert rewards[0].item() == rewards[1].item() < 0.0  # frame 0 stands in
    assert rewards[2].item() == 0.0  # the success at frame 1, paid at frame 2


# --- write round-trip ------------------------------------------------------ #


def test_write_roundtrip(tmp_path):
    obs, qpos, qvel = _stack_episode(
        [[0.02] * 10, [0.10] + [0.02] * 9, [0.10] + [0.02] * 9]
    )
    adapter = adapter_for("LOOT-Stack5-Easy-Dk1")
    plan = SplitPlan(
        adapter,
        variant_from_task_id("LOOT-Stack5-Easy-Dk1"),
        variant_from_task_id("LOOT-Stack3-Easy-Dk1"),
    )
    progress, gate = adapter.progress_and_gate(
        plan.source, _stack_state(obs, qvel, take=10)
    )
    source = pd.DataFrame(
        {
            "observation.state": list(obs),
            "observation.qpos": list(qpos),
            "observation.qvel": list(qvel),
            "observation.robot_state": list(np.zeros((3, 7), np.float32)),
            "action": list(np.zeros((3, 7), np.float32)),
            "next.reward": list(_source_rewards(progress, gate)),
        }
    )

    target = "LOOT-Stack3-Easy-Dk1"
    out_root = str(tmp_path / target / "run")
    summary = split_to_task(
        [(0, source)],
        {"start_pose_noise": 0.0, "episode_seeds": {"0": 17}, "operator_id": 3},
        plan,
        target,
        out_root,
        fps=20,
    )
    assert summary["written"] == 1

    frames = pd.read_parquet(
        next((tmp_path / target / "run" / "data").rglob("*.parquet"))
    )
    assert np.stack(list(frames["observation.state"])).shape[1] == PROPRIO + 6 * 7
    assert np.stack(list(frames["observation.qpos"])).shape[1] == 6 * 7 + 8
    assert np.asarray(frames["next.done"].values).astype(bool).reshape(-1).sum() == 1

    extra = load_extra_info(out_root)
    # The session's own metadata carries over; the task's does not.
    assert extra["start_pose_noise"] == 0.0
    assert extra["operator_id"] == 3
    assert extra["episode_seeds"] == {"0": 17}
    assert extra["split_from"] == "Stack5-Easy"
