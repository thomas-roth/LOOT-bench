"""Tests for the demo splitter (offline relabel to easier same-family tasks).

Everything runs on synthetic episodes: no recorded demo of a splittable
category is committed, and a fabricated one pins the contract more sharply
anyway. Each category's fixture builds an observation in the real layout --
``[joint_pos 8 | joint_vel 8 | ee_pose 9 | pad_contact_force 1 |
object-major sections]``, nothing
after (ADR-016) -- and a reward column matching each frame's own state
(no shift, see ``split_demos``'s "Frame alignment" note), so the source
self-check runs on every relabel.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from loot_bench.scripts.live_bridge.dataset_extra_info import load_extra_info
from loot_bench.scripts.split_demos import (
    SplitPlan,
    adapter_for,
    easier_target_task_ids,
    episode_rewards,
    frame_state,
    main,
    parse_task_id,
    proprio_width,
    relabel_episode,
    split_dataset,
    split_to_task,
    variant_from_task_id,
)
from loot_bench.tasks.common.rewards import shaped_task_reward
from loot_bench.tasks.stack.variants import BLOCK_WIDTH, LEVEL_HEIGHT

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

    Frame ``t`` is paid exactly what the state in frame ``t``'s own
    observation earns -- no shift (see ``split_demos``'s "Frame alignment"
    note).
    """
    return shaped_task_reward(progress, gate).numpy().astype(np.float32)


def _session_frame(
    episode: int, obs: np.ndarray, qpos: np.ndarray, qvel: np.ndarray, reward
) -> pd.DataFrame:
    """One recorded episode as the rows ``load_source_episodes`` reads."""
    frames = len(obs)
    return pd.DataFrame(
        {
            "episode_index": np.full(frames, episode, int),
            "frame_index": np.arange(frames),
            "observation.state": list(obs),
            "observation.robot_state": list(np.zeros((frames, 7), np.float32)),
            "observation.qpos": list(qpos),
            "observation.qvel": list(qvel),
            "action": list(np.zeros((frames, 7), np.float32)),
            "next.reward": [float(r) for r in np.asarray(reward).reshape(-1)],
        }
    )


def _write_session(root: Path, episodes: list[pd.DataFrame], fps: int = 20) -> str:
    """A minimal LeRobot-shaped source session: the parquet chunk and the
    ``info.json`` the splitter reads the frame rate from."""
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir()
    for i, episode in enumerate(episodes):
        episode.to_parquet(root / "data" / "chunk-000" / f"episode_{i:06d}.parquet")
    (root / "meta" / "info.json").write_text(json.dumps({"fps": fps}))
    return str(root)


# --- task-id resolution ---------------------------------------------------- #


class TestTaskIds:
    def test_stack_is_one_family_across_the_level_count(self):
        assert parse_task_id("LOOT-Stack-3Henges-Dk1") == (
            "Stack",
            "3Henges",
            "",
            "Dk1",
        )
        # The level count is the task segment and every variant shares the
        # tail-extending block tiers, so one recording feeds both easier tasks.
        assert easier_target_task_ids("LOOT-Stack-3Henges-Dk1") == [
            "LOOT-Stack-2Henges-Dk1",
            "LOOT-Stack-1Henge-Dk1",
        ]
        assert easier_target_task_ids("LOOT-Stack-2Henges-Dk1") == [
            "LOOT-Stack-1Henge-Dk1"
        ]
        assert easier_target_task_ids("LOOT-Stack-1Henge-Dk1") == []

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
            assert f"LOOT-Sorter-2Shapes-{variation}-Dk1" not in easier_target_task_ids(
                f"LOOT-Sorter-5Shapes-{variation}-Dk1"
            )

    def test_unsplittable_category_and_malformed_id(self):
        with pytest.raises(ValueError, match="not splittable"):
            adapter_for("LOOT-Place-FixedS-FixedT-Dk1")
        with pytest.raises(ValueError, match="not a LOOT task id"):
            parse_task_id("Stack-3Henges")

    def test_obs_section_widths(self):
        assert adapter_for("LOOT-Stack-1Henge-Dk1").obs_section_widths == (7,)
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
    def test_rejects_a_harder_target(self):
        adapter = adapter_for("LOOT-Stack-3Henges-Dk1")
        three = variant_from_task_id("LOOT-Stack-3Henges-Dk1")
        one = variant_from_task_id("LOOT-Stack-1Henge-Dk1")
        SplitPlan(adapter, three, one).validate()
        with pytest.raises(AssertionError, match="not easier"):
            SplitPlan(adapter, one, three).validate()

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

    def test_sorter_2shapes_explicit_target_rejected(self):
        plan = SplitPlan(
            adapter_for("LOOT-Sorter-5Shapes-FixedOrientation-Dk1"),
            variant_from_task_id("LOOT-Sorter-5Shapes-FixedOrientation-Dk1"),
            variant_from_task_id("LOOT-Sorter-2Shapes-FixedOrientation-Dk1"),
        )
        with pytest.raises(AssertionError, match="not splittable"):
            plan.validate()


# --- Stack (henge) --------------------------------------------------------- #

N_BLOCKS = 9
"""3Henges' block count: three tiers of three, navy 0-2, teal 3-5, white 6-8."""

UPRIGHT_Z = 0.0325
"""A standing block's centre: half its length above whatever it stands on."""

UPRIGHT_QUAT = (1.0, 0.0, 0.0, 0.0)
LINTEL_QUAT = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)
"""A 90 deg rotation about y lays the block's long axis along x."""

BUILD_XY = (0.28, 0.0)
UPRIGHT_SPACING = 0.036


def _spawned(i: int) -> list[float]:
    """Block ``i`` standing on its spawn cell, clear of the build zone."""
    return [0.45, -0.3 + 0.075 * i, UPRIGHT_Z, *UPRIGHT_QUAT]


def _upright(level: int, side: float) -> list[float]:
    """A standing block of level ``level`` (1-based), on the levels below it."""
    x, y = BUILD_XY
    z = (level - 1) * LEVEL_HEIGHT + UPRIGHT_Z
    return [x + side * UPRIGHT_SPACING / 2, y, z, *UPRIGHT_QUAT]


def _lintel(level: int) -> list[float]:
    """The lying block of level ``level``, centre at ``k (L + w) - w / 2``."""
    x, y = BUILD_XY
    return [x, y, level * LEVEL_HEIGHT - BLOCK_WIDTH / 2, *LINTEL_QUAT]


def _henge_poses(built: dict[int, tuple[int, int, int]]) -> np.ndarray:
    """(9, 7) block poses with the given levels standing.

    ``built`` maps a level to the three block indices forming it, two
    uprights then the lintel; every other block sits on its spawn cell.
    """
    poses = np.array([_spawned(i) for i in range(N_BLOCKS)], np.float32)
    for level, (left, right, top) in built.items():
        poses[left] = _upright(level, -1.0)
        poses[right] = _upright(level, 1.0)
        poses[top] = _lintel(level)
    return poses


def _henge_episode(
    frames: list[np.ndarray],
    grasp_pos_per_frame=None,
    pad_force: float | list[float] = 0.0,
):
    """Synthetic 3Henges episode from per-frame (9, 7) block poses.

    ``pad_force`` is the recorded finger-pad force, a scalar or one value per
    frame; the env reads release straight off it.
    """
    n_frames = len(frames)
    forces = pad_force if isinstance(pad_force, list) else [pad_force] * n_frames
    grasp = grasp_pos_per_frame or [FAR_AWAY] * n_frames
    obs, qpos, qvel = [], [], []
    for poses, grasp_pos, force in zip(frames, grasp, forces, strict=True):
        robot = ROBOT_QPOS_OPEN if force == 0.0 else ROBOT_QPOS_CLOSED
        obs.append(
            np.concatenate(
                [
                    _head(np.asarray(grasp_pos, np.float32), float(force)),
                    poses.reshape(-1),
                ]
            )
        )
        qpos.append(np.concatenate([poses.reshape(-1), robot]))
        qvel.append(np.zeros(N_BLOCKS * 6 + 8, np.float32))
    return (
        np.array(obs, np.float32),
        np.array(qpos, np.float32),
        np.array(qvel, np.float32),
    )


LEVEL_1 = {1: (0, 1, 2)}
LEVEL_2 = {2: (3, 4, 5)}
LEVEL_3 = {3: (6, 7, 8)}


def _tier_order_frames() -> list[np.ndarray]:
    """The protocol build: one level per frame, each from its own colour tier,
    then one frame of rest so the source's success has a frame to be paid on."""
    return [
        _henge_poses({}),
        _henge_poses(LEVEL_1),
        _henge_poses({**LEVEL_1, **LEVEL_2}),
        _henge_poses({**LEVEL_1, **LEVEL_2, **LEVEL_3}),
        _henge_poses({**LEVEL_1, **LEVEL_2, **LEVEL_3}),
    ]


def _white_first_frames() -> list[np.ndarray]:
    """A protocol break: level 1 from the white blocks, level 3 from navy.

    The source still succeeds (three levels stand), but the first-3 blocks
    never form level 1, so no easier task can be cut out.
    """
    white_low = {1: (6, 7, 8)}
    navy_high = {3: (0, 1, 2)}
    return [
        _henge_poses({}),
        _henge_poses(white_low),
        _henge_poses({**white_low, **LEVEL_2}),
        _henge_poses({**white_low, **LEVEL_2, **navy_high}),
        _henge_poses({**white_low, **LEVEL_2, **navy_high}),
    ]


def _henge_state(obs, qvel, take):
    adapter = adapter_for("LOOT-Stack-3Henges-Dk1")
    return frame_state(
        torch.from_numpy(obs.astype(np.float64)),
        torch.from_numpy(qvel.astype(np.float64)),
        proprio=PROPRIO,
        adapter=adapter,
        n_layout=N_BLOCKS,
        take=take,
    )


def _henge_plan(target: str) -> SplitPlan:
    return SplitPlan(
        adapter_for("LOOT-Stack-3Henges-Dk1"),
        variant_from_task_id("LOOT-Stack-3Henges-Dk1"),
        variant_from_task_id(target),
    )


def _henge_source_rewards(obs, qvel) -> np.ndarray:
    plan = _henge_plan("LOOT-Stack-1Henge-Dk1")
    progress, gate = plan.adapter.progress_and_gate(
        plan.source, _henge_state(obs, qvel, take=N_BLOCKS)
    )
    return _source_rewards(progress, gate)


def _relabel(obs, qpos, qvel, target: str, reward=None):
    frames = len(obs)
    return relabel_episode(
        obs=obs,
        qpos=qpos,
        qvel=qvel,
        action=np.zeros((frames, 7), np.float32),
        robot_state=np.zeros((frames, 7), np.float32),
        reward=_henge_source_rewards(obs, qvel) if reward is None else reward,
        plan=_henge_plan(target),
    )


class TestHengeRecompute:
    def test_progress_counts_standing_levels(self):
        obs, _qpos, qvel = _henge_episode(_tier_order_frames())
        plan = _henge_plan("LOOT-Stack-1Henge-Dk1")
        progress, gate = plan.adapter.progress_and_gate(
            plan.source, _henge_state(obs, qvel, take=N_BLOCKS)
        )
        assert progress.tolist() == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0, 1.0])
        assert gate.tolist() == [False, False, False, True, True]

    @pytest.mark.parametrize(
        ("target", "cut_frame", "k"),
        [("LOOT-Stack-1Henge-Dk1", 1, 3), ("LOOT-Stack-2Henges-Dk1", 2, 6)],
    )
    def test_tier_order_build_cuts_at_each_target_success(self, target, cut_frame, k):
        obs, qpos, qvel = _henge_episode(_tier_order_frames())
        rel = _relabel(obs, qpos, qvel, target)
        assert rel is not None
        assert rel.cut_frame == cut_frame
        assert len(rel.obs) == cut_frame + 1
        # The last frame is absorbing: reward exactly 0 and done set there only.
        assert rel.reward[-1, 0] == 0.0
        assert rel.done[-1, 0] and not rel.done[:-1, 0].any()
        assert (rel.reward[:-1] < 0.0).all()
        # Truncation keeps the head and the first-k blocks of every slice.
        assert rel.obs.shape[1] == PROPRIO + k * 7
        assert rel.qpos.shape[1] == k * 7 + 8
        assert rel.qvel.shape[1] == k * 6 + 8
        assert np.array_equal(rel.obs[:, :PROPRIO], obs[: cut_frame + 1, :PROPRIO])
        assert np.array_equal(
            rel.qpos[:, k * 7 :], qpos[: cut_frame + 1, N_BLOCKS * 7 :]
        )
        # The cut lands the moment the target's levels stand, so the later
        # tiers are still on their spawn cells: nothing dropped was disturbed.
        assert rel.n_dropped_disturbed == 0

    def test_a_held_lintel_earns_no_level(self):
        # The same standing level in both frames; only the grasp site moves
        # onto the lintel with the pads loaded, so the drop is the exclusion.
        frames = [_henge_poses(LEVEL_1)] * 2
        lintel_pos = np.asarray(_lintel(1)[:3], np.float32)
        obs, _qpos, qvel = _henge_episode(
            frames, [FAR_AWAY, lintel_pos], pad_force=[0.0, 5.0]
        )
        plan = _henge_plan("LOOT-Stack-1Henge-Dk1")
        progress, gate = plan.adapter.progress_and_gate(
            plan.target, _henge_state(obs, qvel, take=3)
        )
        assert progress.tolist() == pytest.approx([1.0, 0.0])
        assert gate.tolist() == [True, False]

    def test_white_first_build_relabels_to_nothing(self):
        obs, qpos, qvel = _henge_episode(_white_first_frames())
        assert _relabel(obs, qpos, qvel, "LOOT-Stack-1Henge-Dk1") is None
        assert _relabel(obs, qpos, qvel, "LOOT-Stack-2Henges-Dk1") is None

    def test_self_check_rejects_a_mismatched_reward_column(self):
        obs, qpos, qvel = _henge_episode(_tier_order_frames()[:2])
        with pytest.raises(AssertionError, match="self-check failed"):
            _relabel(
                obs, qpos, qvel, "LOOT-Stack-1Henge-Dk1", reward=np.zeros(2, np.float32)
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


def _domino_state(obs, qvel, take):
    return frame_state(
        torch.from_numpy(obs.astype(np.float64)),
        torch.from_numpy(qvel.astype(np.float64)),
        proprio=PROPRIO,
        adapter=adapter_for("LOOT-Domino-4Pieces-Dk1"),
        n_layout=4,
        take=take,
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
    progress, gate = adapter.progress_and_gate(four, _domino_state(obs, qvel, take=4))
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
    # The first 3 are complete at frame 1, paid there too (no shift).
    assert rel.cut_frame == 1
    assert rel.obs.shape[1] == PROPRIO + 3 * 7 + 3 + 3 * 7
    assert rel.qpos.shape[1] == 3 * 7 + 8
    assert rel.qvel.shape[1] == 3 * 6 + 8


def test_domino_dropped_piece_movement_is_reported():
    # Piece 2 displaced on frame 0 so the target-3 success (and the cut) only
    # holds from frame 1, keeping frame 1 -- where the dropped piece moves --
    # inside the relabeled window.
    obs, qpos, qvel = _domino_episode([(2,), (), ()])
    # Move the dropped 4th piece well past the disturbance tolerance.
    obs[1:, PROPRIO + 3 * 7] += 0.1
    adapter = adapter_for("LOOT-Domino-4Pieces-Dk1")
    four = variant_from_task_id("LOOT-Domino-4Pieces-Dk1")
    three = variant_from_task_id("LOOT-Domino-3Pieces-Dk1")
    progress, gate = adapter.progress_and_gate(four, _domino_state(obs, qvel, take=4))
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
    assert rel.cut_frame == 1  # first-3 latched at frame 1, paid there too
    assert rel.obs.shape[1] == PROPRIO + 7 + 3 * 7 + 3
    # The cube stays: qpos keeps [cube | first-3 pieces | robot].
    assert rel.qpos.shape[1] == 7 + 3 * 7 + 8
    assert rel.qvel.shape[1] == 6 + 3 * 6 + 8


# --- fail loudly on a never-successful relabel ----------------------------- #


def _mixed_henge_session(root: Path) -> str:
    """Episode 0 follows the tier protocol, episode 1 builds level 1 from the
    white blocks."""
    episodes = []
    for index, frames in enumerate((_tier_order_frames(), _white_first_frames())):
        obs, qpos, qvel = _henge_episode(frames)
        episodes.append(
            _session_frame(index, obs, qpos, qvel, _henge_source_rewards(obs, qvel))
        )
    return _write_session(root, episodes)


def _written_episodes(out_root: Path) -> int:
    frames = pd.concat(
        [pd.read_parquet(f) for f in (out_root / "data").rglob("*.parquet")]
    )
    return int(frames["episode_index"].nunique())


class TestSkipsAreFatal:
    def test_good_episodes_are_written_before_the_raise(self, tmp_path):
        source = _mixed_henge_session(tmp_path / "source")
        with pytest.raises(ValueError) as error:
            split_dataset(
                source,
                "LOOT-Stack-3Henges-Dk1",
                str(tmp_path / "out"),
                target_tasks=["LOOT-Stack-1Henge-Dk1"],
                stamp="run",
            )
        message = str(error.value)
        assert "LOOT-Stack-1Henge-Dk1: source episodes [1]" in message
        assert "lintel block [8] on standing blocks [6, 7]" in message
        assert "navy blocks (0-2), level 2 from teal (3-5)" in message
        assert "white (6-8), bottom-up" in message
        # The protocol-following episode is on disk regardless.
        assert (
            _written_episodes(tmp_path / "out" / "LOOT-Stack-1Henge-Dk1" / "run") == 1
        )

    def test_allow_skips_demotes_the_raise_to_a_summary_warning(self, tmp_path):
        source = _mixed_henge_session(tmp_path / "source")
        summaries = split_dataset(
            source,
            "LOOT-Stack-3Henges-Dk1",
            str(tmp_path / "out"),
            stamp="run",
            allow_skips=True,
        )
        assert [s["target_task"] for s in summaries] == [
            "LOOT-Stack-2Henges-Dk1",
            "LOOT-Stack-1Henge-Dk1",
        ]
        for summary in summaries:
            assert summary["written"] == 1
            assert summary["skipped"] == [1]
            assert "lintel block [8]" in summary["skip_diagnostics"][1]

    def test_main_exits_non_zero_unless_skips_are_allowed(self, tmp_path, monkeypatch):
        source = _mixed_henge_session(tmp_path / "source")
        argv = [
            "loot-split-demos",
            "--source-dir",
            source,
            "--source-task",
            "LOOT-Stack-3Henges-Dk1",
            "--target-task",
            "LOOT-Stack-1Henge-Dk1",
            "--output-dir",
        ]
        monkeypatch.setattr(sys, "argv", [*argv, str(tmp_path / "strict")])
        with pytest.raises(SystemExit) as exit_info:
            main()
        assert exit_info.value.code != 0
        assert "source episodes [1]" in str(exit_info.value.code)

        monkeypatch.setattr(
            sys, "argv", [*argv, str(tmp_path / "lenient"), "--allow-skips"]
        )
        main()  # returns normally, which is a zero exit

    def test_source_that_never_stands_level_1_says_so(self, tmp_path):
        # Three levels' worth of frames with nothing ever leaving its cell:
        # the source never succeeds either, so there is no lintel to name.
        obs, qpos, qvel = _henge_episode([_henge_poses({})] * 3)
        source = _write_session(
            tmp_path / "source",
            [_session_frame(0, obs, qpos, qvel, _henge_source_rewards(obs, qvel))],
        )
        with pytest.raises(ValueError, match="never completed level 1 either"):
            split_dataset(
                source,
                "LOOT-Stack-3Henges-Dk1",
                str(tmp_path / "out"),
                target_tasks=["LOOT-Stack-1Henge-Dk1"],
            )

    def test_domino_never_placed_raises_too(self, tmp_path):
        obs, qpos, qvel = _domino_episode([(0, 1, 2, 3)] * 3)
        adapter = adapter_for("LOOT-Domino-4Pieces-Dk1")
        four = variant_from_task_id("LOOT-Domino-4Pieces-Dk1")
        progress, gate = adapter.progress_and_gate(
            four, _domino_state(obs, qvel, take=4)
        )
        source = _write_session(
            tmp_path / "source",
            [_session_frame(0, obs, qpos, qvel, _source_rewards(progress, gate))],
        )
        with pytest.raises(ValueError, match="collection order") as error:
            split_dataset(
                source,
                "LOOT-Domino-4Pieces-Dk1",
                str(tmp_path / "out"),
                target_tasks=["LOOT-Domino-3Pieces-Dk1"],
            )
        assert "LOOT-Domino-3Pieces-Dk1: source episodes [0]" in str(error.value)


# --- frame alignment -------------------------------------------------------- #


def test_episode_rewards_pay_success_the_frame_it_first_holds():
    progress = torch.tensor([0.5, 1.0, 1.0])
    gate = torch.tensor([False, True, True])
    rewards = episode_rewards(progress, gate, last_frame=1)
    assert len(rewards) == 2  # last_frame is inclusive, no extra shift
    assert rewards[0].item() < 0.0
    assert rewards[1].item() == 0.0  # success holds at frame 1, paid there too


# --- write round-trip ------------------------------------------------------ #


def test_write_roundtrip(tmp_path):
    obs, qpos, qvel = _henge_episode(_tier_order_frames())
    plan = _henge_plan("LOOT-Stack-1Henge-Dk1")
    source = _session_frame(0, obs, qpos, qvel, _henge_source_rewards(obs, qvel))

    target = "LOOT-Stack-1Henge-Dk1"
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
    assert summary["skipped"] == []

    frames = pd.read_parquet(
        next((tmp_path / target / "run" / "data").rglob("*.parquet"))
    )
    assert np.stack(list(frames["observation.state"])).shape[1] == PROPRIO + 3 * 7
    assert np.stack(list(frames["observation.qpos"])).shape[1] == 3 * 7 + 8
    assert np.asarray(frames["next.done"].values).astype(bool).reshape(-1).sum() == 1

    extra = load_extra_info(out_root)
    # The session's own metadata carries over; the task's does not.
    assert extra["start_pose_noise"] == 0.0
    assert extra["operator_id"] == 3
    assert extra["episode_seeds"] == {"0": 17}
    assert extra["split_from"] == "Stack-3Henges"
