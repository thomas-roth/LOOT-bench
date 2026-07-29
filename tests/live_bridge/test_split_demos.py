"""Tests for the demo splitter (offline relabel to easier same-family tasks).

Pure-logic tests (parse/enumerate/validate) cover Stack and Domino with no
data. The relabel/write tests run against the committed Stack3-Medium demos and
are skipped if absent (they predate the Stack3/4/5 x Easy/Medium/Hard rename and
still live under the old ``G3A-Stack-Low-Dk1_uni`` directory; the blocks are the
Stack3-Medium ones). Because Stack3 is the easiest registered target height, the
object truncation is exercised with a synthetic target that keeps the first 4 of
the 6 blocks. Domino's recompute is covered by synthetic episodes at several row
angles — the fixed row (the base family) sits at yaw=pi/2, while a `RandomRow`
row is tilted and may have negative yaw, and the recompute recovers that angle
from the stored slot quaternion.
"""

import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from g3a_bench.scripts.live_bridge.dataset_extra_info import load_extra_info
from g3a_bench.scripts.live_bridge.split_demos import (
    _STEP_DT,
    SplitPlan,
    _load_source_episodes,
    adapter_for,
    easier_target_task_ids,
    parse_task_id,
    relabel_episode,
    split_to_task,
    variant_from_task_id,
)
from g3a_bench.tasks.stack.variants import (
    STACK_VARIANTS,
    BlockCfg,
    StackVariantCfg,
)

REPO = Path(__file__).resolve().parents[2]
# Recorded before the rename; geometrically the Stack3-Medium task (6 cubes of
# 0.04 m, target 3 blocks high).
DEMO_DIR = REPO / "dataset" / "demos" / "G3A-Stack-Low-Dk1_uni"

STACK_ADAPTER = adapter_for("G3A-Stack3-Medium-Dk1_uni")
STACK3 = STACK_VARIANTS["Stack3Medium"]
# Synthetic truncation target: keep the first 4 of Stack3Medium's 6 blocks (same
# difficulty, so the same split family).
STACK_TRUNC = StackVariantCfg(
    name="Stack3MediumTrunc",
    difficulty="Medium",
    target_blocks=3,
    blocks=tuple(BlockCfg(difficulty="Medium") for _ in range(4)),
)


def _sessions() -> list[str]:
    return sorted(glob.glob(str(DEMO_DIR / "*")))


needs_demos = pytest.mark.skipif(
    not _sessions(), reason="committed Stack3-Medium demos not present"
)


def _arrays(g: pd.DataFrame) -> dict:
    def col(name):
        return np.stack(list(g[name]))

    return {
        "obs": col("observation.state").astype(np.float32),
        "qpos": col("observation.qpos").astype(np.float32),
        "qvel": col("observation.qvel").astype(np.float32),
        "action": col("action").astype(np.float32),
        "robot_state": col("observation.robot_state").astype(np.float32),
        "reward": np.asarray(g["next.reward"].values, np.float32).reshape(-1),
    }


# --- pure logic (no data) -------------------------------------------------- #


def test_parse_and_enumerate_stack():
    assert parse_task_id("G3A-Stack5-Medium-Dk1_uni") == (
        "Stack5",
        "Medium",
        "",
        "Dk1_uni",
    )
    # The target height rides in the category segment, the block shape is the
    # task, so the splittable family is one difficulty.
    assert easier_target_task_ids("G3A-Stack5-Medium-Dk1_uni") == [
        "G3A-Stack4-Medium-Dk1_uni",
        "G3A-Stack3-Medium-Dk1_uni",
    ]
    assert easier_target_task_ids("G3A-Stack5-Hard-Dk1_uni") == [
        "G3A-Stack4-Hard-Dk1_uni",
        "G3A-Stack3-Hard-Dk1_uni",
    ]
    assert easier_target_task_ids("G3A-Stack3-Medium-Dk1_uni") == []


def test_parse_and_enumerate_domino():
    assert parse_task_id("G3A-Domino-4Pieces-Dk1_uni") == (
        "Domino",
        "4Pieces",
        "",
        "Dk1_uni",
    )
    # base family: 4 -> 3, 2
    assert easier_target_task_ids("G3A-Domino-4Pieces-Dk1_uni") == [
        "G3A-Domino-3Pieces-Dk1_uni",
        "G3A-Domino-2Pieces-Dk1_uni",
    ]
    # DifferentDominoes stays in its own family
    assert easier_target_task_ids("G3A-Domino-4Pieces-DifferentDominoes-Dk1_uni") == [
        "G3A-Domino-3Pieces-DifferentDominoes-Dk1_uni",
        "G3A-Domino-2Pieces-DifferentDominoes-Dk1_uni",
    ]
    # KnockOver is not splittable -> no targets
    assert easier_target_task_ids("G3A-Domino-4Pieces-KnockOver-Dk1_uni") == []
    assert easier_target_task_ids("G3A-Domino-2Pieces-Dk1_uni") == []


def test_parse_and_enumerate_domino_random_row():
    # RandomRow is its own family: the base family's row is fixed, so a base
    # recording must never be relabeled into a RandomRow task or vice versa.
    assert parse_task_id("G3A-Domino-4Pieces-RandomRow-Dk1_uni") == (
        "Domino",
        "4Pieces",
        "RandomRow",
        "Dk1_uni",
    )
    assert easier_target_task_ids("G3A-Domino-4Pieces-RandomRow-Dk1_uni") == [
        "G3A-Domino-3Pieces-RandomRow-Dk1_uni",
        "G3A-Domino-2Pieces-RandomRow-Dk1_uni",
    ]
    assert easier_target_task_ids("G3A-Domino-2Pieces-RandomRow-Dk1_uni") == []
    # the base family's targets stay fixed-row only
    assert "G3A-Domino-3Pieces-RandomRow-Dk1_uni" not in easier_target_task_ids(
        "G3A-Domino-4Pieces-Dk1_uni"
    )


def test_validate_rejects_bad_plans():
    high = variant_from_task_id("G3A-Stack5-Medium-Dk1_uni")
    low = variant_from_task_id("G3A-Stack3-Medium-Dk1_uni")
    hard = variant_from_task_id("G3A-Stack3-Hard-Dk1_uni")
    SplitPlan(STACK_ADAPTER, high, low).validate()  # ok
    with pytest.raises(AssertionError):
        SplitPlan(STACK_ADAPTER, low, high).validate()  # target not easier
    with pytest.raises(AssertionError):
        SplitPlan(STACK_ADAPTER, high, hard).validate()  # different block shape

    d_adapter = adapter_for("G3A-Domino-4Pieces-Dk1_uni")
    d4 = variant_from_task_id("G3A-Domino-4Pieces-Dk1_uni")
    d3 = variant_from_task_id("G3A-Domino-3Pieces-Dk1_uni")
    d4knock = variant_from_task_id("G3A-Domino-4Pieces-KnockOver-Dk1_uni")
    SplitPlan(d_adapter, d4, d3).validate()  # ok
    with pytest.raises(AssertionError):
        SplitPlan(d_adapter, d4knock, d3).validate()  # KnockOver not splittable

    d4rand = variant_from_task_id("G3A-Domino-4Pieces-RandomRow-Dk1_uni")
    d3rand = variant_from_task_id("G3A-Domino-3Pieces-RandomRow-Dk1_uni")
    SplitPlan(d_adapter, d4rand, d3rand).validate()  # ok
    with pytest.raises(AssertionError):
        SplitPlan(d_adapter, d4, d3rand).validate()  # fixed row -> random row
    with pytest.raises(AssertionError):
        SplitPlan(d_adapter, d4rand, d3).validate()  # random row -> fixed row


def test_domino_obs_layout():
    # obs = [proprio | poses G*7 | slot_placed G | command G*7 | actions 7]
    assert adapter_for("G3A-Domino-3Pieces-Dk1_uni").obs_section_widths == (7, 1, 7)
    assert adapter_for("G3A-Stack3-Medium-Dk1_uni").obs_section_widths == (7,)


def _domino_episode(placed_per_frame: list[bool], row_yaw: float = math.pi / 2):
    """Synthetic Domino-4 episode. Each frame either has all pieces exactly on
    their slots (placed=True) or piece 0 displaced out of tolerance.

    ``row_yaw`` is the row direction: pi/2 (along +y) is the fixed row of the
    base family; RandomRow recordings use arbitrary, often negative angles.
    """
    n, proprio = 4, 23
    q = np.array(
        [math.cos(row_yaw / 2), 0, 0, math.sin(row_yaw / 2)], np.float32
    )  # quat_from_euler_xyz(0, 0, row_yaw)
    d = np.array([math.cos(row_yaw), math.sin(row_yaw)])
    slots = np.array(
        [[*(np.array([0.3, -0.075]) + i * 0.05 * d), 0.035, *q] for i in range(n)],
        np.float32,
    )  # upright/row-aligned, z = domino half height
    robot = np.array([0, 1.7, 1.7, -1.5708, 0, 0, 0.0, 0.0], np.float32)  # gripper open
    obs, qpos, qvel = [], [], []
    for placed in placed_per_frame:
        poses = slots.copy()
        if not placed:
            poses[0, 0] += 0.1  # displace piece 0 far past pos_threshold
        row = np.concatenate(
            [
                np.zeros(proprio, np.float32),
                poses.reshape(-1),
                np.zeros(n, np.float32),
                slots.reshape(-1),
                np.zeros(7, np.float32),
            ]
        )
        obs.append(row)
        qpos.append(
            np.concatenate([poses.reshape(-1), robot])
        )  # origin 0 -> world==local
        qvel.append(np.zeros(n * 6 + 8, np.float32))
    return (
        np.array(obs, np.float32),
        np.array(qpos, np.float32),
        np.array(qvel, np.float32),
    )


def test_domino_synthetic_recompute_and_truncation():
    import torch

    from g3a_bench.scripts.live_bridge.split_demos import _STEP_DT, _obs_sections

    adapter = adapter_for("G3A-Domino-4Pieces-Dk1_uni")
    d4 = variant_from_task_id("G3A-Domino-4Pieces-Dk1_uni")
    d3 = variant_from_task_id("G3A-Domino-3Pieces-Dk1_uni")
    obs, qpos, qvel = _domino_episode(
        [False, False, True]
    )  # all placed only at frame 2

    # direct known-answer check of the domino recompute
    o = torch.from_numpy(obs.astype(np.float64))
    sections = _obs_sections(o, 23, adapter.obs_section_widths, 4, 4)
    speed = torch.zeros(3, 4, dtype=torch.float64)
    grip_open = torch.ones(3, dtype=torch.bool)
    prog, gate = adapter.progress_and_gate(d4, sections, speed, grip_open)
    assert prog[2].item() == 1.0 and bool(gate[2])  # all 4 placed, at rest, released
    assert prog[0].item() == 0.75 and not bool(gate[0])  # piece 0 displaced

    # relabel 4->3 with a consistent recorded reward; cut lands at frame 2
    reward = (_STEP_DT * (prog + gate.double())).numpy().astype(np.float32)
    rel = relabel_episode(
        obs=obs,
        qpos=qpos,
        qvel=qvel,
        action=np.zeros((3, 7), np.float32),
        robot_state=np.zeros((3, 7), np.float32),
        reward=reward,
        plan=SplitPlan(adapter, d4, d3),
    )
    assert rel is not None
    assert rel.cut_frame == 2  # first frame the (first-3) row is complete
    # obs = [proprio 23 | poses 3*7 | slot_placed 3 | command 3*7 | actions 7]
    assert rel.obs.shape[1] == 23 + 3 * 7 + 3 + 3 * 7 + 7
    assert rel.qpos.shape[1] == 3 * 7 + 8 and rel.qvel.shape[1] == 3 * 6 + 8


# RandomRow rows are tilted and often negative; the fixed row is exactly pi/2.
# The recompute reads the angle back from the slot quaternion via atan2, so it
# must hold across the whole sampled range (+-pi/2 +- 0.6 rad).
@pytest.mark.parametrize("row_yaw", [math.pi / 2, -math.pi / 2, -1.3388, 2.1, 0.0])
def test_domino_recompute_across_row_angles(row_yaw):
    import torch

    from g3a_bench.scripts.live_bridge.split_demos import _obs_sections

    adapter = adapter_for("G3A-Domino-4Pieces-RandomRow-Dk1_uni")
    d4 = variant_from_task_id("G3A-Domino-4Pieces-RandomRow-Dk1_uni")
    d3 = variant_from_task_id("G3A-Domino-3Pieces-RandomRow-Dk1_uni")
    obs, qpos, qvel = _domino_episode([False, False, True], row_yaw=row_yaw)

    o = torch.from_numpy(obs.astype(np.float64))
    sections = _obs_sections(o, 23, adapter.obs_section_widths, 4, 4)
    speed = torch.zeros(3, 4, dtype=torch.float64)
    grip_open = torch.ones(3, dtype=torch.bool)
    prog, gate = adapter.progress_and_gate(d4, sections, speed, grip_open)
    assert prog[2].item() == 1.0 and bool(gate[2])
    assert prog[0].item() == 0.75 and not bool(gate[0])

    reward = (_STEP_DT * (prog + gate.double())).numpy().astype(np.float32)
    rel = relabel_episode(
        obs=obs,
        qpos=qpos,
        qvel=qvel,
        action=np.zeros((3, 7), np.float32),
        robot_state=np.zeros((3, 7), np.float32),
        reward=reward,
        plan=SplitPlan(adapter, d4, d3),
    )
    assert rel is not None and rel.cut_frame == 2


def test_parse_and_enumerate_sorter():
    assert parse_task_id("G3A-Sorter-5Shapes-Full-Dk1_uni") == (
        "Sorter",
        "5Shapes",
        "Full",
        "Dk1_uni",
    )
    # every spawn family splits the same way (spawn mode doesn't change count)
    for variation in ("FixedOrientation", "NoRotation", "Full", "RandomPositions"):
        assert easier_target_task_ids(f"G3A-Sorter-5Shapes-{variation}-Dk1_uni") == [
            f"G3A-Sorter-4Shapes-{variation}-Dk1_uni",
            f"G3A-Sorter-3Shapes-{variation}-Dk1_uni",
        ]
    assert easier_target_task_ids("G3A-Sorter-3Shapes-Full-Dk1_uni") == []
    # obs = [proprio | cube_pose 7 (head) | piece_poses N*7 | socket_inserted N | acts]
    assert adapter_for("G3A-Sorter-5Shapes-Full-Dk1_uni").obs_section_widths == (7, 1)


def _sorter_episode(latched_per_frame: list[list[int]]):
    """Synthetic Sorter-5 episode: one fixed cube body + 5 pieces, with the
    per-frame persistent latch flags in the socket_inserted obs slice."""
    n, proprio = 5, 23
    cube = np.array([0.3, 0, 0.1, 1, 0, 0, 0], np.float32)
    pieces = np.array(
        [[0.25 + 0.03 * i, 0.1, 0.02, 1, 0, 0, 0] for i in range(n)], np.float32
    )
    robot = np.array([0, 1.7, 1.7, -1.5708, 0, 0, 0.0, 0.0], np.float32)  # gripper open
    obs, qpos, qvel = [], [], []
    for latched in latched_per_frame:
        socket = np.array(latched, np.float32)
        obs.append(
            np.concatenate(
                [
                    np.zeros(proprio, np.float32),
                    cube,
                    pieces.reshape(-1),
                    socket,
                    np.zeros(7, np.float32),
                ]
            )
        )
        qpos.append(np.concatenate([cube, pieces.reshape(-1), robot]))  # cube first
        qvel.append(np.zeros(6 + n * 6 + 8, np.float32))
    return (
        np.array(obs, np.float32),
        np.array(qpos, np.float32),
        np.array(qvel, np.float32),
    )


def test_sorter_synthetic_recompute_and_truncation():
    import torch

    from g3a_bench.scripts.live_bridge.split_demos import _STEP_DT, _obs_sections

    adapter = adapter_for("G3A-Sorter-5Shapes-Full-Dk1_uni")
    s5 = variant_from_task_id("G3A-Sorter-5Shapes-Full-Dk1_uni")
    s3 = variant_from_task_id("G3A-Sorter-3Shapes-Full-Dk1_uni")
    # first-3 complete at frame 1, all-5 at frame 2
    obs, qpos, qvel = _sorter_episode(
        [[0, 0, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    )

    o = torch.from_numpy(obs.astype(np.float64))
    proprio = obs.shape[1] - 5 * sum(adapter.obs_section_widths) - 7
    assert proprio == 30  # proprio 23 + cube_pose 7
    speed = torch.zeros(3, 5, dtype=torch.float64)
    grip = torch.ones(3, dtype=torch.bool)
    p5, g5 = adapter.progress_and_gate(
        s5, _obs_sections(o, proprio, adapter.obs_section_widths, 5, 5), speed, grip
    )
    assert np.allclose(p5.numpy(), [0.0, 0.6, 1.0])
    assert g5.tolist() == [False, False, True]  # all 5 latched only at frame 2
    p3, g3 = adapter.progress_and_gate(
        s3,
        _obs_sections(o, proprio, adapter.obs_section_widths, 5, 3),
        speed[:, :3],
        grip,
    )
    assert np.allclose(p3.numpy(), [0.0, 1.0, 1.0])
    assert g3.tolist() == [False, True, True]  # first 3 latched from frame 1

    reward = (_STEP_DT * (p5 + g5.double())).numpy().astype(np.float32)
    rel = relabel_episode(
        obs=obs,
        qpos=qpos,
        qvel=qvel,
        action=np.zeros((3, 7), np.float32),
        robot_state=np.zeros((3, 7), np.float32),
        reward=reward,
        plan=SplitPlan(adapter, s5, s3),
    )
    assert rel is not None
    assert rel.cut_frame == 1  # first frame the first-3 are all latched
    # obs = [head 30 | piece_poses 3*7 | socket_inserted 3 | actions 7]
    assert rel.obs.shape[1] == 30 + 3 * 7 + 3 + 7
    # qpos = [cube 7 | pieces 3*7 | robot 8]; qvel = [cube 6 | pieces 3*6 | robot 8]
    assert rel.qpos.shape[1] == 7 + 3 * 7 + 8
    assert rel.qvel.shape[1] == 6 + 3 * 6 + 8


# --- relabel core against the committed Stack demos ------------------------ #


@needs_demos
def test_source_self_check_and_truncation():
    plan = SplitPlan(STACK_ADAPTER, STACK3, STACK_TRUNC)
    k = plan.k
    n_cut = 0
    for _ep, g in _load_source_episodes(_sessions()[0]):
        a = _arrays(g)
        rel = relabel_episode(plan=plan, **a)  # asserts the source self-check
        if rel is None:
            continue
        n_cut += 1
        t = len(rel.obs)
        # structured truncation: proprio + first-k blocks + actions preserved
        assert rel.obs.shape[1] == 23 + k * 7 + 7
        assert rel.qpos.shape[1] == k * 7 + 8
        assert rel.qvel.shape[1] == k * 6 + 8
        assert np.array_equal(rel.obs[:, :23], a["obs"][:t, :23])
        assert np.array_equal(
            rel.obs[:, 23 : 23 + k * 7], a["obs"][:t, 23 : 23 + k * 7]
        )
        assert np.array_equal(rel.obs[:, -7:], a["obs"][:t, -7:])
        assert np.array_equal(rel.qpos[:, k * 7 :], a["qpos"][:t, plan.n_src * 7 :])
        assert rel.done[-1, 0] and not rel.done[:-1, 0].any()
        assert rel.reward[-1, 0] > 1.9 * _STEP_DT
    assert n_cut > 0


@needs_demos
def test_write_roundtrip(tmp_path):
    plan = SplitPlan(STACK_ADAPTER, STACK3, STACK_TRUNC)
    session = _sessions()[0]
    out_root = str(tmp_path / "G3A-Stack3-Trunc-Dk1_uni" / "run")
    summary = split_to_task(
        _load_source_episodes(session),
        load_extra_info(session),
        plan,
        "G3A-Stack3-Trunc-Dk1_uni",
        out_root,
    )
    assert summary["written"] > 0
    files = sorted(glob.glob(f"{out_root}/data/chunk-*/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    assert np.stack(list(df["observation.state"])).shape[1] == 23 + plan.k * 7 + 7
    assert np.stack(list(df["observation.qpos"])).shape[1] == plan.k * 7 + 8
    dones = np.asarray(df["next.done"].values).astype(bool).reshape(-1)
    assert dones.sum() == summary["written"]
    extra = load_extra_info(out_root)
    assert len(extra["episode_seeds"]) == summary["written"]
    assert extra["split_from"] == "Stack3-Medium"
