"""Tests for the demo splitter (offline relabel to easier same-family tasks).

Pure-logic tests (parse/enumerate/validate) cover Stack and Domino with no
data. The relabel/write tests run against the committed Stack-Low demos and are
skipped if absent. Because Low is the easiest registered task, the object
truncation is exercised with a synthetic target that keeps the first 4 of the 6
cubes. Domino has no committed demos, so only its pure logic is tested — the
per-episode source self-check in relabel_episode validates the Domino recompute
the first time a real Domino recording is split.
"""

import glob
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
from g3a_bench.tasks.stack.variants import STACK_VARIANTS, StackVariantCfg, _cubes

REPO = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO / "dataset" / "demos" / "G3A-Stack-Low-Dk1_uni"

STACK_ADAPTER = adapter_for("G3A-Stack-Low-Dk1_uni")
STACK3 = STACK_VARIANTS["Stack3"]
# Synthetic truncation target: keep the first 4 of Stack3's 6 cubes (same
# family/height); registered as ("Low", "") by _task_id_parts so validate passes.
STACK_TRUNC = StackVariantCfg(name="Stack3trunc", target_blocks=3, blocks=_cubes(4))


def _sessions() -> list[str]:
    return sorted(glob.glob(str(DEMO_DIR / "*")))


needs_demos = pytest.mark.skipif(
    not _sessions(), reason="committed Stack-Low demos not present"
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
    assert parse_task_id("G3A-Stack-High-Dk1_uni") == ("Stack", "High", "", "Dk1_uni")
    assert parse_task_id("G3A-Stack-Low-TwoStacks-Dk1_uni") == (
        "Stack",
        "Low",
        "TwoStacks",
        "Dk1_uni",
    )
    assert easier_target_task_ids("G3A-Stack-High-Dk1_uni") == [
        "G3A-Stack-Medium-Dk1_uni",
        "G3A-Stack-Low-Dk1_uni",
    ]
    assert easier_target_task_ids("G3A-Stack-High-TwoStacks-Dk1_uni") == [
        "G3A-Stack-Medium-TwoStacks-Dk1_uni",
        "G3A-Stack-Low-TwoStacks-Dk1_uni",
    ]
    assert easier_target_task_ids("G3A-Stack-Low-Dk1_uni") == []


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


def test_validate_rejects_bad_plans():
    high = variant_from_task_id("G3A-Stack-High-Dk1_uni")
    low = variant_from_task_id("G3A-Stack-Low-Dk1_uni")
    two = variant_from_task_id("G3A-Stack-High-TwoStacks-Dk1_uni")
    SplitPlan(STACK_ADAPTER, high, low).validate()  # ok
    with pytest.raises(AssertionError):
        SplitPlan(STACK_ADAPTER, low, high).validate()  # target not easier
    with pytest.raises(AssertionError):
        SplitPlan(STACK_ADAPTER, high, two).validate()  # different family

    d_adapter = adapter_for("G3A-Domino-4Pieces-Dk1_uni")
    d4 = variant_from_task_id("G3A-Domino-4Pieces-Dk1_uni")
    d3 = variant_from_task_id("G3A-Domino-3Pieces-Dk1_uni")
    d4knock = variant_from_task_id("G3A-Domino-4Pieces-KnockOver-Dk1_uni")
    SplitPlan(d_adapter, d4, d3).validate()  # ok
    with pytest.raises(AssertionError):
        SplitPlan(d_adapter, d4knock, d3).validate()  # KnockOver not splittable


def test_domino_obs_layout():
    # obs = [proprio | poses G*7 | slot_placed G | command G*7 | actions 7]
    assert adapter_for("G3A-Domino-3Pieces-Dk1_uni").obs_section_widths == (7, 1, 7)
    assert adapter_for("G3A-Stack-Low-Dk1_uni").obs_section_widths == (7,)


def _domino_episode(placed_per_frame: list[bool]):
    """Synthetic Domino-4 episode. Each frame either has all pieces exactly on
    their slots (placed=True) or piece 0 displaced out of tolerance."""
    import math

    n, proprio = 4, 23
    q = np.array([math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)], np.float32)
    slots = np.array(
        [[0.3, -0.075 + i * 0.05, 0.035, *q] for i in range(n)], np.float32
    )  # row along +y, upright/row-aligned, z = domino half height
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
    out_root = str(tmp_path / "G3A-Stack-Trunc-Dk1_uni" / "run")
    summary = split_to_task(
        _load_source_episodes(session),
        load_extra_info(session),
        plan,
        "G3A-Stack-Trunc-Dk1_uni",
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
    assert tuple(extra["split_from"]) == ("Low", "")
