"""Tests for the Stack demo splitter (offline relabel to easier tasks).

Pure-logic tests need no data. The relabel/write tests run against the
committed Stack-Low demos and are skipped if those are absent. Because Low is
the easiest registered task, the down-split is exercised with a synthetic
Stack2-analog target (keep the first 4 cubes, target height 0.08).
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
    easier_target_task_ids,
    parse_stack_task_id,
    relabel_episode,
    split_to_task,
    variant_from_task_id,
)
from g3a_bench.tasks.stack.variants import STACK_VARIANTS, StackVariantCfg, _cubes

REPO = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO / "dataset" / "demos" / "G3A-Stack-Low-Dk1_uni"

STACK3 = STACK_VARIANTS["Stack3"]
STACK2 = StackVariantCfg(name="Stack2", target_blocks=2, blocks=_cubes(4))  # synthetic


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


def test_parse_and_enumerate():
    assert parse_stack_task_id("G3A-Stack-High-Dk1_uni") == ("High", "", "Dk1_uni")
    assert parse_stack_task_id("G3A-Stack-Low-TwoStacks-Dk1_uni") == (
        "Low",
        "TwoStacks",
        "Dk1_uni",
    )
    assert easier_target_task_ids("G3A-Stack-High-Dk1_uni") == [
        "G3A-Stack-Medium-Dk1_uni",
        "G3A-Stack-Low-Dk1_uni",
    ]
    # Families stay separate; Low has no easier target.
    assert easier_target_task_ids("G3A-Stack-High-TwoStacks-Dk1_uni") == [
        "G3A-Stack-Medium-TwoStacks-Dk1_uni",
        "G3A-Stack-Low-TwoStacks-Dk1_uni",
    ]
    assert easier_target_task_ids("G3A-Stack-Low-Dk1_uni") == []


def test_validate_rejects_bad_plans():
    high = variant_from_task_id("G3A-Stack-High-Dk1_uni")
    low = variant_from_task_id("G3A-Stack-Low-Dk1_uni")
    two = variant_from_task_id("G3A-Stack-High-TwoStacks-Dk1_uni")
    SplitPlan(high, low).validate()  # ok
    with pytest.raises(AssertionError):
        SplitPlan(low, high).validate()  # target not easier
    with pytest.raises(AssertionError):
        SplitPlan(
            high, variant_from_task_id("G3A-Stack-Low-TwoStacks-Dk1_uni")
        ).validate()
    with pytest.raises(AssertionError):
        SplitPlan(two, low).validate()  # cross TwoStacks family


# --- relabel core against the committed demos ------------------------------ #


@needs_demos
def test_source_self_check_and_synthetic_split():
    plan = SplitPlan(STACK3, STACK2)
    n_cut = n_skip = 0
    for _ep, g in _load_source_episodes(_sessions()[0]):
        a = _arrays(g)
        rel = relabel_episode(plan=plan, **a)  # asserts the source self-check
        if rel is None:
            n_skip += 1
            continue
        n_cut += 1
        T = len(rel.obs)
        # structured truncation: proprio + first-k blocks + actions preserved
        assert rel.obs.shape[1] == 23 + plan.k * 7 + 7
        assert rel.qpos.shape[1] == plan.k * 7 + 8
        assert rel.qvel.shape[1] == plan.k * 6 + 8
        assert np.array_equal(rel.obs[:, :23], a["obs"][:T, :23])
        assert np.array_equal(
            rel.obs[:, 23 : 23 + plan.k * 7], a["obs"][:T, 23 : 23 + plan.k * 7]
        )
        assert np.array_equal(rel.obs[:, -7:], a["obs"][:T, -7:])
        assert np.array_equal(
            rel.qpos[:, plan.k * 7 :], a["qpos"][:T, plan.n_src * 7 :]
        )
        # cut is terminal and lands on a target-success frame (~progress 1 + gate)
        assert rel.done[-1, 0] and not rel.done[:-1, 0].any()
        assert rel.reward[-1, 0] > 1.9 * _STEP_DT
    assert n_cut > 0  # some episodes settle at the intermediate height


@needs_demos
def test_write_roundtrip(tmp_path):
    plan = SplitPlan(STACK3, STACK2)
    session = _sessions()[0]
    out_root = str(tmp_path / "G3A-Stack-Test2-Dk1_uni" / "run")
    summary = split_to_task(
        _load_source_episodes(session),
        load_extra_info(session),
        plan,
        "G3A-Stack-Test2-Dk1_uni",
        out_root,
    )
    assert summary["written"] > 0
    files = sorted(glob.glob(f"{out_root}/data/chunk-*/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    assert np.stack(list(df["observation.state"])).shape[1] == 23 + plan.k * 7 + 7
    assert np.stack(list(df["observation.qpos"])).shape[1] == plan.k * 7 + 8
    # one terminal per written episode; seeds carried over unchanged
    dones = np.asarray(df["next.done"].values).astype(bool).reshape(-1)
    assert dones.sum() == summary["written"]
    extra = load_extra_info(out_root)
    assert len(extra["episode_seeds"]) == summary["written"]
    assert extra["split_from"] == "Stack3"
