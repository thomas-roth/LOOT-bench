# pyright: reportAttributeAccessIssue=false
"""Split a recorded Stack demo dataset into easier Stack tasks (offline relabel).

Stack variants tail-extend (an easier task's blocks are the first ``k`` of the
harder one) and reset randomness is seed-prefix safe (``ScatterGrid``), so an
easier task's rollout is embedded in the hardest task's recording. Each source
episode is relabeled into a target task WITHOUT re-simulating:

- truncate the trailing blocks from ``observation.state``/``qpos``/``qvel``,
- recompute the target's ``progress``/``success`` reward from the stored state
  (``reward = step_dt * (progress + gate)``; ``gate = target-reached & at-rest &
  gripper-open``), reusing the env's own success kernels,
- cut the episode at the first frame the target success gate holds (the env's
  success termination fires there).

Only the gripper-open half of the release gate is available offline (the
finger-pad contact force is not recorded); on recorded demos this reproduces
the env's per-frame reward to <1e-4 (see the source self-check below), so the
contact refinement is redundant in practice.

Usage:
    uv run python g3a_bench/scripts/live_bridge/split_demos.py \\
        --source-dir dataset/demos/G3A-Stack-High-Dk1_uni/<stamp> \\
        --source-task G3A-Stack-High-Dk1_uni \\
        --output-dir dataset/demos_split            # all easier same-family tasks
    # or an explicit subset:
        --target-task G3A-Stack-Low-Dk1_uni --target-task G3A-Stack-Medium-Dk1_uni
"""

from __future__ import annotations

import argparse
import glob
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from g3a_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from g3a_bench.scripts.live_bridge.dataset_extra_info import (
    load_extra_info,
    write_extra_info,
)
from g3a_bench.tasks.common.success import OBJECT_VEL_THRESHOLD
from g3a_bench.tasks.stack.config import _task_id_parts
from g3a_bench.tasks.stack.mdp.success import (
    block_top_heights,
    stack_progress,
    stack_reached,
    two_stacks_progress,
    two_stacks_success,
)
from g3a_bench.tasks.stack.variants import (
    FORM_HALF_EXTENTS,
    MIN_STACK_SEPARATION,
    StackVariantCfg,
)

# The reward stored per frame is step_dt * sum(progress, success). If either
# were wrong the per-episode source self-check would fail
_STEP_DT = 0.005 * 4
_ROBOT_GRIPPER_LEFT_QPOS_IDX = 6  # index of gripper_left within the robot qpos block
_OBS_TAIL = 7  # last_action; stack has command_obs=False so nothing else trails
# Mean per-frame reward residual allowed in the source self-check
_REWARD_SELF_CHECK_MEAN_TOL = 1e-4
# A dropped block counts as "disturbed" (the demo may have used it) if it moves
# more than this from its spawn pose before the target-success cut.
_DROPPED_XY_TOL = 0.02
_DROPPED_Z_TOL = 0.02

_GRIPPER_OPEN_THRESHOLD = get_unimanual_dk1_interface().gripper_open_threshold


# --------------------------------------------------------------------------- #
# Task-id <-> variant resolution
# --------------------------------------------------------------------------- #


def _variant_index() -> dict[tuple[str, str], StackVariantCfg]:
    """(task, variation) -> variant, e.g. ("Low", "") / ("High", "TwoStacks")."""
    from g3a_bench.tasks.stack.variants import STACK_VARIANTS

    return {_task_id_parts(v): v for v in STACK_VARIANTS.values()}


def parse_stack_task_id(task_id: str) -> tuple[str, str, str]:
    """Split ``G3A-Stack-<Task>[-<Variation>]-<Robot>`` -> (task, variation, robot)."""
    parts = task_id.split("-")
    if len(parts) < 4 or parts[0] != "G3A" or parts[1] != "Stack":
        raise ValueError(f"not a G3A-Stack task id: {task_id!r}")
    task = parts[2]
    robot = parts[-1]
    variation = "-".join(parts[3:-1])  # "" for the base family
    return task, variation, robot


def variant_from_task_id(task_id: str) -> StackVariantCfg:
    task, variation, _ = parse_stack_task_id(task_id)
    index = _variant_index()
    if (task, variation) not in index:
        raise ValueError(f"unknown Stack variant for task id {task_id!r}")
    return index[(task, variation)]


def robot_from_task_id(task_id: str) -> str:
    return parse_stack_task_id(task_id)[2]


def easier_target_task_ids(source_task_id: str) -> list[str]:
    """All registered same-family Stack tasks with a lower target height."""
    src = variant_from_task_id(source_task_id)
    _, src_variation, robot = parse_stack_task_id(source_task_id)
    out = []
    for (task, variation), v in _variant_index().items():
        if variation == src_variation and v.target_blocks < src.target_blocks:
            suffix = f"-{variation}" if variation else ""
            out.append(f"G3A-Stack-{task}{suffix}-{robot}")
    # Hardest-first so the largest cut is listed first.
    out.sort(key=lambda t: variant_from_task_id(t).target_blocks, reverse=True)
    return out


# --------------------------------------------------------------------------- #
# Split plan + validation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SplitPlan:
    source: StackVariantCfg
    target: StackVariantCfg

    @property
    def n_src(self) -> int:
        return len(self.source.blocks)

    @property
    def k(self) -> int:
        return len(self.target.blocks)

    def validate(self) -> None:
        s, t = self.source, self.target
        assert t.two_stacks == s.two_stacks, "cannot split across TwoStacks families"
        assert t.target_blocks < s.target_blocks, "target is not easier than source"
        assert self.k <= self.n_src, "target has more blocks than source"
        # Tail-extending: target blocks must be the source's block prefix.
        for i in range(self.k):
            assert t.blocks[i].form == s.blocks[i].form, (
                f"block {i} form differs ({t.blocks[i].form} vs {s.blocks[i].form}): "
                "not the same tail-extending family"
            )


def _half_heights(variant: StackVariantCfg) -> torch.Tensor:
    return torch.tensor(
        [FORM_HALF_EXTENTS[b.form][2] for b in variant.blocks], dtype=torch.float64
    )


# --------------------------------------------------------------------------- #
# Per-frame recompute (reuses the env's success kernels; env-local heights)
# --------------------------------------------------------------------------- #


def _progress_and_gate(
    block_local: torch.Tensor,  # (T, N, 7) env-local pos+quat
    block_speed: torch.Tensor,  # (T, N) linear speed
    gripper_open: torch.Tensor,  # (T,) bool
    variant: StackVariantCfg,
    half_h: torch.Tensor,  # (N,)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (progress in [0,1], success gate bool) for the given block set.

    Heights are env-local (local_z = world_z - env_origin_z, table_z = 0), which
    equals the env's height-above-table since the origin offset cancels.
    """
    pos_z = block_local[:, :, 2]
    xy = block_local[:, :, :2]
    top_z = block_top_heights(pos_z, half_h)
    target = variant.target_height
    tol = variant.height_tol
    reached = stack_reached(top_z, 0.0, target, tol)
    if variant.two_stacks:
        task_done = two_stacks_success(reached, xy, MIN_STACK_SEPARATION)
        progress = two_stacks_progress(top_z, xy, 0.0, target, MIN_STACK_SEPARATION)
    else:
        task_done = reached.any(dim=-1)
        progress = stack_progress(top_z, 0.0, target)
    at_rest = block_speed.amax(dim=-1) <= OBJECT_VEL_THRESHOLD
    gate = task_done & at_rest & gripper_open
    return progress, gate


@dataclass
class RelabeledEpisode:
    obs: np.ndarray
    robot_state: np.ndarray
    action: np.ndarray
    reward: np.ndarray  # (T,1) float32
    done: np.ndarray  # (T,1) bool
    qpos: np.ndarray
    qvel: np.ndarray
    n_dropped_disturbed: int  # dropped blocks that moved before the cut
    cut_frame: int  # index of the target-success frame (inclusive)


def relabel_episode(
    *,
    obs: np.ndarray,  # (T, D_src)
    qpos: np.ndarray,  # (T, Q)
    qvel: np.ndarray,  # (T, V)
    action: np.ndarray,  # (T, 7)
    robot_state: np.ndarray,  # (T, 7)
    reward: np.ndarray,  # (T,) recorded next.reward
    plan: SplitPlan,
) -> RelabeledEpisode | None:
    """Relabel one source episode into the target task, or None if the target
    success is never reached (episode cannot be split to this task)."""
    plan.validate()
    n_src, k = plan.n_src, plan.k
    proprio = obs.shape[1] - n_src * 7 - _OBS_TAIL
    assert proprio > 0, "unexpected obs layout"

    o = torch.from_numpy(obs.astype(np.float64))
    q = torch.from_numpy(qpos.astype(np.float64))
    v = torch.from_numpy(qvel.astype(np.float64))
    rew = torch.from_numpy(reward.astype(np.float64))

    # Object-major sections. obs: [proprio | blocks N*7 | actions]. qpos/qvel:
    # [blocks first | robot last].
    blk = o[:, proprio : proprio + n_src * 7].reshape(-1, n_src, 7)
    bvel = v[:, : n_src * 6].reshape(-1, n_src, 6)
    speed = bvel[:, :, :3].norm(dim=-1)  # (T, N)
    grip_l = q[:, n_src * 7 + _ROBOT_GRIPPER_LEFT_QPOS_IDX]
    gripper_open = grip_l > _GRIPPER_OPEN_THRESHOLD

    # --- self-check: recomputed SOURCE reward must match the recording ---
    src_prog, src_gate = _progress_and_gate(
        blk, speed, gripper_open, plan.source, _half_heights(plan.source)
    )
    src_reward = _STEP_DT * (src_prog + src_gate.double())
    resid = (src_reward - rew).abs()
    mean_resid = resid.mean().item()
    assert mean_resid <= _REWARD_SELF_CHECK_MEAN_TOL, (
        f"source reward self-check failed (mean residual {mean_resid:.2e}, "
        f"max {resid.max().item():.2e}); obs/qpos layout, gripper index/threshold "
        "or step_dt is wrong"
    )

    # --- target relabel ---
    tgt_half = _half_heights(plan.target)
    tgt_prog, tgt_gate = _progress_and_gate(
        blk[:, :k], speed[:, :k], gripper_open, plan.target, tgt_half
    )
    if not bool(tgt_gate.any()):
        return None
    cut = int(torch.argmax(tgt_gate.int()).item())  # first success frame
    sl = slice(0, cut + 1)

    tgt_reward = (_STEP_DT * (tgt_prog[sl] + tgt_gate[sl].double())).numpy()
    done = np.zeros((cut + 1, 1), dtype=bool)
    done[-1, 0] = True

    # Structured truncation: obs keeps [proprio | first-k blocks | actions];
    # qpos/qvel keep [first-k blocks | robot].
    new_obs = np.concatenate(
        [
            obs[sl, :proprio],
            obs[sl, proprio : proprio + k * 7],
            obs[sl, proprio + n_src * 7 :],
        ],
        axis=1,
    ).astype(np.float32)
    new_qpos = np.concatenate(
        [qpos[sl, : k * 7], qpos[sl, n_src * 7 :]], axis=1
    ).astype(np.float32)
    new_qvel = np.concatenate(
        [qvel[sl, : k * 6], qvel[sl, n_src * 6 :]], axis=1
    ).astype(np.float32)

    # --- dropped-block interference check (Ordered-collection sanity) ---
    n_disturbed = 0
    if k < n_src:
        dropped = blk[sl, k:]  # (cut+1, n_src-k, 7)
        d_xy = (dropped[:, :, :2] - dropped[0:1, :, :2]).norm(dim=-1).amax(dim=0)
        d_z = (dropped[:, :, 2] - dropped[0:1, :, 2]).abs().amax(dim=0)
        n_disturbed = int(
            ((d_xy > _DROPPED_XY_TOL) | (d_z > _DROPPED_Z_TOL)).sum().item()
        )

    return RelabeledEpisode(
        obs=new_obs,
        robot_state=robot_state[sl].astype(np.float32),
        action=action[sl].astype(np.float32),
        reward=tgt_reward.reshape(-1, 1).astype(np.float32),
        done=done,
        qpos=new_qpos,
        qvel=new_qvel,
        n_dropped_disturbed=n_disturbed,
        cut_frame=cut,
    )


# --------------------------------------------------------------------------- #
# Dataset IO
# --------------------------------------------------------------------------- #


def _load_source_episodes(source_dir: str) -> list[tuple[int, pd.DataFrame]]:
    files = sorted(glob.glob(f"{source_dir}/data/chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {source_dir}/data")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    out: list[tuple[int, pd.DataFrame]] = []
    for ep, g in df.groupby("episode_index", sort=True):
        out.append((int(ep), g))  # type: ignore[arg-type]
    return out


def _col(g: pd.DataFrame, name: str) -> np.ndarray:
    return np.stack(list(g[name]))


def _features(obs_dim: int, qpos_dim: int, qvel_dim: int) -> dict:
    return {
        "observation.state": {"dtype": "float32", "shape": (obs_dim,), "names": None},
        "observation.robot_state": {"dtype": "float32", "shape": (7,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "success": {"dtype": "bool", "shape": (1,), "names": None},
        "observation.qpos": {"dtype": "float32", "shape": (qpos_dim,), "names": None},
        "observation.qvel": {"dtype": "float32", "shape": (qvel_dim,), "names": None},
    }


def split_to_task(
    source_episodes: list[tuple[int, pd.DataFrame]],
    source_extra: dict,
    plan: SplitPlan,
    target_task: str,
    out_root: str,
    fps: int = 50,
) -> dict:
    """Relabel every source episode into ``target_task`` and write a LeRobot
    dataset at ``out_root``. Returns a summary dict."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src_seeds = source_extra.get("episode_seeds", {})
    dataset: LeRobotDataset | None = None
    new_seeds: dict[str, int] = {}
    written = 0
    skipped: list[int] = []
    disturbed_eps: list[int] = []
    total_cut, total_src = 0, 0

    for src_ep, g in source_episodes:
        obs = _col(g, "observation.state").astype(np.float32)
        qpos = _col(g, "observation.qpos").astype(np.float32)
        qvel = _col(g, "observation.qvel").astype(np.float32)
        action = _col(g, "action").astype(np.float32)
        robot_state = _col(g, "observation.robot_state").astype(np.float32)
        reward = np.asarray(g["next.reward"].values, dtype=np.float32).reshape(-1)

        rel = relabel_episode(
            obs=obs,
            qpos=qpos,
            qvel=qvel,
            action=action,
            robot_state=robot_state,
            reward=reward,
            plan=plan,
        )
        total_src += len(g)
        if rel is None:
            skipped.append(src_ep)
            continue
        if rel.n_dropped_disturbed:
            disturbed_eps.append(src_ep)
        total_cut += rel.cut_frame + 1

        if dataset is None:
            dataset = LeRobotDataset.create(
                target_task,
                fps=fps,
                features=_features(
                    rel.obs.shape[1], rel.qpos.shape[1], rel.qvel.shape[1]
                ),
                root=out_root,
            )
        label = np.array([True], dtype=bool)  # target success reached (cut point)
        for i in range(len(rel.obs)):
            dataset.add_frame(
                {
                    "observation.state": rel.obs[i],
                    "observation.robot_state": rel.robot_state[i],
                    "action": rel.action[i],
                    "next.reward": rel.reward[i],
                    "next.done": rel.done[i],
                    "success": label,
                    "observation.qpos": rel.qpos[i],
                    "observation.qvel": rel.qvel[i],
                    "task": target_task,
                }
            )
        dataset.save_episode()
        # Seed carries over unchanged: seed-prefix reset reproduces the first-k
        # block layout in the target env.
        if str(src_ep) in src_seeds:
            new_seeds[str(written)] = int(src_seeds[str(src_ep)])
        written += 1

    if dataset is not None:
        extra = {
            k: source_extra[k]
            for k in ("gravity_compensate", "seed_device", "operator_id")
            if k in source_extra
        }
        if new_seeds:
            extra["episode_seeds"] = new_seeds
        extra["split_from"] = plan.source.name
        write_extra_info(dataset.meta.root, extra)
        dataset.finalize()

    return {
        "target_task": target_task,
        "written": written,
        "skipped": skipped,
        "disturbed_eps": disturbed_eps,
        "mean_cut_frames": (total_cut / written) if written else 0,
        "source_frames": total_src,
    }


def split_dataset(
    source_dir: str,
    source_task: str,
    output_dir: str,
    target_tasks: list[str] | None = None,
    stamp: str | None = None,
) -> list[dict]:
    source_variant = variant_from_task_id(source_task)
    targets = target_tasks or easier_target_task_ids(source_task)
    if not targets:
        raise ValueError(
            f"{source_task} is already the easiest task in its family — nothing to "
            "split. Use a harder source (e.g. a Medium or High recording)."
        )
    source_episodes = _load_source_episodes(source_dir)
    source_extra = load_extra_info(source_dir)
    stamp = stamp or time.strftime("%Y-%m-%d_%H-%M-%S")

    summaries = []
    for target_task in targets:
        plan = SplitPlan(source_variant, variant_from_task_id(target_task))
        out_root = str(Path(output_dir) / target_task / stamp)
        summary = split_to_task(
            source_episodes, source_extra, plan, target_task, out_root
        )
        summary["out_root"] = out_root
        summaries.append(summary)
    return summaries


def _print_summary(source_task: str, summaries: list[dict]) -> None:
    print(f"\n[split] source: {source_task}")
    for s in summaries:
        print(
            f"  -> {s['target_task']}: wrote {s['written']} ep "
            f"(mean {s['mean_cut_frames']:.0f} frames/ep), "
            f"skipped {len(s['skipped'])} (no target success)"
        )
        if s["disturbed_eps"]:
            print(
                f"     WARN dropped-block movement in eps {s['disturbed_eps']} "
                "(a non-prefix block may have been used; check collection order)"
            )
        print(f"     {s['out_root']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", required=True, help="source LeRobot dataset root")
    p.add_argument("--source-task", required=True, help="e.g. G3A-Stack-High-Dk1_uni")
    p.add_argument(
        "--target-task",
        action="append",
        default=None,
        help="easier target task id (repeatable); default = all easier same-family",
    )
    p.add_argument("--output-dir", required=True, help="output root for split datasets")
    args = p.parse_args()

    # Expected user errors (wrong/easiest source, non-easier target) print a
    # clean message, not a traceback.
    try:
        summaries = split_dataset(
            source_dir=args.source_dir,
            source_task=args.source_task,
            output_dir=args.output_dir,
            target_tasks=args.target_task,
        )
    except (ValueError, AssertionError) as e:
        raise SystemExit(f"[split] {e}") from None
    _print_summary(args.source_task, summaries)


if __name__ == "__main__":
    main()
