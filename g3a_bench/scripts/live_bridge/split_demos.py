# pyright: reportAttributeAccessIssue=false
"""Split a recorded demo dataset into easier same-family tasks (offline relabel).

Supported categories: **Stack** (``Stack3``/``Stack4``/``Stack5``),
**Domino** (base, RandomRow and DifferentDominoes; KnockOver is not
splittable — its topple is a global chain over all pieces) and **Sorter** (all
spawn variations). Each tail-extends (an easier task's objects are the first
``k`` of the harder one), so an easier task's rollout is embedded in the
hardest task's recording. Splits stay inside one family: Stack blocks only
tail-extend within one difficulty (the block shape differs between Easy,
Medium and Hard), the base Domino tasks use the fixed row and ``RandomRow``
the sampled one, so their recordings are never mixed. Each source episode is
relabeled WITHOUT re-simulating:

- truncate the trailing objects from every object-major slice of
  ``observation.state`` and from ``qpos``/``qvel``,
- recompute the target's ``progress``/``success`` reward from the stored state
  (``reward = step_dt * (progress + gate)``; ``gate = task-done & at-rest &
  gripper-open``), reusing the env's own success kernels,
- cut the episode at the first frame the target success gate holds.

Only the gripper-open half of the release gate is available offline (the
finger-pad contact force is not recorded); on recorded Stack demos this
reproduces the env reward to <1e-4. A per-episode source self-check asserts the
recomputed source reward matches the recording, so a wrong obs/qpos layout,
gripper index/threshold or step_dt is caught immediately (it also validates a
category the first time real demos of it are split).

Usage:
    uv run python g3a_bench/scripts/live_bridge/split_demos.py \\
        --source-dir dataset/demos/G3A-Stack5-Medium-Dk1_uni/<stamp> \\
        --source-task G3A-Stack5-Medium-Dk1_uni \\
        --output-dir dataset/demos_split            # all easier same-family tasks
    # or an explicit subset:
        --target-task G3A-Domino-2Pieces-Dk1_uni
"""

from __future__ import annotations

import argparse
import glob
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from g3a_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from g3a_bench.scripts.live_bridge.dataset_extra_info import (
    load_extra_info,
    write_extra_info,
)
from g3a_bench.tasks.common.success import OBJECT_VEL_THRESHOLD
from g3a_bench.tasks.domino.mdp.success import domino_upright_aligned
from g3a_bench.tasks.stack.mdp.success import (
    block_top_heights,
    stack_progress,
    stack_reached,
)
from g3a_bench.tasks.stack.variants import TARGET_BLOCKS

# Env sim constants (tasks/common/env_cfg.py: timestep=0.005, decimation=4).
# reward stored per frame is step_dt * (progress + success); a wrong value here
# fails the per-episode source self-check.
_STEP_DT = 0.005 * 4
_ROBOT_GRIPPER_LEFT_QPOS_IDX = 6  # index of gripper_left within the robot qpos block
_OBS_TAIL = 7  # trailing last_action, shared by every category
# Mean per-frame reward residual allowed in the source self-check. A wrong
# layout/gripper-index/step_dt corrupts ~every frame (mean ~ 0.01-2.0); only
# rare gripper-threshold float ties (a `>` compare that can't be reproduced
# bit-exactly offline) slip through, and they barely move the mean.
_REWARD_SELF_CHECK_MEAN_TOL = 1e-4
# A dropped object counts as "disturbed" (the demo may have used it) if it moves
# more than this from its spawn pose before the target-success cut.
_DROPPED_XY_TOL = 0.02
_DROPPED_Z_TOL = 0.02
# Domino placement tolerances (DominoCommandCfg defaults; not overridden by the
# env cfg).
_DOMINO_POS_THRESHOLD = 0.015
_DOMINO_UPRIGHT_DEG = 15.0

_GRIPPER_OPEN_THRESHOLD = get_unimanual_dk1_interface().gripper_open_threshold


# --------------------------------------------------------------------------- #
# Category adapters: everything that differs between Stack, Domino and Sorter.
# --------------------------------------------------------------------------- #


def _stack_progress_and_gate(
    variant: Any,
    sections: list[torch.Tensor],
    speed: torch.Tensor,
    grip_open: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack: height-only success over the block-pose section."""
    blk = sections[0]  # (T, k, 7) env-local pos+quat
    half_h = torch.tensor(
        [b.half_extents[2] for b in variant.blocks], dtype=torch.float64
    )
    top_z = block_top_heights(blk[:, :, 2], half_h)
    target, tol = variant.target_height, variant.height_tol
    task_done = stack_reached(top_z, 0.0, target, tol).any(dim=-1)
    progress = stack_progress(top_z, 0.0, target)
    at_rest = speed.amax(dim=-1) <= OBJECT_VEL_THRESHOLD
    return progress, task_done & at_rest & grip_open


def _domino_progress_and_gate(
    variant: Any,
    sections: list[torch.Tensor],
    speed: torch.Tensor,
    grip_open: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Domino (base/DifferentDominoes): every piece placed at its row slot.

    The per-slot target pose lives in the command obs section; the row yaw is
    recovered from its (shared) upright/row-aligned quaternion.
    """
    poses = sections[0]  # (T, k, 7) piece pose env-local
    command = sections[2]  # (T, k, 7) per-slot target pose env-local
    dom_pos, dom_quat = poses[:, :, :3], poses[:, :, 3:7]
    slot_pos, slot_quat = command[:, :, :3], command[:, :, 3:7]
    # slot_quat = quat_from_euler_xyz(0, 0, row_yaw) = [cos(y/2), 0, 0, sin(y/2)].
    row_yaw = 2.0 * torch.atan2(slot_quat[:, 0, 3], slot_quat[:, 0, 0])  # (T,)
    margin = variant.margin
    error = (slot_pos - dom_pos).norm(dim=-1)  # (T, k)
    at_goal = (error < _DOMINO_POS_THRESHOLD * margin) & domino_upright_aligned(
        dom_quat, row_yaw, _DOMINO_UPRIGHT_DEG * margin
    )
    task_done = at_goal.all(dim=-1)
    progress = at_goal.double().mean(dim=-1)
    at_rest = speed.amax(dim=-1) <= OBJECT_VEL_THRESHOLD
    return progress, task_done & at_rest & grip_open


def _sorter_progress_and_gate(
    variant: Any,
    sections: list[torch.Tensor],
    speed: torch.Tensor,
    grip_open: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shape sorter: fraction of pieces latched into their own hole.

    The env stores the persistent per-socket latch as the ``socket_inserted``
    obs slice, and a piece latches into its own hole independently of the
    others, so the first-k latch state is exactly the target task's. There is
    no at-rest requirement (the env's gate uses vel_threshold=inf); success is
    every piece latched and the gripper released.
    """
    latched = sections[1][:, :, 0] > 0.5  # (T, k) from socket_inserted
    progress = latched.double().mean(dim=-1)
    task_done = latched.all(dim=-1)
    return progress, task_done & grip_open


@dataclass(frozen=True)
class CategoryAdapter:
    categories: tuple[str, ...]
    """Category segments this adapter owns; Stack spans one per target height."""
    obs_section_widths: tuple[int, ...]
    """Per-object width of each object-major obs slice, in obs order.
    Stack: (7,) = block poses. Domino: (7, 1, 7) = piece poses, slot_placed,
    per-slot command."""
    variants: dict[str, Any]
    id_parts: Callable[[Any], tuple[str, str]]  # variant -> (task, variation)
    category_of: Callable[[Any], str]  # variant -> category segment
    family: Callable[[Any], Any]
    """Split family key: only variants of one family tail-extend each other
    (Stack: the block shape; Domino/Sorter: the variation)."""
    n_objects: Callable[[Any], int]
    object_signatures: Callable[
        [Any], tuple
    ]  # per-object identity for the prefix check
    splittable: Callable[[Any], bool]
    progress_and_gate: Callable[
        [Any, list[torch.Tensor], torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]
    qpos_fixed_bodies: int = 0
    """Free bodies kept verbatim before the truncatable objects in qpos/qvel.
    Sorter = 1 (the cube, also carried in the obs head as cube_pose)."""

    def task_id_middle(self, variant: Any) -> str:
        """The ``<Category>-<Task>[-<Variation>]`` part of the task id."""
        task, variation = self.id_parts(variant)
        return "-".join(p for p in (self.category_of(variant), task, variation) if p)

    def variant_index(self) -> dict[str, Any]:
        return {self.task_id_middle(v): v for v in self.variants.values()}


def _make_stack_adapter() -> CategoryAdapter:
    from g3a_bench.tasks.stack.config import _category, _task_id_parts
    from g3a_bench.tasks.stack.variants import STACK_VARIANTS

    return CategoryAdapter(
        categories=_STACK_CATEGORIES,
        obs_section_widths=(7,),
        variants=STACK_VARIANTS,
        id_parts=_task_id_parts,
        category_of=_category,
        family=lambda v: v.difficulty,
        n_objects=lambda v: len(v.blocks),
        object_signatures=lambda v: tuple(b.difficulty for b in v.blocks),
        splittable=lambda v: True,
        progress_and_gate=_stack_progress_and_gate,
    )


def _make_domino_adapter() -> CategoryAdapter:
    from g3a_bench.tasks.domino.config import _task_id_parts
    from g3a_bench.tasks.domino.variants import DOMINO_VARIANTS

    return CategoryAdapter(
        categories=("Domino",),
        obs_section_widths=(7, 1, 7),
        variants=DOMINO_VARIANTS,
        id_parts=_task_id_parts,
        category_of=lambda v: "Domino",
        family=lambda v: _task_id_parts(v)[1],
        n_objects=lambda v: v.num_dominoes,
        object_signatures=lambda v: v.half_extents,
        splittable=lambda v: not v.knock_over,
        progress_and_gate=_domino_progress_and_gate,
    )


def _make_sorter_adapter() -> CategoryAdapter:
    from g3a_bench.tasks.shape_sorter.config import _task_id_parts
    from g3a_bench.tasks.shape_sorter.variants import SHAPE_SORTER_VARIANTS

    return CategoryAdapter(
        categories=("Sorter",),
        # obs = [proprio | cube_pose 7 (rides in the head) | piece_poses N*7 |
        # socket_inserted N | actions]; command_obs=False.
        obs_section_widths=(7, 1),
        variants=SHAPE_SORTER_VARIANTS,
        id_parts=_task_id_parts,
        category_of=lambda v: "Sorter",
        family=lambda v: _task_id_parts(v)[1],
        n_objects=lambda v: v.num_pieces,
        object_signatures=lambda v: v.shapes,
        splittable=lambda v: True,
        progress_and_gate=_sorter_progress_and_gate,
        qpos_fixed_bodies=1,  # the cube, before the pieces
    )


_STACK_CATEGORIES = tuple(f"Stack{n}" for n in TARGET_BLOCKS)

_ADAPTERS: dict[str, Callable[[], CategoryAdapter]] = {
    **dict.fromkeys(_STACK_CATEGORIES, _make_stack_adapter),
    "Domino": _make_domino_adapter,
    "Sorter": _make_sorter_adapter,
}


# --------------------------------------------------------------------------- #
# Task-id <-> (adapter, variant) resolution
# --------------------------------------------------------------------------- #


def parse_task_id(task_id: str) -> tuple[str, str, str, str]:
    """Split ``G3A-<Category>-<Task>[-<Variation>]-<Robot>`` into its parts."""
    parts = task_id.split("-")
    if len(parts) < 4 or parts[0] != "G3A":
        raise ValueError(f"not a G3A task id: {task_id!r}")
    category, task, robot = parts[1], parts[2], parts[-1]
    variation = "-".join(parts[3:-1])  # "" for the base family
    if category not in _ADAPTERS:
        raise ValueError(f"category {category!r} is not splittable")
    return category, task, variation, robot


def adapter_for(task_id: str) -> CategoryAdapter:
    return _ADAPTERS[parse_task_id(task_id)[0]]()


def variant_from_task_id(task_id: str) -> Any:
    category, task, variation, _ = parse_task_id(task_id)
    middle = "-".join(p for p in (category, task, variation) if p)
    index = adapter_for(task_id).variant_index()
    if middle not in index:
        raise ValueError(f"unknown variant for task id {task_id!r}")
    return index[middle]


def easier_target_task_ids(source_task_id: str) -> list[str]:
    """All registered same-family, splittable tasks with fewer objects."""
    robot = parse_task_id(source_task_id)[3]
    adapter = adapter_for(source_task_id)
    src = variant_from_task_id(source_task_id)
    if not adapter.splittable(src):
        return []
    n_src = adapter.n_objects(src)
    out = [
        f"G3A-{middle}-{robot}"
        for middle, v in adapter.variant_index().items()
        if adapter.family(v) == adapter.family(src)
        and adapter.splittable(v)
        and adapter.n_objects(v) < n_src
    ]
    out.sort(key=lambda t: adapter.n_objects(variant_from_task_id(t)), reverse=True)
    return out


# --------------------------------------------------------------------------- #
# Split plan + validation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SplitPlan:
    adapter: CategoryAdapter
    source: Any
    target: Any

    @property
    def n_src(self) -> int:
        return self.adapter.n_objects(self.source)

    @property
    def k(self) -> int:
        return self.adapter.n_objects(self.target)

    def validate(self) -> None:
        a = self.adapter
        assert a.family(self.source) == a.family(self.target), (
            "source and target are not the same family"
        )
        assert a.splittable(self.source) and a.splittable(self.target), (
            "variant is not splittable (e.g. Domino KnockOver)"
        )
        assert self.k < self.n_src, "target is not easier than source"
        # Tail-extending: target objects must be the source's object prefix.
        assert (
            a.object_signatures(self.target)
            == a.object_signatures(self.source)[: self.k]
        ), "target objects are not the source's prefix (not the same family)"


def plan_for(source_task_id: str, target_task_id: str) -> SplitPlan:
    return SplitPlan(
        adapter_for(source_task_id),
        variant_from_task_id(source_task_id),
        variant_from_task_id(target_task_id),
    )


# --------------------------------------------------------------------------- #
# Per-frame recompute (reuses the env's success kernels; env-local heights)
# --------------------------------------------------------------------------- #


def _obs_sections(
    obs: torch.Tensor, proprio: int, widths: tuple[int, ...], n_layout: int, take: int
) -> list[torch.Tensor]:
    """Extract the first ``take`` objects of each object-major obs slice.

    ``n_layout`` is the object count the obs is stored with (section spacing);
    ``take`` (<= n_layout) is how many objects to return per slice.
    """
    t = obs.shape[0]
    off = proprio
    out = []
    for w in widths:
        out.append(obs[:, off : off + take * w].reshape(t, take, w))
        off += n_layout * w
    return out


@dataclass
class RelabeledEpisode:
    obs: np.ndarray
    robot_state: np.ndarray
    action: np.ndarray
    reward: np.ndarray  # (T,1) float32
    done: np.ndarray  # (T,1) bool
    qpos: np.ndarray
    qvel: np.ndarray
    n_dropped_disturbed: int
    cut_frame: int


def relabel_episode(
    *,
    obs: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    action: np.ndarray,
    robot_state: np.ndarray,
    reward: np.ndarray,
    plan: SplitPlan,
) -> RelabeledEpisode | None:
    """Relabel one source episode into the target task, or None if the target
    success is never reached (episode cannot be split to this task)."""
    plan.validate()
    a = plan.adapter
    n_src, k = plan.n_src, plan.k
    widths = a.obs_section_widths
    proprio = obs.shape[1] - n_src * sum(widths) - _OBS_TAIL
    assert proprio > 0, "unexpected obs layout"

    o = torch.from_numpy(obs.astype(np.float64))
    q = torch.from_numpy(qpos.astype(np.float64))
    v = torch.from_numpy(qvel.astype(np.float64))
    rew = torch.from_numpy(reward.astype(np.float64))

    # qpos/qvel layout: [fixed leading free bodies | objects | robot], each free
    # body 7 qpos / 6 qvel. Sorter has one fixed body (the cube) before the
    # truncatable pieces; Stack/Domino have none.
    f = a.qpos_fixed_bodies
    speed_all = (
        v[:, f * 6 : (f + n_src) * 6].reshape(-1, n_src, 6)[:, :, :3].norm(dim=-1)
    )  # (T, N)
    grip_l = q[:, (f + n_src) * 7 + _ROBOT_GRIPPER_LEFT_QPOS_IDX]
    grip_open = grip_l > _GRIPPER_OPEN_THRESHOLD

    # --- self-check: recomputed SOURCE reward must match the recording ---
    src_sections = _obs_sections(o, proprio, widths, n_src, n_src)
    src_prog, src_gate = a.progress_and_gate(
        plan.source, src_sections, speed_all, grip_open
    )
    resid = (_STEP_DT * (src_prog + src_gate.double()) - rew).abs()
    mean_resid = resid.mean().item()
    assert mean_resid <= _REWARD_SELF_CHECK_MEAN_TOL, (
        f"source reward self-check failed (mean residual {mean_resid:.2e}, "
        f"max {resid.max().item():.2e}); obs/qpos layout, gripper index/threshold "
        "or step_dt is wrong"
    )

    # --- target relabel ---
    tgt_sections = _obs_sections(o, proprio, widths, n_src, k)
    tgt_prog, tgt_gate = a.progress_and_gate(
        plan.target, tgt_sections, speed_all[:, :k], grip_open
    )
    if not bool(tgt_gate.any()):
        return None
    cut = int(torch.argmax(tgt_gate.int()).item())  # first success frame
    sl = slice(0, cut + 1)

    tgt_reward = (_STEP_DT * (tgt_prog[sl] + tgt_gate[sl].double())).numpy()
    done = np.zeros((cut + 1, 1), dtype=bool)
    done[-1, 0] = True

    # Structured truncation: obs keeps [proprio | first-k of each section | tail];
    # qpos/qvel keep [first-k objects | robot].
    obs_parts = [obs[sl, :proprio]]
    off = proprio
    for w in widths:
        obs_parts.append(obs[sl, off : off + k * w])
        off += n_src * w
    obs_parts.append(obs[sl, off:])  # tail (off is now the tail start)
    new_obs = np.concatenate(obs_parts, axis=1).astype(np.float32)
    # Keep [fixed bodies + first-k objects | robot], drop the trailing objects.
    new_qpos = np.concatenate(
        [qpos[sl, : (f + k) * 7], qpos[sl, (f + n_src) * 7 :]], axis=1
    ).astype(np.float32)
    new_qvel = np.concatenate(
        [qvel[sl, : (f + k) * 6], qvel[sl, (f + n_src) * 6 :]], axis=1
    ).astype(np.float32)

    # --- dropped-object interference check (Ordered-collection sanity) ---
    n_disturbed = 0
    if k < n_src:
        dropped = src_sections[0][sl, k:]  # (cut+1, n_src-k, 7) pose section
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
        rel = relabel_episode(
            obs=_col(g, "observation.state").astype(np.float32),
            qpos=_col(g, "observation.qpos").astype(np.float32),
            qvel=_col(g, "observation.qvel").astype(np.float32),
            action=_col(g, "action").astype(np.float32),
            robot_state=_col(g, "observation.robot_state").astype(np.float32),
            reward=np.asarray(g["next.reward"].values, dtype=np.float32).reshape(-1),
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
        # objects in the target env.
        if str(src_ep) in src_seeds:
            new_seeds[str(written)] = int(src_seeds[str(src_ep)])
        written += 1

    if dataset is not None:
        extra = {
            key: source_extra[key]
            for key in ("gravity_compensate", "seed_device", "operator_id")
            if key in source_extra
        }
        if new_seeds:
            extra["episode_seeds"] = new_seeds
        extra["split_from"] = plan.adapter.task_id_middle(plan.source)
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
    adapter = adapter_for(source_task)
    source_variant = variant_from_task_id(source_task)
    if not adapter.splittable(source_variant):
        raise ValueError(
            f"{source_task} is not splittable (e.g. Domino KnockOver: the topple "
            "is a global chain over all pieces)."
        )
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
        plan = SplitPlan(adapter, source_variant, variant_from_task_id(target_task))
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
                f"     WARN dropped-object movement in eps {s['disturbed_eps']} "
                "(a non-prefix object may have been used; check collection order)"
            )
        print(f"     {s['out_root']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", required=True, help="source LeRobot dataset root")
    p.add_argument(
        "--source-task", required=True, help="e.g. G3A-Stack5-Medium-Dk1_uni"
    )
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
