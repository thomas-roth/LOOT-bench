# pyright: reportAttributeAccessIssue=false
"""Relabel a recorded demo into the easier same-family tasks, without a sim.

Every splittable category tail-extends: an easier task's objects are the first
``k`` of a harder one's, and the reset draws per object in index order, so an
easier task's rollout is already embedded in the harder task's recording. This
script cuts it out, turning one hard session into a dataset per easier task at
no extra teleop time.

Supported: **Stack** (``1Pi``/``2Pis``/``3Pis``, one family: the
colour tiers navy/teal/white are the cut boundaries), **Domino** (base,
``RandomRow``, ``DifferentDominoes``; ``KnockOver`` is not splittable, its
topple being one chain over all pieces) and **Sorter** (every spawn
variation; excluding 2Shapes, custom-recorded). Splits never cross families: the
base Domino row is fixed where ``RandomRow``'s is sampled.

Each source episode is relabeled without re-simulating:

- drop the trailing objects from every object-major slice of
  ``observation.state`` and from ``qpos``/``qvel``,
- recompute the target's reward from the stored state, through the same
  ``shaped_task_reward`` and success kernels the env uses,
- cut at the target's first success, the frame it is paid ``0`` on.

**Frame alignment.** A row is ``(obs_t, action_t, reward_t)`` with no shift:
the recorder reads the observation before ``env.step()``
(``recording_mixin.py``), which cancels ADR-006's env-internal reward lag.
Do not re-apply it here. ``qpos``/``qvel`` are the exception -- cloned
*after* the step, so row ``t``'s sim state gates row ``t + 1`` and is
shifted back one to line up with ``observation.state[t]``.

The env's real release gate survives offline: the finger-pad contact force
is a recorded observation term, so the relabel applies the env's own gate
straight out of the observation. A per-episode self-check recomputes the
*source* reward and asserts it matches the recording, so a wrong obs/qpos
layout, pad-force column, success kernel, or frame alignment fails loudly on
the first real run.

A source episode in which the target's success is never paid is not a demo
of the target and is dropped. That is a protocol break (a pi level built
from the wrong colour tier, a domino placed out of order), so the run
**fails**: every relabelable episode of every target is written first, then
one error lists the dropped episodes per target with a per-episode
diagnostic, and the process exits non-zero. ``--allow-skips`` demotes that
to a warning for a session whose loss is understood.

Usage:
    uv run loot-split-demos \\
        --source-dir dataset/demos/LOOT-Stack-3Pis-Dk1/<stamp> \\
        --source-task LOOT-Stack-3Pis-Dk1 \\
        --output-dir dataset/demos_split          # all easier same-family tasks
    # or an explicit subset:
        --target-task LOOT-Domino-2Pieces-Dk1
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from loot_bench.scripts.live_bridge.dataset_extra_info import (
    load_extra_info,
    write_extra_info,
)
from loot_bench.tasks.common.commands import MIN_GRIP_FORCE
from loot_bench.tasks.common.rewards import shaped_task_reward
from loot_bench.tasks.common.success import OBJECT_VEL_THRESHOLD
from loot_bench.tasks.domino.mdp.success import domino_upright_aligned
from loot_bench.tasks.stack.mdp.commands import GRIP_REACH
from loot_bench.tasks.stack.mdp.success import (
    block_is_horizontal,
    pi_levels_complete,
    pi_progress,
)
from loot_bench.tasks.stack.variants import BLOCK_LENGTH, BLOCKS_PER_LEVEL

_EE_POSE_WIDTH = 9
"""Width of ``ee_pose``, the second-to-last proprioceptive obs term (ADR-007)."""

_PAD_FORCE_WIDTH = 1
"""Width of ``pad_contact_force``, the last proprioceptive obs term."""

_REWARD_SELF_CHECK_MEAN_TOL = 1e-4
"""Mean per-frame reward residual allowed by the source self-check.

A wrong layout, gripper index or threshold corrupts nearly every frame (mean
residual 0.01 upwards). Only rare gripper-threshold float ties slip through,
and they barely move the mean."""

# A dropped object counts as disturbed (the demo may have used it) once it
# moves this far from its spawn pose before the target-success cut.
_DROPPED_XY_TOL = 0.02
_DROPPED_Z_TOL = 0.02

# DominoCommandCfg defaults; no env cfg overrides them.
_DOMINO_POS_THRESHOLD = 0.015
_DOMINO_UPRIGHT_DEG = 15.0


@dataclass(frozen=True)
class FrameState:
    """One episode's per-frame state, narrowed to a task's first ``k`` objects.

    ``sections`` holds those objects' slice of each object-major obs term,
    ``(T, k, width)`` each, in obs order.
    """

    sections: list[torch.Tensor]
    speed: torch.Tensor
    """(T, k) object linear speed."""
    released: torch.Tensor
    """(T,) pads force-free: the recorded pad contact force at or below
    ``MIN_GRIP_FORCE``.

    The env's real release condition, read straight out of the observation
    rather than reconstructed from the gripper joints."""
    grasp_pos: torch.Tensor
    """(T, 3) grasp-site position, env-local -- the obs poses' frame."""


# --------------------------------------------------------------------------- #
# Category adapters: everything that differs between Stack, Domino and Sorter.
# --------------------------------------------------------------------------- #


def _pi_levels(
    variant: Any, blocks: torch.Tensor, state: FrameState, levels: int
) -> torch.Tensor:
    """(T, levels) per-level lintel test over the first ``levels`` levels.

    Env-local poses with the table at ``0.0``, the held block masked out
    (ADR-017), through the env's own kernel so the tilt gate is never copied.
    """
    level_heights = torch.tensor(
        variant.level_centre_heights[:levels], dtype=blocks.dtype
    )
    return pi_levels_complete(
        blocks[:, :, 2],
        blocks[:, :, 3:7],
        0.0,
        level_heights,
        variant.height_tol,
        exclude=_held_block_mask(blocks, state),
    )


def _pi_progress_and_gate(
    variant: Any, state: FrameState
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack (pi): a horizontal, unheld lintel at every level's height."""
    blocks = state.sections[0]  # (T, k, 7) env-local pos+quat
    complete = _pi_levels(variant, blocks, state, variant.levels)
    task_done = complete.all(dim=-1)
    at_rest = state.speed.amax(dim=-1) <= OBJECT_VEL_THRESHOLD
    return pi_progress(complete), task_done & at_rest & state.released


def _pi_skip_diagnostic(variant: Any, state: FrameState) -> str:
    """Which blocks formed the source's first level 1, if it ever stood.

    A pi relabel fails when level 1 was built from a later colour tier:
    the target keeps only the first-``k`` blocks, so a white lintel is cut
    out of the observation and the level it made never counts. Naming the
    blocks that actually formed level 1 turns the failure into "block 8 was
    the lintel", which the operator can act on.
    """
    blocks = state.sections[0]  # (T, n_src, 7)
    level_1 = _pi_levels(variant, blocks, state, levels=1)[:, 0]  # (T,)
    if not bool(level_1.any()):
        return "the source never completed level 1 either"
    frame = int(torch.argmax(level_1.int()).item())
    poses = blocks[frame]
    horizontal = block_is_horizontal(poses[:, 3:7])
    at_level = (poses[:, 2] - variant.level_centre_height(1)).abs() <= (
        variant.height_tol
    )
    held = _held_block_mask(blocks, state)[frame]
    lintels = (horizontal & at_level & ~held).nonzero().reshape(-1)
    # Supports are the standing blocks under the lintel's footprint: a lintel
    # overhangs at most half its length on either side of its centre.
    lintel_xy = poses[lintels, :2]  # (n_lintels, 2)
    under = (poses[:, None, :2] - lintel_xy[None]).norm(dim=-1) <= BLOCK_LENGTH / 2
    supports = (~horizontal & under.any(dim=-1)).nonzero().reshape(-1)
    return (
        f"level 1 first stood at frame {frame} with lintel block "
        f"{lintels.tolist()} on standing blocks {supports.tolist()}"
    )


def _generic_skip_diagnostic(variant: Any, state: FrameState) -> str:
    del variant, state
    return "the target's first-k objects were never all in place"


def _held_block_mask(blocks: torch.Tensor, state: FrameState) -> torch.Tensor:
    """(T, k) bool marking the block in the gripper, as ``PiCommand`` does.

    A held block earns no progress, so the offline reward has to find the same
    one: the nearest block within ``GRIP_REACH`` while the pads report grip
    force.
    """
    dist = (blocks[:, :, :3] - state.grasp_pos.unsqueeze(1)).norm(dim=-1)  # (T, k)
    nearest = dist.argmin(dim=-1)
    frame = torch.arange(dist.shape[0])
    held = ~state.released & (dist[frame, nearest] <= GRIP_REACH)
    mask = torch.zeros_like(dist, dtype=torch.bool)
    mask[frame, nearest] = held
    return mask


def _domino_progress_and_gate(
    variant: Any, state: FrameState
) -> tuple[torch.Tensor, torch.Tensor]:
    """Domino: every piece placed on its row slot.

    The per-slot target pose is the command obs section; the row yaw comes back
    out of its (shared) upright, row-aligned quaternion.
    """
    poses, command = state.sections[0], state.sections[2]
    piece_pos, piece_quat = poses[:, :, :3], poses[:, :, 3:7]
    slot_pos, slot_quat = command[:, :, :3], command[:, :, 3:7]
    # slot_quat = quat_from_euler_xyz(0, 0, row_yaw) = [cos(y/2), 0, 0, sin(y/2)].
    row_yaw = 2.0 * torch.atan2(slot_quat[:, 0, 3], slot_quat[:, 0, 0])  # (T,)
    margin = variant.margin
    at_goal = (slot_pos - piece_pos).norm(dim=-1) < _DOMINO_POS_THRESHOLD * margin
    at_goal &= domino_upright_aligned(piece_quat, row_yaw, _DOMINO_UPRIGHT_DEG * margin)
    at_rest = state.speed.amax(dim=-1) <= OBJECT_VEL_THRESHOLD
    gate = at_goal.all(dim=-1) & at_rest & state.released
    return at_goal.double().mean(dim=-1), gate


def _sorter_progress_and_gate(
    variant: Any, state: FrameState
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shape sorter: fraction of pieces latched into their own hole.

    A piece latches independently of the others, so the first-k flags of the
    persistent ``socket_inserted`` slice are exactly the target task's. The
    env's gate has no at-rest term (``vel_threshold=inf`` in
    ``ShapeSorterCommand``): the socket holds a latched piece, so (unlike the
    domino adapter above) no ``OBJECT_VEL_THRESHOLD`` check applies. Release
    still gates, read off the recorded pad force like every category.
    """
    del variant  # every variant scores alike; only the piece count differs
    latched = state.sections[1][:, :, 0] > 0.5  # (T, k)
    return latched.double().mean(dim=-1), latched.all(dim=-1) & state.released


@dataclass(frozen=True)
class CategoryAdapter:
    categories: tuple[str, ...]
    """Category segments this adapter owns."""
    obs_section_widths: tuple[int, ...]
    """Per-object width of each object-major obs slice, in obs order.

    Stack (7,) = block poses. Domino (7, 1, 7) = piece poses, slot_placed,
    per-slot command. Sorter (7, 1) = piece poses, socket_inserted; its
    fixed-width cube_pose rides in the proprioceptive head."""
    variants: dict[str, Any]
    id_parts: Callable[[Any], tuple[str, str]]
    category_of: Callable[[Any], str]
    family: Callable[[Any], Any]
    """Split family key: only variants of one family tail-extend each other
    (Stack is one family, Domino/Sorter one per variation)."""
    n_objects: Callable[[Any], int]
    object_signatures: Callable[[Any], tuple]
    """Per-object identity, compared prefix-wise to reject a cross-family plan."""
    splittable: Callable[[Any], bool]
    progress_and_gate: Callable[[Any, FrameState], tuple[torch.Tensor, torch.Tensor]]
    skip_cause: str
    """Why a source episode never pays a target's success: the recording
    protocol the operator has to follow for the relabel to work."""
    skip_diagnostic: Callable[[Any, FrameState], str] = _generic_skip_diagnostic
    """Per-episode evidence for a skip, from the *source* variant and the
    source episode's full state (all ``n_src`` objects)."""
    qpos_fixed_bodies: int = 0
    """Free bodies before the truncatable objects in qpos/qvel, kept verbatim.
    Sorter has 1 (the cube, which also rides in the obs head)."""

    def task_id_middle(self, variant: Any) -> str:
        """The ``<Category>-<Task>[-<Variation>]`` part of a task id."""
        task, variation = self.id_parts(variant)
        return "-".join(p for p in (self.category_of(variant), task, variation) if p)

    def variant_index(self) -> dict[str, Any]:
        return {self.task_id_middle(v): v for v in self.variants.values()}


_ORDER_CAUSE = (
    "the target's first-k objects were never all placed/inserted before the "
    "recording ended; check the collection order"
)


def _pi_skip_cause(variants: dict[str, Any]) -> str:
    """The tier protocol, spelled from the hardest variant's block colours so
    the message cannot drift from ``order_color``."""
    hardest = max(variants.values(), key=lambda v: v.levels)
    tiers = []
    for level in range(1, hardest.levels + 1):
        first = (level - 1) * BLOCKS_PER_LEVEL
        last = first + BLOCKS_PER_LEVEL - 1
        colour = hardest.block_colors[first]
        tiers.append(
            f"level {level} from the {colour} blocks ({first}-{last})"
            if level == 1
            else f"level {level} from {colour} ({first}-{last})"
        )
    return "build " + ", ".join(tiers) + ", bottom-up"


def _make_stack_adapter() -> CategoryAdapter:
    from loot_bench.tasks.stack.config import task_id_parts
    from loot_bench.tasks.stack.variants import PI_VARIANTS

    return CategoryAdapter(
        categories=("Stack",),
        obs_section_widths=(7,),
        variants=PI_VARIANTS,
        id_parts=task_id_parts,
        category_of=lambda v: "Stack",
        family=lambda v: task_id_parts(v)[1],
        n_objects=lambda v: v.num_blocks,
        object_signatures=lambda v: v.block_colors,
        splittable=lambda v: True,
        progress_and_gate=_pi_progress_and_gate,
        skip_cause=_pi_skip_cause(PI_VARIANTS),
        skip_diagnostic=_pi_skip_diagnostic,
    )


def _make_domino_adapter() -> CategoryAdapter:
    from loot_bench.tasks.domino.config import task_id_parts
    from loot_bench.tasks.domino.variants import DOMINO_VARIANTS

    return CategoryAdapter(
        categories=("Domino",),
        obs_section_widths=(7, 1, 7),
        variants=DOMINO_VARIANTS,
        id_parts=task_id_parts,
        category_of=lambda v: "Domino",
        family=lambda v: task_id_parts(v)[1],
        n_objects=lambda v: v.num_dominoes,
        object_signatures=lambda v: v.half_extents,
        splittable=lambda v: not v.knock_over,
        progress_and_gate=_domino_progress_and_gate,
        skip_cause=_ORDER_CAUSE,
    )


def _make_sorter_adapter() -> CategoryAdapter:
    from loot_bench.tasks.shape_sorter.config import task_id_parts
    from loot_bench.tasks.shape_sorter.variants import SHAPE_SORTER_VARIANTS

    return CategoryAdapter(
        categories=("Sorter",),
        obs_section_widths=(7, 1),
        variants=SHAPE_SORTER_VARIANTS,
        id_parts=task_id_parts,
        category_of=lambda v: "Sorter",
        family=lambda v: task_id_parts(v)[1],
        n_objects=lambda v: v.num_pieces,
        object_signatures=lambda v: v.shapes,
        splittable=lambda v: v.num_pieces > 2,
        progress_and_gate=_sorter_progress_and_gate,
        skip_cause=_ORDER_CAUSE,
        qpos_fixed_bodies=1,  # the cube, before the pieces
    )


_ADAPTERS: dict[str, Callable[[], CategoryAdapter]] = {
    "Stack": _make_stack_adapter,
    "Domino": _make_domino_adapter,
    "Sorter": _make_sorter_adapter,
}


# --------------------------------------------------------------------------- #
# Task-id <-> (adapter, variant) resolution
# --------------------------------------------------------------------------- #


def parse_task_id(task_id: str) -> tuple[str, str, str, str]:
    """Split ``LOOT-<Category>-<Task>[-<Variation>]-<Robot>`` into its parts."""
    parts = task_id.split("-")
    if len(parts) < 4 or parts[0] != "LOOT":
        raise ValueError(f"not a LOOT task id: {task_id!r}")
    category, task, robot = parts[1], parts[2], parts[-1]
    variation = "-".join(parts[3:-1])  # "" for a base family
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
    source = variant_from_task_id(source_task_id)
    if not adapter.splittable(source):
        return []
    n_src = adapter.n_objects(source)
    targets = [
        f"LOOT-{middle}-{robot}"
        for middle, variant in adapter.variant_index().items()
        if adapter.family(variant) == adapter.family(source)
        and adapter.splittable(variant)
        and adapter.n_objects(variant) < n_src
    ]
    targets.sort(key=lambda t: adapter.n_objects(variant_from_task_id(t)), reverse=True)
    return targets


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
        adapter = self.adapter
        assert adapter.family(self.source) == adapter.family(self.target), (
            "source and target are not the same family"
        )
        assert adapter.splittable(self.source) and adapter.splittable(self.target), (
            "variant is not splittable (e.g. Domino KnockOver)"
        )
        assert self.k < self.n_src, "target is not easier than source"
        assert (
            adapter.object_signatures(self.target)
            == adapter.object_signatures(self.source)[: self.k]
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


def proprio_width(obs_dim: int, adapter: CategoryAdapter, n_objects: int) -> int:
    """Width of the proprioceptive head of a recorded observation.

    Nothing follows the object-major sections -- there is no trailing
    ``actions`` term (ADR-016) -- so the head is what is left over.
    """
    width = obs_dim - n_objects * sum(adapter.obs_section_widths)
    assert width > _EE_POSE_WIDTH, f"unexpected obs layout: head would be {width}"
    return width


def frame_state(
    obs: torch.Tensor,
    qvel: torch.Tensor,
    *,
    proprio: int,
    adapter: CategoryAdapter,
    n_layout: int,
    take: int,
) -> FrameState:
    """Narrow one episode's recorded state to its first ``take`` objects.

    ``n_layout`` is the object count the recording is laid out with (the
    section spacing); ``take`` (<= ``n_layout``) is how many to keep.
    Release reads out of the recorded observation: the pad force sits last
    in the proprioceptive head, so it is ``obs`` at ``proprio - 1`` and the
    grasp-site position (the first 3 of ``ee_pose``) sits one ``ee_pose``
    width before that.
    """
    frames = obs.shape[0]
    sections, offset = [], proprio
    for width in adapter.obs_section_widths:
        section = obs[:, offset : offset + take * width]
        sections.append(section.reshape(frames, take, width))
        offset += n_layout * width
    fixed = adapter.qpos_fixed_bodies
    speed = qvel[:, fixed * 6 : (fixed + take) * 6].reshape(frames, take, 6)
    ee_pos = proprio - _PAD_FORCE_WIDTH - _EE_POSE_WIDTH
    return FrameState(
        sections=sections,
        speed=speed[:, :, :3].norm(dim=-1),
        released=obs[:, proprio - _PAD_FORCE_WIDTH] <= MIN_GRIP_FORCE,
        grasp_pos=obs[:, ee_pos : ee_pos + 3],
    )


def episode_state(
    obs: np.ndarray, qvel: np.ndarray, plan: SplitPlan, take: int
) -> FrameState:
    """A recorded episode's state, laid out for the plan's source and narrowed
    to its first ``take`` objects."""
    return frame_state(
        torch.from_numpy(obs.astype(np.float64)),
        torch.from_numpy(qvel.astype(np.float64)),
        proprio=proprio_width(obs.shape[1], plan.adapter, plan.n_src),
        adapter=plan.adapter,
        n_layout=plan.n_src,
        take=take,
    )


def episode_rewards(
    progress: torch.Tensor, gate: torch.Tensor, last_frame: int
) -> torch.Tensor:
    """Rewards for frames ``0..last_frame``, each from its own state.

    The recording carries no obs/reward shift (see the module docstring's
    "Frame alignment" note), so nothing is re-aligned here.
    """
    return shaped_task_reward(progress[: last_frame + 1], gate[: last_frame + 1])


def _check_source_reward(
    plan: SplitPlan, state: FrameState, recorded: torch.Tensor
) -> None:
    """Assert the recompute reproduces the source episode's own reward.

    The one thing that validates the obs/qpos layout, the pad-force column,
    the frame alignment, and every success kernel against real recorded data.
    Compared frame-for-frame: row ``t``'s recorded reward is exactly what the
    state in row ``t``'s observation earns (see the module docstring).
    """
    progress, gate = plan.adapter.progress_and_gate(plan.source, state)
    residual = (shaped_task_reward(progress, gate) - recorded).abs()
    mean = residual.mean().item()
    assert mean <= _REWARD_SELF_CHECK_MEAN_TOL, (
        f"source reward self-check failed (mean residual {mean:.2e}, max "
        f"{residual.max().item():.2e}); the obs/qpos layout, the pad-force "
        "column, the frame alignment, or a success kernel disagrees with the "
        "recording"
    )


def _count_disturbed(state: FrameState, dropped_start: int, last_frame: int) -> int:
    """How many dropped objects moved before the cut.

    Demos are collected in index order, so the objects an easier task does not
    have should still sit on their spawn cells when it succeeds. One that moved
    means the operator used it, and the relabeled episode is then a demo of
    something the target task cannot do.
    """
    poses = state.sections[0][: last_frame + 1, dropped_start:, :3]
    if poses.shape[1] == 0:
        return 0
    moved_xy = (poses[:, :, :2] - poses[0:1, :, :2]).norm(dim=-1).amax(dim=0)
    moved_z = (poses[:, :, 2] - poses[0:1, :, 2]).abs().amax(dim=0)
    return int(((moved_xy > _DROPPED_XY_TOL) | (moved_z > _DROPPED_Z_TOL)).sum().item())


@dataclass
class RelabeledEpisode:
    obs: np.ndarray
    robot_state: np.ndarray
    action: np.ndarray
    reward: np.ndarray  # (T, 1) float32
    done: np.ndarray  # (T, 1) bool
    qpos: np.ndarray
    qvel: np.ndarray
    n_dropped_disturbed: int
    cut_frame: int
    """Index of the last frame kept: the one the target's success is paid on."""


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
    """Relabel one source episode into the target task.

    Returns None when the target's success is never reached early enough to be
    paid inside the recording, i.e. this episode is not a demo of the target.
    """
    plan.validate()
    adapter = plan.adapter
    n_src, k = plan.n_src, plan.k
    proprio = proprio_width(obs.shape[1], adapter, n_src)

    obs_t = torch.from_numpy(obs.astype(np.float64))
    qvel_t = torch.from_numpy(qvel.astype(np.float64))
    recorded = torch.from_numpy(reward.astype(np.float64)).reshape(-1)

    # qvel is cloned post-step, one frame ahead of the pre-step obs at the
    # same row (see the module docstring's "Frame alignment" note); shift it
    # back one so the at-rest check reads the qvel for the state obs_t[t]
    # describes. Row 0 has no predecessor, so it stands in for itself -- exact
    # whenever nothing has moved yet. Only this recompute copy shifts; the
    # written dataset below keeps the recorder's original post-step qvel.
    qvel_t = torch.cat([qvel_t[:1], qvel_t[:-1]])

    # qpos/qvel layout: [fixed leading free bodies | objects | robot], each free
    # body 7 qpos and 6 qvel. Release reads out of the recorded observation,
    # so the robot's qpos block is only truncated below, never interpreted.
    fixed = adapter.qpos_fixed_bodies

    def state_of(take: int) -> FrameState:
        return frame_state(
            obs_t,
            qvel_t,
            proprio=proprio,
            adapter=adapter,
            n_layout=n_src,
            take=take,
        )

    _check_source_reward(plan, state_of(n_src), recorded)

    target = state_of(k)
    progress, gate = adapter.progress_and_gate(plan.target, target)
    if not bool(gate.any()):
        return None
    # The success is paid on the frame the gate first holds, which is also
    # the frame the env terminates on (no extra shift, see the module
    # docstring), so that frame is the last one kept.
    cut = int(torch.argmax(gate.int()).item())
    kept = slice(0, cut + 1)

    done = np.zeros((cut + 1, 1), dtype=bool)
    done[-1, 0] = True

    # obs keeps [proprio | first-k of each section]; qpos/qvel keep
    # [fixed bodies + first-k objects | robot].
    obs_parts, offset = [obs[kept, :proprio]], proprio
    for width in adapter.obs_section_widths:
        obs_parts.append(obs[kept, offset : offset + k * width])
        offset += n_src * width
    rewards = episode_rewards(progress, gate, cut).numpy().reshape(-1, 1)

    return RelabeledEpisode(
        obs=np.concatenate(obs_parts, axis=1).astype(np.float32),
        robot_state=robot_state[kept].astype(np.float32),
        action=action[kept].astype(np.float32),
        reward=rewards.astype(np.float32),
        done=done,
        qpos=np.concatenate(
            [qpos[kept, : (fixed + k) * 7], qpos[kept, (fixed + n_src) * 7 :]], axis=1
        ).astype(np.float32),
        qvel=np.concatenate(
            [qvel[kept, : (fixed + k) * 6], qvel[kept, (fixed + n_src) * 6 :]], axis=1
        ).astype(np.float32),
        n_dropped_disturbed=_count_disturbed(state_of(n_src), k, cut),
        cut_frame=cut,
    )


# --------------------------------------------------------------------------- #
# Dataset IO
# --------------------------------------------------------------------------- #


def load_source_episodes(source_dir: str) -> list[tuple[int, pd.DataFrame]]:
    files = sorted(glob.glob(f"{source_dir}/data/chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {source_dir}/data")
    frames = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    frames = frames.sort_values(["episode_index", "frame_index"])
    return [
        # pandas types a group key as Hashable; this column holds integers.
        (int(cast(int, episode)), group)
        for episode, group in frames.reset_index(drop=True).groupby(
            "episode_index", sort=True
        )
    ]


def source_fps(source_dir: str) -> int:
    """The source dataset's frame rate, carried over to every split output.

    Read rather than assumed: the control rate has changed once already
    (ADR-016), and a wrong fps silently rescales every timestamp.
    """
    info = json.loads(Path(source_dir, "meta", "info.json").read_text())
    return int(info["fps"])


def _column(group: pd.DataFrame, name: str) -> np.ndarray:
    return np.stack(list(group[name])).astype(np.float32)


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


# Sidecar keys that describe the recording session rather than its task, so
# they survive a relabel. `episode_seeds` is rebuilt from the kept episodes.
_CARRIED_EXTRA_KEYS = ("start_pose_noise", "seed_device", "operator_id")


def split_to_task(
    source_episodes: list[tuple[int, pd.DataFrame]],
    source_extra: dict,
    plan: SplitPlan,
    target_task: str,
    out_root: str,
    fps: int,
) -> dict:
    """Relabel every source episode into ``target_task`` and write it out.

    Returns a summary dict; the LeRobot dataset lands at ``out_root``. A
    source episode that never pays the target's success is skipped, with a
    diagnostic under ``skip_diagnostics``; every other episode is still
    written, so the caller decides whether the skips are fatal after the good
    ones are safely on disk.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source_seeds = source_extra.get("episode_seeds", {})
    dataset: LeRobotDataset | None = None
    seeds: dict[str, int] = {}
    written = 0
    skipped: list[int] = []
    skip_diagnostics: dict[int, str] = {}
    disturbed: list[int] = []
    kept_frames, source_frames = 0, 0

    for source_episode, group in source_episodes:
        source_frames += len(group)
        obs = _column(group, "observation.state")
        qvel = _column(group, "observation.qvel")
        episode = relabel_episode(
            obs=obs,
            qpos=_column(group, "observation.qpos"),
            qvel=qvel,
            action=_column(group, "action"),
            robot_state=_column(group, "observation.robot_state"),
            reward=np.asarray(group["next.reward"].values, dtype=np.float32).reshape(
                -1
            ),
            plan=plan,
        )
        if episode is None:
            skipped.append(source_episode)
            skip_diagnostics[source_episode] = plan.adapter.skip_diagnostic(
                plan.source, episode_state(obs, qvel, plan, plan.n_src)
            )
            continue
        if episode.n_dropped_disturbed:
            disturbed.append(source_episode)
        kept_frames += len(episode.obs)

        if dataset is None:
            dataset = LeRobotDataset.create(
                target_task,
                fps=fps,
                features=_features(
                    episode.obs.shape[1], episode.qpos.shape[1], episode.qvel.shape[1]
                ),
                root=out_root,
            )
        success = np.array([True], dtype=bool)  # the cut is the target's success
        for i in range(len(episode.obs)):
            dataset.add_frame(
                {
                    "observation.state": episode.obs[i],
                    "observation.robot_state": episode.robot_state[i],
                    "action": episode.action[i],
                    "next.reward": episode.reward[i],
                    "next.done": episode.done[i],
                    "success": success,
                    "observation.qpos": episode.qpos[i],
                    "observation.qvel": episode.qvel[i],
                    "task": target_task,
                }
            )
        dataset.save_episode()
        # The seed carries over unchanged: the reset draws per object in index
        # order, so the same seed reproduces the first-k spawns in the target.
        if str(source_episode) in source_seeds:
            seeds[str(written)] = int(source_seeds[str(source_episode)])
        written += 1

    if dataset is not None:
        extra = {k: source_extra[k] for k in _CARRIED_EXTRA_KEYS if k in source_extra}
        if seeds:
            extra["episode_seeds"] = seeds
        extra["split_from"] = plan.adapter.task_id_middle(plan.source)
        write_extra_info(dataset.meta.root, extra)
        dataset.finalize()

    return {
        "target_task": target_task,
        "written": written,
        "skipped": skipped,
        "skip_diagnostics": skip_diagnostics,
        "disturbed_eps": disturbed,
        "mean_cut_frames": (kept_frames / written) if written else 0,
        "source_frames": source_frames,
    }


def _skip_report(adapter: CategoryAdapter, summaries: list[dict]) -> str:
    """One message for every target with skips: which source episodes never
    paid the target's success, the protocol that explains it, and the
    per-episode evidence."""
    lines = []
    for summary in summaries:
        lines.append(
            f"{summary['target_task']}: source episodes {summary['skipped']} "
            f"never paid the target's success and were not written "
            f"({summary['written']} were, under {summary['out_root']})."
        )
        for episode, diagnostic in summary["skip_diagnostics"].items():
            lines.append(f"    ep {episode}: {diagnostic}")
    lines.append(f"likely cause: {adapter.skip_cause}")
    lines.append(
        "The written episodes are complete; re-run with --allow-skips to "
        "accept the loss."
    )
    return "\n".join(lines)


def split_dataset(
    source_dir: str,
    source_task: str,
    output_dir: str,
    target_tasks: list[str] | None = None,
    stamp: str | None = None,
    allow_skips: bool = False,
) -> list[dict]:
    """Split one recording into every target, then fail on any skip.

    Every target is written before the check, so a session where most
    episodes relabel cleanly is kept, and one error names all the skips at
    once rather than the first. ``allow_skips`` demotes that error to the
    summary's warning; the skips are still reported.
    """
    adapter = adapter_for(source_task)
    source_variant = variant_from_task_id(source_task)
    if not adapter.splittable(source_variant):
        raise ValueError(
            f"{source_task} is not splittable (Domino KnockOver: the topple is "
            "one chain over all pieces)."
        )
    targets = target_tasks or easier_target_task_ids(source_task)
    if not targets:
        raise ValueError(
            f"{source_task} is already the easiest task in its family, so there "
            "is nothing to split. Record a harder source."
        )
    source_episodes = load_source_episodes(source_dir)
    source_extra = load_extra_info(source_dir)
    fps = source_fps(source_dir)
    stamp = stamp or time.strftime("%Y-%m-%d_%H-%M-%S")

    summaries = []
    for target_task in targets:
        plan = SplitPlan(adapter, source_variant, variant_from_task_id(target_task))
        out_root = str(Path(output_dir) / target_task / stamp)
        summary = split_to_task(
            source_episodes, source_extra, plan, target_task, out_root, fps
        )
        summary["out_root"] = out_root
        summaries.append(summary)

    with_skips = [s for s in summaries if s["skipped"]]
    if with_skips and not allow_skips:
        raise ValueError(_skip_report(adapter, with_skips))
    return summaries


def _print_summary(
    source_task: str, adapter: CategoryAdapter, summaries: list[dict]
) -> None:
    print(f"\n[split] source: {source_task}")
    for summary in summaries:
        print(
            f"  -> {summary['target_task']}: wrote {summary['written']} ep "
            f"(mean {summary['mean_cut_frames']:.0f} frames/ep), "
            f"skipped {len(summary['skipped'])} (target success never paid)"
        )
        if summary["skipped"]:
            print(
                f"     WARN skipped source eps {summary['skipped']}; likely "
                f"cause: {adapter.skip_cause}"
            )
            for episode, diagnostic in summary["skip_diagnostics"].items():
                print(f"       ep {episode}: {diagnostic}")
        if summary["disturbed_eps"]:
            print(
                f"     WARN dropped objects moved in eps {summary['disturbed_eps']}; "
                "the operator may have used one, check the collection order"
            )
        print(f"     {summary['out_root']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source-dir", required=True, help="source LeRobot dataset root"
    )
    parser.add_argument("--source-task", required=True, help="e.g. LOOT-Stack-3Pis-Dk1")
    parser.add_argument(
        "--target-task",
        action="append",
        default=None,
        help="easier target task id (repeatable); default = all easier same-family",
    )
    parser.add_argument(
        "--output-dir", required=True, help="output root for the split datasets"
    )
    parser.add_argument(
        "--allow-skips",
        action="store_true",
        help="only warn about source episodes that never pay a target's success "
        "(default: write the rest, then exit non-zero)",
    )
    args = parser.parse_args()

    # Expected user errors (wrong or easiest source, non-easier target, a
    # protocol-breaking take) get a clean message rather than a traceback.
    try:
        summaries = split_dataset(
            source_dir=args.source_dir,
            source_task=args.source_task,
            output_dir=args.output_dir,
            target_tasks=args.target_task,
            allow_skips=args.allow_skips,
        )
    except (ValueError, AssertionError) as error:
        raise SystemExit(f"[split] {error}") from None
    _print_summary(args.source_task, adapter_for(args.source_task), summaries)


if __name__ == "__main__":
    main()
