"""Shared fixed-spawn mechanism every category spawns from.

Stack, domino and sorter place object ``i`` on cell ``i`` of their
hand-written table; place draws over the same envelope; the archived
``peg_insertion`` seats its single peg on its one-cell table. This module
states the pieces those tables need:

- the spawn envelope as the arm's real shape (a near bound, a radial reach
  bound and the table's half-width), stated once here;
- the minimum-pitch rule derived from the gripper, stated once here;
- :func:`sample_fixed_spawns`, which places object ``i`` on cell ``i`` with
  jitter, in index order (one ``rand`` draw per object, so an easier task's
  reset is a prefix of a harder one's);
- :func:`validate_spawn_table`, the table checks the per-category tests reuse.

The caller passes sampler output to :func:`scatter.write_upright_spawn`,
which sits entities upright on their cells and zeros their velocity.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

SPAWN_JITTER = 0.01
"""Per-axis jitter (m) around each spawn cell. One number every category."""

SPAWN_X_MIN = 0.20
"""Near bound (m): no jittered pose may sit closer to the base than this."""

SPAWN_REACH_RADIUS = 0.49
"""Radial reach bound (m): ``hypot(x, y)`` of any jittered pose stays inside."""

SPAWN_TABLE_HALF_WIDTH = 0.35
"""Table half-width (m): ``abs(y) + object_radius`` of any jittered pose."""

GRIPPER_CLEARANCE = 0.04
"""Lateral clearance (m) between neighbouring spawn cells.

The **closed** gripper's swept half-span (measured 39 mm). The gripper
descends *open* (swept half-span 80 mm, full span 152 mm), so anyone who
re-measures will otherwise read this constant as wrong. Sizing on the open
span is not required: yaw is free, so the fingers can be oriented away from
the nearest neighbour, and OGBench uses a 0.10-0.15 m band for a same-size
cube under the same +/-1 cm jitter.
"""


def spawn_pitch(
    radius_a: float, radius_b: float, jitter: float = SPAWN_JITTER
) -> float:
    """Minimum centre-to-centre pitch (m) for two neighbouring spawn cells.

    Set by the gripper, not by object radius:

    ``pitch = 2 * jitter + r_a + r_b + GRIPPER_CLEARANCE``.
    """
    return 2.0 * jitter + radius_a + radius_b + GRIPPER_CLEARANCE


def spawn_pose_in_envelope(x: float, y: float, object_radius: float) -> bool:
    """Whether one jittered ``(x, y)`` pose is inside the shared envelope."""
    if x < SPAWN_X_MIN - 1e-6:
        return False
    if math.hypot(x, y) > SPAWN_REACH_RADIUS + 1e-6:
        return False
    return abs(y) + object_radius <= SPAWN_TABLE_HALF_WIDTH + 1e-6


def spawn_cell_in_envelope(
    cx: float, cy: float, object_radius: float, jitter: float = SPAWN_JITTER
) -> bool:
    """Whether a cell is inside the envelope at all four jitter corners."""
    return all(
        spawn_pose_in_envelope(cx + sx * jitter, cy + sy * jitter, object_radius)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
    )


def min_pairwise_separation(cells: torch.Tensor) -> float:
    """Minimum centre-to-centre distance (m) over all cell pairs."""
    n = cells.shape[0]
    assert n >= 2, "need at least two cells to measure a separation"
    dist = torch.cdist(cells, cells)
    dist.fill_diagonal_(torch.inf)
    return float(dist.min())


def sample_fixed_spawns(
    cells: torch.Tensor,
    num_envs: int,
    count: int,
    *,
    jitter: float = SPAWN_JITTER,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw jittered poses with object ``i`` on cell ``i``.

    One ``rand(E, 3)`` draw per object (jitter x/y, yaw), in index order, so
    the draw count and the draw-to-pose mapping are independent of ``count``
    (seed-prefix rule: an easier task's reset is a prefix of a harder
    task's). Pass the output to :func:`scatter.write_upright_spawn` for the
    upright, zero-velocity write.

    Returns:
        ``(xy, yaw)``: (num_envs, count, 2) and (num_envs, count).
    """
    num_cells = cells.shape[0]
    assert count <= num_cells, f"cannot place {count} objects on {num_cells} cells"
    dev = cells.device
    xy = torch.empty(num_envs, count, 2, device=dev)
    yaw = torch.empty(num_envs, count, device=dev)
    for i in range(count):
        u = torch.rand(num_envs, 3, device=dev)
        xy[:, i] = cells[i] + (u[:, :2] - 0.5) * 2.0 * jitter
        yaw[:, i] = u[:, 2] * 2.0 * math.pi
    return xy, yaw


def validate_spawn_table(
    cells: torch.Tensor,
    *,
    object_radius: float,
    pitch: float,
    max_objects: int,
    jitter: float = SPAWN_JITTER,
    is_clear: Callable[[float, float], bool] | None = None,
) -> list[str]:
    """Check a category's spawn table; return a list of violations (``[]`` ok).

    Args:
        cells: (C, 2) cell-centre xy positions, object ``i`` takes cell ``i``.
        object_radius: bounding-circle radius of the largest object.
        pitch: minimum required centre-to-centre separation (see
            :func:`spawn_pitch`).
        max_objects: the category's maximum object count; the table must hold
            at least this many cells.
        is_clear: optional work-zone predicate on a jittered ``(x, y)`` pose.
            The caller bakes the halo (object radius + jitter + margin) into
            it; every cell is checked at all four jitter corners.
    """
    errors: list[str] = []
    num_cells = cells.shape[0]
    if num_cells < max_objects:
        errors.append(
            f"table holds {num_cells} cells, needs at least {max_objects} "
            "for the category's maximum object count"
        )
    if num_cells >= 2:
        separation = min_pairwise_separation(cells)
        if separation < pitch - 1e-6:
            errors.append(
                f"minimum separation {separation:.4f} below pitch {pitch:.4f}"
            )
    for i in range(num_cells):
        cx, cy = float(cells[i, 0]), float(cells[i, 1])
        if not spawn_cell_in_envelope(cx, cy, object_radius, jitter=jitter):
            errors.append(f"cell {i} ({cx:.3f}, {cy:.3f}) outside the envelope")
            continue
        if is_clear is not None and not all(
            is_clear(cx + sx * jitter, cy + sy * jitter)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
        ):
            errors.append(f"cell {i} ({cx:.3f}, {cy:.3f}) inside the work zone")
    return errors
