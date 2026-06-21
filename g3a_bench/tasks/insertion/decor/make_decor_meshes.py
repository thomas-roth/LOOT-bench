"""One-time generator for the insertion-scene decoration meshes.

Writes the binary-STL files referenced by ``workstation_decor.xml`` into
``decor/meshes/``. Currently just the open-top parts tray ("Werkbox") that the
pegs are spawned into. The tray is purely decorative (no collision), so its
footprint only needs to match the peg spawn region in
``insertion.assets`` / ``insertion.mdp.commands`` and the placement in
``workstation_decor.xml``; watertightness/quality only matters for rendering.

This is a build-time tool (like ``robots/trlc_dk1/utils/convert_glb_meshes_to_obj.py``),
not part of the runtime. Re-run it to regenerate the committed STLs:

    uv run python g3a_bench/tasks/insertion/decor/make_decor_meshes.py
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

MESH_DIR = Path(__file__).resolve().parent / "meshes"

Vec = tuple[float, float, float]
Tri = tuple[Vec, Vec, Vec]


# --- STL writing. ---------------------------------------------------------


def _write_stl(path: Path, name: str, tris: list[Tri]) -> None:
    """Write *tris* as a binary STL (the only format MuJoCo's decoder accepts)."""
    buf = bytearray()
    buf += f"binary STL: {name}".encode("ascii").ljust(80, b"\0")[:80]
    buf += struct.pack("<I", len(tris))
    for a, b, c in tris:
        n = np.cross(np.subtract(b, a), np.subtract(c, a))
        ln = float(np.linalg.norm(n))
        nrm = (n / ln) if ln > 1e-12 else np.zeros(3)
        buf += struct.pack("<3f", *nrm)
        for v in (a, b, c):
            buf += struct.pack("<3f", *v)
        buf += struct.pack("<H", 0)
    path.write_bytes(bytes(buf))


# --- Primitive triangle builders. -----------------------------------------


def _quad(p0: Vec, p1: Vec, p2: Vec, p3: Vec) -> list[Tri]:
    """Two triangles for quad p0->p1->p2->p3 (consistent winding)."""
    return [(p0, p1, p2), (p0, p2, p3)]


def _box(cx: float, cy: float, cz: float, hx: float, hy: float, hz: float) -> list[Tri]:
    """Axis-aligned box centred at (cx,cy,cz) with the given half-extents."""
    x0, x1 = cx - hx, cx + hx
    y0, y1 = cy - hy, cy + hy
    z0, z1 = cz - hz, cz + hz
    v = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),  # bottom 0-3
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),  # top 4-7
    ]
    tris: list[Tri] = []
    tris += _quad(v[0], v[3], v[2], v[1])  # bottom (-z)
    tris += _quad(v[4], v[5], v[6], v[7])  # top (+z)
    tris += _quad(v[0], v[1], v[5], v[4])  # -y
    tris += _quad(v[2], v[3], v[7], v[6])  # +y
    tris += _quad(v[1], v[2], v[6], v[5])  # +x
    tris += _quad(v[3], v[0], v[4], v[7])  # -x
    return tris


# --- Decor pieces. --------------------------------------------------------


# Tray outer footprint (half extents) and heights. Long axis along y so the
# spawned pegs span the robot's comfortable lateral reach. Keep in sync with
# ``insertion.assets.TRAY_*`` and the placement in ``workstation_decor.xml``.
TRAY_OUTER_HALF = (0.080, 0.155)
TRAY_WALL = 0.008
TRAY_HEIGHT = 0.034
TRAY_FLOOR = 0.006


def tool_tray() -> list[Tri]:
    """Open-top rectangular parts tray, bottom resting on z=0."""
    hx, hy = TRAY_OUTER_HALF
    height = TRAY_HEIGHT
    wall = TRAY_WALL
    floor = TRAY_FLOOR
    tris: list[Tri] = []
    tris += _box(0.0, 0.0, floor / 2, hx, hy, floor / 2)  # floor slab
    # Four rim walls spanning the full height.
    tris += _box(hx - wall / 2, 0.0, height / 2, wall / 2, hy, height / 2)  # +x
    tris += _box(-hx + wall / 2, 0.0, height / 2, wall / 2, hy, height / 2)  # -x
    tris += _box(0.0, hy - wall / 2, height / 2, hx, wall / 2, height / 2)  # +y
    tris += _box(0.0, -hy + wall / 2, height / 2, hx, wall / 2, height / 2)  # -y
    return tris


def main() -> None:
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    pieces = {"tool_tray": tool_tray()}
    for name, tris in pieces.items():
        out = MESH_DIR / f"{name}.stl"
        _write_stl(out, name, tris)
        print(f"wrote {out.relative_to(MESH_DIR.parent)}  ({len(tris)} facets)")


if __name__ == "__main__":
    main()
