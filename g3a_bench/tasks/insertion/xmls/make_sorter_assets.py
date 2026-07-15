# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "mapbox-earcut>=1.0"]
# ///
"""One-time generator for the STL shape-sorter assets.

Converts the user-designed CAD STLs (committed under ``cad/``; mm units,
uncentered assembly offsets) into the committed runtime assets under
``tasks/insertion/xmls/``:

- ``meshes/sorter_visual.stl``      pretty sorter mesh (metres, centred)
- ``meshes/sorter_col_<face>_<k>.stl``  exact convex collision parts, one set
  per face panel (MuJoCo collides meshes as convex hulls, so the holed cube
  must be split; the split is exact, not an approximation, because the hole
  clearance ~1-2 mm *is* the task)
- ``meshes/piece_<shape>.stl``      piece meshes, centred at the cross-section
  centroid, extrusion axis +z
- ``meshes/piece_star_col_<k>.stl`` convex parts of the (non-convex) star piece
- ``sorter.xml`` / ``piece_<shape>.xml``  freejoint MJCF bodies
- ``sorter_layout.json``            hole mouths/axes + piece dims in the cube
  body frame -- the single source both the XML geometry and the success metric
  derive from (see ``insertion.assets.load_sorter_layout``)

Build-time tool (like ``decor/make_decor_meshes.py``), not part of the runtime.
Re-run to regenerate the committed assets:

    uv run g3a_bench/tasks/insertion/xmls/make_sorter_assets.py
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

# PEP 723 inline dep (see header) resolved by `uv run <script>`, not the venv.
import mapbox_earcut  # pyright: ignore[reportMissingImports]
import numpy as np

OUT_DIR = Path(__file__).resolve().parent
DEFAULT_SRC = OUT_DIR / "cad"

SCALE = 1e-3  # CAD mm -> m
SORTER_MASS = 0.6
PIECE_MASS = 0.06
SORTER_FRICTION = "1.0 0.005 0.0001"
SORTER_RGBA = "0.72 0.6 0.42 1"  # light wood, matches assets.BOARD_RGBA

# CAD file name -> repo shape name (matches variants.Shape).
PIECE_FILES = {
    "cylinder.stl": "circle",
    "cube.stl": "square",
    "rectangle.stl": "rect",
    "triangle.stl": "triangle",
    "hexagon.stl": "hexagon",
    "star.stl": "star",
}

# The six cube faces: (name, axis index, sign of the outward normal).
FACES = [
    ("xmin", 0, -1.0),
    ("xmax", 0, 1.0),
    ("ymin", 1, -1.0),
    ("ymax", 1, 1.0),
    ("zmin", 2, -1.0),
    ("zmax", 2, 1.0),
]

Tri = tuple[np.ndarray, np.ndarray, np.ndarray]


# --- STL I/O. ---------------------------------------------------------------


def read_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Binary STL -> (unique vertices (V,3) float64, faces (F,3) int)."""
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 80)[0]
    raw = np.zeros((n, 9))
    off = 84
    for i in range(n):
        raw[i] = struct.unpack_from("<12f", data, off)[3:]
        off += 50
    flat = raw.reshape(-1, 3)
    key = np.round(flat / 1e-4).astype(np.int64)  # dedup at 0.1 um (CAD-exact)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    verts = flat[idx]
    faces = inv.reshape(-1, 3)
    return verts, faces


def write_stl(path: Path, name: str, tris: list[Tri]) -> None:
    """Write binary STL (the only format MuJoCo's decoder accepts)."""
    buf = bytearray()
    buf += f"binary STL: {name}".encode("ascii").ljust(80, b"\0")[:80]
    buf += struct.pack("<I", len(tris))
    for a, b, c in tris:
        n = np.cross(b - a, c - a)
        ln = float(np.linalg.norm(n))
        nrm = (n / ln) if ln > 1e-12 else np.zeros(3)
        buf += struct.pack("<3f", *nrm)
        for v in (a, b, c):
            buf += struct.pack("<3f", *v)
        buf += struct.pack("<H", 0)
    path.write_bytes(bytes(buf))


def mesh_tris(verts: np.ndarray, faces: np.ndarray) -> list[Tri]:
    return [(verts[f[0]], verts[f[1]], verts[f[2]]) for f in faces]


# --- 2D polygon helpers. ----------------------------------------------------


def cross2(u: np.ndarray, v: np.ndarray) -> float:
    return float(u[0] * v[1] - u[1] * v[0])


def poly_area(p: np.ndarray) -> float:
    """Signed area (positive = CCW)."""
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def poly_centroid(p: np.ndarray) -> np.ndarray:
    x, y = p[:, 0], p[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = 0.5 * np.sum(cross)
    cx = np.sum((x + np.roll(x, -1)) * cross) / (6.0 * a)
    cy = np.sum((y + np.roll(y, -1)) * cross) / (6.0 * a)
    return np.array([cx, cy])


def is_convex(p: np.ndarray, tol: float = 1e-9) -> bool:
    """CCW ring convex (collinear vertices allowed)?"""
    if poly_area(p) <= 0:
        return False
    e = np.roll(p, -1, axis=0) - p
    cross = (
        e[:, 0] * np.roll(e, -1, axis=0)[:, 1] - e[:, 1] * np.roll(e, -1, axis=0)[:, 0]
    )
    return bool(np.all(cross >= -tol))


def point_in_poly(pt: np.ndarray, poly: np.ndarray) -> bool:
    """Ray-cast containment test; works for concave polygons."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xi = x0 + (y - y0) / (y1 - y0) * (x1 - x0)
            if x < xi:
                inside = not inside
    return inside


def dist_to_boundary(pt: np.ndarray, poly: np.ndarray) -> float:
    """Distance from *pt* to the closest polygon edge."""
    best = math.inf
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        t = float(np.clip(np.dot(pt - a, ab) / (np.dot(ab, ab) + 1e-18), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(pt - (a + t * ab))))
    return best


# --- Boundary-loop extraction. ----------------------------------------------


def face_loops(
    verts: np.ndarray, faces: np.ndarray, axis: int, plane: float, tol: float = 1e-6
) -> list[list[int]]:
    """Closed boundary loops of the triangulated region on a coordinate plane.

    Returns vertex-index loops of the once-used (boundary) edges of all
    triangles lying entirely on ``verts[:, axis] == plane``.
    """
    on_plane = np.abs(verts[:, axis] - plane) < tol
    tri_on = on_plane[faces].all(axis=1)
    edge_count: dict[tuple[int, int], int] = {}
    directed: dict[tuple[int, int], int] = {}
    for f in faces[tri_on]:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            edge_count[(min(a, b), max(a, b))] = (
                edge_count.get((min(a, b), max(a, b)), 0) + 1
            )
            directed[(int(a), int(b))] = 1
    boundary = {e for e, c in edge_count.items() if c == 1}
    # Recover direction of each boundary edge from the triangle winding.
    nxt: dict[int, int] = {}
    for a, b in directed:
        if (min(a, b), max(a, b)) in boundary:
            assert a not in nxt, "non-manifold boundary"
            nxt[a] = b
    loops: list[list[int]] = []
    seen: set[int] = set()
    for start in sorted(nxt):
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        cur = nxt[start]
        while cur != start:
            loop.append(cur)
            seen.add(cur)
            cur = nxt[cur]
        loops.append(loop)
    return loops


def classify_hole(poly: np.ndarray) -> str:
    """Map a hole outline to a repo shape name by vertex count/aspect."""
    # Drop collinear vertices first (triangulation may add steiner-like points).
    keep: list[int] = []
    n = len(poly)
    for i in range(n):
        a, b, c = poly[i - 1], poly[i], poly[(i + 1) % n]
        if abs(cross2(b - a, c - b)) > 1e-10:
            keep.append(i)
    p = poly[keep]
    n = len(p)
    if n >= 16:
        return "circle"
    if not is_convex(p if poly_area(p) > 0 else p[::-1]):
        return "star"  # only non-convex hole in the set (any point count)
    if n == 3:
        return "triangle"
    if n == 4:
        ext = p.max(axis=0) - p.min(axis=0)
        return "square" if abs(ext[0] - ext[1]) < 5e-3 else "rect"
    if n == 6:
        return "hexagon"
    raise ValueError(f"cannot classify convex hole with {n} corners")


# --- Convex decomposition (earcut + Hertel-Mehlhorn). -----------------------


def triangulate(rings: list[np.ndarray]) -> tuple[np.ndarray, list[list[int]]]:
    """Earcut-triangulate outer ring + hole rings -> (verts2d, CCW triangles)."""
    verts = np.vstack(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    idx = mapbox_earcut.triangulate_float64(verts, ends)
    tris: list[list[int]] = []
    for k in range(0, len(idx), 3):
        t = [int(idx[k]), int(idx[k + 1]), int(idx[k + 2])]
        if poly_area(verts[t]) < 0:
            t = t[::-1]
        tris.append(t)
    return verts, tris


def hertel_mehlhorn(verts: np.ndarray, polys: list[list[int]]) -> list[list[int]]:
    """Greedily merge convex polygons across shared edges while convex."""
    polys = [list(p) for p in polys]
    merged = True
    while merged:
        merged = False
        edge_owner: dict[tuple[int, int], int] = {}
        for pi, p in enumerate(polys):
            for i in range(len(p)):
                edge_owner[(p[i], p[(i + 1) % len(p)])] = pi
        for pi, p in enumerate(polys):
            done = False
            for i in range(len(p)):
                u, v = p[i], p[(i + 1) % len(p)]
                qi = edge_owner.get((v, u))
                if qi is None or qi == pi:
                    continue
                q = polys[qi]
                j = next(
                    k for k in range(len(q)) if q[k] == v and q[(k + 1) % len(q)] == u
                )
                ring_a = p[i + 1 :] + p[: i + 1]  # starts v ... ends u
                rot_b = q[j + 1 :] + q[: j + 1]  # starts u ... ends v
                cand = ring_a + rot_b[1:-1]
                if is_convex(verts[cand]):
                    polys[pi] = cand
                    polys.pop(qi)
                    merged = True
                    done = True
                    break
            if done:
                break
    # Drop collinear vertices to keep the meshes minimal.
    out: list[list[int]] = []
    for p in polys:
        keep: list[int] = []
        for i in range(len(p)):
            a, b, c = verts[p[i - 1]], verts[p[i]], verts[p[(i + 1) % len(p)]]
            if abs(cross2(b - a, c - b)) > 1e-12:
                keep.append(p[i])
        out.append(keep)
    return out


def extrude(poly: np.ndarray, frame: np.ndarray, d0: float, d1: float) -> list[Tri]:
    """Convex CCW polygon (in the (c0,c1) plane) -> closed prism triangles.

    ``frame`` rows are (c0, c1, axis); the prism spans axis coords [d0, d1]
    (d0 < d1); polygon coords are (u, v) in the c0/c1 plane.
    """
    c0, c1, a = frame
    lo = [u * c0 + v * c1 + d0 * a for u, v in poly]
    hi = [u * c0 + v * c1 + d1 * a for u, v in poly]
    n = len(poly)
    tris: list[Tri] = []
    for i in range(1, n - 1):  # caps (fan; polygon is convex)
        tris.append((lo[0], lo[i + 1], lo[i]))
        tris.append((hi[0], hi[i], hi[i + 1]))
    for i in range(n):
        j = (i + 1) % n
        tris.append((lo[i], lo[j], hi[j]))
        tris.append((lo[i], hi[j], hi[i]))
    return tris


def face_frame(axis_out: np.ndarray) -> np.ndarray:
    """Same (c0, c1, axis_out) convention as ``insertion.assets.frame_axes``."""
    a = axis_out / np.linalg.norm(axis_out)
    t = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    c0 = np.cross(t, a)
    c0 /= np.linalg.norm(c0)
    c1 = np.cross(a, c0)
    return np.stack([c0, c1, a])


# --- XML emit. ---------------------------------------------------------------


def emit_xml(path: Path, model: str, body: str, assets: list[str], geoms: list[str]):
    lines = [
        f'<mujoco model="{model}">',
        "  <!-- GENERATED by make_sorter_assets.py - do not edit by hand. -->",
        '  <compiler meshdir="meshes" angle="radian" />',
        "  <asset>",
        *[f"    {a}" for a in assets],
        "  </asset>",
        "  <worldbody>",
        f'    <body name="{body}">',
        f'      <freejoint name="{body}_joint" />',
        *[f"      {g}" for g in geoms],
        "    </body>",
        "  </worldbody>",
        "</mujoco>",
        "",
    ]
    path.write_text("\n".join(lines))


# --- Main pipeline. ----------------------------------------------------------


def process_sorter(src: Path, sorter_stl: str, prefix: str, mesh_dir: Path) -> dict:
    verts, faces = read_stl(src / sorter_stl)
    verts = verts * SCALE
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
    verts = verts - center
    half = float(np.round((verts.max(axis=0) - verts.min(axis=0)).max() / 2.0, 6))
    print(f"{sorter_stl}: half_size={half * 1e3:.1f} mm")

    write_stl(mesh_dir / f"{prefix}_visual.stl", prefix, mesh_tris(verts, faces))

    holes: dict[str, dict] = {}
    col_geoms: list[tuple[str, float]] = []  # (mesh name, part volume)
    total_vol = 0.0
    parts_out: dict[str, list[Tri]] = {}

    for fname, axis, sign in FACES:
        axis_out = np.zeros(3)
        axis_out[axis] = sign
        frame = face_frame(axis_out)
        outer_plane = sign * half

        loops = face_loops(verts, faces, axis, outer_plane)
        assert len(loops) == 2, f"face {fname}: expected outer+hole, got {len(loops)}"
        # Project both loops into the face frame; larger |area| = outer square.
        origin = axis_out * half
        loops2d = []
        for lp in loops:
            p3 = verts[lp] - origin
            p2 = np.stack([p3 @ frame[0], p3 @ frame[1]], axis=1)
            if poly_area(p2) < 0:
                p2 = p2[::-1]
            loops2d.append(p2)
        loops2d.sort(key=lambda p: abs(poly_area(p)), reverse=True)
        outer2d, hole2d = loops2d

        # Wall thickness: nearest well-populated vertex plane inward.
        coords = verts[:, axis]
        inner_candidates = sorted(
            c
            for c in np.unique(np.round(coords, 6))
            if 0 < (outer_plane - c) * sign < 0.02
            and np.sum(np.abs(coords - c) < 1e-6) >= 6
        )
        assert inner_candidates, f"face {fname}: no inner wall plane found"
        wall_t = float(min((outer_plane - c) * sign for c in inner_candidates))

        shape = classify_hole(hole2d)
        centroid = poly_centroid(hole2d)
        mouth = origin + centroid[0] * frame[0] + centroid[1] * frame[1]
        bounding_r = float(np.max(np.linalg.norm(hole2d - centroid, axis=1)))
        ext = hole2d.max(axis=0) - hole2d.min(axis=0)
        holes[shape] = {
            "face": fname,
            "axis_out": [float(x) for x in axis_out],
            "mouth": [float(np.round(x, 6)) for x in mouth],
            "wall_thickness": round(wall_t, 6),
            "bounding_radius": round(bounding_r, 6),
            "extents": [round(float(e), 6) for e in ext],
            "polygon": [[round(float(u), 6), round(float(v), 6)] for u, v in hole2d],
        }

        # Exact convex decomposition of the panel (outer square minus hole).
        # Earcut wants the hole ring in the opposite winding of the outer ring.
        v2d, tris = triangulate([outer2d, hole2d[::-1]])
        polys = hertel_mehlhorn(v2d, tris)
        for k, p in enumerate(polys):
            poly = v2d[p]
            assert is_convex(poly), f"face {fname} part {k} not convex"
            name = f"{prefix}_col_{fname}_{k}"
            # Prism spans [half - wall_t, half] along the outward axis.
            parts_out[name] = [
                (
                    t[0] + origin - axis_out * half,
                    t[1] + origin - axis_out * half,
                    t[2] + origin - axis_out * half,
                )
                for t in extrude(poly, frame, half - wall_t, half)
            ]
            vol = abs(poly_area(poly)) * wall_t
            col_geoms.append((name, vol))
            total_vol += vol
        print(
            f"  {fname}: hole={shape:8s} extents="
            f"{ext[0] * 1e3:.1f}x{ext[1] * 1e3:.1f} mm  wall={wall_t * 1e3:.1f} mm"
            f"  parts={len(polys)}"
        )

    assert set(holes) == {"circle", "square", "rect", "triangle", "hexagon", "star"}

    for name, tris in parts_out.items():
        write_stl(mesh_dir / f"{name}.stl", name, tris)

    assets = [f'<mesh name="{prefix}_visual" file="{prefix}_visual.stl" />'] + [
        f'<mesh name="{n}" file="{n}.stl" />' for n, _ in col_geoms
    ]
    geoms = [
        f'<geom name="{prefix}_visual" type="mesh" mesh="{prefix}_visual" '
        f'group="2" contype="0" conaffinity="0" mass="0" rgba="{SORTER_RGBA}" />'
    ] + [
        f'<geom name="{n}" type="mesh" mesh="{n}" group="3" '
        f'mass="{SORTER_MASS * vol / total_vol:.6f}" '
        f'friction="{SORTER_FRICTION}" rgba="0.72 0.6 0.42 0" />'
        for n, vol in col_geoms
    ]
    emit_xml(OUT_DIR / f"{prefix}.xml", prefix, prefix, assets, geoms)
    print(f"  -> {prefix}.xml  ({len(col_geoms)} collision parts)")

    return {
        "half_size": half,
        "wall_thickness": min(h["wall_thickness"] for h in holes.values()),
        "mass": SORTER_MASS,
        "holes": holes,
    }


def rot_about_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def best_fit(cross: np.ndarray, hole: np.ndarray) -> tuple[float, float]:
    """Best (clearance, angle) of the centred *cross* inside the centred *hole*
    over sampled z-rotations; clearance is -inf if it never fits."""
    best_clear, best_ang = -math.inf, 0.0
    for deg in range(360):
        ang = math.radians(deg)
        r = rot_about_z(ang)[:2, :2]
        pts = cross @ r.T
        if not all(point_in_poly(pt, hole) for pt in pts):
            continue
        clear = min(
            min(dist_to_boundary(pt, hole) for pt in pts),
            min(dist_to_boundary(pt, pts) for pt in hole),
        )
        if clear > best_clear:
            best_clear, best_ang = clear, ang
    return best_clear, best_ang


# Proper rotations mapping each candidate CAD prism axis onto +z.
_AXIS_TO_Z = {
    0: np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    1: np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    2: np.eye(3),
}


def process_piece(
    src: Path, cad_name: str, shape: str, hole: dict, mesh_dir: Path
) -> dict:
    verts, faces = read_stl(src / cad_name)
    verts = verts * SCALE
    hole2d = np.array(hole["polygon"])
    hole2d = hole2d - poly_centroid(hole2d)

    # The CAD prism axis differs per piece (triangle/hexagon extrude along y).
    # Detect it: an axis is a candidate if both end planes carry a single
    # closed outline; among candidates keep the one whose cross-section fits
    # the matching hole (some yaw), and bake that yaw into the mesh so an
    # identity-orientation piece drops straight into its hole.
    found = None
    for axis in (2, 1, 0):
        lo_loops = face_loops(verts, faces, axis, float(verts[:, axis].min()))
        hi_loops = face_loops(verts, faces, axis, float(verts[:, axis].max()))
        if len(lo_loops) != 1 or len(hi_loops) != 1:
            continue
        v_rot = verts @ _AXIS_TO_Z[axis].T
        top = float(v_rot[:, 2].max())
        f_rot = faces
        loops = face_loops(v_rot, f_rot, 2, top)
        cross = v_rot[loops[0]][:, :2]
        if poly_area(cross) < 0:
            cross = cross[::-1]
        centroid = poly_centroid(cross)
        cross = cross - centroid
        clear, ang = best_fit(cross, hole2d)
        if clear > 0:
            found = (axis, v_rot, cross, centroid, clear, ang)
            break
    assert found is not None, f"{cad_name}: no prism axis fits the '{shape}' hole"
    axis, verts, cross, centroid, clear, ang = found

    zmin, zmax = float(verts[:, 2].min()), float(verts[:, 2].max())
    # Center at (cross centroid, mid-height) so the body origin coincides with
    # the hole mouth centroid when the piece sits perfectly in its hole, then
    # yaw so identity orientation aligns with the hole polygon frame.
    verts = verts - np.array([centroid[0], centroid[1], (zmin + zmax) / 2.0])
    verts = verts @ rot_about_z(ang).T
    cross = cross @ rot_about_z(ang)[:2, :2].T
    half_len = (zmax - zmin) / 2.0
    bounding_r = float(np.max(np.linalg.norm(cross, axis=1)))
    ext = cross.max(axis=0) - cross.min(axis=0)
    print(
        f"  piece {shape:8s} axis={'xyz'[axis]} len={2 * half_len * 1e3:.1f} mm"
        f"  fit clearance={clear * 1e3:.2f} mm at yaw={math.degrees(ang):.0f} deg"
    )

    write_stl(mesh_dir / f"piece_{shape}.stl", shape, mesh_tris(verts, faces))

    assets = [f'<mesh name="piece_{shape}" file="piece_{shape}.stl" />']
    if shape == "star":
        # Non-convex: visual mesh + exact convex prism parts for collision.
        v2d, tris = triangulate([cross])
        polys = hertel_mehlhorn(v2d, tris)
        frame = np.eye(3)
        areas = [abs(poly_area(v2d[p])) for p in polys]
        geoms = [
            '<geom name="piece_visual" type="mesh" mesh="piece_star" '
            'group="2" contype="0" conaffinity="0" mass="0" />'
        ]
        for k, (p, area) in enumerate(zip(polys, areas, strict=True)):
            assert is_convex(v2d[p])
            name = f"piece_star_col_{k}"
            write_stl(
                mesh_dir / f"{name}.stl",
                name,
                extrude(v2d[p], frame, -half_len, half_len),
            )
            assets.append(f'<mesh name="{name}" file="{name}.stl" />')
            geoms.append(
                f'<geom name="{name}" type="mesh" mesh="{name}" group="3" '
                f'mass="{PIECE_MASS * area / sum(areas):.6f}" rgba="0.5 0.5 0.5 0" />'
            )
        print(f"  star: {len(polys)} convex collision parts")
    else:
        geoms = [
            f'<geom name="piece_geom" type="mesh" mesh="piece_{shape}" '
            f'mass="{PIECE_MASS}" />'
        ]
    emit_xml(OUT_DIR / f"piece_{shape}.xml", f"piece_{shape}", "piece", assets, geoms)

    return {
        "half_len": round(half_len, 6),
        "cross_bounding_radius": round(bounding_r, 6),
        "cross_extents": [round(float(e), 6) for e in ext],
        "mass": PIECE_MASS,
        "_cross": cross,  # for the fit matrix; stripped before JSON dump
    }


def fit_matrix(holes: dict[str, dict], pieces: dict[str, dict]) -> None:
    """Print best-over-yaw piece-in-hole clearances (centroids aligned)."""
    names = ["circle", "square", "rect", "triangle", "hexagon", "star"]
    print("\nFit matrix (best min-clearance over yaw, mm; '-' = never fits):")
    print("  piece \\ hole " + "".join(f"{h:>9s}" for h in names))
    for pn in names:
        cross = pieces[pn]["_cross"]
        row = ""
        for hn in names:
            hole = np.array(holes[hn]["polygon"])
            hole = hole - poly_centroid(hole)
            clear, _ = best_fit(cross, hole)
            row += f"{clear * 1e3:9.2f}" if clear > -math.inf else f"{'-':>9s}"
        print(f"  {pn:12s} {row}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--sorter-stl", default="sorter.stl")
    ap.add_argument("--prefix", default="sorter")
    args = ap.parse_args()

    mesh_dir = args.out / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    cube = process_sorter(args.src, args.sorter_stl, args.prefix, mesh_dir)
    pieces = {
        shape: process_piece(args.src, cad, shape, cube["holes"][shape], mesh_dir)
        for cad, shape in PIECE_FILES.items()
    }
    fit_matrix(cube["holes"], pieces)

    layout = {
        "cube": {k: v for k, v in cube.items() if k != "holes"},
        "holes": cube["holes"],
        "pieces": {
            n: {k: v for k, v in p.items() if not k.startswith("_")}
            for n, p in pieces.items()
        },
    }
    out = args.out / f"{args.prefix}_layout.json"
    out.write_text(json.dumps(layout, indent=2) + "\n")
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
