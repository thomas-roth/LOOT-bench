"""One-time conversion of the official TRLC-DK1 URDF meshes to MuJoCo-loadable
assets.

Source: https://github.com/robot-learning-co/trlc-dk1 (Apache-2.0), urdf/follower.

MuJoCo cannot load GLB, so each visual GLB is split into one OBJ per material
color (scene-graph transforms applied) so the MJCF can color the parts.

Usage:
    python convert_glb_meshes_to_obj.py <path-to-trlc-dk1-repo>
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh

LINKS = [
    "base_link",
    "link1-2",
    "link2-3",
    "link3-4",
    "link4-5",
    "link5-6",
    "link6-7",
    "finger_left",
    "finger_right",
]

ASSETS_DIR = Path(__file__).resolve().parent.parent / "xmls" / "assets"


def linear_to_srgb(c: float) -> float:
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * c ** (1 / 2.4) - 0.055


def convert_visual(glb_path: Path, link: str) -> list[dict]:
    scene = trimesh.load(glb_path, force="scene")
    assert isinstance(scene, trimesh.Scene)

    # Gather transformed geometry, grouped by base color.
    groups: dict[tuple, list[trimesh.Trimesh]] = {}
    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node_name]
        geom = scene.geometry[geom_name]
        assert isinstance(geom, trimesh.Trimesh)
        mat = getattr(geom.visual, "material", None)
        color = getattr(mat, "baseColorFactor", None)
        key = tuple(int(c) for c in color[:3]) if color is not None else (128, 128, 128)
        mesh = trimesh.Trimesh(
            vertices=geom.vertices.copy(), faces=geom.faces.copy(), process=False
        )
        mesh.apply_transform(transform)
        groups.setdefault(key, []).append(mesh)

    entries = []
    merged_groups = [
        (color, trimesh.util.concatenate(meshes)) for color, meshes in groups.items()
    ]
    # Drop degenerate fragments (e.g. flat decal strips) that MuJoCo's mesh
    # compiler may reject.
    merged_groups = [
        (color, mesh) for color, mesh in merged_groups if len(mesh.vertices) >= 10
    ]
    for i, (color, merged) in enumerate(sorted(merged_groups, key=lambda x: x[0])):
        name = f"{link}_{i}"
        merged.export(ASSETS_DIR / f"{name}.obj", include_color=False)
        rgba = [round(linear_to_srgb(c / 255.0), 4) for c in color] + [1.0]
        entries.append(
            {
                "mesh": name,
                "file": f"{name}.obj",
                "rgba": rgba,
                "vertices": int(merged.vertices.shape[0]),
            }
        )
    return entries


def main() -> None:
    src = Path(sys.argv[1]).expanduser() / "urdf" / "follower" / "meshes"
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, list[dict]] = {}
    for link in LINKS:
        manifest[link] = convert_visual(src / "visual" / f"{link}.glb", link)
        col_src = src / "collision" / f"{link}.stl"
        shutil.copyfile(col_src, ASSETS_DIR / f"{link}_col.stl")
        print(f"{link}: {len(manifest[link])} color groups")

    # Remove stray .mtl files the OBJ exporter may have written.
    for mtl in ASSETS_DIR.glob("*.mtl"):
        mtl.unlink()

    with open(ASSETS_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {ASSETS_DIR / 'manifest.json'}")

    np.set_printoptions(suppress=True)


if __name__ == "__main__":
    main()
