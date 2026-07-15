"""Shared offscreen-render helper for the task verification scripts.

Rendering is deliberately decoupled from each script's PASS/FAIL logic: a
missing GL context (headless CI, some remote sessions) must not turn a task
that builds and scores correctly into a reported failure. The single entry
point therefore returns a warning string on failure instead of raising.

Images are saved full-HD under ``out/render/<domain>/`` (see ``domain_of``).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import imageio.v3 as iio
from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer.offscreen_renderer import OffscreenRenderer

RENDER_WIDTH = 1920
RENDER_HEIGHT = 1080

# Camera tuned for a clean, close, slightly-elevated side view of the DK1 workspace
_CAM_DISTANCE = 1.1
_CAM_ELEVATION = -30.0
_CAM_AZIMUTH = 90.0


def domain_of(task_id: str) -> str:
    """Task domain from a ``G3a-<Domain>-<Variant>-<Robot>`` id, lower-cased.

    Maps straight onto an ``out/render/<domain>/`` folder. Domains are *not*
    enumerated anywhere: a new task category (beyond ball/cube/insertion) is
    picked up automatically from its id.
    """
    return task_id.split("-")[1].lower()


def save_env_render(
    env: ManagerBasedRlEnv, path: Path, *, draw_goals: bool = True
) -> str | None:
    """Render environment 0 of *env* to *path* (full-HD PNG).

    Uses the task's own ``cfg.viewer`` for camera tracking but overrides the
    framing (resolution, distance, angle) for a close, unoccluded workspace
    shot, with ``max_extra_envs`` forced to 0 for a clean single-env image.
    When *draw_goals* is true the command debug visualizers (goal markers) are
    overlaid. Returns ``None`` on success or a short warning string if
    rendering/saving failed.
    """
    try:
        viewer = dataclasses.replace(
            env.cfg.viewer,
            width=RENDER_WIDTH,
            height=RENDER_HEIGHT,
            max_extra_envs=0,
            distance=_CAM_DISTANCE,
            elevation=_CAM_ELEVATION,
            azimuth=_CAM_AZIMUTH,
            enable_shadows=False,
        )
        renderer = OffscreenRenderer(
            model=env.sim.mj_model,
            cfg=viewer,
            scene=env.scene,
            sim_model=env.sim.model,
            expanded_fields=env.sim.expanded_fields,
        )
        renderer.initialize()
        callback = env.update_visualizers if draw_goals else None
        renderer.update(env.sim.data, debug_vis_callback=callback)
        img = renderer.render()
        renderer.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(path, img)
        return None
    except Exception as exc:  # noqa: BLE001 - render failure must not abort the test
        return f"{type(exc).__name__}: {exc}"
