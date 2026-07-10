"""Play entry point: mjlab's player with the g3a-bench tasks registered."""

import functools
import time

import mjlab.scripts.play as _mjlab_play
import mujoco.viewer
from mjlab.viewer import NativeMujocoViewer

import g3a_bench  # noqa: F401  (registers benchmark tasks)


class _ThrottledOverlayNativeViewer(NativeMujocoViewer):
    """NativeMujocoViewer whose status overlay can't throttle the main loop.

    Stock mjlab calls ``viewer.set_texts(...)`` on every rendered frame, and
    MuJoCo's passive-viewer ``set_texts`` spin-waits until the render thread
    (vsync'd to the monitor) consumes it, so the whole python loop is silently
    capped at the monitor refresh rate no matter what ``frame_rate`` is
    requested. Refreshing the overlay at a fixed low rate removes that coupling;
    the overlay's contents (FPS/RTF stats) only change every 0.5 s anyway.
    """

    OVERLAY_INTERVAL = 0.25  # seconds between set_texts calls (4 Hz)

    _last_overlay_time = 0.0

    def _set_status_overlay(self, viewer: mujoco.viewer.Handle) -> None:
        now = time.perf_counter()
        if now - self._last_overlay_time < self.OVERLAY_INTERVAL:
            return
        self._last_overlay_time = now
        super()._set_status_overlay(viewer)


VIEWER_FPS = 240.0


def main():
    """Wrap mjlab's play entry point with an uncapped-FPS native viewer.

    mjlab's NativeMujocoViewer hardcodes ``frame_rate=60`` with no CLI
    override, and its per-frame status overlay additionally locks the loop
    to the monitor refresh rate (see _ThrottledOverlayNativeViewer). We
    always swap in the throttled-overlay viewer at VIEWER_FPS.
    """
    _mjlab_play.NativeMujocoViewer = functools.partial(
        _ThrottledOverlayNativeViewer, frame_rate=VIEWER_FPS
    )
    _mjlab_play.main()


if __name__ == "__main__":
    main()
