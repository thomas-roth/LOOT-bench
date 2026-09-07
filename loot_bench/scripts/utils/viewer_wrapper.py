"""Native viewer with safe teardown and uncapped FPS.

close() waits for GLFW teardown; overlays refresh at low fixed rates.
"""

import sys
import time

import mujoco.viewer
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.native.viewer import compute_viewports

TEARDOWN_TIMEOUT = 5.0  # seconds to wait for teardown of viewer thread


def wait_for_viewer_teardown(viewer_handle: object) -> bool:
    """Wait for GLFW teardown; True if done within TEARDOWN_TIMEOUT."""

    sim_ref = getattr(viewer_handle, "_sim", None)
    if sim_ref is None:
        print(
            "[loot] WARNING: viewer handle has no _sim weakref (mujoco API change?); "
            "teardown wait skipped. Run tests/viewer/check_viewer_shutdown.py",
            file=sys.stderr,
        )
        return False

    deadline = time.perf_counter() + TEARDOWN_TIMEOUT
    while sim_ref() is not None:
        if time.perf_counter() > deadline:
            print(
                f"[loot] WARNING: viewer teardown not finished after {TEARDOWN_TIMEOUT:g}s; "
                "exiting anyway. May lead to segfault.",
                file=sys.stderr,
            )
            return False
        time.sleep(0.01)
    return True


class NativeViewerWrapper(NativeMujocoViewer):
    """NativeMujocoViewer with safe teardown and throttled overlays."""

    VIEWER_FRAME_RATE = 1000.0  # frame rate of viewer (1000 is functionally uncapped)
    OVERLAY_INTERVAL = 0.25  # seconds between set_texts calls (4 Hz)
    FIGURES_INTERVAL = 0.067  # seconds between set_figures calls (~15 Hz)

    def __init__(
        self,
        env,
        policy,
        frame_rate: float = VIEWER_FRAME_RATE,
        **kwargs,
    ):
        super().__init__(env, policy, frame_rate=frame_rate, **kwargs)
        self._last_overlay_time = 0.0
        self._last_figures_time = 0.0

    def _setup_camera(self) -> None:
        # Stock mjlab forces opt.frame = mjFRAME_WORLD, which paints a permanent
        # RGB axis triad at the world origin. Clear it; the frame key still cycles.
        super()._setup_camera()
        if self.viewer is not None:
            self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE.value

    def close(self) -> None:
        # Read handle first: super().close() clears self.viewer.
        viewer_handle = self.viewer
        super().close()

        if viewer_handle is not None:
            wait_for_viewer_teardown(viewer_handle)

    def _set_status_overlay(self, viewer: mujoco.viewer.Handle) -> None:
        # Throttled to avoid vsync cap on monitor refresh.

        now = time.perf_counter()
        if now - self._last_overlay_time < self.OVERLAY_INTERVAL:
            return
        self._last_overlay_time = now

        super()._set_status_overlay(viewer)

    def _update_reward_figures(self, viewer: mujoco.viewer.Handle) -> None:
        # Throttled to avoid vsync cap on monitor refresh.

        if not self._show_plots or not self._term_names:
            if self._figures_dirty:
                viewer.set_figures([])
                self._figures_dirty = False
            return

        if not self._is_paused:
            terms = self.env.unwrapped.reward_manager.get_active_iterable_terms(
                self.env_idx
            )
            for name, arr in terms:
                if name in self._histories:
                    self._append_point(name, float(arr[0]))

        now = time.perf_counter()
        if now - self._last_figures_time < self.FIGURES_INTERVAL:
            return
        self._last_figures_time = now

        for name in self._term_names:
            if name in self._histories:
                self._write_history_to_figure(name)

        viewports = compute_viewports(
            len(self._term_names), viewer.viewport, self._plot_cfg
        )
        viewport_figs = [
            (viewports[i], self._figures[self._term_names[i]])
            for i in range(
                min(len(viewports), len(self._term_names), self._plot_cfg.max_viewports)
            )
        ]
        viewer.set_figures(viewport_figs)
        self._figures_dirty = True
