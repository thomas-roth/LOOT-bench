"""Play entry point: mjlab's player with the g3a-bench tasks registered."""

import functools
import os
import time

import mjlab.scripts.play as _mjlab_play
import mujoco.viewer
from mjlab.viewer.native.viewer import compute_viewports

import g3a_bench  # noqa: F401  (registers benchmark tasks)
from g3a_bench.scripts.native_viewer_close import SafeCloseNativeViewer


class _ThrottledOverlayNativeViewer(SafeCloseNativeViewer):
    """NativeMujocoViewer whose UI overlays can't throttle the main loop.

    Stock mjlab calls ``viewer.set_texts(...)`` on every rendered frame (and
    ``viewer.set_figures(...)`` on every frame while reward plots are shown).
    MuJoCo's passive-viewer ``set_texts``/``set_figures`` spin-wait until the
    render thread (vsync'd to the monitor) consumes them, so the whole python
    loop is silently capped at the monitor refresh rate no matter what
    ``frame_rate`` is requested. Refreshing the overlays at a fixed low rate
    removes that coupling; the overlay's stats only change every 0.5 s anyway,
    and reward-plot points are still appended every frame — only the push to
    the render thread is throttled.
    """

    OVERLAY_INTERVAL = 0.25  # seconds between set_texts/set_figures calls (4 Hz)

    _last_overlay_time = 0.0
    _last_figures_time = 0.0

    def _set_status_overlay(self, viewer: mujoco.viewer.Handle) -> None:
        now = time.perf_counter()
        if now - self._last_overlay_time < self.OVERLAY_INTERVAL:
            return
        self._last_overlay_time = now
        super()._set_status_overlay(viewer)

    def _update_reward_figures(self, viewer: mujoco.viewer.Handle) -> None:
        # Reimplements NativeMujocoViewer._update_reward_figures with the
        # spin-waiting set_figures call throttled to OVERLAY_INTERVAL.
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
        if now - self._last_figures_time < self.OVERLAY_INTERVAL:
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


DEFAULT_VIEWER_FPS = 240.0


def main():
    """Wrap mjlab's play entry point with an uncapped-FPS native viewer.

    mjlab's NativeMujocoViewer hardcodes ``frame_rate=60`` with no CLI
    override, and its per-frame status overlay additionally locks the loop
    to the monitor refresh rate (see _ThrottledOverlayNativeViewer). We
    always swap in the throttled-overlay viewer; the frame-rate cap defaults
    to DEFAULT_VIEWER_FPS and can be overridden via G3A_VIEWER_FPS.
    """
    frame_rate = float(os.environ.get("G3A_VIEWER_FPS", DEFAULT_VIEWER_FPS))
    print(
        f"[g3a] native viewer frame_rate={frame_rate:g} (override with G3A_VIEWER_FPS)"
    )
    _mjlab_play.NativeMujocoViewer = functools.partial(
        _ThrottledOverlayNativeViewer, frame_rate=frame_rate
    )
    _mjlab_play.main()


if __name__ == "__main__":
    main()
