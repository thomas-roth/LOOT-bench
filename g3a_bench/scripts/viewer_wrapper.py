"""Safe shutdown + uncapped-FPS native viewer.

The two fixes over mjlab's stock ``NativeMujocoViewer``:

**Safe teardown.** ``mujoco.viewer.Handle.close()`` only *requests* teardown:
the actual window destruction happens afterwards on the viewer's daemon GLFW
thread. If the Python interpreter exits while that thread is still inside
``eglSwapBuffers`` (routine right after a close request), the C runtime's
shared-library destructors (mesa/EGL) tear the display down under it and the
thread segfaults. The kernel then writes a core dump of the whole process —
multiple GB with a loaded sim — during which the window sits frozen (the
compositor reports the viewer as not responding) and the machine stutters.
Ctrl+C in the terminal hits this window almost every time, because the main
thread reaches interpreter exit milliseconds after ``Handle.close()``.
The fix is to block until teardown actually finished before letting the
interpreter exit.

**Throttled overlays.** Stock mjlab calls ``viewer.set_texts(...)`` on every
rendered frame (and ``viewer.set_figures(...)`` on every frame while reward
plots are shown). MuJoCo's passive-viewer ``set_texts``/``set_figures``
spin-wait until the render thread (vsync'd to the monitor) consumes them, so
the whole python loop is silently capped at the monitor refresh rate no matter
what ``frame_rate`` is requested. Refreshing the overlays at a fixed low rate
removes that coupling.
"""

import os
import sys
import time

import mujoco.viewer
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.native.viewer import compute_viewports

TEARDOWN_TIMEOUT = 5.0


def _wait_for_passive_viewer_teardown(
    handle: object, timeout: float = TEARDOWN_TIMEOUT
) -> bool:
    """Block until the passive viewer's GLFW thread finished tearing down.

    The ``Handle`` keeps only a weakref to the C++ ``Simulate`` object; the
    strong reference lives in the GLFW thread's launch frame and is dropped
    once ``simulate.destroy()`` completed. The weakref clearing is therefore
    a reliable "teardown done" signal. Returns True if teardown completed
    within ``timeout``, False if the wait was skipped or timed out (in both
    cases exiting may still segfault); failures are also warned to stderr so
    they don't pass silently.
    """
    sim_ref = getattr(handle, "_sim", None)
    if sim_ref is None:
        print(
            "[g3a] WARNING: viewer Handle has no _sim weakref (mujoco API "
            "change?); teardown wait skipped — run "
            "tests/viewer/check_viewer_shutdown.py",
            file=sys.stderr,
        )
        return False
    deadline = time.perf_counter() + timeout
    while sim_ref() is not None:
        if time.perf_counter() > deadline:
            print(
                f"[g3a] WARNING: viewer teardown not finished after {timeout:g}s; "
                "exiting anyway (may segfault)",
                file=sys.stderr,
            )
            return False
        time.sleep(0.01)
    return True


class NativeViewerWrapper(NativeMujocoViewer):
    """NativeMujocoViewer with safe teardown and uncapped FPS.

    Combines two fixes over the stock viewer:

    1. ``close()`` waits for the GLFW thread teardown to finish before
       returning, preventing a segfault on Ctrl+C or fast interpreter exit.
    2. UI overlays (``set_texts``) are refreshed at a fixed low rate
       (``OVERLAY_INTERVAL`` = 0.25 s) and reward figures (``set_figures``)
       at ``FIGURES_INTERVAL`` = 0.1 s so neither can spin-wait on the render
       thread and silently cap the simulation loop to the monitor refresh rate.

    The frame-rate defaults to ``VIEWER_FPS`` (1000) and can be overridden
    via the ``G3A_VIEWER_FPS`` environment variable.
    """

    VIEWER_FPS = 1000.0
    OVERLAY_INTERVAL = 0.25  # seconds between set_texts calls (4 Hz)
    FIGURES_INTERVAL = 0.067  # seconds between set_figures calls (~15 Hz)

    def __init__(
        self,
        env,
        policy,
        frame_rate: float | None = None,
        **kwargs,
    ):
        if frame_rate is None:
            frame_rate = float(os.environ.get("G3A_VIEWER_FPS", self.VIEWER_FPS))
        super().__init__(env, policy, frame_rate=frame_rate, **kwargs)
        self._last_overlay_time = 0.0
        self._last_figures_time = 0.0

    def close(self) -> None:
        handle = self.viewer
        super().close()
        if handle is not None:
            _wait_for_passive_viewer_teardown(handle)

    def _set_status_overlay(self, viewer: mujoco.viewer.Handle) -> None:
        # Throttle set_texts to avoid spin-wait cap (vsync lock at 60 Hz).
        # Would otherwise result in sim being capped to monitor's refresh rate when left overlay shown.
        now = time.perf_counter()
        if now - self._last_overlay_time < self.OVERLAY_INTERVAL:
            return
        self._last_overlay_time = now
        super()._set_status_overlay(viewer)

    def _update_reward_figures(self, viewer: mujoco.viewer.Handle) -> None:
        # Throttle set_figures to avoid spin-wait cap (same reason as overlay).
        # Would otherwise result in sim being capped to monitor's refresh rate when reward plots shown.
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
