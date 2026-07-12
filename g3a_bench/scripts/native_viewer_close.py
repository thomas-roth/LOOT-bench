"""Safe shutdown for MuJoCo's passive native viewer.

``mujoco.viewer.Handle.close()`` only *requests* teardown: the actual window
destruction happens afterwards on the viewer's daemon GLFW thread. If the
Python interpreter exits while that thread is still inside
``eglSwapBuffers`` (routine right after a close request), the C runtime's
shared-library destructors (mesa/EGL) tear the display down under it and the
thread segfaults. The kernel then writes a core dump of the whole process —
multiple GB with a loaded sim — during which the window sits frozen (the
compositor reports the viewer as not responding) and the machine stutters.

Ctrl+C in the terminal hits this window almost every time, because the main
thread reaches interpreter exit milliseconds after ``Handle.close()``.
Closing via the window manager rarely does: teardown completes while the env
is still being closed. The fix is to block until teardown actually finished
before letting the interpreter exit.
"""

import sys
import time

from mjlab.viewer import NativeMujocoViewer

TEARDOWN_TIMEOUT = 5.0


def wait_for_passive_viewer_teardown(
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


class SafeCloseNativeViewer(NativeMujocoViewer):
    """NativeMujocoViewer whose close() waits for the GLFW thread teardown."""

    def close(self) -> None:
        handle = self.viewer
        super().close()
        if handle is not None:
            wait_for_passive_viewer_teardown(handle)
