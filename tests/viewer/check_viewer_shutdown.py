"""Check that closing the native viewer doesn't segfault at interpreter exit
(needs a display; not for CI).

Regression check for the Ctrl+C shutdown crash — see the module docstring of
``g3a_bench/scripts/viewer_wrapper.py`` for the full diagnosis. The
child process below opens a tiny passive viewer, closes it, and exits
immediately — the exact race. The parent asserts a clean, prompt exit.

Usage (from repo root):
  uv run python tests/viewer/check_viewer_shutdown.py            # expect PASS
  uv run python tests/viewer/check_viewer_shutdown.py --show-bug # skips the
      teardown wait; passes when the child segfaults (bug reproduced)
"""

import argparse
import subprocess
import sys
import time

CHILD = """
import sys, time
import mujoco, mujoco.viewer

XML = '''
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="2 2 .1"/>
    <body pos="0 0 1"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
  </worldbody>
</mujoco>
'''
m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)
v = mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False)
for _ in range(60):
    mujoco.mj_step(m, d)
    v.sync()
    time.sleep(0.005)
v.close()
if "--safe" in sys.argv:
    from g3a_bench.scripts.viewer_wrapper import (
        _wait_for_passive_viewer_teardown as wait_for_passive_viewer_teardown,
    )
    wait_for_passive_viewer_teardown(v)
# interpreter exits here; without the wait the GLFW thread is still
# mid-swap and mesa's exit-time destructors segfault it
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--show-bug",
        action="store_true",
        help="run the child without the teardown wait to demonstrate the crash",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="max seconds for the child (a segfault core dump takes far longer)",
    )
    args = ap.parse_args()

    child_args = [sys.executable, "-c", CHILD]
    if not args.show_bug:
        child_args.append("--safe")

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(child_args, timeout=args.timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        print(f"FAIL: child still running after {args.timeout:g}s (hung teardown)")
        return 1
    dt = time.perf_counter() - t0

    sig = f" (signal {-rc})" if rc < 0 else ""
    if args.show_bug:
        # Demo mode: the crash is the expected outcome.
        if rc != 0:
            print(f"PASS: bug reproduced — child exited rc={rc}{sig} in {dt:.1f}s")
            return 0
        print(
            f"FAIL: show-bug child exited cleanly in {dt:.1f}s — race did not "
            "reproduce (timing, or the viewer stack changed)"
        )
        return 1

    if rc == 0:
        print(f"PASS: safe-close child exited cleanly in {dt:.1f}s")
        return 0
    print(f"FAIL: safe-close child exited rc={rc}{sig} after {dt:.1f}s")
    return 1


if __name__ == "__main__":
    sys.exit(main())
