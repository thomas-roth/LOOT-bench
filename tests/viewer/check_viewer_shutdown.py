"""Check that closing the native viewer doesn't segfault at interpreter exit.

See tests/viewer/README.md.
"""

import argparse
import subprocess
import sys
import time

CLOSE_MARKER = "CHILD_CLOSING"


def run_child(safe: bool) -> None:
    """Open a tiny passive viewer, close it, and return straight into
    interpreter exit — the race the safe teardown wait prevents."""
    import mujoco
    import mujoco.viewer

    from g3a_bench.scripts.viewer_wrapper import (
        NativeViewerWrapper,
        _wait_for_viewer_teardown,
    )

    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <light pos="0 0 3"/>
        <geom type="plane" size="2 2 .1"/>
        <body pos="0 0 0.5"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
      </worldbody>
    </mujoco>
    """)

    data = mujoco.MjData(model)
    viewer = mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    )

    for _ in range(60):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.005)

    print(CLOSE_MARKER, flush=True)
    viewer.close()

    if safe:
        _wait_for_viewer_teardown(viewer, NativeViewerWrapper.TEARDOWN_TIMEOUT)
    # interpreter exits right after this returns; without the wait the GLFW
    # thread is still mid-swap and mesa's exit-time destructors segfault it


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
        default=15.0,
        help="max seconds between the child's close() and its exit",
    )
    # Internal flags for the self-spawned child process.
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--safe", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        run_child(safe=args.safe)
        return 0

    child_args = [sys.executable, __file__, "--child"]
    if not args.show_bug:
        child_args.append("--safe")

    proc = subprocess.Popen(child_args, stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.rstrip("\n") == CLOSE_MARKER:
            break
        print(line, end="")  # pass through any other child output
    else:
        # stdout hit EOF without the marker: child died before close().
        return_code = proc.wait()
        print(f"FAIL: child exited (return code {return_code}) before reaching close()")
        return 1

    t_close = time.perf_counter()
    try:
        proc.communicate(timeout=args.timeout)
        return_code = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        print(
            f"FAIL: child still running {args.timeout:g}s after close() (hung teardown)"
        )
        return 1
    dt = time.perf_counter() - t_close

    if args.show_bug:
        if return_code != 0:
            print(
                f"PASS: show-bug child exited (return code {return_code}) {dt:.1f}s after close()"
            )
            return 0
        print(
            f"FAIL: show-bug child exited cleanly (return code {return_code}) {dt:.1f}s after close(). "
            "Race did not reproduce (timing, or the viewer stack changed)"
        )
        return 1

    if return_code == 0:
        print(
            f"PASS: safe-close child exited cleanly (return code {return_code}) {dt:.1f}s after close()"
        )
        return 0
    print(
        f"FAIL: safe-close child exited (return code {return_code}) {dt:.1f}s after close()"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
