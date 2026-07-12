# Native-viewer verification

Manual checks for the native MuJoCo viewer as wrapped by `g3a-play`. Both
need a display and are **not for CI**; run them from the **repo root**.

---

## FPS check (`measure_viewer_fps.py`)

Measures the real frame rate of the native MuJoCo viewer (the one `g3a-play`
uses) and prints the median. Run it after touching
`g3a_bench/scripts/play.py` or bumping mjlab, and on any machine where the
viewer seems stuck at the monitor refresh rate (60 fps).

It runs a 1-env zero-policy task at 1/32x slow-motion by default, so the render
loop is not physics-bound even on CPU-only machines — it measures the renderer
cap, not sim throughput. `--viewer stock` swaps in mjlab's unpatched viewer,
which is expected to pin at the monitor refresh rate (that's the bug the play
wrapper fixes); `--plots` exercises the reward-figures path, which has the same
spin-wait coupling.

```bash
uv run python tests/viewer/measure_viewer_fps.py --min-fps 61   # expect >> 60
uv run python tests/viewer/measure_viewer_fps.py --viewer stock # shows the 60 cap
uv run python tests/viewer/measure_viewer_fps.py --plots --min-fps 61
```

---

## Shutdown check (`check_viewer_shutdown.py`)

Checks that closing the native viewer and exiting immediately (the Ctrl+C
path) does not segfault. Run it after touching
`g3a_bench/scripts/native_viewer_close.py` or upgrading mjlab/mujoco/glfw —
the fix relies on mujoco's private `Handle._sim` weakref, so a dependency
upgrade can silently disable it (it warns on stderr if that happens).

The why and the fix live in the module docstring of
`g3a_bench/scripts/native_viewer_close.py`: `Handle.close()` only *requests*
teardown, and exiting while the GLFW thread is still rendering segfaults in
mesa's exit-time destructors. `SafeCloseNativeViewer` waits for the teardown
before returning from `close()`; every native-viewer subclass in this repo
(the g3a-play wrapper, the live-bridge recorders/replayers, and these test
scripts) derives from it.

```bash
uv run python tests/viewer/check_viewer_shutdown.py            # expect PASS
uv run python tests/viewer/check_viewer_shutdown.py --show-bug # passes when
    # the child segfaults (demonstrates the bug the safe close prevents)
```
