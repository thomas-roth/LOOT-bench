# Native-viewer verification

Manual checks for the native MuJoCo viewer as wrapped by `g3a-play`. Both
need a display and are **not for CI**; run them from the **repo root**.

---

## FPS check (`measure_viewer_fps.py`)

Measures the real frame rate of the native MuJoCo viewer wrapper: it prints
the smoothed FPS samples the viewer computes every 0.5 s and reports the median.
Run it after touching `g3a_bench/scripts/viewer_wrapper.py` / `play.py` or
bumping mjlab, and on any machine where the viewer seems stuck at the monitor
refresh rate.

It runs a 1-env zero-policy task at 1/32x slow-motion, so the render loop is
not physics-bound even on CPU-only machines not sim throughput.
`--viewer stock` swaps in mjlab's unpatched native viewer, which is expected to
pin at the monitor refresh rate; `--reward-plots` exercises the reward-figures
path, which has the same spin-wait coupling.

```bash
uv run python tests/viewer/measure_viewer_fps.py                 # expect >> cap
uv run python tests/viewer/measure_viewer_fps.py --viewer stock  # shows the cap
uv run python tests/viewer/measure_viewer_fps.py --reward-plots  # shows reward plots
uv run python tests/viewer/measure_viewer_fps.py --frame-rate 20 # capped low
```

---

## Shutdown check (`check_viewer_shutdown.py`)

Checks that closing the native viewer and exiting immediately (the Ctrl+C
path) does not segfault. Run it after touching
`g3a_bench/scripts/viewer_wrapper.py` or upgrading mjlab/mujoco/glfw. The fix
relies on mujoco's private `Handle._sim` weakref, so a dependency upgrade can
silently disable it (it warns on stderr if that happens).

The script re-invokes itself as a child process that opens a tiny passive
viewer, closes it, and exits immediately. The parent asserts a clean, prompt
exit; `--timeout` is measured from the moment the child reaches `close()` (the
child signals it on stdout), so slow startup/imports don't count against it.

The why and the fix live in the module docstring of
`g3a_bench/scripts/viewer_wrapper.py`: `Handle.close()` only *requests*
teardown, and exiting while the GLFW thread is still rendering segfaults in
mesa's exit-time destructors. `NativeViewerWrapper` waits for the teardown
before returning from `close()`; every native-viewer subclass in this repo
derives from it, and the child here calls the same wait helper directly.

```bash
uv run python tests/viewer/check_viewer_shutdown.py            # expect clean closing
uv run python tests/viewer/check_viewer_shutdown.py --show-bug # expect segfault
```
