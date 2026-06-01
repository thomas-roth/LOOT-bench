import argparse
import time

import genesis as gs
import numpy as np

gs.init()


def setup_franka(franka):
    # set control gains
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    end_effector = franka.get_link("hand")
    return end_effector


def run_ik_benchmark(steps: int, ik_stride: int, show_viewer: bool) -> None:
    # Rebuild the scene with requested viewer setting.
    scene = gs.Scene(show_viewer=show_viewer)

    scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(0.08, 0.08, 0.08),
            pos=(0.65, 0.0, 0.13),
        )
    )

    scene.build()
    end_effector = setup_franka(franka)

    ik_times_ms = []
    last_qpos = None
    start = time.perf_counter()

    for step_idx in range(steps):
        if step_idx % ik_stride == 0:
            # Use current EEF pose as the target to mimic live pose conversion.
            pos = end_effector.get_pos()
            quat = end_effector.get_quat()

            start = time.perf_counter()
            last_qpos = franka.inverse_kinematics(
                link=end_effector,
                pos=pos,
                quat=quat,
            )
            ik_times_ms.append((time.perf_counter() - start) * 1000.0)

        if last_qpos is not None:
            franka.control_dofs_position(last_qpos)

        scene.step()

    ik_times = np.array(ik_times_ms, dtype=np.float64)
    if ik_times.size == 0:
        print("No IK calls executed. Increase steps or decrease ik_stride.")
        return

    summary = {
        "calls": int(ik_times.size),
        "mean_ms": float(np.mean(ik_times)),
        "median_ms": float(np.median(ik_times)),
        "min_ms": float(np.min(ik_times)),
        "p50_ms": float(np.percentile(ik_times, 50)),
        "p90_ms": float(np.percentile(ik_times, 90)),
        "p99_ms": float(np.percentile(ik_times, 99)),
        "max_ms": float(np.max(ik_times)),
        "max_index": float(np.argmax(ik_times)),
    }
    print(ik_times_ms)
    print("IK benchmark results:")
    for key, value in summary.items():
        if key == "calls":
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure IK latency every N steps.")
    parser.add_argument("--steps", type=int, default=2000, help="Total simulation steps.")
    parser.add_argument("--ik-stride", type=int, default=1, help="Run IK every N steps.")
    parser.add_argument(
        "--show-viewer",
        default=True,
        action="store_true",
        help="Enable the viewer (slower, but useful for debugging).",
    )
    args = parser.parse_args()

    run_ik_benchmark(steps=args.steps, ik_stride=args.ik_stride, show_viewer=args.show_viewer)
