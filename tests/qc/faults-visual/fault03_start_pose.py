"""Fault 03, before and after: do the demonstrations and the tests start together?

Reads the first frame of every episode of the two recorded sessions - one from
before the rebuild, one from after - and compares it with where an evaluation
episode actually begins. The observation block is ``joint_pos_rel``, the joint
position minus the keyframe, so a demonstration that starts where the
evaluation starts reads zero.

- **before**: the operator aligned the leader arm to the rest pose, while the
  reset put the robot at the keyframe. 1.7 rad apart on two joints, and the
  same pose in every episode.
- **after**: both are the home pose, plus the task's own start-pose sample.

Writes a per-joint gap chart and the two starting poses side by side. See
tests/qc/faults-visual/README.md.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from _visual import (
    AFTER_COLOR,
    BEFORE_COLOR,
    DATASET_AFTER,
    DATASET_BEFORE,
    build_env,
    common_parser,
    first_frames,
    montage,
    obs_slice,
    pick_device,
    render_pose,
    save_fig,
    write_result,
)


def _reset_starts(env, resets: int) -> np.ndarray:
    """Where an evaluation episode really begins, over *resets* resets."""
    robot = env.scene["robot"]
    starts = []
    for _ in range(resets):
        env.reset()
        starts.append(
            (robot.data.joint_pos[0] - robot.data.default_joint_pos[0])
            .detach()
            .cpu()
            .numpy()
        )
    return np.stack(starts)


def main() -> None:
    parser = common_parser(__doc__ or "")
    parser.add_argument("--resets", type=int, default=20)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    env = build_env(pick_device(args))
    robot = env.scene["robot"]
    names = list(robot.joint_names)
    resets = _reset_starts(env, args.resets)

    # Both sessions put ``joint_pos`` first, although the rest of the
    # observation changed shape between them, so one slice reads both.
    joints = obs_slice(env, "joint_pos")
    before = first_frames(DATASET_BEFORE, joints.stop - joints.start)
    after = first_frames(DATASET_AFTER, joints.stop - joints.start)

    gap_before = np.abs(before).max(axis=0)
    gap_after = np.abs(after).max(axis=0)
    spread_before = np.ptp(before, axis=0)
    spread_after = np.ptp(after, axis=0)

    images = []
    if not args.no_render:
        keyframe = robot.data.default_joint_pos.clone()
        demo_old = keyframe.clone()
        demo_old[0, : before.shape[1]] += torch.as_tensor(
            before[0], dtype=keyframe.dtype, device=keyframe.device
        )
        demo_new = keyframe.clone()
        demo_new[0, : after.shape[1]] += torch.as_tensor(
            after[0], dtype=keyframe.dtype, device=keyframe.device
        )
        paths = [
            out / "fault03_before_demo.png",
            out / "fault03_before_eval.png",
            out / "fault03_after_demo.png",
            out / "fault03_after_eval.png",
        ]
        warn = render_pose(env, demo_old, paths[0])
        warn = render_pose(env, keyframe, paths[1]) or warn
        warn = render_pose(env, demo_new, paths[2]) or warn
        env.reset()
        warn = render_pose(env, robot.data.joint_pos.clone(), paths[3]) or warn
        if warn:
            print(f"  (render skipped: {warn})")
        montage(
            paths[:2],
            [
                "BEFORE: where a demonstration began",
                "BEFORE: where a test episode began",
            ],
            out / "fault03_before.png",
            "Fault 03 - the two starting poses before the rebuild",
        )
        montage(
            paths[2:],
            [
                "AFTER: where a demonstration begins",
                "AFTER: where a test episode begins",
            ],
            out / "fault03_after.png",
            "Fault 03 - the two starting poses today",
        )
        images += [out / "fault03_before.png", out / "fault03_after.png"]

    index = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].bar(index - 0.2, gap_before, 0.4, color=BEFORE_COLOR, label="before")
    axes[0].bar(index + 0.2, gap_after, 0.4, color=AFTER_COLOR, label="after")
    axes[0].set_title("distance from the demo start to the test start [rad]")
    axes[0].annotate(
        "after: the 0.05 is the start-pose sample, on purpose",
        xy=(0.02, 0.86),
        xycoords="axes fraction",
        fontsize=9,
        color=AFTER_COLOR,
    )
    axes[1].bar(index - 0.2, spread_before, 0.4, color=BEFORE_COLOR, label="before")
    axes[1].bar(index + 0.2, spread_after, 0.4, color=AFTER_COLOR, label="after")
    axes[1].set_title("variety of the demo start, peak to peak [rad]")
    for ax in axes:
        ax.set_xticks(index)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.legend()
    fig.suptitle(
        f"Fault 03 - {DATASET_BEFORE.name} ({before.shape[0]} episodes) against "
        f"{DATASET_AFTER.name} ({after.shape[0]} episodes)",
        fontsize=12,
    )
    save_fig(fig, out / "fault03_gap.png")
    images.insert(0, out / "fault03_gap.png")

    worst_before = int(np.argmax(gap_before))
    write_result(
        out,
        "fault03",
        "Fault 03 - demonstrations and tests began 1.7 rad apart",
        "Does a demonstration begin where a test episode begins?",
        [
            f"before: largest gap {gap_before[worst_before]:.3f} rad on "
            f"{names[worst_before]}; start variety {spread_before.max():.3f} rad "
            f"over {before.shape[0]} episodes",
            f"after:  largest gap {gap_after.max():.3f} rad; start variety "
            f"{spread_after.max():.3f} rad over {after.shape[0]} episodes",
            f"live resets ({args.resets}): largest distance from the keyframe "
            f"{np.abs(resets).max():.3f} rad, spread "
            f"{np.ptp(resets, axis=0).max():.3f} rad",
            "peak to peak, never a standard deviation: float32 samples that never "
            "vary still return about 1e-6 (ADR-008)",
        ],
        images,
    )
    env.close()


if __name__ == "__main__":
    main()
