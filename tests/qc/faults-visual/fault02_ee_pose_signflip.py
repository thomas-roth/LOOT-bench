"""Fault 02, before and after: does the gripper pose jump while the arm is still?

Drives the arm through one smooth motion and reads the grasp site both ways:

- **before**: the site quaternion, which mjlab builds with an ``argmax`` over
  four candidates that tie at this pose, so float noise decides the sign.
- **after**: the 6D rotation the observation publishes today, which covers
  each rotation once and has no sign to choose.

Writes the two traces with their step-to-step change, and the two rendered
frames on either side of the first jump - the same arm, twice. See
tests/qc/faults-visual/README.md.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from _visual import (
    AFTER_COLOR,
    BEFORE_COLOR,
    apply_pose,
    build_env,
    common_parser,
    montage,
    pick_device,
    render_pose,
    save_fig,
    write_result,
)
from mjlab.utils.lab_api.math import matrix_from_quat

from g3a_bench.tasks.common.observations import rot6d_from_matrix

_FLIP = 1.0
"""Step-to-step change [-] above which the pose can only have changed sign:
the pose itself moves by about 0.005 per step on this sweep."""


def _short(values: torch.Tensor) -> str:
    """A tensor as a short, fixed-width row of numbers."""
    return "(" + ", ".join(f"{float(v):+.3f}" for v in values) + ")"


def _sweep(env, joint: int, low: float, high: float, samples: int):
    """Move one joint smoothly and read the grasp site at every sample."""
    robot = env.scene["robot"]
    site = list(robot.site_names).index("grasp_site")
    poses, quats = [], []
    for value in np.linspace(low, high, samples):
        q = robot.data.default_joint_pos.clone()
        q[0, joint] = float(value)
        apply_pose(env, q)
        poses.append(q.clone())
        quats.append(robot.data.site_quat_w[0, site].clone())
    return poses, torch.stack(quats)


def main() -> None:
    parser = common_parser(__doc__ or "")
    parser.add_argument("--joint", type=int, default=4, help="Joint to sweep [4].")
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    env = build_env(pick_device(args))
    poses, quat = _sweep(env, args.joint, -0.6, 0.6, args.samples)
    joint_name = env.scene["robot"].joint_names[args.joint]

    # Exactly the two forms the observation can publish. The 6D form goes
    # through the matrix, which drops the sign the branch invented.
    rot6d = rot6d_from_matrix(matrix_from_quat(quat))
    quat_step = (quat[1:] - quat[:-1]).norm(dim=-1).cpu().numpy()
    rot6d_step = (rot6d[1:] - rot6d[:-1]).norm(dim=-1).cpu().numpy()
    flips = np.flatnonzero(quat_step > _FLIP)

    images = []
    if not args.no_render and flips.size:
        k = int(flips[0])
        before = out / "fault02_before.png"
        after = out / "fault02_after.png"
        warn = render_pose(env, poses[k], before)
        warn = render_pose(env, poses[k + 1], after) or warn
        if warn:
            print(f"  (render skipped: {warn})")
        montage(
            [before, after],
            [
                f"step {k}: quaternion {_short(quat[k])}",
                f"step {k + 1}: quaternion {_short(quat[k + 1])} - "
                "the same rotation, negated",
            ],
            out / "fault02_frames.png",
            f"Fault 02 - one step of {joint_name}: the arm barely moves, the "
            f"quaternion changes by {quat_step[k]:.2f}",
        )
        images.append(out / "fault02_frames.png")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    steps = np.arange(quat.shape[0])
    for column in range(4):
        axes[0].plot(steps, quat[:, column].cpu().numpy(), lw=1.2)
    axes[0].set_title(
        f"BEFORE: site quaternion (w, x, y, z) - {flips.size} sign flips in "
        f"{len(steps) - 1} steps",
        color=BEFORE_COLOR,
    )
    for column in range(6):
        axes[1].plot(steps, rot6d[:, column].cpu().numpy(), lw=1.2)
    axes[1].set_title(
        "AFTER: the 6D rotation the observation publishes", color=AFTER_COLOR
    )
    axes[2].semilogy(
        steps[1:],
        np.maximum(quat_step, 1e-9),
        color=BEFORE_COLOR,
        lw=1.2,
        label="quaternion",
    )
    axes[2].semilogy(
        steps[1:],
        np.maximum(rot6d_step, 1e-9),
        color=AFTER_COLOR,
        lw=1.2,
        label="6D rotation",
    )
    axes[2].axhline(_FLIP, color="#77828f", ls=":", lw=1.0)
    axes[2].set_title("change per step - one physical step of the same smooth motion")
    axes[2].set_xlabel(f"sample along a {joint_name} sweep of 1.2 rad")
    axes[2].legend(loc="center right")
    fig.suptitle(
        "Fault 02 - the same pose reaching the policy as two observations", fontsize=13
    )
    save_fig(fig, out / "fault02_trace.png")
    images.insert(0, out / "fault02_trace.png")

    write_result(
        out,
        "fault02",
        "Fault 02 - the gripper orientation flipped sign",
        "Does one smooth motion produce a jump in what the policy reads?",
        [
            f"before: {flips.size} jumps above {_FLIP} in {len(steps) - 1} steps "
            f"({100.0 * flips.size / (len(steps) - 1):.0f} % of steps), "
            f"largest {quat_step.max():.2f}, median step {np.median(quat_step):.4f}",
            f"after:  largest step {rot6d_step.max():.4f}, "
            f"median {np.median(rot6d_step):.4f}, no jump",
            f"swept {joint_name} from -0.6 to 0.6 rad in {args.samples} samples, "
            "kinematics only",
        ],
        images,
    )
    env.close()


if __name__ == "__main__":
    main()
