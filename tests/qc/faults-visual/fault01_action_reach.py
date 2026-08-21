"""Fault 01, before and after: could the policy command the demonstrated pose?

Takes the recorded actions of one demonstration and asks, joint by joint,
which of them each action convention can command:

- **before**: mjlab's residual convention, ``target = action * 0.07 rad +
  keyframe``. A bounded policy lives in ``[-1, 1]``, so it can only reach a
  narrow band around the keyframe.
- **after**: the current convention, where ``-1`` is a joint's lower limit
  and ``+1`` its upper limit.

Writes a band plot (the demonstration against both reachable bands) and a pair
of pictures: the demonstrated pose at the grasp, and the closest pose the old
convention could have commanded. See tests/qc/faults-visual/README.md.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _visual import (
    AFTER_COLOR,
    BEFORE_COLOR,
    DATASET_AFTER,
    DATASET_BEFORE,
    action_joint_ids,
    build_env,
    common_parser,
    full_pose,
    grasp_site_pos,
    montage,
    obs_slice,
    old_residual_scale,
    pick_device,
    place_object,
    render_pose,
    save_fig,
    write_result,
)


def _xyz(values: np.ndarray) -> str:
    return "(" + ", ".join(f"{float(v):.3f}" for v in values) + ")"


def _legacy_check(
    dataset: Path, names: list[str], out: Path, images: list[Path]
) -> list[str]:
    """The same question asked of the session recorded *before* the rebuild.

    That session stores its actions in the residual convention, normalized by
    the fixed physical bounds it also stores. Undoing that normalization gives
    the raw action the operator flew, and a tanh policy can only emit values
    inside ``[-1, 1]`` of it.
    """
    if not dataset.exists():
        return [f"pre-rebuild session not found at {dataset}, skipped"]
    data = np.load(dataset, allow_pickle=True)
    if "action_min" not in data.files:
        return [f"{dataset.name} carries no residual-convention bounds, skipped"]
    low, high = data["action_min"], data["action_max"]
    raw = (data["actions"] + 1.0) / 2.0 * (high - low) + low
    reachable = int((np.abs(raw) <= 1.0).all(axis=1).sum())

    fig, ax = plt.subplots(figsize=(11, 4.2))
    index = np.arange(raw.shape[1])
    ax.barh(
        index,
        raw.max(axis=0) - raw.min(axis=0),
        left=raw.min(axis=0),
        height=0.5,
        color=BEFORE_COLOR,
        label="actions the operator flew",
    )
    ax.axvspan(-1.0, 1.0, color=AFTER_COLOR, alpha=0.35, label="a bounded policy")
    ax.set_yticks(index)
    ax.set_yticklabels(names[: raw.shape[1]], fontsize=9)
    ax.set_xlabel(
        "action, residual convention - each bar is one joint's whole range, "
        "and a frame counts only when every joint is inside the band at once"
    )
    ax.legend(loc="lower right")
    ax.set_title(
        f"{dataset.name}: {reachable} of {raw.shape[0]} recorded frames sit "
        "inside the box a bounded policy can emit"
    )
    save_fig(fig, out / "fault01_legacy.png")
    images.append(out / "fault01_legacy.png")
    return [
        f"pre-rebuild session ({dataset.name}): {reachable} of {raw.shape[0]} "
        f"frames inside [-1, 1]; the recording runs from {raw.min():.1f} to "
        f"{raw.max():.1f}"
    ]


def main() -> None:
    parser = common_parser(__doc__ or "")
    parser.add_argument("--dataset", type=Path, default=DATASET_AFTER)
    parser.add_argument("--episode", type=int, default=0, help="Episode to plot [0].")
    parser.add_argument(
        "--legacy-dataset",
        type=Path,
        default=DATASET_BEFORE,
        help="The session recorded before the rebuild, read as a second check.",
    )
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    data = np.load(args.dataset, allow_pickle=True)
    # The dataset stores the action, and the mapping it was recorded under, so
    # the commanded joint target is exact rather than inferred.
    target = data["actions"] * data["action_scale"] + data["action_offset"]
    episode = data["episode_index"] == args.episode
    target_ep = target[episode]

    env = build_env(pick_device(args))
    robot = env.scene["robot"]
    ids = action_joint_ids(env)
    names = [robot.joint_names[i] for i in ids]
    term = env.action_manager.get_term("joint_pos")
    keyframe = robot.data.default_joint_pos[0, ids].detach().cpu().numpy()

    old_scale = old_residual_scale(names)
    old_low, old_high = keyframe - old_scale, keyframe + old_scale
    new_low = (-1.0 * term.scale + term.offset)[0].detach().cpu().numpy()
    new_high = (1.0 * term.scale + term.offset)[0].detach().cpu().numpy()

    inside_old = (target >= old_low) & (target <= old_high)
    inside_new = (target >= new_low - 1e-6) & (target <= new_high + 1e-6)
    frames_old = int(inside_old.all(axis=1).sum())
    frames_new = int(inside_new.all(axis=1).sum())
    total = int(target.shape[0])

    # The grasp frame: where the demonstration closes the gripper hardest.
    grip = target_ep[:, names.index("gripper_left")]
    frame = int(np.argmin(grip))
    demo_q = full_pose(env, target_ep[frame], ids)
    old_q = full_pose(env, np.clip(target_ep[frame], old_low, old_high), ids)

    # Put the cube back where the operator saw it, so both pictures show the
    # pose against the object it was aiming at.
    cube = data["observations"][
        np.flatnonzero(episode)[frame], obs_slice(env, "cube_pose")
    ]
    place_object(env, "cube", cube)

    images = []
    if not args.no_render:
        before = out / "fault01_before.png"
        after = out / "fault01_after.png"
        warn = render_pose(env, old_q, before)
        old_site = grasp_site_pos(env)
        warn = render_pose(env, demo_q, after) or warn
        demo_site = grasp_site_pos(env)
        if warn:
            print(f"  (render skipped: {warn})")
        montage(
            [before, after],
            [
                "BEFORE: the closest pose the old action space could command",
                "AFTER: the demonstrated pose, commandable exactly",
            ],
            out / "fault01_poses.png",
            "Fault 01 - the demonstrated grasp pose, under both action conventions",
        )
        images.append(out / "fault01_poses.png")
    else:
        from _visual import apply_pose

        apply_pose(env, old_q)
        old_site = grasp_site_pos(env)
        apply_pose(env, demo_q)
        demo_site = grasp_site_pos(env)

    fig, axes = plt.subplots(4, 2, figsize=(12, 11), sharex=True)
    steps = np.arange(target_ep.shape[0])
    for slot, name in enumerate(names):
        ax = axes.flat[slot]
        ax.axhspan(new_low[slot], new_high[slot], color=AFTER_COLOR, alpha=0.13)
        ax.axhspan(old_low[slot], old_high[slot], color=BEFORE_COLOR, alpha=0.5)
        ax.plot(steps, target_ep[:, slot], color="#12181f", lw=1.4)
        share = 100.0 * inside_old[episode][:, slot].mean()
        ax.set_title(f"{name} - old space holds {share:.0f} % of the demo", fontsize=10)
        ax.set_ylabel("rad")
    axes.flat[4].set_xlabel("step in the episode")
    axes.flat[5].set_xlabel("step in the episode")
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0.0,
        0.5,
        f"black: the demonstration\n"
        f"red band: reachable before (keyframe +/- residual scale)\n"
        f"green band: reachable after (the joint's own travel)\n\n"
        f"whole frames commandable before: {frames_old} of {total}\n"
        f"whole frames commandable after:  {frames_new} of {total}",
        fontsize=11,
        va="center",
    )
    fig.suptitle(
        f"Fault 01 - one demonstration ({args.dataset.name}, episode {args.episode}) "
        "against each action space",
        fontsize=13,
    )
    save_fig(fig, out / "fault01_bands.png")
    images.insert(0, out / "fault01_bands.png")

    legacy_line = _legacy_check(args.legacy_dataset, names, out, images)

    gap = float(np.linalg.norm(demo_site - old_site))
    write_result(
        out,
        "fault01",
        "Fault 01 - the arm could not reach the table",
        "Which demonstrated poses can each action convention command?",
        [
            f"before: {frames_old} of {total} demonstrated frames commandable "
            f"({100.0 * frames_old / total:.1f} %)",
            f"after:  {frames_new} of {total} demonstrated frames commandable "
            f"({100.0 * frames_new / total:.1f} %)",
            "residual scale per joint [rad]: "
            + ", ".join(f"{n}={s:.3f}" for n, s in zip(names, old_scale, strict=True)),
            *legacy_line,
            f"at the demonstrated grasp the old space misses the gripper "
            f"position by {gap:.3f} m (demo {_xyz(demo_site)}, "
            f"best old {_xyz(old_site)})",
        ],
        images,
    )
    env.close()


if __name__ == "__main__":
    main()
