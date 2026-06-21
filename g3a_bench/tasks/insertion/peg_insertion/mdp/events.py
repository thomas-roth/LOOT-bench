from __future__ import annotations

import mujoco
import torch

# MetaWorld's SawyerXYZEnv._reset_hand(): mocap target + open-gripper ctrl
# while the <weld body1="mocap" body2="hand"> equality constraint pulls the
# arm joints (right_j0-right_j6) from qpos0=0 into a reaching pose.
_HAND_INIT_POS = (0.0, 0.6, 0.2)
_HAND_INIT_QUAT = (1.0, 0.0, 1.0, 0.0)
_HAND_RESET_STEPS = 100


# MetaWorld peg-insert-side-v3 sampling ranges (obj_low/high, goal_low/high)
# from sawyer_peg_insertion_side_v3.py. Used when ``randomize=True``.
_PEG_X_RANGE = (0.0, 0.2)
_PEG_Y_RANGE = (0.5, 0.7)
_BOX_X_RANGE = (-0.35, -0.25)
_BOX_Y_RANGE = (0.4, 0.7)
_MIN_PEG_BOX_DIST = 0.1  # native rejects samples closer than this in xy

# Deterministic fallback poses (used when ``randomize=False``): the peg lines
# up directly in front of the hole (shared y) for a clean, reproducible scene.
_PEG_FIXED_XY = (0.04132, 0.66940)
_BOX_FIXED_XY = (-0.28088, 0.66940)

# Box orientation: 90 deg about z (matches the old euler="0 0 1.57"), as a
# (w, x, y, z) quaternion. Applied via the box's mocap pose.
_BOX_QUAT = (0.7073883, 0.0, 0.0, 0.7068252)


def _sample_peg_box_xy() -> tuple[tuple[float, float], tuple[float, float]]:
    """Sample peg and box xy from MetaWorld's ranges, rejecting pairs that are
    closer than _MIN_PEG_BOX_DIST in xy (same rejection loop as native)."""
    while True:
        peg_xy = (
            float(torch.empty(1).uniform_(*_PEG_X_RANGE)),
            float(torch.empty(1).uniform_(*_PEG_Y_RANGE)),
        )
        box_xy = (
            float(torch.empty(1).uniform_(*_BOX_X_RANGE)),
            float(torch.empty(1).uniform_(*_BOX_Y_RANGE)),
        )
        dx, dy = peg_xy[0] - box_xy[0], peg_xy[1] - box_xy[1]
        if (dx * dx + dy * dy) ** 0.5 >= _MIN_PEG_BOX_DIST:
            return peg_xy, box_xy


def reset_peg_and_box(
    env,
    env_ids: torch.Tensor,
    randomize: bool = False,
) -> None:
    # Peg / box / goal / mocap now live in separate mjlab entities, each
    # attached under its own "<entity>/" name prefix.
    sim = env.unwrapped.sim
    mj_model = sim.mj_model

    peg_jnt_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "peg/peg_free")
    peg_qpos_adr = int(mj_model.jnt_qposadr[peg_jnt_id])
    peg_dof_adr = int(mj_model.jnt_dofadr[peg_jnt_id])
    goal_sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "world/goal")
    peg_head_sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "peg/pegHead")
    mocap_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "sawyer/mocap")
    mocap_idx = int(mj_model.body_mocapid[mocap_bid])
    # The box is a fixed-base entity, so mjlab wraps it in a "box/mocap_base"
    # mocap body; its world pose is driven through that mocap, not body_pos.
    box_mocap_bid = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_BODY, "box/mocap_base"
    )
    box_mocap_idx = int(mj_model.body_mocapid[box_mocap_bid])

    device = env_ids.device
    num_envs = env.num_envs

    if not hasattr(env.unwrapped, "target_pos"):
        env.unwrapped.target_pos = torch.zeros(num_envs, 3, device=device)
        env.unwrapped.in_place_margin = torch.ones(num_envs, device=device)
        env.unwrapped.obj_init_pos = torch.zeros(num_envs, 3, device=device)

    # Reset the hand first, while the peg/box are still at their stable
    # XML-default poses, exactly as MetaWorld's reset_model() does
    # (_reset_hand() runs before _set_obj_xyz()/box placement).
    for env_id in env_ids:
        i = int(env_id)
        sim.data.mocap_pos[i, mocap_idx] = torch.tensor(_HAND_INIT_POS, device=device)
        sim.data.mocap_quat[i, mocap_idx] = torch.tensor(_HAND_INIT_QUAT, device=device)

    sim.data.ctrl[env_ids, 0] = -1.0
    sim.data.ctrl[env_ids, 1] = 1.0
    for _ in range(_HAND_RESET_STEPS):
        sim.step()

    # Now teleport the peg/box/goal into their randomized poses (no further
    # stepping), matching MetaWorld's _set_obj_xyz() + box/goal placement.
    for env_id in env_ids:
        i = int(env_id)

        # Like MetaWorld: randomize=True samples peg/box from the task ranges
        # (with the peg-box distance rejection); randomize=False uses a fixed,
        # reproducible layout with the peg lined up in front of the hole.
        if randomize:
            (peg_x, peg_y), (box_x, box_y) = _sample_peg_box_xy()
        else:
            (peg_x, peg_y), (box_x, box_y) = _PEG_FIXED_XY, _BOX_FIXED_XY
        peg_z = 0.02
        box_z = 0.0001

        sim.data.qpos[i, peg_qpos_adr : peg_qpos_adr + 7] = torch.tensor(
            [peg_x, peg_y, peg_z, 1.0, 0.0, 0.0, 0.0], device=device
        )
        sim.data.qvel[i, peg_dof_adr : peg_dof_adr + 6] = 0.0
        env.unwrapped.obj_init_pos[i] = torch.tensor(
            [peg_x, peg_y, peg_z], device=device
        )

        sim.data.mocap_pos[i, box_mocap_idx] = torch.tensor(
            [box_x, box_y, box_z], device=device
        )
        sim.data.mocap_quat[i, box_mocap_idx] = torch.tensor(_BOX_QUAT, device=device)

        target = torch.tensor([box_x + 0.030, box_y, box_z + 0.130], device=device)
        sim.model.site_pos[i, goal_sid] = target
        env.unwrapped.target_pos[i] = target

    sim.forward()

    for env_id in env_ids:
        i = int(env_id)
        peg_head_pos = sim.data.site_xpos[i, peg_head_sid].clone()
        env.unwrapped.in_place_margin[i] = torch.norm(
            peg_head_pos - env.unwrapped.target_pos[i]
        ).clamp(min=1e-3)
