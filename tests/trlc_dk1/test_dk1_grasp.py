# pyright: reportAttributeAccessIssue=false
"""DK1 grasp contact contract on the compiled MuJoCo model.

The sim gripper is a force-capped saturating servo (see dk1_actuators.py):
stiff gains reach the effort cap after a small position error, and the cap
is lowered from the rated 114 N to 20 N. At the rated force the pads sank
~4.5 mm into task pieces and ejected them from the grip (261e0ca); a soft
natural-frequency PD held gently but closed too slowly.

Guards the resulting grasp contract: closing the gripper on a task-piece-
sized box (a) reaches the piece quickly, (b) keeps pad penetration small —
transiently at impact and at steady state, (c) holds the piece at the grasp
site instead of squeezing it out, and (d) keeps the grip force gentle but
firmly nonzero. Physics options mirror the play/live-bridge env, where the
original regression was observed.
"""

import mujoco
import numpy as np
from mjlab.entity.entity import Entity

from g3a_bench.robots.trlc_dk1.dk1_actuators import GRIPPER_COMMAND_RANGE
from g3a_bench.robots.trlc_dk1.dk1_constants import (
    HOME_JOINT_POS,
    get_unimanual_dk1_robot_cfg,
)

_PIECE_HALF_SIZE = 0.02  # [m], 40 mm box, tasks/stack base-cube sized.
_PIECE_MASS = 0.1  # [kg], tasks/stack FORM_MASS["cube"].
_SETTLE_STEPS = 300  # 3 s at the play timestep.
_STEADY_WINDOW = 50

_MAX_STEADY_PENETRATION = 1.5e-3  # [m], pad half-thickness.
_MAX_PEAK_PENETRATION = 3e-3  # [m], impact transient bound.
_MAX_PIECE_DISPLACEMENT = 5e-3  # [m], piece must stay at the grasp site.
_MAX_TIME_TO_CONTACT = 0.3  # [s], fingers must reach the piece quickly.
# Gentle-force regime: ~10 N/finger at the teleop full-close command. The
# ceiling catches a return to crush-force tuning (rated cap: ~57 N/finger),
# the floor catches a gripper too weak to actually grip.
_MIN_GRIP_FORCE_PER_FINGER = 3.0  # [N]
_MAX_GRIP_FORCE_PER_FINGER = 30.0  # [N]


def _build_grasp_model() -> mujoco.MjModel:
    """Unimanual DK1 plus a free box, physics matching the play env."""
    spec = Entity(get_unimanual_dk1_robot_cfg()).spec
    body = spec.worldbody.add_body(name="piece")
    body.add_freejoint()
    body.add_geom(
        name="piece",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(_PIECE_HALF_SIZE,) * 3,
        mass=_PIECE_MASS,
        friction=(1.0, 0.01, 0.0001),
    )
    spec.option.timestep = 0.01
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_NATIVECCD
    # Zero gravity keeps the box at the grasp site without a table, so the
    # measured contacts are exclusively pad<->piece.
    spec.option.gravity = (0.0, 0.0, 0.0)
    return spec.compile()


def test_grasp_holds_piece_without_crushing() -> None:
    model = _build_grasp_model()
    data = mujoco.MjData(model)

    for name, value in HOME_JOINT_POS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(model, data)

    # Float the box at the grasp site, aligned with the gripper frame so its
    # faces meet the pads flat.
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    ee_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6-7")
    adr = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "piece")
    ]
    data.qpos[adr : adr + 3] = data.site_xpos[site_id]
    data.qpos[adr + 3 : adr + 7] = data.xquat[ee_body]

    # Hold the arm at home, command the gripper fully closed (the teleop
    # full-close command, mapping the leader's gripper_closed_pos).
    for name in HOME_JOINT_POS:
        act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        data.ctrl[act] = HOME_JOINT_POS[name]
    gripper_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_left")
    data.ctrl[gripper_act] = GRIPPER_COMMAND_RANGE["closed"]

    piece_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "piece")
    finger_bodies = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
        for b in ("finger_left", "finger_right")
    }

    penetrations, forces = [], []
    force6 = np.zeros(6)
    time_to_contact = float("inf")
    for step in range(_SETTLE_STEPS):
        mujoco.mj_step(model, data)
        penetration, force = 0.0, 0.0
        for i in range(data.ncon):
            contact = data.contact[i]
            if piece_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == piece_geom else contact.geom1
            if model.geom_bodyid[other] not in finger_bodies:
                continue
            penetration = max(penetration, -contact.dist)
            mujoco.mj_contactForce(model, data, i, force6)
            force += abs(force6[0])
        if penetration > 0.0 and np.isinf(time_to_contact):
            time_to_contact = (step + 1) * model.opt.timestep
        penetrations.append(penetration)
        forces.append(force)

    steady_penetration = float(np.mean(penetrations[-_STEADY_WINDOW:]))
    steady_force_per_finger = float(np.mean(forces[-_STEADY_WINDOW:])) / 2
    piece_displacement = float(
        np.linalg.norm(data.qpos[adr : adr + 3] - data.site_xpos[site_id])
    )

    assert time_to_contact < _MAX_TIME_TO_CONTACT, (
        f"fingers took {time_to_contact:.2f} s to reach the piece — "
        "gripper closes too slowly"
    )
    assert max(penetrations) < _MAX_PEAK_PENETRATION, (
        f"impact transient sinks {1e3 * max(penetrations):.2f} mm into the "
        "piece — closing impact too hard"
    )
    assert steady_penetration < _MAX_STEADY_PENETRATION, (
        f"pads sink {1e3 * steady_penetration:.2f} mm into the piece — "
        "grip force too high or pad contacts too soft"
    )
    assert piece_displacement < _MAX_PIECE_DISPLACEMENT, (
        f"piece squeezed {1e3 * piece_displacement:.2f} mm out of the grasp — "
        "grip force too high for a stable hold"
    )
    assert (
        _MIN_GRIP_FORCE_PER_FINGER
        < steady_force_per_finger
        < _MAX_GRIP_FORCE_PER_FINGER
    ), (
        f"grip force {steady_force_per_finger:.1f} N/finger outside the graded "
        f"regime ({_MIN_GRIP_FORCE_PER_FINGER}-{_MAX_GRIP_FORCE_PER_FINGER} N)"
    )
