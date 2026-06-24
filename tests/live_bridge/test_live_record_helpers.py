import mujoco


def test_ghost_spec_joint_names_prefixed():
    """Every joint in the compiled ghost spec must start with 'ghost_'."""
    from g3a_bench.robots.trlc_dk1.dk1_constants import get_ghost_spec

    spec = get_ghost_spec()
    model = spec.compile()
    bare = [
        model.joint(i).name
        for i in range(model.njnt)
        if model.joint(i).name and not model.joint(i).name.startswith("ghost_")
    ]
    assert bare == [], f"Un-prefixed joints found: {bare}"


def test_ghost_spec_geoms_non_colliding():
    """All ghost geoms must have contype=0 and conaffinity=0."""
    from g3a_bench.robots.trlc_dk1.dk1_constants import get_ghost_spec

    spec = get_ghost_spec()
    model = spec.compile()
    for i in range(model.ngeom):
        name = model.geom(i).name
        if name.startswith("ghost_"):
            assert model.geom_contype[i] == 0, (
                f"{name}: contype={model.geom_contype[i]}"
            )
            assert model.geom_conaffinity[i] == 0, (
                f"{name}: conaffinity={model.geom_conaffinity[i]}"
            )


def test_ghost_spec_geoms_start_invisible():
    """All ghost geoms must start with alpha=0 (hidden)."""
    from g3a_bench.robots.trlc_dk1.dk1_constants import get_ghost_spec

    spec = get_ghost_spec()
    model = spec.compile()
    for i in range(model.ngeom):
        name = model.geom(i).name
        if name.startswith("ghost_"):
            alpha = model.geom_rgba[i, 3]
            assert alpha == 0.0, f"{name}: alpha={alpha}"


def test_ghost_spec_arm_joints_spring_at_home():
    """Arm joints must have stiffness>0 and springref matching _HOME_JOINT_POS."""
    from g3a_bench.robots.trlc_dk1.dk1_constants import _HOME_JOINT_POS, get_ghost_spec

    spec = get_ghost_spec()
    model = spec.compile()

    arm_joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    for orig in arm_joints:
        jname = f"ghost_{orig}"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        assert jid >= 0, f"Joint {jname} not found in compiled model"
        assert model.jnt_stiffness[jid] > 0, f"{jname}: stiffness=0"
        qadr = model.jnt_qposadr[jid]
        springref = model.qpos_spring[qadr]
        expected = _HOME_JOINT_POS[orig]
        assert abs(springref - expected) < 1e-5, (
            f"{jname}: springref={springref:.4f}, expected={expected:.4f}"
        )


# ── Task 4: ask_operator and _arm_at_home helpers ────────────────────────────


def test_ask_operator_valid(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "2")
    from g3a_bench.scripts.live_bridge.live_record import ask_operator

    assert ask_operator() == 2


def test_ask_operator_retries_on_bad_input(monkeypatch):
    responses = iter(["0", "5", "abc", "3"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    from g3a_bench.scripts.live_bridge.live_record import ask_operator

    assert ask_operator() == 3


def test_arm_at_home_true_within_threshold():
    from g3a_bench.scripts.live_bridge.live_record import _arm_at_home
    from g3a_bench.scripts.live_bridge.mapper import HOME_JOINT_POS

    obs = HOME_JOINT_POS[:6].copy() + 0.05
    assert _arm_at_home(obs, threshold=0.1) is True


def test_arm_at_home_false_outside_threshold():
    from g3a_bench.scripts.live_bridge.live_record import _arm_at_home
    from g3a_bench.scripts.live_bridge.mapper import HOME_JOINT_POS

    obs = HOME_JOINT_POS[:6].copy()
    obs[2] += 0.5
    assert _arm_at_home(obs, threshold=0.2) is False


def test_arm_at_home_none_returns_false():
    from g3a_bench.scripts.live_bridge.live_record import _arm_at_home

    assert _arm_at_home(None, threshold=0.2) is False
