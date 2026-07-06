# TRLC-DK1 visual scene checks

Two ways to eyeball the DK1 robot on a floor with a skybox. Run both from the **repo root** so the relative mesh paths resolve.

---

## Via the mjlab config pipeline (`basic_scene.py`)

Builds the robot through `get_unimanual_dk1_robot_cfg()` -> `Entity`, i.e. the same path training uses. Use this to check the assembled `EntityCfg`.

```bash
uv run python tests/trlc_dk1/basic_scene.py
```

---

## Via raw MJCF (`basic_scene.xml`)

Loads `dk1.xml` standalone (no mjlab), so it also checks the `<include>` and `meshdir` resolution.

- Directly pass the file with `--mjcf`.
- Omit the file in the command and drag-and-drop the XML file into the viewer.

```bash
uv run python -m mujoco.viewer --mjcf=tests/trlc_dk1/basic_scene.xml
```
