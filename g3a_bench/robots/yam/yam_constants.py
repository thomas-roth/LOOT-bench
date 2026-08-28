"""YAM constants.

Robot decisions for the mjlab asset-zoo YAM. The actuator/entity configs live in
mjlab; this file only owns the values get_yam_interface needs that mjlab does not
expose.
"""

from g3a_bench.robots.interface import gripper_open_threshold

YAM_NAME = "Yam"
YAM_GRASP_SITE = "grasp_site"
YAM_EE_SUBTREE = "link_6"
YAM_VIEWER_BODY = "arm"
# The grippy finger-pad geoms mjlab's own asset-zoo collision tuning already
# identifies (mjlab/asset_zoo/robots/i2rt_yam/yam_constants.py's
# FULL_COLLISION/GRIPPER_ONLY_COLLISION condim/friction overrides) -- reused
# here as the contact-release sensor's primary pattern.
YAM_PAD_GEOMS_PATTERN = r"[lr]f_down(6|7|8|9|10|11)_collision"
# Only left_finger is actuated; right_finger is coupled via equality.
_GRIPPER_MECHANICAL_RANGE = {"closed": -0.00205, "open": 0.037524}
YAM_GRIPPER_JOINTS = ("left_finger",)
YAM_GRIPPER_OPEN_THRESHOLD = gripper_open_threshold(_GRIPPER_MECHANICAL_RANGE)
# The actuator split the two action terms are built from, mirroring mjlab's
# own ARM_ACTUATORS / ACTUATOR_DM_4310_LINEAR_CRANK. Enough to keep YAM
# building; no task is registered against it (ADR-016 scopes it out).
YAM_ARM_ACTUATORS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
YAM_GRIPPER_ACTUATORS = ("left_finger",)
YAM_IK_MAX_DQ = 0.28
"""[rad] differential IK's max_dq: the arm's PD saturation error.

A placeholder equal to the DK1's. mjlab tunes YAM's stiffness per joint from
effective inertias, so the honest derivation (min effort_limit / stiffness
over ARM_ACTUATORS) needs numbers this file would have to reach into mjlab's
asset zoo for. No task is registered against YAM, so it is enough to keep the
interface building -- derive it properly before wiring YAM to a task. See
``ManipulatorInterface.ik_max_dq``."""
