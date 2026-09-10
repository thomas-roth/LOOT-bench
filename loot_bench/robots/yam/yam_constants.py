"""YAM constants for get_yam_interface from mjlab's asset zoo."""

YAM_NAME = "Yam"
YAM_GRASP_SITE = "grasp_site"
YAM_EE_SUBTREE = "link_6"
YAM_VIEWER_BODY = "arm"
# Pad geoms from mjlab asset-zoo tuning; contact-release primary.
YAM_PAD_GEOMS_PATTERN = r"[lr]f_down(6|7|8|9|10|11)_collision"
# The actuator split the two action terms are built from, mirroring mjlab's
# own ARM_ACTUATORS / ACTUATOR_DM_4310_LINEAR_CRANK. Enough to keep YAM
# building; no task is registered against it (ADR-016 scopes it out).
YAM_ARM_ACTUATORS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
YAM_GRIPPER_ACTUATORS = ("left_finger",)
YAM_IK_MAX_DQ = 0.0
"""[rad] differential IK's max_dq: the arm's PD saturation error.

Not yet derived -- mjlab tunes YAM's stiffness per joint from effective
inertias, so the honest derivation (min effort_limit / stiffness over
ARM_ACTUATORS) needs numbers this file would have to reach into mjlab's asset
zoo for. 0.0 rather than a borrowed guess: EndEffectorDeltaActionCfg.build
rejects it, so wiring YAM to a task fails loud instead of quietly running
under the DK1's torque cap. No task is registered against YAM today
(ADR-016 scopes it out). See ``ManipulatorInterface.ik_max_dq``."""
