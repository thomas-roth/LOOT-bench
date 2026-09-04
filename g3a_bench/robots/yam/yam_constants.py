"""YAM constants for get_yam_interface from mjlab's asset zoo."""

from g3a_bench.robots.interface import gripper_open_threshold

YAM_NAME = "Yam"
YAM_GRASP_SITE = "grasp_site"
YAM_EE_SUBTREE = "link_6"
YAM_VIEWER_BODY = "arm"
# Pad geoms from mjlab asset-zoo tuning; contact-release primary.
YAM_PAD_GEOMS_PATTERN = r"[lr]f_down(6|7|8|9|10|11)_collision"
# Only left_finger is actuated; right_finger is coupled via equality.
_GRIPPER_MECHANICAL_RANGE = {"closed": -0.00205, "open": 0.037524}
YAM_GRIPPER_JOINTS = ("left_finger",)
YAM_GRIPPER_OPEN_THRESHOLD = gripper_open_threshold(_GRIPPER_MECHANICAL_RANGE)
