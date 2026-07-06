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
YAM_PAD_GEOM_PATTERN = r"[lr]f_down(6|7|8|9|10|11)_collision"
# Gripper slide range, open at the upper limit; must match left_finger in the
# mjlab asset-zoo yam.xml. Only left_finger is actuated; right_finger is
# coupled via equality.
_GRIPPER_JOINT_RANGE = (-0.00205, 0.037524)
YAM_GRIPPER_JOINTS = ("left_finger",)
YAM_GRIPPER_OPEN_THRESHOLD = gripper_open_threshold(_GRIPPER_JOINT_RANGE)
