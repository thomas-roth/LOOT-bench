"""YAM constants.

Robot decisions for the mjlab asset-zoo YAM. The actuator/entity configs live in
mjlab; this file only owns the values get_yam_interface needs that mjlab does not
expose.
"""

YAM_NAME = "Yam"
YAM_GRASP_SITE = "grasp_site"
YAM_EE_SUBTREE = "link_6"
YAM_VIEWER_BODY = "arm"
YAM_PAD_GEOM_PATTERN = r"[lr]f_down(6|7|8|9|10|11)_collision"
