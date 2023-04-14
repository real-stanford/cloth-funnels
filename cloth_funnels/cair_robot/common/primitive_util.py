import numpy as np
from scipy.spatial.transform import Rotation
from cair_robot.common.geometry import pos_rot_to_pose, rot_from_directions, pos_rot_to_mat, normalize

def get_base_fling_poses(
        place_y=0.0,
        stroke=0.6, 
        lift_height=0.45, 
        swing_angle=np.pi/4,
        place_height=0.05
    ):
    """
    Basic fling trajectory: single trajectory on y-plane.
    From -y to +y, x=0
    Waypoint 1 is at place_y

                  z
    ----stroke----^
    --------------->y
    |             |
    |2     0     1|lift_height
    |             |
    |            3|place_height
    ---------------
    """

    base_fling_pos = np.array([
        [0,0,lift_height],
        [0,place_y,lift_height],
        [0,place_y-stroke,lift_height],
        [0,place_y,place_height]
    ])
    init_rot = Rotation.from_rotvec([0,np.pi,0])
    base_fling_rot = Rotation.from_euler('xyz',[
        [0,0,0],
        [swing_angle,0,0],
        [-swing_angle,0,0],
        [swing_angle/8,0,0]
    ])
    fling_rot = base_fling_rot * init_rot
    fling_pose = pos_rot_to_pose(base_fling_pos, fling_rot)
    return fling_pose


def points_to_action_frame(left_point, right_point):
    """
    Compute transfrom from action frame to world
    Action frame: centered on the mid-point between gripers,
    with the y-axis facing fling direction (i.e. forward)
    * left_point
    |---> y
    * right_point
    |
    x
    """
    left_point, right_point = left_point.copy(), right_point.copy()
    center_point = (left_point + right_point) / 2
    # enforce z
    left_point[2] = center_point[2]
    right_point[2] = center_point[2]
    # compute forward direction
    forward_direction = np.cross(
        np.array([0,0,1]), (right_point - left_point))
    forward_direction = forward_direction / np.linalg.norm(forward_direction)
    # default facing +y
    rot = rot_from_directions(
        np.array([0,1,0]), forward_direction)
    tx_world_action = pos_rot_to_mat(center_point, rot)
    return tx_world_action

def center_to_action_frame(center_point, to_point):
    """
    Compute transfrom from action frame to world
    Action frame: centered on the mid-point between gripers,
    with the y-axis facing fling direction (i.e. forward)
    center_point
    *---*>y
    |   to_point
    x
    """
    # enforce z
    to_point = to_point.copy()
    to_point[2] = center_point[2]
    # compute forward direction
    forward_direction = normalize(to_point - center_point)
    # default facing +y
    rot = rot_from_directions(
        np.array([0,1,0]), forward_direction)
    tx_world_action = pos_rot_to_mat(center_point, rot)
    return tx_world_action