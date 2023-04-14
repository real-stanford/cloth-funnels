import numpy as np
from cair_robot.cameras.kinect_client import KinectClient
from cair_robot.scenes.dual_arm_table_scene import DualArmTableScene
from cair_robot.robots.ur5_robot import UR5RTDE
from cair_robot.robots.grippers import WSG50
from cair_robot.envs.transformed_view_env import TransformedViewEnv, is_coord_valid_robot, is_coord_valid_table
import threading
import sys
from argparse import ArgumentParser

def scan_grid(scene, env, grid, is_left):
    name = 'left' if is_left else 'right'
    print(f"{name} has number of points: {grid.shape[0]}")
    for i, pt in enumerate(grid):
        print(f"{name} picking point:", i, pt)
        env.pick_and_place(scene, is_left, pt, pt + np.array([0, 0, 0.1]))
    



if __name__ == '__main__':

    parser = ArgumentParser("Dynamic Cloth Manipulation")
    parser.add_argument('--arm', type=str, default='right', choices=['left', 'right'])
    parser.add_argument('--grid_width', type=int, default=4)
    parser.add_argument('--mat_thickness', type=float, default=0.052)
    parser.add_argument('--left_picker_offset', type=float, default=0.02)

    args = parser.parse_args()

    tx_table_camera = np.loadtxt('real/cam_pose/cam2table_pose.txt')
    tx_left_camera = np.loadtxt('real/cam_pose/cam2left_pose.txt')
    tx_right_camera = np.loadtxt('real/cam_pose/cam2right_pose.txt')
    tx_camera_view = np.loadtxt('real/cam_pose/view2cam.txt')

    camera = KinectClient('128.59.23.32', '8080', fielt_bg=False)
    # scene = PyBulletDualArmTableScene(tx_table_camera, tx_left_camera, tx_right_camera)
    env = TransformedViewEnv(kinect_camera=camera, 
        tx_camera_img=tx_camera_view, tx_world_camera=tx_table_camera,
        tx_left_camera=tx_left_camera, tx_right_camera=tx_right_camera,
        must_pick_on_obj=False, 
        mat_thickness=args.mat_thickness,
        left_picker_offset=args.left_picker_offset)
    
    wsg50 = WSG50('192.168.0.231', 1006)
    print('WSG50 ready')
    left_ur5 = UR5RTDE('192.168.0.139', wsg50) # latte
    print('Latte ready')
    right_ur5 = UR5RTDE('192.168.0.204', 'rg2') # oolong
    print('Oolong ready')
    wsg50.home()
    wsg50.open()
    scene = DualArmTableScene(
        tx_table_camera=tx_table_camera,
        tx_left_camera=tx_left_camera,
        tx_right_camera=tx_right_camera,
        left_robot=left_ur5,
        right_robot=right_ur5,
        confirm_actions=False,
    )
    scene.home(speed=0.5)

    grid_pts = 4
    start, end = -0.6, 0.6
    interval = (end-start)/grid_pts
    xy = np.meshgrid(np.arange(start, end, interval), np.arange(start, end, interval))
    zeros = np.zeros(shape=(grid_pts, grid_pts))
    grid = np.stack([xy[0], xy[1], zeros], axis=2)

    left_valid_coords = env.is_coord_valid_robot(grid, is_left=True)
    right_valid_coords = env.is_coord_valid_robot(grid, is_left=False)

    table_valid_coords = is_coord_valid_table(grid)

    left_grid_filter = np.logical_and(table_valid_coords, left_valid_coords)
    right_grid_filter = np.logical_and(table_valid_coords, right_valid_coords)

    left_grid = grid[left_grid_filter]
    right_grid = grid[right_grid_filter]

    if args.arm == "right":
        scan_grid(scene, env, right_grid, is_left=False)
    if args.arm == "left":
        scan_grid(scene, env, left_grid, is_left=True)

    print(args.mat_thickness, args.left_picker_offset)
