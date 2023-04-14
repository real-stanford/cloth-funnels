import numpy as np
from cair_robot.cameras.kinect_client import KinectClient
from cair_robot.scenes.dual_arm_table_scene import DualArmTableScene
from cair_robot.robots.ur5_robot import UR5RTDE
from cair_robot.robots.grippers import WSG50
from cair_robot.envs.transformed_view_env import TransformedViewEnv


def main():
    tx_table_camera = np.loadtxt('cam_pose/cam2table_pose.txt')
    tx_left_camera = np.loadtxt('cam_pose/cam2left_pose.txt')
    tx_right_camera = np.loadtxt('cam_pose/cam2right_pose.txt')
    tx_camera_view = np.loadtxt('cam_pose/view2cam.txt')

    camera = KinectClient('128.59.23.32', '8080', fielt_bg=False)
    # scene = PyBulletDualArmTableScene(tx_table_camera, tx_left_camera, tx_right_camera)
    env = TransformedViewEnv(kinect_camera=camera, 
        tx_camera_img=tx_camera_view, tx_world_camera=tx_table_camera,
        tx_left_camera=tx_left_camera, tx_right_camera=tx_right_camera,
        must_pick_on_obj=False, 

        left_picker_offset=0.04,)
    
    print("Connecting to WSG50")
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
        right_robot=right_ur5
    )
    scene.home(speed=0.5)

    env.initialize_fling_primitive()
    rs = np.random.RandomState(0)

    try:
        while True:
            if rs.random() > 0.4:
                # might fail due to cloth off center
                r = env.random_pick_and_fling(scene=scene, rs=rs)
                if not r:
                    r = env.random_pick_and_place(scene=scene, rs=rs)
                assert(r)
            else:
                r = env.random_pick_and_place(scene=scene, rs=rs)
                assert(r)
    except Exception as e:
        scene.disconnect()
        raise e

if __name__ == '__main__':
    main()
