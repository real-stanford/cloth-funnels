from operator import is_
import threading
import numpy as np
from cair_robot.robots.ur5_robot import UR5RTDE
from cair_robot.robots.grippers import WSG50
from cair_robot.common.geometry import pos_rot_to_mat, transform_pose, pose_to_mat, pos_rot_to_pose, rot_from_directions, normalize, transform_points
from scipy.spatial.transform import Rotation
import time
from cair_robot.common.primitive_util import get_base_fling_poses, points_to_action_frame


def points_to_gripper_pose(left_point, right_point, max_width=None):

    tx_world_action = points_to_action_frame(left_point, right_point)

    width = np.linalg.norm((left_point - right_point)[:2])
    if max_width is not None:
        width = min(width, max_width)
    left_pose_action = np.array([-width/2,0,0,0,np.pi,0])
    right_pose_action = np.array([width/2,0,0, 0,np.pi,0])
    left_pose = transform_pose(tx_world_action, left_pose_action)
    right_pose = transform_pose(tx_world_action, right_pose_action)
    return left_pose, right_pose


def points_to_fling_path(
        left_point, right_point,
        width=None,   
        swing_stroke=0.6, 
        swing_height=0.45, 
        swing_angle=np.pi/4,
        lift_height=0.4,
        place_height=0.05):
    tx_world_action = points_to_action_frame(left_point, right_point)
    tx_world_fling_base = tx_world_action.copy()
    # height is managed by get_base_fling_poses
    tx_world_fling_base[2,3] = 0
    base_fling = get_base_fling_poses(
        swing_stroke=swing_stroke,
        swing_height=swing_height,
        swing_angle=swing_angle,
        lift_height=lift_height,
        place_height=place_height)
    if width is None:
        width = np.linalg.norm((left_point - right_point)[:2])
    left_path = base_fling.copy()
    left_path[:,0] = -width/2
    right_path = base_fling.copy()
    right_path[:,0] = width/2
    left_path_w = transform_pose(tx_world_fling_base, left_path)
    right_path_w = transform_pose(tx_world_fling_base, right_path)
    return left_path_w, right_path_w


class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        def function():
            self.result = target(*args, **kwargs)
        super().__init__(group=group, target=function, name=name, daemon=daemon)


class DualArmTableScene:
    """
    All parameters in world(table) space.
    """

    def __init__(self,
        tx_table_camera,
        tx_left_camera,
        tx_right_camera,
        left_robot,
        right_robot,
        confirm_actions=True,
        ):

        tx_left_world = tx_left_camera @ np.linalg.inv(tx_table_camera)
        tx_right_world = tx_right_camera @ np.linalg.inv(tx_table_camera)

        self.tx_left_world = tx_left_world
        self.tx_right_world = tx_right_world

        self.left_robot = left_robot
        self.right_robot = right_robot

        self.confirm_actions = confirm_actions

        r = self.open_grippers(sleep_time=0, blocking=False)
        if not r:
            raise Exception("Could not open grippers")


    
    def disconnect(self):
        self.left_robot.disconnect()
        self.right_robot.disconnect()

    def get_obs(self):
        pass

    def single_arm_movel(self, is_left, p, speed=0.25, acceleration=1.2, blocking=True, avoid_singularity=False):
        tx = self.tx_left_world if is_left else self.tx_right_world
        robot = self.left_robot if is_left else self.right_robot
        rp = transform_pose(tx, p)
        result = robot.movel(rp, speed, acceleration, blocking, avoid_singularity=avoid_singularity)
        return result
    
    def single_arm_movej_ik(self, is_left, p, speed=0.25, acceleration=1.2, blocking=True):
        tx = self.tx_left_world if is_left else self.tx_right_world
        robot = self.left_robot if is_left else self.right_robot
        rp = transform_pose(tx, p)
        result = robot.movej_ik(rp, speed, acceleration, blocking)
        return result

    def dual_arm_movel(self, p_left, p_right, speed=0.25, acceleration=1.2, blocking=True, avoid_singularity=False):
        t1 = ThreadWithResult(target=self.single_arm_movel, args=(True, p_left, speed, acceleration, blocking, avoid_singularity))
        t2 = ThreadWithResult(target=self.single_arm_movel, args=(False, p_right, speed, acceleration, blocking, avoid_singularity))
        t1.start()
        t2.start()
        if blocking:
            t1.join()
            t2.join()
            return t1.result and t2.result
        else:
            return True

    def dual_arm_fling(self, left_path, right_path, speed, acceleration):
        r = self.dual_arm_movel(left_path[0], right_path[0], speed=speed, acceleration=acceleration)
        if not r: return False
        r = self.dual_arm_movel(left_path[1:], right_path[1:], speed=speed, acceleration=acceleration)
        return r

    def get_tcp_distance(self):
        left_tcp_pose = transform_pose(np.linalg.inv(self.tx_left_world),
            self.left_robot.get_tcp_pose())
        right_tcp_pose = transform_pose(np.linalg.inv(self.tx_right_world),
            self.right_robot.get_tcp_pose())
        tcp_distance = np.linalg.norm((right_tcp_pose - left_tcp_pose)[:3])
        return tcp_distance

    def dual_arm_strech(self, 
        left_pose, right_pose,
        force=12, init_force=30,
        max_speed=0.15, 
        max_width=0.7, max_time=5,
        speed_threshold=0.001):
        """
        Assuming specific gripper and tcp orientation.
        """
        r = self.dual_arm_movel(left_pose, right_pose, speed=max_speed)
        if not r: return False

        left_tcp_pose = self.left_robot.get_tcp_pose()
        right_tcp_pose = self.right_robot.get_tcp_pose()

        # task_frame = [0, 0, 0, 0, 0, 0]
        selection_vector = [1, 0, 0, 0, 0, 0]
        force_type = 2
        # speed for compliant axis, deviation for non-compliant axis
        limits = [max_speed, 2, 2, 1, 1, 1]
        dt = 1.0/125

        # enable force mode on both robots
        tcp_distance = self.get_tcp_distance()
        with self.left_robot.start_force_mode() as left_force_guard:
            with self.right_robot.start_force_mode() as right_force_guard:
                start_time = time.time()
                prev_time = start_time
                max_acutal_speed = 0
                while (time.time() - start_time) < max_time:
                    f = force
                    if max_acutal_speed < max_speed/20:
                        f = init_force
                    left_wrench = [f, 0, 0, 0, 0, 0]
                    right_wrench = [-f, 0, 0, 0, 0, 0]

                    # apply force
                    r = left_force_guard.apply_force(left_tcp_pose, selection_vector, 
                        left_wrench, force_type, limits)
                    if not r: return False
                    r = right_force_guard.apply_force(right_tcp_pose, selection_vector, 
                        right_wrench, force_type, limits)
                    if not r: return False

                    # check for distance
                    tcp_distance = self.get_tcp_distance()
                    if tcp_distance >= max_width:
                        print('Max distance reached: {}'.format(tcp_distance))
                        break

                    # check for speed
                    l_speed = np.linalg.norm(self.left_robot.get_tcp_speed()[:3])
                    r_speed = np.linalg.norm(self.right_robot.get_tcp_speed()[:3])
                    actual_speed = max(l_speed, r_speed)
                    max_acutal_speed = max(max_acutal_speed, actual_speed)
                    if max_acutal_speed > (max_speed * 0.4):
                        if actual_speed < speed_threshold:
                            print('Action stopped at acutal_speed: {} with  max_acutal_speed: {}'.format(
                                actual_speed, max_acutal_speed))
                            break

                    curr_time = time.time()
                    duration = curr_time - prev_time
                    if duration < dt:
                        time.sleep(dt - duration)
        return r

    def single_arm_pick_and_place(self,
            is_left,
            start, end,
            min_pick_height=0.01,
            lift_height=0.4,
            speed=0.5,
            acceleration=1):

        start_pose = np.array([0,0,0,0,np.pi,0])
        end_pose = start_pose.copy()
        start_pose[:2] = start[:2]
        start_pose[2] = max(start[2], min_pick_height)
        end_pose[:2] = end[:2]
        end_pose[2] = max(end[2], min_pick_height)

        start_lift_pose = start_pose.copy()
        start_lift_pose[2] = lift_height
        end_lift_pose = end_pose.copy()
        end_lift_pose[2] = lift_height

        robot = self.left_robot if is_left else self.right_robot

        # ready_to_lift = False
        # while not ready_to_lift:

        robot.open_gripper(0)
        
        # start_seq = []
        # start_seq.extend([start_lift_pose,start_pose])
        r = self.single_arm_movel(is_left, 
            p=np.array([start_lift_pose]), 
            speed=speed, acceleration=acceleration,
            avoid_singularity=True)
        if not r: return False

        while True:
            r = self.single_arm_movel(is_left, 
            p=np.array([start_pose]), 
            speed=speed, acceleration=acceleration,
            avoid_singularity=True)

            if not r: return r

            robot.close_gripper(1)

            if self.confirm_actions:
                response = input("Enter grip offset (x):")
                if len(response) == 0:
                    # good grip
                    print("Continuing...")
                    break
                else:
                    response = [float(x) for x in response.split(" ")]
                    start_pose[2] += np.array(response[0])
                    robot.open_gripper(0)
            else:
                break



            

        r = self.single_arm_movel(is_left, 
            p=np.array([start_lift_pose, end_lift_pose, end_pose]), 
            speed=speed, acceleration=acceleration)
        if not r: return r

        robot.open_gripper(1)
        r = self.single_arm_movel(is_left, 
            p=end_lift_pose, 
            speed=speed, acceleration=acceleration)
        return r

    def dual_arm_strech_and_fling(self, 
            left_point, 
            right_point,
            strech_force=20,
            strech_max_speed=0.15,
            strech_max_width=0.7, 
            strech_max_time=5,
            swing_stroke=0.6,
            swing_height=0.45,
            swing_angle=np.pi/4,
            lift_height=0.4,
            place_height=0.05,
            fling_speed=1.3,
            fling_acceleration=5
            ):
        left_pose, right_pose = points_to_gripper_pose(
            left_point, right_point, max_width=strech_max_width)

        # strech
        r = self.dual_arm_strech(left_pose, right_pose, 
            force=strech_force, 
            max_speed=strech_max_speed, 
            max_width=strech_max_width,
            max_time=strech_max_time)
        if not r: return False
        width = self.get_tcp_distance()
        print('Width: {}'.format(width))
        
        # fling
        left_path, right_path = points_to_fling_path(
            left_point=left_point,
            right_point=right_point,
            width=width,
            swing_stroke=swing_stroke,
            swing_height=swing_height,
            swing_angle=swing_angle,
            lift_height=lift_height,
            place_height=place_height
        )
        return self.dual_arm_fling(left_path, right_path, 
            fling_speed, fling_acceleration)
    
    def dual_arm_pick(self, left_point, right_point,
            min_pick_height=0.01, lift_height=0.3, grip_sleep_time=1,
            speed=1.2, acceleration=1):
        left_pose, right_pose = points_to_gripper_pose(
            left_point, right_point)

        left_pose[2] = left_point[2]
        right_pose[2] = right_point[2]
        
        left_lift_pose = left_pose.copy()
        right_lift_pose = right_pose.copy()
        left_lift_pose[2] = lift_height
        right_lift_pose[2] = lift_height
        left_pick_pose = left_pose.copy()
        right_pick_pose = right_pose.copy()
        left_pick_pose[2] = max(left_pick_pose[2], min_pick_height)
        right_pick_pose[2] = max(right_pick_pose[2], min_pick_height)

        # prepare

        r = self.open_grippers(sleep_time=0, blocking=False)
        if not r: return False
        r = self.dual_arm_movel(left_lift_pose, right_lift_pose, 
            speed=speed, acceleration=acceleration, avoid_singularity=True)
        if not r: return False
        # drop
        while True:
            r = self.dual_arm_movel(left_pick_pose, right_pick_pose, 
                speed=speed, acceleration=acceleration)
            if not r: return False
            #wsg50 otherwise won't close
            r = self.close_grippers(sleep_time=0, blocking=True)
            # self.right_robot.close_gripper(1)
            # self.left_robot.close_gripper(1)
            if self.confirm_actions:
                response = input("Enter grip offset (x y):")
                if len(response) == 0:
                    # good grip
                    print("Continuing...")
                    break
                else:
                    response = [float(x) for x in response.split(" ")]
                    print(response)
                    left_pick_pose[2] += np.array(response[0])
                    right_pick_pose[2] += np.array(response[1])

                    r = self.open_grippers(sleep_time=0, blocking=True)
                    if not r: return False
            else:
                break

        if not r: return False
        # lift
        r = self.dual_arm_movel(left_lift_pose, right_lift_pose, 
            speed=speed, acceleration=acceleration)
        return r
    
    def open_grippers(self, sleep_time=1, blocking=True):
        t1 = ThreadWithResult(target=self.left_robot.open_gripper, args=(0,))
        t2 = ThreadWithResult(target=self.right_robot.open_gripper, args=(0,))
        t1.start()
        t2.start()
        if blocking:
            t1.join()
            t2.join()
            time.sleep(sleep_time)
            return t1.result and t2.result
        else:
            return True
    
    def close_grippers(self, sleep_time=1, blocking=True):
        t1 = ThreadWithResult(target=self.left_robot.close_gripper, args=(1,))
        t2 = ThreadWithResult(target=self.right_robot.close_gripper, args=(1,))
        t1.start()
        t2.start()
        if blocking:
            t1.join()
            t2.join()
            time.sleep(sleep_time)
            return t1.result and t2.result
        else:
            return True

    def home(self, speed=1.5, acceleration=1, blocking=True):
        t1 = ThreadWithResult(target=self.left_robot.home, args=(speed, acceleration, blocking))
        t2 = ThreadWithResult(target=self.right_robot.home, args=(speed, acceleration, blocking))
        t1.start()
        t2.start()
        if blocking:
            t1.join()
            t2.join()
            return t1.result and t2.result
        else:
            return True
    
    def lift_grippers(self, height=0.1, speed=0.5, acceleration=1):
        left_pose = transform_pose(
            np.linalg.inv(self.tx_left_world),
            self.left_robot.get_tcp_pose())
        right_pose = transform_pose(
            np.linalg.inv(self.tx_right_world),
            self.right_robot.get_tcp_pose())
        left_pose[2] = max(left_pose[2], height)
        right_pose[2] = max(right_pose[2], height)
        return self.dual_arm_movel(left_pose, right_pose, 
            speed=speed, acceleration=acceleration)
    


def test():
    tx_table_camera = np.loadtxt('cam_pose/cam2table_pose.txt')
    tx_left_camera = np.loadtxt('cam_pose/cam2left_pose.txt')
    tx_right_camera = np.loadtxt('cam_pose/cam2right_pose.txt')

    wsg50 = WSG50('192.168.0.231', 1002)
    left_ur5 = UR5RTDE('192.168.0.139', wsg50) # latte
    right_ur5 = UR5RTDE('192.168.0.204', 'rg2') # oolong
    wsg50.home()
    wsg50.grip()

    scene = DualArmTableScene(
        tx_table_camera=tx_table_camera,
        tx_left_camera=tx_left_camera,
        tx_right_camera=tx_right_camera,
        left_robot=left_ur5,
        right_robot=right_ur5
    )

    p_left = np.array([-0.1,0,0.3,0,3.14,0])
    p_right = np.array([0.1,0,0.3,0,3.14,0])

    tx_rot = pose_to_mat(np.array([0,0,0,0,0,np.pi/2]))
    # tx_rot = pose_to_mat(np.array([0,0,0,0,0,np.pi/4]))
    p_left = transform_pose(tx_rot, p_left)
    p_right= transform_pose(tx_rot, p_right)

    scene.home()
    # scene.single_arm_movel(is_left=True, p=p_left)
    # scene.single_arm_movel(is_left=False, p=p_right)
    scene.dual_arm_movel(p_left, p_right)
    time.sleep(5.0)
    scene.dual_arm_strech(p_left, p_right, force=20)
    width = scene.get_tcp_distance()

    base_fling = get_base_fling_poses(swing_stroke=0.5)
    left_path = base_fling.copy()
    left_path[:,0] = -width/2
    right_path = base_fling.copy()
    right_path[:,0] = width/2
    left_path = transform_pose(tx_rot, left_path)
    right_path = transform_pose(tx_rot, right_path)
    
    # i = 0
    # scene.dual_arm_movel(left_path[i], right_path[i], speed=0.4, acceleration=1)
    # scene.dual_arm_movel(left_path[1:], right_path[1:], speed=0.05, acceleration=1)
    # scene.dual_arm_fling(left_path, right_path, speed=0.05, acceleration=1)
    scene.dual_arm_fling(left_path, right_path, speed=1.3, acceleration=5)

    scene.disconnect()

    scene.open_grippers()
    scene.lift_grippers()
    scene.home()

    time.sleep(5)
    scene.dual_arm_pick(
        left_point=p_left[:3],
        right_point=p_right[:3],
        pick_height=0.05
    )
    left_point = p_left[:3]
    right_point = p_right[:3]
    scene.dual_arm_strech_and_fling(
        p_left[:3], p_right[:3],
        swing_stroke=0.6,
        strech_force=15)


