import pybullet
import pybullet_data
from pybullet_utils.bullet_client import BulletClient
from scipy.spatial.transform import Rotation
import numpy as np
from cair_robot.common.errors import RobotError
from cair_robot.common.geometry import transform_pose, pose_to_mat
from cair_robot.robots.ur5_pybullet_robot import UR5PyBulletRobot, get_joints_max_speed_acceleration

class PyBulletDualArmTableScene:
    def __init__(self,
            tx_table_camera,
            tx_left_camera,
            tx_right_camera,
            gui_enabled=True,
            th_tool_deviation=1e-3,
            th_joint_speed=3.14,
            dt=1/125,
            step_per_sim=1):

        if gui_enabled:
            bc = BulletClient(connection_mode=pybullet.GUI)
            bc.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        else:
            bc = BulletClient(connection_mode=pybullet.DIRECT)
        bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        bc.setGravity(0, 0, -9.8)
        tx_left_world = tx_left_camera @ np.linalg.inv(tx_table_camera)
        tx_right_world = tx_right_camera @ np.linalg.inv(tx_table_camera)

        # load table
        table_body_id = bc.loadURDF('assets/table/table.urdf')

        self.bc = bc
        self.left_robot = UR5PyBulletRobot(bc, np.linalg.inv(tx_left_world),
            th_tool_deviation=th_tool_deviation, th_joint_speed=th_joint_speed, dt=dt)
        self.right_robot = UR5PyBulletRobot(bc, np.linalg.inv(tx_right_world),
            th_tool_deviation=th_tool_deviation, th_joint_speed=th_joint_speed, dt=dt)
        self.tx_left_world = tx_left_world
        self.tx_right_world = tx_right_world
        self.th_tool_deviation = th_tool_deviation
        self.th_joint_speed = th_joint_speed
        self.dt = dt
        self.step_per_sim = step_per_sim
    
    def __del__(self):
        self.bc.disconnect()
    
    def step(self, n=1):
        for _ in range(int(n)):
            self.bc.stepSimulation()

    def has_collision(self):
        return len(self.bc.getContactPoints()) > 0
    
    def single_arm_movel(self, is_left, p, speed=0.25, acceleration=1.2, blocking=True):
        tx = self.tx_left_world if is_left else self.tx_right_world
        robot = self.left_robot if is_left else self.right_robot
        rp = transform_pose(tx, p)
        return robot.movel(rp, speed=speed, acceleration=acceleration)
    

    def dual_arm_movej_traj(self, left_joint_trajectory, right_joint_trajectory):
        dt = self.dt
        dq_max, ddq_max = get_joints_max_speed_acceleration(
            left_joint_trajectory, dt=dt)
        if dq_max > self.th_joint_speed:
            raise RobotError('Left arm joint speed limit exceeded! {} rad/s'.format(dq_max))
        dq_max, ddq_max = get_joints_max_speed_acceleration(
            right_joint_trajectory, dt=dt)
        if dq_max > self.th_joint_speed:
            raise RobotError('Right arm joint speed limit exceeded! {} rad/s'.format(dq_max))
        for i in range(max(len(left_joint_trajectory), len(right_joint_trajectory))):
            left_i = min(i, len(left_joint_trajectory)-1)
            right_i = min(i, len(right_joint_trajectory)-1)
            left_q = left_joint_trajectory[left_i]
            right_q = right_joint_trajectory[right_i]
            self.left_robot.set_joints(left_q)
            self.right_robot.set_joints(right_q)
            if (i % self.step_per_sim) == 0:
                self.step()
                n_col = len(self.bc.getContactPoints())
                if n_col > 0:
                    raise RobotError('Collision!')
        return True

    def dual_arm_movel_traj(self, left_pose_traj, right_pose_traj):
        left_pose_traj_local = transform_pose(self.tx_left_world, left_pose_traj)
        right_pose_traj_local = transform_pose(self.tx_right_world, right_pose_traj)

        left_joint_trajectory, deviations = self.left_robot.trajectory_ik(left_pose_traj_local)
        if np.max(deviations) > self.th_tool_deviation:
            raise RobotError('Left arm unreachable pose!')
        right_joint_trajectory, deviations = self.right_robot.trajectory_ik(right_pose_traj_local)
        if np.max(deviations) > self.th_tool_deviation:
            raise RobotError('Right arm unreachable pose!')
        return self.dual_arm_movej_traj(left_joint_trajectory, right_joint_trajectory)

    def dual_arm_movel(self, left_path, right_path, speed, acceleration):
        left_pose_traj = self.left_robot.get_movel_pose_trajectory(left_path, 
            speed=speed, acceleration=acceleration)
        right_pose_traj = self.right_robot.get_movel_pose_trajectory(right_path,
            speed=speed, acceleration=acceleration)
        return self.dual_arm_movel_traj(left_pose_traj, right_pose_traj)

    def dual_arm_home_set(self):
        self.left_robot.home()
        self.right_robot.home()
    
def test():
    from cair_robot.common.primitive_util import get_base_fling_poses

    tx_table_camera = np.loadtxt('cam_pose/cam2table_pose.txt')
    tx_left_camera = np.loadtxt('cam_pose/cam2left_pose.txt')
    tx_right_camera = np.loadtxt('cam_pose/cam2right_pose.txt')

    scene = PyBulletDualArmTableScene(
        tx_table_camera, tx_left_camera, tx_right_camera, 
        gui_enabled=False,
        th_joint_speed=5.0,
        dt=1/125)
    scene.step(10)

    width = 0.5
    base_fling = get_base_fling_poses(swing_stroke=0.5)
    left_path = base_fling.copy()
    left_path[:,0] = -width/2
    right_path = base_fling.copy()
    right_path[:,0] = width/2

    # tx_rot = pose_to_mat(np.array([0,0,0,0,0,np.pi/2]))
    # left_path = transform_pose(tx_rot, left_path)
    # right_path = transform_pose(tx_rot, right_path)

    import time
    s = time.perf_counter()
    n = 100
    for i in range(n):
        scene.dual_arm_movel(left_path, right_path, speed=1.3, acceleration=5)
    print((time.perf_counter() - s) / n)

    # scene.dual_arm_movel(left_path, right_path, speed=1.3, acceleration=5)

    # left_path_local = transform_pose(scene.tx_left_world, left_path)

    # scene.left_robot.movel(left_path_local)
    # scene.left_robot.set_joints(scene.left_robot.home_joints)


# %%
