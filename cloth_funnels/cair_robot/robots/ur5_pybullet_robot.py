# %%
import os
import sys
proj_dir = os.path.expanduser('~/dev/urscript_sim')
os.chdir(proj_dir)
sys.path.append(proj_dir)

# %%
import pybullet
import pybullet_data
from pybullet_utils.bullet_client import BulletClient
import numpy as np
from scipy.spatial.transform import Rotation
from cair_robot.urscript_sim.trajectory import gen_movel_noblend_trajectory
from cair_robot.common.geometry import pos_rot_to_pose, transform_pose
from cair_robot.common.errors import RobotError

def get_joints_max_speed_acceleration(q, dt):
    dq = np.gradient(q, dt, axis=0)
    ddq = np.gradient(dq, dt, axis=0)
    max_dq = np.max(np.abs(dq))
    max_ddq = np.max(np.abs(ddq))
    return max_dq, max_ddq


class UR5PyBulletRobot:
    def __init__(self,
            bc: BulletClient, 
            tx_world_robot: np.ndarray,
            th_tool_deviation=1e-3,
            th_joint_speed=3.14,
            dt=1/125
            ):
        pos = tx_world_robot[:3,3]
        rot = Rotation.from_matrix(tx_world_robot[:3,:3])
        # pos, rot = mat_to_pos_rot(tx_world_robot)
        robot_body_id = bc.loadURDF(
            "assets/ur5/ur5.urdf", pos, 
            bc.getQuaternionFromEuler(rot.as_euler('xyz')),
            flags=pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
        
        robot_joint_info = [bc.getJointInfo(robot_body_id, i) for i in range(
            bc.getNumJoints(robot_body_id))]
        robot_joint_indices = [
            x[0] for x in robot_joint_info if x[2] == bc.JOINT_REVOLUTE]
        home_joints = np.array([np.pi/2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])

        self.bc = bc
        self.tx_world_robot = tx_world_robot
        self.robot_body_id = robot_body_id
        self.robot_joint_indices = robot_joint_indices
        self.home_joints = home_joints
        self.robot_end_effector_link_index = 9
        self.th_tool_deivation = th_tool_deviation
        self.th_joint_speed = th_joint_speed
        self.dt = dt
        self.ik_max_iter = 20
        self.ik_th = 0.01
        self.ik_solver = pybullet.IK_SDLS
        self.set_joints(self.home_joints)

    def get_movel_pose_trajectory(self, path, 
            speed=1.2, acceleration=5):
        dt = self.dt
        world_path = path
        init_pose = world_path[0]
        param_path = np.zeros((len(path)-1,8))
        param_path[:,:6] = world_path[1:]
        param_path[:,6] = speed
        param_path[:,7] = acceleration

        tool_trajectory = gen_movel_noblend_trajectory(
            init_pose, param_path)
        pose_samples = pos_rot_to_pose(*tool_trajectory.sample(dt=dt))
        return pose_samples
        
    def get_movel_joint_trajectory(self, path, 
            speed=1.2, acceleration=5):
        dt = self.dt
        pose_samples = self.get_movel_pose_trajectory(path, speed, acceleration, dt)
        joint_trajectory, deviations = self.trajectory_ik(pose_samples)
        return joint_trajectory, deviations
    
    def trajectory_ik(self, pose_trajectory, init_joint=None, max_iter=100):
        bc = self.bc
        world_pose_trajectory = transform_pose(
            self.tx_world_robot, pose_trajectory)
        if init_joint:
            self.set_joints(init_joint)
        joint_trajectory = np.zeros(pose_trajectory.shape)
        deviations = np.zeros(len(pose_trajectory))
        angles = np.linalg.norm(world_pose_trajectory[:,3:], axis=-1)
        axes = (world_pose_trajectory[:,3:].T / angles).T
        for i in range(len(pose_trajectory)):
            position = world_pose_trajectory[i,:3]
            orientation = bc.getQuaternionFromAxisAngle(
                axes[i], angles[i])
            ik = bc.calculateInverseKinematics(
                self.robot_body_id, 
                self.robot_end_effector_link_index, 
                position, orientation, maxNumIterations=max_iter)
            joint_trajectory[i] = ik
            self.set_joints(ik)
            fk = self.get_tcp_pose_world()
            deviation = np.linalg.norm(fk[:3] - position)
            deviations[i] = deviation
        return joint_trajectory, deviations

    def set_joints(self, joints):
        bc = self.bc
        for joint, value in zip(self.robot_joint_indices, joints):
            bc.resetJointState(self.robot_body_id, joint, value)

    def step(self, n=1):
        for _ in range(int(n)):
            self.bc.stepSimulation()

    def movej_traj(self, joint_trajectory) -> bool:
        dt = self.dt
        dq_max, ddq_max = get_joints_max_speed_acceleration(
            joint_trajectory, dt=dt)
        if dq_max > self.th_joint_speed:
            raise RobotError('Joint speed limit exceeded! {} rad/s'.format(dq_max))
        for q in joint_trajectory:
            self.set_joints(q)
            self.step()
            n_col = len(self.bc.getContactPoints())
            if n_col > 0:
                raise RobotError('Collision!')
        return True

    def movel_traj(self, pose_trajectory) -> bool:
        joint_trajectory, deviations = self.trajectory_ik(pose_trajectory)
        if np.max(deviations) > self.th_tool_deivation:
            raise RobotError('Unreachable pose!')
        return self.movej_traj(joint_trajectory)

    def movel(self, path, 
            speed=1.2, acceleration=5):
        pose_path = None
        if len(path.shape) == 1:
            # single
            pose_path = np.stack([self.get_tcp_pose(), path], axis=0)
        else:
            # multi
            pose_path = path
            # pose_path = np.concatenate([self.get_tcp_pose().reshape(1,-1), path], axis=0)
        pose_traj = self.get_movel_pose_trajectory(pose_path, 
            speed=speed, acceleration=acceleration)
        return self.movel_traj(pose_traj)

    def get_tcp_pose_world(self):
        bc = self.bc
        link_state = bc.getLinkState(self.robot_body_id, 
            self.robot_end_effector_link_index)
        pos = np.array(link_state[0])
        rot = Rotation.from_matrix(np.array(
            bc.getMatrixFromQuaternion(link_state[1])).reshape(3,3))
        pose = pos_rot_to_pose(pos, rot)
        return pose
    
    def get_tcp_pose(self):
        pose = self.get_tcp_pose_world()
        local_pose = transform_pose(np.linalg.inv(self.tx_world_robot), pose)
        return local_pose
    
    def home(self):
        self.set_joints(self.home_joints)

def test():
    from pybullet_utils.bullet_client import BulletClient
    from cair_robot.common.primitive_util import get_base_fling_poses
    from matplotlib import pyplot as plt

    # bc = BulletClient(connection_mode=pybullet.GUI)
    # bc.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
    bc = BulletClient(connection_mode=pybullet.DIRECT)
    bc.setGravity(0, 0, -9.8)

    tx_world_robot = np.eye(4)
    tx_world_robot[:3, 3] = [-0.4, 0.05, 0.1]

    robot = UR5PyBulletRobot(bc, tx_world_robot, th_joint_speed=5.0)
    robot.set_joints(robot.home_joints)
    
    import time
    # while True:
    #     bc.stepSimulation()
    #     time.sleep(1/60)

    dt = 1/125
    base_fling_path = get_base_fling_poses()
    fling_path = base_fling_path
    # fling_path[:,0] = -0.2
    fling_path_local = transform_pose(np.linalg.inv(robot.tx_world_robot), fling_path)

    s = time.perf_counter()
    n = 100
    for i in range(n):
        robot.movel(fling_path_local)
    print((time.perf_counter() - s) / n)


    joint_trajectory, deviations = robot.get_movel_joint_trajectory(fling_path_local, dt=dt)


    max_q_diff = np.max(deviations)
    max_dq, max_ddq = get_joints_max_speed_acceleration(joint_trajectory, dt)

    time.sleep(1)
    for j in joint_trajectory:
        robot.set_joints(j)
        time.sleep(1/60)



# %%
