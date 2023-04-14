import ikfastpy
import numpy as np


def validate_joint_trajectory(joints, dt, joint_speed_limit=3.14):
    joint_diff = np.diff(joints, axis=0)
    joint_vel = np.abs(joint_diff) / dt
    return np.all(joint_vel < joint_speed_limit)


def validate_tool_trajectory(pos, rot, dt, init_joint=None,
        joint_speed_limit=3.14):
    N = len(pos)
    poses = np.zeros((N,3,4))
    poses[:,:,3] = pos
    poses[:,:3,:3] = rot.as_matrix()
    
    ur5_kin = ikfastpy.PyKinematics()
    n_joints = ur5_kin.getDOF()

    prev_joint = init_joint
    for i, pose in enumerate(poses):
        print(i)
        joint_configs = ur5_kin.inverse(pose.reshape(-1).tolist())
        n_solutions = int(len(joint_configs)/n_joints)
        if n_solutions == 0:
            return False
        joint_configs = np.asarray(joint_configs).reshape(n_solutions,n_joints)
        if prev_joint is None:
            prev_joint = joint_configs[0]
        dists = np.mean(np.abs(joint_configs - prev_joint), axis=-1)
        best_idx = np.argmin(dists)
        best_joint = joint_configs[best_idx]
        # check
        joint_speed = np.abs(best_joint - prev_joint) / dt
        print(np.max(joint_speed))
        # if np.any(joint_speed > joint_speed_limit):
        #     return False
        prev_joint = best_joint
    return True


def test():
    proj_path = '/Users/cchi/dev/urscript_sim'
    import os
    import sys
    os.chdir(proj_path)
    sys.path.append(proj_path)

    from cair_robot.urscript_sim.trajectory import gen_movel_trajectory
    from scipy.spatial.transform import Rotation

    pos_traj = np.array([
        [0.4,0,0.2],
        [0.4,-0.2,0.2],
        [0.4,0.2,0.3],
        [0.4,0,0]
    ])

    init_rot = Rotation.from_euler('z', -np.pi/2) * Rotation.from_rotvec([np.pi, 0, 0])
    left_rotvec = (Rotation.from_euler('z', -np.pi/2) * Rotation.from_rotvec([np.pi, 0, 0])).as_rotvec()
    left_rotvec_tilt = (Rotation.from_euler('y', -np.pi/6) * Rotation.from_euler('z', -np.pi/2) * Rotation.from_rotvec([np.pi, 0, 0])).as_rotvec()
    left_rotvec_pre_fling = (Rotation.from_euler('x', -np.pi/4) * Rotation.from_euler('z', -np.pi/2) * Rotation.from_rotvec([np.pi, 0, 0])).as_rotvec()
    left_rotvec_after_fling = (Rotation.from_euler('x', np.pi/4) * Rotation.from_euler('z', -np.pi/2) * Rotation.from_rotvec([np.pi, 0, 0])).as_rotvec()

    rot_traj = np.array([
        left_rotvec,
        left_rotvec_pre_fling,
        left_rotvec_after_fling,
        left_rotvec
    ])

    speed = 1.2
    acceleration = 5
    radius = 0.05

    all_path = np.zeros((4,9))
    all_path[:,:3] = pos_traj
    all_path[:,3:6] = rot_traj
    all_path[:,6:] = [speed,acceleration,radius]

    init_pose = all_path[0,:6]
    path = all_path[1:]

    dt = 0.01
    trajectory = gen_movel_trajectory(init_pose, path)
    pos, rot = trajectory.sample(dt)

    is_valid = validate_tool_trajectory(pos, rot, dt)


def test():
    proj_path = '/Users/cchi/dev/urscript_sim'
    import os
    import sys
    os.chdir(proj_path)
    sys.path.append(proj_path)

    from scipy.spatial.transform import Rotation
    from cair_robot.urscript_sim.trajectory import gen_movel_trajectory


    rec_joints = np.array([
        [5.75, -86.42, 89.07, -92.74, -90.01, -84.25],
        [31.73,-66.69,64.40,-87.78,-90.05,-58.27],
        [-38,-78,56.65,-68.78,-89.94,-128],
        [5.75,-64.78,117.56,-142.86,-90.01,-84.25]
    ])
    rec_pose = np.array([
        [-0.5,-0.16,0.2,0,3.143,0],
        [-0.5,-0.437,0.2,0,3.143,0],
        [-0.5,0.252,0.352,0,3.143,0],
        [-0.5,-0.16,-0.133,0,3.143,0]
    ])

    init_joint = rec_joints[0] / 180 * np.pi

    init_pose = rec_pose[0]
    path = np.zeros((3,9))
    path[:,:6] = rec_pose[1:]

    speed = 1.2
    acceleration = 5
    radius = 0.05
    path[:,6:] = [speed,acceleration,radius]

    dt = 0.01
    trajectory = gen_movel_trajectory(init_pose, path)
    pos, rot = trajectory.sample(dt)

    is_valid = validate_tool_trajectory(pos, rot, dt)


    ur5_kin = ikfastpy.PyKinematics()
    n_joints = ur5_kin.getDOF()
    ee_pose = ur5_kin.forward(init_joint)
    ee_pose = np.asarray(ee_pose).reshape(3,4)
    fk_pose = np.zeros((6,))
    fk_pose[:3] = ee_pose[:,-1]
    fk_pose[3:] = Rotation.from_matrix(ee_pose[:,:3]).as_rotvec()

    flange_pose = init_pose.copy()
    flange_pose[2] += 0.21
    diff = fk_pose[:3] - flange_pose[:3]
