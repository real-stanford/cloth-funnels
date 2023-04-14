import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from cair_robot.urscript_sim.geodesic_path import LinearPath, CircularPath, OrientationPath
from cair_robot.urscript_sim.speed_profile import RampSpeedProfile, sequential_speed_profile, TrapezoidalSpeedProfile
from scipy.interpolate import interp1d
import itertools
import numba


@numba.jit(nopython=True)
def gen_intervals(lengths, use_inf=True):
    assert(len(lengths.shape) == 1)
    N = lengths.shape[0]
    begin = np.zeros_like(lengths)
    end = np.zeros_like(lengths)
    curr_dist = 0
    for i, length in enumerate(lengths):
        if use_inf and (i == 0):
            begin[i] = -np.inf
        else:
            begin[i] = curr_dist
        
        curr_dist += length

        if use_inf and (i == (N-1)):
            end[i] = np.inf
        else:
            end[i] = curr_dist
    
    return begin, end

@numba.jit(nopython=True)
def interval_search(begin, end, value):
    """
    assume sorted non-overlapping intervals [begin, end)
    """
    assert(len(begin.shape) == 1)
    assert(begin.shape == end.shape)
    assert(len(value.shape) == 1)
    N = len(begin)
    B = len(value)

    idxs = np.zeros(B, dtype=np.int64)
    for i in range(B):
        x = value[i]
        left_idx = 0
        right_idx = N
        while left_idx != right_idx:
            mid_idx = (left_idx + right_idx) // 2
            if x < begin[mid_idx]:
                right_idx = mid_idx - 1
            elif x >= end[mid_idx]:
                left_idx = mid_idx + 1
            else:
                left_idx = mid_idx
                break
        idxs[i] = left_idx
    return idxs

@numba.jit(nopython=True)
def _sample_q_progress(t, speed_profiles, idxs, real_begin, left_distance):
    pos = np.zeros_like(t)
    for i, idx in enumerate(idxs):
        ts = t[i] - real_begin[idx]
        q = speed_profiles[idx].q(ts) + left_distance[idx]
        pos[i] = q
    return pos

def sample_q_progress(t, speed_profiles):
    """
    t: monotonic increasing
    """
    durations = np.array([x.duration for x in speed_profiles])
    distances = np.array([x.distance for x in speed_profiles])
    begin, end = gen_intervals(durations, use_inf=True)
    real_begin, _ = gen_intervals(durations, use_inf=False)
    left_distance, _ = gen_intervals(distances, use_inf=False)
    idxs = interval_search(begin, end, t)
    pos = _sample_q_progress(t, numba.typed.List(speed_profiles), idxs, real_begin, left_distance)
    return pos


class PoseTrajectory:
    def __init__(self, pos_paths, rot_paths, speed_profiles):
        self.pos_paths = pos_paths
        self.rot_paths = rot_paths
        self.speed_profiles = speed_profiles
        self.distance = sum([x.distance for x in pos_paths])
        self.duration = sum([x.duration for x in speed_profiles])
    
    def sample(self, dt):
        n_samples = int(self.duration / dt)
        t = np.arange(n_samples) * dt
        q_progress = sample_q_progress(t, self.speed_profiles)

        distances = np.array([x.distance for x in self.pos_paths])
        begin, end = gen_intervals(distances, use_inf=True)
        left_distance, _ = gen_intervals(distances, use_inf=False)
        idxs = interval_search(begin, end, q_progress)

        pos_segments = list()
        rot_segments = list()
        for i in range(len(self.rot_paths)):
            mask = (idxs == i)
            local_q = q_progress[mask] - left_distance[i]
            pos = self.pos_paths[i](local_q)
            rot = self.rot_paths[i](local_q)
            pos_segments.append(pos)
            rot_segments.append(rot)

        pos = np.concatenate(pos_segments, axis=0)
        rot = Rotation.concatenate(rot_segments)
        return pos, rot


class JointTrajectory:
    def __init__(self, joint_paths, speed_profiles):
        self.joint_paths = joint_paths
        self.speed_profiles = speed_profiles
        self.distance = sum([x.distance for x in joint_paths])
        self.duration = sum([x.duration for x in speed_profiles])
    
    def sample(self, dt):
        n_samples = int(self.duration / dt)
        t = np.arange(n_samples) * dt
        q_progress = sample_q_progress(t, self.speed_profiles)

        distances = np.array([x.distance for x in self.joint_paths])
        begin, end = gen_intervals(distances, use_inf=True)
        left_distance, _ = gen_intervals(distances, use_inf=False)
        idxs = interval_search(begin, end, q_progress)

        joint_segments = list()
        for i in range(len(self.joint_paths)):
            mask = (idxs == i)
            local_q = q_progress[mask] - left_distance[i]
            joint = self.joint_paths[i](local_q)
            joint_segments.append(joint)

        joints = np.concatenate(joint_segments, axis=0)
        return joints


def gen_movel_trajectory(init_pose, path):
    """
    path: Nx9 array
    3xpos, 3xrotvec, speed, acceleartion, radius
    """
    all_pose = np.concatenate([init_pose.reshape(1,-1),path[:,:6]])
    all_pos = all_pose[:,:3]
    all_rot = Rotation.from_rotvec(all_pose[:,3:])

    if len(path) == 1:
        # single linear path
        assert(False)

    path_diff = np.diff(all_pos, axis=0)
    path_lengths = np.linalg.norm(path_diff, axis=-1)

    # check radius overlap
    via_radius = path[:-1,8]
    assert(path_lengths[0] > path[0,8])
    assert(path_lengths[-1] > path[-2,8])
    if len(path) > 2:
        via_radius_sum = via_radius[:-1] + via_radius[1:]
        via_dists = path_lengths[:-1]
        assert(np.all(via_radius_sum < via_dists))

    # construct one linear path per raw path
    linear_pos_paths = list()
    linear_rot_paths = list()
    for i in range(len(path_lengths)):
        # per each segment between two poses
        path_length = path_lengths[i]
        if i == 0:
            # beginning segment
            start_dist = 0
            end_dist = path_length - via_radius[i]
        elif i == (len(path_lengths) - 1):
            # last segment
            start_dist = via_radius[i-1]
            end_dist = path_length
        else:
            # middle_sement
            start_dist = via_radius[i-1]
            end_dist = path_length - via_radius[i]

        dist_to_rot = OrientationPath(
            all_rot[i], all_rot[i+1], path_length)
        dist_to_pos = interp1d([0, path_length], 
            all_pos[i:i+2], axis=0, 
            bounds_error='extrapolate')
        start_pos = dist_to_pos(start_dist)
        end_pos = dist_to_pos(end_dist)
        start_rot = dist_to_rot(start_dist)
        end_rot = dist_to_rot(end_dist)

        this_pos_path = LinearPath(start_pos, end_pos)
        this_rot_path = OrientationPath(
            start_rot, end_rot, end_dist-start_dist)
        linear_pos_paths.append(this_pos_path)
        linear_rot_paths.append(this_rot_path)

    # construct one arc path per via point
    arc_pos_paths = list()
    arc_rot_paths = list()
    for i in range(len(via_radius)):
        radius = via_radius[i]
        in_dist_to_rot = OrientationPath(
            all_rot[i], all_rot[i+1], path_lengths[i])
        out_dist_to_rot = OrientationPath(
            all_rot[i+1], all_rot[i+2], path_lengths[i+1])

        in_cut_dist = path_lengths[i] - radius
        out_cut_dist = radius

        via_pos = all_pos[i+1]
        in_pos = all_pos[i]
        out_pos = all_pos[i+2]
        this_pos_path = CircularPath.from_corner_blend(
            corner_pos=via_pos, 
            in_pos=in_pos, out_pos=out_pos,
            blend_radius=radius)

        in_rot = in_dist_to_rot(in_cut_dist)
        out_rot = out_dist_to_rot(out_cut_dist)
        this_rot_path = OrientationPath(
            in_rot, out_rot, 
            distance=this_pos_path.distance)

        arc_pos_paths.append(this_pos_path)
        arc_rot_paths.append(this_rot_path)

    assert(len(linear_pos_paths) == (len(arc_pos_paths) + 1))

    # interleave linear and arc path to get full geodesic path
    all_pos_paths = [linear_pos_paths[0]] \
        + list(itertools.chain(*zip(arc_pos_paths, linear_pos_paths[1:])))
    all_rot_paths = [linear_rot_paths[0]] \
        + list(itertools.chain(*zip(arc_rot_paths, linear_rot_paths[1:])))

    # create time profile
    params = path[:,-3:]
    speed = params[:,0]
    acceleration = params[:,0]

    speed_profile_segments = list()
    for i in range(len(path_lengths)):
        linear_length = linear_pos_paths[i].distance
        arc_length = 0
        if i != 0:
            arc_length += arc_pos_paths[i-1].distance / 2
        if i != (len(path_lengths) - 1):
            arc_length += arc_pos_paths[i].distance / 2
        segment_length = linear_length + arc_length

        if i == 0:
            start_speed = 0
        else:
            start_speed = speed_profile_segments[-1].end_speed
        
        if i == (len(path_lengths) - 1):
            end_speed = 0
        else:
            end_speed = speed[i]
        
        this_sp = RampSpeedProfile(distance=segment_length, 
            start_speed=start_speed, end_speed=end_speed,
            acceleration=acceleration[i])
        speed_profile_segments.append(this_sp)
    
    trajectory = PoseTrajectory(
        pos_paths=all_pos_paths, 
        rot_paths=all_rot_paths, 
        speed_profiles=speed_profile_segments)
    return trajectory


def gen_movel_noblend_trajectory(init_pose, path):
    all_pose = np.concatenate([init_pose.reshape(1,-1), path[:,:6]])
    all_pos = all_pose[:,:3]
    all_rot = Rotation.from_rotvec(all_pose[:,3:])

    path_params = path[:,6:]
    pos_paths = list()
    rot_paths = list()
    speed_profiles = list()
    for i in range(len(path_params)):
        speed, acceleration = path_params[i]
        start_pos = all_pos[i]
        end_pos = all_pos[i+1]
        path = LinearPath(start_pos, end_pos)
        distance = path.distance
        start_rot = all_rot[i]
        end_rot = all_rot[i+1]
        rot_path = OrientationPath(start_rot, end_rot, distance)
        profile = TrapezoidalSpeedProfile(distance, 
            speed=speed, 
            acceleration=acceleration)
        
        pos_paths.append(path)
        rot_paths.append(rot_path)
        speed_profiles.append(profile)

    trajectory = PoseTrajectory(
        pos_paths=pos_paths, 
        rot_paths=rot_paths, 
        speed_profiles=speed_profiles)
    return trajectory


def gen_movej_noblend_trajectory(init_joint, joint_path):
    all_joints = np.concatenate([init_joint.reshape(1,-1), joint_path[:,:6]])

    path_params = joint_path[:,6:]
    joint_paths = list()
    speed_profiles = list()
    for i in range(len(path_params)):
        speed, acceleration = path_params[i]
        start_joint = all_joints[i]
        end_joint = all_joints[i+1]
        path = LinearPath(start_joint, end_joint, norm='linf')
        distance = path.distance
        profile = TrapezoidalSpeedProfile(distance, 
            speed=speed, 
            acceleration=acceleration)
        joint_paths.append(path)
        speed_profiles.append(profile)
    
    trajectory = JointTrajectory(
        joint_paths=joint_paths,
        speed_profiles=speed_profiles)
    return trajectory


def test():
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

    trajectory = gen_movel_trajectory(init_pose, path)
    pos, rot = trajectory.sample(0.01)

    init_j = np.zeros(6)
    joint_path = np.ones((1,8))
    traj = gen_movej_noblend_trajectory(init_j, joint_path)
    joints = traj.sample(0.01)

