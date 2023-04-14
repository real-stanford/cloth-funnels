import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
from scipy.spatial import geometric_slerp


class LinearPath:
    """
    Geodesic interpolation of linear path
    """

    def __init__(self, start_pos, end_pos, norm='l2'):
        if norm == 'l2':
            distance = np.linalg.norm(end_pos - start_pos)
        elif norm == 'linf':
            distance = np.max(np.abs(end_pos - start_pos))
        else:
            raise RuntimeError('Invalid Norm Type')
        pos_interp = interp1d([0, distance], [start_pos, end_pos], 
            axis=0, fill_value='extrapolate')
        self.distance = distance
        self.interp = pos_interp
    
    def __len__(self):
        return self.distance

    def __call__(self, d):
        return self.interp(d)

class CircularPath:
    """
    Geodesic interpolation of an arc
    """
    @classmethod
    def from_corner_blend(cls, corner_pos, in_pos, out_pos, blend_radius, tol=1e-5):
        in_vec = corner_pos - in_pos
        out_vec = out_pos - corner_pos
        in_dir = in_vec / np.linalg.norm(in_vec)
        out_dir = out_vec / np.linalg.norm(out_vec)
        normal_vec = np.cross(in_dir, out_dir)
        normal_norm = np.linalg.norm(normal_vec)
        start_pos = corner_pos - in_dir * blend_radius
        end_pos = corner_pos + out_dir * blend_radius
        # check if angle too small
        if normal_norm < tol:
            return LinearPath(start_pos, end_pos)
        
        turn_angle = np.arccos(np.dot(in_dir, out_dir))
        hinge_angle = np.pi - turn_angle
        radius = blend_radius * np.tan(hinge_angle / 2)

        normal = normal_vec / normal_norm
        start_ray = np.cross(in_dir, normal)
        end_ray = np.cross(out_dir, normal)
        pivot = start_pos - start_ray * radius
        assert(np.linalg.norm(end_pos - (end_ray*radius+pivot)) < tol)
        return cls(start_ray, end_ray, pivot, radius, turn_angle)

    @classmethod
    def from_tangents(cls, start_pos, end_pos, start_direction, end_direction, tol=1e-5):
        start_direction = start_direction / np.linalg.norm(start_direction)
        end_direction = end_direction / np.linalg.norm(end_direction)
        normal_vec = np.cross(start_direction, end_direction)
        normal_norm = np.linalg.norm(normal_vec)
        if normal_norm < tol:
            return LinearPath(start_pos, end_pos)
        normal = normal_vec / normal_norm

        angle = np.arccos(np.dot(start_direction, end_direction))
        diff = end_pos - start_pos
        linear_distance = np.linalg.norm(diff)
        radius = 0.5 * linear_distance / np.sin(angle / 2)

        start_ray = np.cross(start_direction, normal)
        end_ray = np.cross(end_direction, normal)
        pivot = start_pos - start_ray * radius
        assert(np.linalg.norm(end_pos - (pivot + end_ray * radius)) < tol)
        return cls(start_ray, end_ray, pivot, radius, angle)
    
    def __init__(self, start_ray, end_ray, pivot, radius, angle):
        assert(np.allclose(np.linalg.norm(start_ray), 1))
        assert(np.allclose(np.linalg.norm(end_ray), 1))
        self.distance = radius * angle
        self.start_ray = start_ray
        self.end_ray = end_ray
        self.pivot = pivot
        self.radius = radius
        self.angle = angle
    
    def __len__(self):
        return self.distance
    
    def __call__(self, d):
        phase = d / self.distance
        ray = geometric_slerp(self.start_ray, self.end_ray, t=phase)
        point = ray * self.radius + self.pivot
        return point


class OrientationPath:
    def __init__(self, start_rot, end_rot, distance):
        self.interp = Slerp([0,1],
            Rotation.concatenate([start_rot, end_rot]))
        self.distance = distance

    def __len__(self):
        return self.distance
    
    def __call__(self, d):
        phase = d / self.distance
        rot = self.interp(phase)
        return rot


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

    path = np.zeros((4,9))
    path[:,:3] = pos_traj
    path[:,3:6] = rot_traj
    path[:,6:] = [speed,acceleration,radius]

    linear = LinearPath(pos_traj[0], pos_traj[1], rot_traj[0], rot_traj[1])

    circle = CircularPath.from_tangents(
        start_pos=np.array([0,0,0]),
        end_pos=np.array([1,1,0]),
        start_direction=np.array([0,1,0]),
        end_direction=np.array([1,0,0])
    )
    d = np.linspace(0, circle.distance, 100)
    samples = circle(d)

