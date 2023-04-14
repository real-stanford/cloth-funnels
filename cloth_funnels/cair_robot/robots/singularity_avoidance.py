import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation


def path_avoid_singularity(path, radius=0.25, 
        detour_ratio=np.sqrt(2)):
    if np.min(np.linalg.norm(path[:,:2], axis=-1)) < radius:
        raise RuntimeError('Path within singularity radius.')

    result_path = list()
    for i in range(len(path)-1):
        start_pose = path[i]
        end_pose = path[i+1]
        result_path.append(start_pose)

        # decide if linear path intersects with singularity cylindar
        pstart = start_pose[:2]
        pend = end_pose[:2]
        diff = pend - pstart
        length = np.linalg.norm(diff)

        direction = diff / length
        to_origin = -pstart
        proj_dist = np.dot(to_origin, direction)
        if 0 < proj_dist < length:
            nn_point = proj_dist * direction + pstart
            origin_dist = np.linalg.norm(nn_point)
            if origin_dist < radius:
                # need to insert waypoint
                detour_point = nn_point / origin_dist * (detour_ratio * radius)
                pos_interp = interp1d([0, length], [start_pose[:3], end_pose[:3]], axis=0)
                rot_interp = Slerp([0,length], 
                    rotations=Rotation.from_rotvec([
                        start_pose[-3:],
                        end_pose[-3:]
                    ]))
                pos = pos_interp(proj_dist)
                pos[:2] = detour_point
                rot = rot_interp(proj_dist).as_rotvec()
                detour_pose = np.zeros(6)
                detour_pose[:3] = pos
                detour_pose[-3:] = rot
                result_path.append(detour_pose)
        result_path.append(end_pose)
    result_path = np.array(result_path)
    return result_path
