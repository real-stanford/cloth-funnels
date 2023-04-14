import numpy as np
from scipy.spatial.transform import Rotation
import skimage.transform as st
import scipy.interpolate as si

def pos_rot_to_mat(pos, rot):
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4,4), dtype=pos.dtype)
    mat[...,:3,3] = pos
    mat[...,:3,:3] = rot.as_matrix()
    mat[...,3,3] = 1
    return mat

def mat_to_pos_rot(mat):
    pos = (mat[...,:3,3].T / mat[...,3,3].T).T
    rot = Rotation.from_matrix(mat[...,:3,:3])
    return pos, rot

def pos_rot_to_pose(pos, rot):
    shape = pos.shape[:-1]
    pose = np.zeros(shape+(6,), dtype=pos.dtype)
    pose[...,:3] = pos
    pose[...,3:] = rot.as_rotvec()
    return pose

def pose_to_pos_rot(pose):
    pos = pose[...,:3]
    rot = Rotation.from_rotvec(pose[...,3:])
    return pos, rot

def pose_to_mat(pose):
    return pos_rot_to_mat(*pose_to_pos_rot(pose))

def mat_to_pose(mat):
    return pos_rot_to_pose(*mat_to_pos_rot(mat))

def transform_pose(tx, pose):
    """
    tx: tx_new_old
    pose: tx_old_obj
    result: tx_new_obj
    """
    pose_mat = pose_to_mat(pose)
    tf_pose_mat = tx @ pose_mat
    tf_pose = mat_to_pose(tf_pose_mat)
    return tf_pose

def normalize(vec, tol=1e-7):
    return vec / np.maximum(np.linalg.norm(vec), tol)

def rot_from_directions(from_vec, to_vec):
    from_vec = normalize(from_vec)
    to_vec = normalize(to_vec)
    axis = np.cross(from_vec, to_vec)
    axis = normalize(axis)
    angle = np.arccos(np.dot(from_vec, to_vec))
    rotvec = axis * angle
    rot = Rotation.from_rotvec(rotvec)
    return rot

def get_center_affine(img_shape, rotation, scale, **kwargs):
    """
    (x,y) convention, upper left (0,0)
    """
    # scale > 1 will make image appear smaller
    assert(len(img_shape) == 2)
    offset = np.array(img_shape[::-1]) / 2
    pre = st.AffineTransform(translation=-offset).params
    aff = st.AffineTransform(rotation=rotation, scale=scale, **kwargs).params
    post = st.AffineTransform(translation=offset).params
    mat = post @ aff @ pre
    tf = st.AffineTransform(matrix=mat)
    return tf


def transform_points(pts, tx_new_old):
    dim = pts.shape[-1]
    assert (dim == 3) or (dim == 2)
    return pts @ tx_new_old[:dim,:dim].T + tx_new_old[:dim,dim]


def transform_pointcloud(xyz_pts, rigid_transform):
    """Apply rigid transformation to 3D pointcloud.

    Args:
        xyz_pts: Nx3 float array of 3D points
        rigid_transform: 3x4 or 4x4 float array defining a rigid transformation (rotation and translation)

    Returns:
        xyz_pts: Nx3 float array of transformed 3D points
    """
    xyz_pts = np.dot(rigid_transform[:3,:3],xyz_pts.T) # apply rotation
    xyz_pts = xyz_pts+np.tile(rigid_transform[:3,3].reshape(3,1),(1,xyz_pts.shape[1])) # apply translation
    return xyz_pts.T


def get_depth_interpolator(depth_im):
    x_coords = np.arange(depth_im.shape[0])
    y_coords = np.arange(depth_im.shape[1])
    d_interp = si.interp2d(
        x=x_coords, y=y_coords, z=depth_im.T, copy=False,
        fill_value=0)
    return d_interp


def pixel_to_3d(depth_im, pix, cam_pose, cam_intr, depth_scale=1):
    pix_y = np.clip(pix[:,0].astype(np.int64), 0, depth_im.shape[0]-1)
    pix_x = np.clip(pix[:,1].astype(np.int64), 0, depth_im.shape[1]-1)
    cam_pts_z = depth_im[pix_y, pix_x]
    cam_pts_z *= depth_scale
    cam_pts_x = (pix[:, 1]-cam_intr[0, 2]) * cam_pts_z/cam_intr[0, 0]
    cam_pts_y = (pix[:, 0]-cam_intr[1, 2]) * cam_pts_z/cam_intr[1, 1]
    cam_pts = np.array([cam_pts_x,cam_pts_y,cam_pts_z]).T
    wrd_pts = transform_points(cam_pts, cam_pose)
    return wrd_pts


def get_pointcloud(depth_img,  cam_intr, cam_pose=None):
    """Get 3D pointcloud from depth image.
    
    Args:
        depth_img: HxW float array of depth values in meters aligned with color_img
        cam_intr: 3x3 float array of camera intrinsic parameters
        cam_pose: (optional) 3x4 float array of camera pose matrix
        
    Returns:
        cam_pts: Nx3 float array of 3D points in camera/world coordinates
        color_pts: Nx3 uint8 array of color points
    """
    
    # Project depth into 3D pointcloud in camera coordinates
    pixel_y, pixel_x = np.indices(depth_img.shape[:2], dtype=depth_img.dtype)
    cam_intr = cam_intr.astype(depth_img.dtype)
    cam_pts_x = np.multiply(pixel_x-cam_intr[0,2],depth_img/cam_intr[0,0])
    cam_pts_y = np.multiply(pixel_y-cam_intr[1,2],depth_img/cam_intr[1,1])
    cam_pts_z = depth_img
    cam_pts = np.stack([cam_pts_x,cam_pts_y,cam_pts_z]).transpose(1,2,0)

    if cam_pose is not None:
        cam_pts = transform_points(cam_pts, cam_pose.astype(depth_img.dtype))
    return cam_pts
