from torch import cat, tensor
from matplotlib import pyplot as plt
import pickle
import numpy as np
from imageio import get_writer
from cloth_funnels.learning.utils import rewards_from_group
import trimesh
import pyflex
from os import devnull
import subprocess
import os
import imageio
# import OpenEXR
from Imath import PixelType
import random
import cv2
from scipy import ndimage as nd
from math import ceil
import io
from PIL import Image, ImageDraw, ImageFont
import skimage.morphology as morph
from pathlib import Path
from cloth_funnels.learning.nocs_unet_inference import nocs_pred, get_flow_correspondence
import torch
import cv2
from cloth_funnels.learning.utils import rewards_from_group

#################################################
################# RENDER UTILS ##################
#################################################


def find_max_indices(vmaps, mask):

    flat_indices = torch.tensor(np.indices(vmaps.shape)).reshape(4, -1).to(vmaps.device)

    masked_flat_vmaps = vmaps.reshape(1, -1).masked_select(mask.flatten())
    masked_flat_indices = flat_indices.masked_select(torch.unsqueeze(mask.flatten(), 0)).reshape(4, -1)
    max_flat_indices = torch.argsort(masked_flat_vmaps, descending=True)

    indices = masked_flat_indices[:, max_flat_indices].T
    return indices

def grid_index(x, y, dimx):
    return y*dimx + x


def get_cloth_mesh(
        dimx,
        dimy,
        base_index=0):
    if dimx == -1 or dimy == -1:
        positions = pyflex.get_positions().reshape((-1, 4))
        vertices = positions[:, :3]
        faces = pyflex.get_faces().reshape((-1, 3))
    else:
        positions = pyflex.get_positions().reshape((-1, 4))
        faces = []
        vertices = positions[:, :3]
        for y in range(dimy):
            for x in range(dimx):
                if x > 0 and y > 0:
                    faces.append([
                        base_index + grid_index(x-1, y-1, dimx),
                        base_index + grid_index(x, y-1, dimx),
                        base_index + grid_index(x, y, dimx)
                    ])
                    faces.append([
                        base_index + grid_index(x-1, y-1, dimx),
                        base_index + grid_index(x, y, dimx),
                        base_index + grid_index(x-1, y, dimx)])
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def blender_render_cloth(cloth_mesh, resolution):
    output_prefix = '/tmp/' + str(os.getpid())
    obj_path = output_prefix + '.obj'
    cloth_mesh.export(obj_path)
    commands = [
        'blender',
        'cloth.blend',
        '-noaudio',
        '-E', 'BLENDER_EEVEE',
        '--background',
        '--python',
        'render_rgbd.py',
        obj_path,
        output_prefix,
        str(resolution)]
    with open(devnull, 'w') as FNULL:
        while True:
            try:
                # render images
                subprocess.check_call(
                    commands,
                    stdout=FNULL)
                break
            except Exception as e:
                print(e)
    # get images
    output_dir = Path(output_prefix)
    color = imageio.imread(str(list(output_dir.glob('*.png'))[0]))
    color = color[:, :, :3]
    # depth = OpenEXR.InputFile(str(list(output_dir.glob('*.exr'))[0]))
    redstr = depth.channel('R', PixelType(PixelType.FLOAT))
    depth = np.fromstring(redstr, dtype=np.float32)
    depth = depth.reshape(resolution, resolution)
    return color, depth


def render_lift_cloth(cloth_mesh, resolution):
    output_prefix = '/tmp/' + str(os.getpid())
    obj_path = output_prefix + '.obj'
    cloth_mesh.export(obj_path)
    commands = [
        'blender',
        'lifted_cloth.blend',
        '-noaudio',
        '-E', 'BLENDER_EEVEE',
        '--background',
        '--python',
        'render_rgbd.py',
        obj_path,
        output_prefix,
        str(resolution)]
    with open(devnull, 'w') as FNULL:
        while True:
            try:
                # render images
                subprocess.check_call(
                    commands,
                    stdout=FNULL)
                break
            except Exception as e:
                print(e)
    # get images
    output_dir = Path(output_prefix)
    color = imageio.imread(str(list(output_dir.glob('*.png'))[0]))
    color = color[:, :, :3]
    depth = OpenEXR.InputFile(str(list(output_dir.glob('*.exr'))[0]))
    redstr = depth.channel('R', PixelType(PixelType.FLOAT))
    depth = np.fromstring(redstr, dtype=np.float32)
    depth = depth.reshape(resolution, resolution)
    return color, depth

#################################################
################ TRANSFORM UTILS ################
#################################################


def rot2d(angle, degrees=True):
    if degrees:
        angle = np.pi*angle/180
    return np.array([
        [np.cos(angle), np.sin(angle), 0],
        [-np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ]).T


def translate2d(translation):
    return np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1],
    ]).T


def scale2d(scale):
    return np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1],
    ]).T


def get_transform_matrix(original_dim, resized_dim, rotation, scale):
    # resize
    resize_mat = scale2d(original_dim/resized_dim)
    # scale
    scale_mat = np.matmul(
        np.matmul(
            translate2d(-np.ones(2)*(resized_dim//2)),
            scale2d(scale),
        ), translate2d(np.ones(2)*(resized_dim//2)))
    # rotation
    rot_mat = np.matmul(
        np.matmul(
            translate2d(-np.ones(2)*(resized_dim//2)),
            rot2d(rotation),
        ), translate2d(np.ones(2)*(resized_dim//2)))
    return np.matmul(np.matmul(scale_mat, rot_mat), resize_mat)


def compute_pose(pos, lookat, up=[0, 0, 1]):
    norm = np.linalg.norm
    if type(lookat) != np.array:
        lookat = np.array(lookat)
    if type(pos) != np.array:
        pos = np.array(pos)
    if type(up) != np.array:
        up = np.array(up)
    f = (lookat - pos)
    f = f/norm(f)
    u = up / norm(up)
    s = np.cross(f, u)
    s = s/norm(s)
    u = np.cross(s, f)
    view_matrix = [
        s[0], u[0], -f[0], 0,
        s[1], u[1], -f[1], 0,
        s[2], u[2], -f[2], 0,
        -np.dot(s, pos), -np.dot(u, pos), np.dot(f, pos), 1
    ]
    view_matrix = np.array(view_matrix).reshape(4, 4).T
    pose_matrix = np.linalg.inv(view_matrix)
    pose_matrix[:, 1:3] = -pose_matrix[:, 1:3]
    return pose_matrix


def compute_intrinsics(fov, image_size):
    image_size = float(image_size)
    focal_length = (image_size / 2)\
        / np.tan((np.pi * fov / 180) / 2)
    return np.array([[focal_length, 0, image_size / 2],
                     [0, focal_length, image_size / 2],
                     [0, 0, 1]])


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


def get_pointcloud(depth_img, color_img, cam_intr, cam_pose=None):
    """Get 3D pointcloud from depth image.
    
    Args:
        depth_img: HxW float array of depth values in meters aligned with color_img
        color_img: HxWx3 uint8 array of color image
        cam_intr: 3x3 float array of camera intrinsic parameters
        cam_pose: (optional) 3x4 float array of camera pose matrix
        
    Returns:
        cam_pts: Nx3 float array of 3D points in camera/world coordinates
        color_pts: Nx3 uint8 array of color points
    """

    img_h = depth_img.shape[0]
    img_w = depth_img.shape[1]

    # Project depth into 3D pointcloud in camera coordinates
    pixel_x,pixel_y = np.meshgrid(np.linspace(0,img_w-1,img_w),
                                  np.linspace(0,img_h-1,img_h))
    cam_pts_x = np.multiply(pixel_x-cam_intr[0,2],depth_img/cam_intr[0,0])
    cam_pts_y = np.multiply(pixel_y-cam_intr[1,2],depth_img/cam_intr[1,1])
    cam_pts_z = depth_img
    cam_pts = np.array([cam_pts_x,cam_pts_y,cam_pts_z]).transpose(1,2,0).reshape(-1,3)

    if cam_pose is not None:
        cam_pts = transform_pointcloud(cam_pts, cam_pose)
    color_pts = None if color_img is None else color_img.reshape(-1, 3)

    return cam_pts, color_pts

def pixel_to_3d(depth_im, x, y,
                pose_matrix,
                fov=39.5978,
                depth_scale=1):
    intrinsics_matrix = compute_intrinsics(fov, depth_im.shape[0])
    click_z = depth_im[y, x]
    click_z *= depth_scale
    click_x = (x-intrinsics_matrix[0, 2]) * \
        click_z/intrinsics_matrix[0, 0]
    click_y = (y-intrinsics_matrix[1, 2]) * \
        click_z/intrinsics_matrix[1, 1]
    if click_z == 0:
        raise Exception('Invalid pick point')
    # 3d point in camera coordinates
    point_3d = np.asarray([click_x, click_y, click_z])
    point_3d = np.append(point_3d, 1.0).reshape(4, 1)
    # Convert camera coordinates to world coordinates
    target_position = np.dot(pose_matrix, point_3d)
    target_position = target_position[0:3, 0]
    target_position[0] = - target_position[0]
    return target_position


def pixels_to_3d_positions(
        transform_pixels, scale, rotation, pretransform_depth,
        transformed_depth, pose_matrix=None,
        pretransform_pix_only=False, **kwargs):

    # print("\n\n")
    # print("transform rotation: ", rotation)
    # print("transform scale: ", scale)
    # print("original dimensions: ", pretransform_depth.shape[0])
    # print("transformed dimensions: ", transformed_depth.shape[0]) 

    mat = get_transform_matrix(
        original_dim=pretransform_depth.shape[0],
        resized_dim=transformed_depth.shape[0],
        rotation=-rotation,  # TODO bug
        scale=scale)

    # print("Pixels before matmul: ", transform_pixels)
    pixels = np.concatenate((transform_pixels, np.array([[1], [1]])), axis=1)
    pixels = np.matmul(pixels, mat)[:, :2].astype(int)
    pix_1, pix_2 = pixels
    max_idx = pretransform_depth.shape[0]
    transformed_depth[transform_pixels[0][0], transform_pixels[0][1]] = 0
    transformed_depth[transform_pixels[1][0], transform_pixels[1][1]] = 1
    
    if (pixels < 0).any() or (pixels >= max_idx).any():
        print("pixels out of bounds", pixels, "\n\n\n")
        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(transformed_depth)
        # axs[0].set_title(transform_pixels)
        # axs[1].imshow(pretransform_depth)
        # axs[1].set_title(pretransform_depth.mean())
        # plt.savefig('pixels_fig.png')
        # exit(1)
        return {
            'valid_action': False,
            'p1': None, 'p2': None,
            'pretransform_pixels': np.array([pix_1, pix_2])
        }
    # if pretransform_pix_only:
    #     return {
    #         'valid_action': True,
    #         'pretransform_pixels': np.array([pix_1, pix_2])
    #     }
    # Note this order of x,y is not a bug
    x, y = pix_1
    p1 = pixel_to_3d(depth_im=pretransform_depth,
                     x=x, y=y,
                     pose_matrix=pose_matrix)
    # Same here
    x, y = pix_2
    p2 = pixel_to_3d(depth_im=pretransform_depth,
                     x=x, y=y,
                     pose_matrix=pose_matrix)

    return {
        'valid_action': p1 is not None and p2 is not None,
        'p1': p1,
        'p2': p2,
        'pretransform_pixels': np.array([pix_1, pix_2])
    }


#################################################
############ VISUALIZATION UTILS ################
#################################################

def draw_circled_lines(pixels, shape=None, img=None, thickness=1):
    if shape is not None:
        shape = list(shape) + [4]
        img = np.zeros(shape)
    else:
        assert img is not None
    left, right = pixels
    img = cv2.circle(
        img=img, center=(int(left[1]), int(left[0])),
        radius=thickness*2, color=(0, 1, 0, 1), thickness=thickness)
    img = cv2.line(
        img=img,
        pt1=(int(left[1]), int(left[0])),
        pt2=(int(right[1]), int(right[0])),
        color=(1, 1, 0, 1), thickness=thickness)
    img = cv2.circle(
        img=img, center=(int(right[1]), int(right[0])),
        radius=thickness*2, color=(1, 0, 0, 1), thickness=thickness)
    return img


def draw_circled_lines_with_arrow(
        pixels, shape=None, img=None, thickness=1):
    if shape is not None:
        shape = list(shape) + [4]
        img = np.zeros(shape)
    else:
        assert img is not None
    left, right = pixels
    img = cv2.circle(
        img=img, center=(int(left[1]), int(left[0])),
        radius=thickness*2, color=(1, 0, 1, 1), thickness=thickness)
    img = cv2.line(
        img=img,
        pt1=(int(left[1]), int(left[0])),
        pt2=(int(right[1]), int(right[0])),
        color=(1, 1, 0, 1), thickness=thickness)
    img = cv2.circle(
        img=img, center=(int(right[1]), int(right[0])),
        radius=thickness*2, color=(0, 1, 1, 1), thickness=thickness)
    direction = np.cross((left - right).tolist() +
                         [0], np.array([0, 0, 1]))[:2]
    center_start = ((left + right) / 2).astype(int)
    center_end = center_start + direction
    img = cv2.arrowedLine(
        img=img,
        pt1=(int(center_start[1]), int(center_start[0])),
        pt2=(int(center_end[1]), int(center_end[0])),
        color=(1, 0, 0, 1), thickness=thickness)
    return img


def draw_arrow(pixels, shape=None, img=None, thickness=1, color=(0, 1, 1, 1)):
    if shape is not None:
        shape = list(shape) + [4]
        img = np.zeros(shape)
    else:
        assert img is not None
    start, end = pixels
    img = cv2.arrowedLine(
        img=img,
        pt1=(int(start[1]), int(start[0])),
        pt2=(int(end[1]), int(end[0])),
        color=color, thickness=thickness)
    return img


def draw_action(action_primitive, shape, pixels, **kwargs):
    if action_primitive == 'fling':
        return draw_circled_lines(
            shape=shape, pixels=pixels, **kwargs)
    elif action_primitive == 'stretchdrag':
        return draw_circled_lines_with_arrow(
            shape=shape, pixels=pixels, **kwargs)
    elif action_primitive == 'drag':
        return draw_arrow(
            shape=shape, pixels=pixels,
            color=(1, 0, 1, 1), **kwargs)
    elif action_primitive == 'place':
        return draw_arrow(
            shape=shape, pixels=pixels,
            color=(0, 1, 1, 1), **kwargs)
    else:
        raise NotImplementedError()


def visualize_action(
    action_primitive, transformed_pixels,
    pretransform_pixels,
    rotation, scale, pretransform_depth,
    pretransform_rgb,
    transformed_rgb, value_map=None,
    all_value_maps=None, mask=None,
        **kwargs):
    if value_map is None and all_value_maps is None:
        # final resized
        if pretransform_rgb.shape[0] != pretransform_rgb.shape[1]:
            pretransform_rgb = get_square_crop(pretransform_rgb.copy())
        plt.imshow(pretransform_rgb)
        action = draw_action(
            action_primitive=action_primitive,
            shape=pretransform_depth.shape[:2],
            pixels=pretransform_pixels,
            thickness=3)
        plt.imshow(action, alpha=0.9)
        plt.title(f'Final {action_primitive}')
    else:
        fig, axes = plt.subplots(1, 3)
        fig.set_figheight(3.5)
        fig.set_figwidth(9)
        for ax in axes.flatten():
            ax.axis('off')
        if value_map is not None:
            value_map = value_map.to('cpu')

            masked_vmap = value_map[mask]
            if masked_vmap.numel() == 0:
                min_vmap = value_map.min()
                max_vmap = value_map.max()
            else:
                min_vmap = masked_vmap.min()
                max_vmap = masked_vmap.max()

            imshow = axes[0].imshow(
                value_map * (mask.int()) + min_vmap * (1-mask.int()), cmap='jet',
                vmin=min_vmap, vmax=max_vmap)
            axes[0].set_title('Value Map')
            fig.colorbar(mappable=imshow, ax=axes[0], shrink=0.8)
        else:
            axes[0].set_title('No Value Map')
        axes[1].imshow(
            np.swapaxes(np.swapaxes(transformed_rgb, 0, -1), 0, 1))
        action = draw_action(
            action_primitive=action_primitive,
            shape=transformed_rgb.shape[-2:],
            pixels=transformed_pixels)
        axes[1].imshow(action, alpha=0.9)
        axes[1].set_title(action_primitive)
        # final resized
        if pretransform_rgb.shape[0] != pretransform_rgb.shape[1]:
            pretransform_rgb = get_square_crop(pretransform_rgb.copy())
        axes[2].imshow(pretransform_rgb)
        action = draw_action(
            action_primitive=action_primitive,
            shape=pretransform_depth.shape[:2],
            pixels=pretransform_pixels,
            thickness=3)
        if action.shape[0] != action.shape[1]:
            action = get_square_crop(action.copy())
        axes[2].imshow(action, alpha=0.9)
        axes[2].set_title(f'Final {action_primitive}')
    plt.tight_layout(pad=0)
    # dump to image in memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    action_visualization = np.array(Image.open(buf)).astype(np.uint8)
    if value_map is not None or all_value_maps is not None:
        plt.close(fig)

    return action_visualization


def plot_before_after(group, fontsize=16, output_path=None, step=None):
    fontsize=40
    fig, axs = \
        plt.subplots(1, 2, figsize=(10, 5))
    fig.set_figheight(10)
    fig.set_figwidth(20)
    
    for ax in axs.flatten():
        ax.axis('off')

    def get_img(key):
        return np.swapaxes(np.swapaxes(np.array(group[key]), 0, -1), 0, 1)


    def log_data(image, prefix, colors=None):

        if colors == None:
            colors = [(255, 255, 255)] * 4

        image = np.ascontiguousarray(image * 255).astype(np.uint8)
        #put text on the top left corner of this 480x480 image
        cv2.putText(image, f"R: {group.attrs[f'{prefix}_icp_distance']:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colors[0], 2)
        cv2.putText(image, f"D: {group.attrs[f'{prefix}_l2_distance']:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colors[1], 2) 
        cv2.putText(image, f"W: {group.attrs[f'{prefix}_weighted_distance']:.3f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colors[2], 2) 
        cv2.putText(image, f"L: {group.attrs[f'{prefix}_pointwise_distance']:.3f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colors[3], 2) 

        return image
        

    preaction_img = log_data(get_img('pretransform_observations')[..., :3], 'preaction')
    comparisons = [(group.attrs[f'postaction_{d}'] < group.attrs[f'preaction_{d}']) for d in ['icp_distance', 'l2_distance', 'weighted_distance', 'pointwise_distance']]
    colors = [(0, 255, 0) if c else (255, 0, 0) for c in comparisons]
    postaction_img = log_data(get_img('next_observations')[..., :3], 'postaction', colors)

    axs[0].imshow(preaction_img)
    # axs[0].set_title(f"[{step}] W: {group.attrs['preaction_weighted_distance']:.3f}, D: {group.attrs['preaction_l2_distance']:.3f} ", fontsize=fontsize)

    axs[1].imshow(postaction_img)
    # axs[1].set_title(f"[{step}] W: {(group.attrs['postaction_weighted_distance']):.3f}, D: {group.attrs['postaction_l2_distance']:.3f}", fontsize=fontsize)

    fig.tight_layout()
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path)
        plt.close(fig)
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf)


def visualize_grasp(group, key, path_prefix, dir_path, fontsize=30, include_videos=True):
    step = int(key.split('step')[-1].split('_last')[0])
    episode_id = int(key.split('step')[0][:-1])
    action_mask = torch.tensor(np.array(group['action_mask'])).bool()
    predicted = float(torch.tensor(np.array(group["value_map"])).masked_select(action_mask).item())
    output = f'<td> Episode {episode_id}, Step {step} \
    <div> <strong> Reward </strong>: {"{:.3f}".format(-1 * group.attrs["postaction_weighted_distance"] + 1)} </div> \
    <div> <strong> Delta </strong>: {" ".join([str(x.item())[:4] for x in rewards_from_group(group).items()])} </div> \
    <div> <strong> Predicted </strong>: {predicted:.3f} </div> \
    <div> <strong> Error </strong> : {abs(rewards_from_group(group)["weighted"] - prediction):.3f} </div> \
    </td><td> '

    # Plot all observations and value maps
    if 'value_maps' in group and 'all_obs' in group:
        output_path = path_prefix + '_all.png'
        output += f'<img src="{output_path}" height="256px"> </td> <td>'
        if not os.path.exists(dir_path+output_path):
            value_maps = np.array(group['value_maps'])
            masks = np.array(group['masks'])

            value_maps = value_maps * masks + (1-masks) * value_maps[masks].min()

            fig, axes = plt.subplots(5, 12)
            axes = axes.transpose().flatten()
            fig.set_figheight(5)
            fig.set_figwidth(12)
            max_value = value_maps[:, 5:-5, 5:-5].max()
            max_index = np.where(value_maps == max_value)
            min_value = value_maps.min()
            for ax, value_map in zip(axes, value_maps):
                ax.axis('off')
                ax.imshow(value_map, cmap='jet',
                          vmin=min_value, vmax=max_value)

                if (value_map == max_value).any():
                    circle = np.zeros(value_map.shape)
                    center = value_map.shape[0]//2
                    circle = cv2.circle(
                        img=circle,
                        center=(center, center),
                        radius=center,
                        color=1,
                        thickness=3)
                    ax.imshow(circle, alpha=circle, cmap='Blues')
            plt.tight_layout(pad=0)
            plt.savefig(dir_path + output_path)
            plt.close(fig)

    if 'action_visualization' in group:
        action_vis = group['action_visualization']
        output_path = path_prefix+'_action.png'
        if not os.path.exists(dir_path+output_path):
            imageio.imwrite(dir_path + output_path, action_vis)
        output += f'<img src="{output_path}" height="256px"></td>'

    if 'visualization_dir' in group.attrs and step == 0 and include_videos:
        output += '<td style="display: flex; flex-direction: row;" >'
        vis_dir_path = group.attrs['visualization_dir']
        for video_path in Path(vis_dir_path).glob('*.mp4'):
            video_path = str(video_path)
            video_path = video_path.split('/')[-2] \
                + '/' + video_path.split('/')[-1]
            output += """
            <video height=256px autoplay loop controls muted>
                <source src="{}" type="video/mp4">
            </video>
            """.format(video_path)
    else:
        output += f'<td>Step {step}'
    if 'last' in key:
        message = "No Errors"
        if ('failed_grasp' in group.attrs and
                group.attrs['failed_grasp']):
            message = "Failed Grasp"
        elif ('cloth_stuck' in group.attrs and
                group.attrs['cloth_stuck']):
            message = "Cloth Stuck"
        elif ('timed_out' in group.attrs and
                group.attrs['timed_out']):
            message = "Timed out"
        output += f':{message}'
    output += '</td><td>'

    output_path = path_prefix + '.png'
    if not os.path.exists(dir_path+output_path):
        plot_before_after(group,
                          output_path=dir_path + output_path,
                          fontsize=fontsize)
    output += f'<img src="{output_path}" height="256px"> </td>'
    if 'faces' in group and 'gripper_states' in group and 'states' in group:
        output_pkl = {
            'faces': np.array(group['faces']),
            'gripper_states': [],
            'states': [],
        }
        for k in group['gripper_states']:
            output_pkl['gripper_states'].append(
                np.array(group['gripper_states'][k]))
        for k in group['states']:
            output_pkl['states'].append(np.array(group['states'][k]))
        output_path = dir_path + path_prefix + '.pkl'
        pickle.dump(output_pkl, open(output_path, 'wb'))
        output += f'<td> {output_path} </td>'
    return output


def add_text_to_image(image, text,
                      color='rgb(255, 255, 255)', fontsize=12):
    image = Image.fromarray(image)
    ImageDraw.Draw(image).text(
        (0, 0), text,
        fill=color,
        font=ImageFont.truetype(
            "/usr/share/fonts/truetype/lato/Lato-Black.ttf", fontsize))
    return np.array(image)


def preprocess_obs(rgb, d):
    preprocessed_obs =  cat((tensor(rgb).float()/255,
                tensor(d).unsqueeze(dim=2).float()),
               dim=2).permute(2, 0, 1)
    return preprocessed_obs


def get_largest_component(arr):
    # label connected components for mask
    labeled_arr, num_components = \
        morph.label(
            arr, return_num=True,
            background=0)
    masks = [(i, (labeled_arr == i).astype(np.uint8))
             for i in range(0, num_components)]
    masks.append((
        len(masks),
        1-(np.sum(mask for i, mask in masks) != 0)))
    sorted_volumes = sorted(
        masks, key=lambda item: np.count_nonzero(item[1]),
        reverse=True)
    for i, mask in sorted_volumes:
        if arr[mask == 1].sum() == 0:
            continue
        return mask



def step_env(all_envs, ready_envs, ready_actions, remaining_observations):
    remaining_observations.extend([e.step.remote(a)
                                   for e, a in zip(ready_envs, ready_actions)])
    step_retval = []
    start = time()
    total_time = 0
    while True:
        ready, remaining_observations = ray.wait(
            remaining_observations, num_returns=1)
        if len(ready) == 0:
            continue
        step_retval.extend(ready)
        total_time = time() - start
        if (total_time > 0.01 and len(step_retval) > 0)\
                or len(step_retval) == len(all_envs):
            break

    observations = []
    ready_envs = []

    for obs, env_id in ray.get(step_retval):
        observations.append(obs)
        ready_envs.append(env_id['val'])

    return ready_envs, observations, remaining_observations


def shift_tensor(tensor, offset):
    new_tensor = torch.zeros_like(tensor).bool()
    #shifted up
    if offset > 0:
        new_tensor[:, :-offset, :] = tensor[:, offset:, :]
    #shifted down
    elif offset < 0:
        offset *= -1
        new_tensor[:, offset:, :] = tensor[:, :-offset, :]
    return new_tensor

def generate_workspace_mask(left_mask, right_mask, action_primitives, pix_place_dist, pix_grasp_dist):
                                
    workspace_masks = {}
    for primitive in action_primitives:
        if primitive == 'place':

            lowered_left_primitive_mask = shift_tensor(left_mask, -pix_place_dist)
            lowered_right_primitive_mask = shift_tensor(right_mask, -pix_place_dist)
            #WORKSPACE CONSTRAINTS (ensures that both the pickpoint and the place points are located within the workspace)
            left_primitive_mask = torch.logical_and(left_mask, lowered_left_primitive_mask)
            right_primitive_mask = torch.logical_and(right_mask, lowered_right_primitive_mask)
            primitive_workspace_mask = torch.logical_or(left_primitive_mask, right_primitive_mask)

        elif primitive == 'fling' or primitive == 'drag' or primitive == 'stretchdrag':

            raised_left_primitive_mask = shift_tensor(left_mask, pix_grasp_dist)
            lowered_left_primitive_mask = shift_tensor(left_mask, -pix_grasp_dist)
            raised_right_primitive_mask = shift_tensor(right_mask, pix_grasp_dist)
            lowered_right_primitive_mask = shift_tensor(right_mask, -pix_grasp_dist)
            #WORKSPACE CONSTRAINTS
            aligned_workspace_mask = torch.logical_and(raised_left_primitive_mask, lowered_right_primitive_mask)
            opposite_workspace_mask = torch.logical_and(raised_right_primitive_mask, lowered_left_primitive_mask)
            primitive_workspace_mask = torch.logical_or(aligned_workspace_mask, opposite_workspace_mask)
        
        workspace_masks[primitive] = primitive_workspace_mask

    return workspace_masks 

def generate_primitive_cloth_mask(cloth_mask, action_primitives, pix_place_dist, pix_grasp_dist):
    cloth_masks = {}
    for primitive in action_primitives:
        if primitive == 'place':
            primitive_cloth_mask = cloth_mask
        elif primitive == 'fling' or primitive == 'drag' or primitive == 'stretchdrag':
            #CLOTH MASK (both pickers grasp the cloth)
            raised_primitive_cloth_mask = shift_tensor(cloth_mask, pix_grasp_dist)
            lowered_primitive_cloth_mask = shift_tensor(cloth_mask, -pix_grasp_dist)
            primitive_cloth_mask = torch.logical_and(raised_primitive_cloth_mask, lowered_primitive_cloth_mask)
        else:
            raise NotImplementedError
        cloth_masks[primitive] = primitive_cloth_mask
    return cloth_masks

def shirt_keypoints(mask):

    canvas = np.zeros_like(mask)
    positive_coords = np.argwhere(mask)
    mean_pix = np.mean(positive_coords, axis=0).astype(int)

    positive_coords = positive_coords - mean_pix

    top_right_idx = np.argmax((positive_coords).sum(axis=1))
    top_right_coord = positive_coords[top_right_idx]

    top_left_idx = np.argmax((positive_coords * np.array([-1, 1]).reshape(1, 2)).sum(axis=1))
    top_left_coord = positive_coords[top_left_idx]

    bottom_right_idx = np.argmax((positive_coords * np.array([1, -2])).sum(axis=1))
    bottom_right_coord = positive_coords[bottom_right_idx]

    bottom_left_idx = np.argmax((positive_coords * np.array([-1, -2]).reshape(1, 2)).sum(axis=1))
    bottom_left_coord = positive_coords[bottom_left_idx]

    left_to_right_vector = top_left_coord - top_right_coord

    rotation = np.pi - np.arctan2(left_to_right_vector[1], left_to_right_vector[0])
    rotation_matrix = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
    inverse_rotation_matrix = np.array([[np.cos(-rotation), -np.sin(-rotation)], [np.sin(-rotation), np.cos(-rotation)]])

    rotated_positive_coords = (positive_coords @ rotation_matrix.T).astype(int) 
    rotated_bottom_left_coord = (bottom_left_coord @ rotation_matrix.T)
    rotated_bottom_right_coord = (bottom_right_coord @ rotation_matrix.T)

    coordinates_on_bottom_right_axis = rotated_positive_coords[rotated_positive_coords[:, 0] == int(rotated_bottom_right_coord[0])]
    coordinates_on_bottom_left_axis = rotated_positive_coords[rotated_positive_coords[:, 0] == int(rotated_bottom_left_coord[0])]

    rotated_right_shoulder_coord_idx = coordinates_on_bottom_right_axis[:, 1].argmax()
    rotated_right_shoulder_coord = coordinates_on_bottom_right_axis[rotated_right_shoulder_coord_idx, :]
    right_shoulder_coord = (rotated_right_shoulder_coord @ inverse_rotation_matrix.T)

    rotated_left_shoulder_coord_idx = coordinates_on_bottom_left_axis[:, 1].argmax()
    rotated_left_shoulder_coord = coordinates_on_bottom_left_axis[rotated_left_shoulder_coord_idx, :]
    left_shoulder_coord = (rotated_left_shoulder_coord @ inverse_rotation_matrix.T)

    top_left_coord += mean_pix
    top_right_coord += mean_pix
    bottom_left_coord += mean_pix
    bottom_right_coord += mean_pix

    right_shoulder_coord += mean_pix
    left_shoulder_coord += mean_pix

    top_left_coord = top_left_coord.astype(int)
    top_right_coord = top_right_coord.astype(int)
    bottom_left_coord = bottom_left_coord.astype(int)
    bottom_right_coord = bottom_right_coord.astype(int)
    
    right_shoulder_coord = right_shoulder_coord.astype(int)
    left_shoulder_coord = left_shoulder_coord.astype(int)

    # fig, axs = plt.subplots(1, 2)

    # canvas[top_left_coord[0]-4:top_left_coord[0]+4, top_left_coord[1]-4:top_left_coord[1]+4] = 1
    # canvas[top_right_coord[0]-4:top_right_coord[0]+4, top_right_coord[1]-4:top_right_coord[1]+4] = 1
    # canvas[bottom_left_coord[0]-4:bottom_left_coord[0]+4, bottom_left_coord[1]-4:bottom_left_coord[1]+4] = 1
    # canvas[bottom_right_coord[0]-4:bottom_right_coord[0]+4, bottom_right_coord[1]-4:bottom_right_coord[1]+4] = 1
    # canvas[right_shoulder_coord[0]-4:right_shoulder_coord[0]+4, right_shoulder_coord[1]-4:right_shoulder_coord[1]+4] = 1
    # canvas[left_shoulder_coord[0]-4:left_shoulder_coord[0]+4, left_shoulder_coord[1]-4:left_shoulder_coord[1]+4] = 1

    # axs[0].imshow(mask)
    # axs[1].imshow(canvas)
    # plt.show()
    # plt.close()

    return {
        'top_left': top_left_coord,
        'top_right': top_right_coord,
        'bottom_left': bottom_left_coord,
        'bottom_right': bottom_right_coord,
        'right_shoulder': right_shoulder_coord,
        'left_shoulder': left_shoulder_coord
    }


def shirt_folding_heuristic(keypoint_positions):

    top_midpoint = (keypoint_positions['top_right'] + keypoint_positions['top_left'])/2
    bottom_midpoint = (keypoint_positions['bottom_right'] + keypoint_positions['bottom_left'])/2

    alpha = 0.9
    bottom_right_quarter_point = alpha * bottom_midpoint + (1-alpha) * keypoint_positions['bottom_right']
    bottom_left_quarter_point = alpha * bottom_midpoint + (1-alpha) * keypoint_positions['bottom_left']

    right_midpoint = (keypoint_positions['right_shoulder'] + keypoint_positions['bottom_right'])/2 
    left_midpoint = (keypoint_positions['left_shoulder'] + keypoint_positions['bottom_left'])/2 

    right_arm_length = np.linalg.norm(keypoint_positions['right_shoulder'] - keypoint_positions['top_right'])
    left_arm_length = np.linalg.norm(keypoint_positions['left_shoulder'] - keypoint_positions['top_left'])

    arm_length = max([right_arm_length, left_arm_length])

    right_shoulder_to_arm_fold = (bottom_right_quarter_point - keypoint_positions['right_shoulder'])
    right_arm_place_point = keypoint_positions['right_shoulder'] + (right_shoulder_to_arm_fold/(np.linalg.norm(right_shoulder_to_arm_fold)+ 1e-6 )) * arm_length

    left_shoulder_to_arm_fold = (bottom_left_quarter_point - keypoint_positions['left_shoulder'])
    left_arm_place_point = keypoint_positions['left_shoulder'] + (left_shoulder_to_arm_fold/(np.linalg.norm(left_shoulder_to_arm_fold)+ 1e-6 )) * arm_length

    right_double_pick = keypoint_positions['bottom_right'] + (keypoint_positions['right_shoulder'] - keypoint_positions['bottom_right']) * 0.95
    left_double_pick = keypoint_positions['bottom_left'] + (keypoint_positions['left_shoulder'] -  keypoint_positions['bottom_left']) * 0.95

    right_axis_gap = keypoint_positions['right_shoulder'] - keypoint_positions['bottom_right']
    left_axis_gap = keypoint_positions['left_shoulder'] - keypoint_positions['bottom_left']

    # avg_axis_gap

    right_double_place = keypoint_positions['bottom_right'] + left_axis_gap * 0.2
    left_double_place = keypoint_positions['bottom_left'] + right_axis_gap * 0.2

    
    is_right_arm_folded = np.cross(top_midpoint - bottom_midpoint, keypoint_positions['top_right'] - keypoint_positions['right_shoulder'])[1] > 0
    is_left_arm_folded = np.cross(top_midpoint - bottom_midpoint, keypoint_positions['top_left'] - keypoint_positions['left_shoulder'])[1] < 0
    #If the right arm is really on the right
    
    actions = []

    actions.append([{"pick":keypoint_positions['top_right'], "place":right_arm_place_point}])
    actions.append([{"pick":keypoint_positions['top_left'], "place":left_arm_place_point}])
    actions.append( [{"place":left_double_place, "pick":left_double_pick},
                  {"place":right_double_place, "pick":right_double_pick}])

    return actions