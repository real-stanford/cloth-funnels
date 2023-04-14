import torch
from torchvision import transforms
import h5py
from tqdm import tqdm
import torchvision

from typing import Dict, Tuple, Optional
import os
import pathlib
import copy

import imutils
import igl
import numpy as np
import pandas as pd
import zarr
import torch
import math
import torch.nn.functional as F
# from torch_geometric.data import Dataset, Data, DataLoader
# from torch.utils.data import Subset
from scipy.spatial.transform import Rotation
import pytorch_lightning as pl
from torch.utils.data import Subset
import open3d as o3d
from functools import lru_cache
import ray
import time
from copy import deepcopy


DELTA_WEIGHTED_REWARDS_MEAN = -0.0018245290312917787
DELTA_WEIGHTED_REWARDS_STD = 0.072
DELTA_L2_STD = 0.019922712535836946
DELTA_POINTWISE_REWARDS_STD = 0.12881897698788683

def rewards_from_group(group):

    deformable_weight = group.attrs["deformable_weight"]

    delta_l2_distance = group.attrs['preaction_l2_distance'] - group.attrs['postaction_l2_distance']
    delta_l2_distance /= DELTA_WEIGHTED_REWARDS_STD
    deformable_reward = delta_l2_distance

    delta_icp_distance = group.attrs['preaction_icp_distance'] - group.attrs['postaction_icp_distance']
    delta_icp_distance /= DELTA_WEIGHTED_REWARDS_STD
    rigid_reward = delta_icp_distance

    delta_pointwise_distance = group.attrs['preaction_pointwise_distance'] - group.attrs['postaction_pointwise_distance']
    delta_pointwise_distance /= DELTA_POINTWISE_REWARDS_STD
    l2_reward = delta_pointwise_distance

    weighted_reward = deformable_weight * deformable_reward + (1-deformable_weight) * rigid_reward

    preaction_coverage = group.attrs['postaction_coverage'] - group.attrs['preaction_coverage']

    return {'weighted':torch.tensor(weighted_reward).float(), \
            'deformable': torch.tensor(deformable_reward).float(), \
            'rigid': torch.tensor(rigid_reward).float(),        \
            'l2':torch.tensor(l2_reward).float(),
            'coverage': torch.tensor(preaction_coverage).float()}
    

def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

def superimpose(current_verts, goal_verts, indices=None, symmetric_goal=False):

    current_verts = current_verts.copy()
    goal_verts = goal_verts.copy()
    # flipped_goal_verts = goal_verts.copy()
    # flipped_goal_verts[:, 0] = 2*np.mean(flipped_goal_verts[:, 0]) - flipped_goal_verts[:, 0]

    if indices is not None:
        R, t = rigid_transform_3D(current_verts[indices].T, goal_verts[indices].T)
    else:
        R, t = rigid_transform_3D(current_verts.T, goal_verts.T)

    icp_verts = (R @ current_verts.T + t).T

    return icp_verts


def deformable_distance(goal_verts, current_verts, max_coverage, deformable_weight=0.65, flip_x=True, icp_steps=1000, scale=None):

    goal_verts = goal_verts.copy()
    current_verts = current_verts.copy()

    #flatten goals
    goal_verts[:, 1] = 0
    current_verts[:, 1] = 0
    flipped_goal_verts = goal_verts.copy()
    flipped_goal_verts[:, 0] =  -1 * flipped_goal_verts[:, 0]

    real_l2_distance = np.mean(np.linalg.norm(goal_verts - current_verts, axis=1))
    real_l2_distance_flipped = np.mean(np.linalg.norm(flipped_goal_verts - current_verts, axis=1))
    if real_l2_distance_flipped < real_l2_distance:
        real_l2_distance = real_l2_distance_flipped


    #GOAL is RED
    goal_vert_cloud = o3d.geometry.PointCloud()
    goal_vert_cloud.points = o3d.utility.Vector3dVector(goal_verts.copy())
    goal_vert_cloud.paint_uniform_color([1, 0, 0])

    normal_init_vert_cloud = deepcopy(goal_vert_cloud)

    flipped_goal_vert_cloud = o3d.geometry.PointCloud()
    flipped_goal_vert_cloud.points = o3d.utility.Vector3dVector(flipped_goal_verts.copy())
    flipped_goal_vert_cloud.paint_uniform_color([0, 1, 1])

    goal_vert_cloud += flipped_goal_vert_cloud
    #CURRENT is GREEN
    verts_cloud = o3d.geometry.PointCloud()
    verts_cloud.points = o3d.utility.Vector3dVector(current_verts.copy())
    verts_cloud.paint_uniform_color([0, 1, 0])

    THRESHOLD_COEFF = 0.3
    threshold = np.sqrt(max_coverage) * THRESHOLD_COEFF
    #superimpose current to goal
    icp_verts = superimpose(current_verts, goal_verts)
    for i in range(5):
        threshold = THRESHOLD_COEFF * np.sqrt(max_coverage)
        indices = np.linalg.norm(icp_verts - goal_verts, axis=1) < threshold
        icp_verts = superimpose(icp_verts, goal_verts, indices=indices)

    #superimpose reverse goal to current
    reverse_goal_verts = goal_verts.copy()
    R, t = rigid_transform_3D(reverse_goal_verts.T, current_verts.T)
    reverse_goal_verts = (R @ reverse_goal_verts.T + t).T
    indices = np.linalg.norm(reverse_goal_verts - current_verts, axis=1) < threshold
    reverse_goal_verts = superimpose(reverse_goal_verts, current_verts, indices=indices)

    reverse_goal_cloud = o3d.geometry.PointCloud()
    reverse_goal_cloud.points = o3d.utility.Vector3dVector(reverse_goal_verts.copy())
    reverse_goal_cloud.paint_uniform_color([1, 0, 1])

    icp_verts_cloud = o3d.geometry.PointCloud()
    icp_verts_cloud.points = o3d.utility.Vector3dVector(icp_verts.copy())
    icp_verts_cloud.paint_uniform_color([0, 0, 1])

    l2_regular = np.mean(np.linalg.norm(icp_verts - goal_verts, axis=1))
    l2_flipped = np.mean(np.linalg.norm(icp_verts - flipped_goal_verts, axis=1))
    l2_distance = min(l2_regular, l2_flipped)

    icp_distance_regular = np.mean(np.linalg.norm(goal_verts - reverse_goal_verts, axis=1))
    icp_distance_flipped = np.mean(np.linalg.norm(flipped_goal_verts - reverse_goal_verts, axis=1))
    icp_distance = min(icp_distance_regular, icp_distance_flipped)

    #make reward scale invariant
    assert(max_coverage != 0 or scale != 0)
    if scale is None:
        l2_distance /= np.sqrt(max_coverage)
        icp_distance /= np.sqrt(max_coverage)
        real_l2_distance /= np.sqrt(max_coverage)
    else:
        l2_distance /= scale
        icp_distance /= scale
        real_l2_distance /= scale

    weighted_distance = deformable_weight * l2_distance + (1 - deformable_weight) * icp_distance

    return weighted_distance, l2_distance, icp_distance, real_l2_distance, {"init_vert_cloud": goal_vert_cloud, "normal_init_vert_cloud": normal_init_vert_cloud , "verts_cloud": verts_cloud, 'icp_verts_cloud': icp_verts_cloud, "reverse_init_verts_cloud": reverse_goal_cloud}


def generate_positional_encoding(scale, dim, type="linear"):
    assert(type == "linear" or type == "cosine" or type == "square" or type == "constant")
    x, y = np.meshgrid(np.linspace(-scale, scale, 128),np.linspace(-scale, scale, 128))
    if type == "linear":
        map = np.sqrt(x**2 + y**2)
    if type == "square":
        NORMALIZER = 3
        map = (x**2 + y**2)/NORMALIZER
    if type == "cosine":
        MULTIPLER = 10 * np.pi
        map = np.cos(MULTIPLER * np.sqrt(x**2 + y**2))
    if type == "constant":
        map = np.zeros((dim, dim))
    return map


class GraspDataset(torch.utils.data.Dataset):
    def __init__(self,
                 hdf5_path: str,
                 num_rotations: int,
                 scale_factors=[1.0, 1.5, 2.0, 2.5, 3.0],
                 check_validity=False,
                 filter_fn=None,
                 obs_color_jitter=True,
                 use_normalized_coverage=True,
                 replay_buffer_size=2000,
                 fixed_replay_buffer=False,
                 positional_encoding=None,
                 reward_type=None,
                 action_primitives=None,
                 episode_length=None,
                 gamma=0.0,
                 **kwargs):

        self.hdf5_path = hdf5_path
        self.filter_fn = filter_fn
        self.use_normalized_coverage = use_normalized_coverage
        self.rgb_transform = transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1,
                    saturation=0.2, hue=0.5),
                transforms.RandomAdjustSharpness(1.1, p=0.25),
                ])\
            if obs_color_jitter else lambda x: x

        self.replay_buffer_size = replay_buffer_size
        self.action_primitives = action_primitives
        self.episode_length = episode_length
        self.supervised_training = kwargs['pretrain_dataset_path'] is not None
        self.gamma = gamma

        if check_validity:
            for k in tqdm(self.keys, desc='Checking validity'):
                self.check_validity(k)

        self.keys = self.get_keys()
        print("Number of keys:", len(self.keys))
        self.size = len(self.keys)

        self.num_rotations = num_rotations
        self.scale_factors = np.array(scale_factors)
        self.positional_encoding = positional_encoding

        self.reward_type = reward_type

        if not fixed_replay_buffer:
            self.replay_buffer_size = 100000

    def get_keys(self):
        # dataset_length = len(dataset)
        # replay_buffer_size = min(len(dataset), self.replay_buffer_size)
        with h5py.File(self.hdf5_path, "r") as dataset:
           
            if not self.supervised_training:
                min_index = len(dataset) - self.replay_buffer_size
            else:
                min_index = 0

            print("[Dataloader] min_index: ", min_index)
            keys = []
            for i, k in tqdm(enumerate(dataset)):
                if i < min_index:
                    continue
                attrs = dataset[k].attrs
                if self.filter_fn is None or self.filter_fn(attrs) and \
                    ('postaction_weighted_distance' in attrs):
                    keys.append(k)
   
            # print("keys from get_keys", keys, "hdf5 path", self.hdf5_path)
            return keys

    def check_validity(self, key):
        with h5py.File(self.hdf5_path, "a") as dataset:
            group = dataset.get(key)
            if 'actions' not in group or 'observations' not in group \
                or 'postaction_coverage' not in group.attrs:
                del dataset[key]
                return

    def __len__(self):
        return len(self.keys)

    # @profile
    def __getitem__(self, index):
        with h5py.File(self.hdf5_path, "r") as dataset:
            group = dataset.get(self.keys[index])
            # primitive_id = self.action_primitives.index(group.attrs['action_primitive'])
            
            #  = reward_function_from_group(group, setting=self.reward_type)
            rewards_dict = rewards_from_group(group)
            weighted_reward, deformable_reward, rigid_reward, l2_reward = \
                rewards_dict['weighted'], rewards_dict['deformable'], rewards_dict['rigid'], rewards_dict['l2']
            coverage_reward = rewards_dict['coverage']
            obs = torch.tensor(np.array(group['observations']))
            # obs[:3, :, :] = self.rgb_transform(obs[:3, :, :])

            action = torch.tensor(np.array(group['actions'])).bool()

            key = self.keys[index]
            episode = int(key.split("_")[0])
            step = int(key.split("_")[1][4:])
          
            is_terminal = "last" in key

            retval = {
                'obs': obs,
                'action': action,
                'weighted_reward': weighted_reward,
                'deformable_reward': deformable_reward,
                'rigid_reward': rigid_reward,
                'l2_reward': l2_reward,
                'is_terminal': is_terminal,
                'coverage_reward': coverage_reward
            }

            for key, value in retval.items():
                if np.isnan(value).any() or np.isinf(value).any():
                    print("NaN or Inf detected in sample: ", key)
                    new_index = np.random.randint(0, len(self.keys))
                    return self.__getitem__(new_index)

            return retval

            # return obs, action, reward, next_obs, next_place_mask, next_fling_mask, deformable_reward, rigid_reward, l2_reward, torch.tensor(is_terminal).bool()
            


#takes in numpy array of shape (H, W, 4) and outputs tensor of shape (1, 4, H, W)
def rgbd_to_tensor(rgbd):
    out_tensor = torch.empty(
        rgbd.shape[2], rgbd.shape[0], rgbd.shape[1])
    out_tensor[:3, :, :] = torchvision.transforms.ToTensor()(rgbd[:, :, :3].astype(np.uint8))
    depth = rgbd[:, :, 3]
    out_tensor[3, :, :] = torch.tensor(
        (depth - depth.mean())/(depth.std() + 1e-8))
    out_tensor = torch.reshape(out_tensor, (1, out_tensor.shape[0], \
        out_tensor.shape[1], out_tensor.shape[2]))
    return out_tensor

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


def crop_center(img, crop):
    startx = img.shape[1]//2-(crop//2)
    starty = img.shape[0]//2-(crop//2)
    return img[starty:starty+crop, startx:startx+crop, ...]


def pad(img, size):
    n = (size-img.shape[0])//2
    return cv2.copyMakeBorder(img, n, n, n, n, cv2.BORDER_REPLICATE)

    
def generate_coordinate_map(dim, rotation, scale, normalize=False):

    MAX_SCALE=5
    scale = 1

    coordinate_dim = int(dim * (MAX_SCALE/scale))
    x, y = np.meshgrid(MAX_SCALE * np.linspace(-1 , 1, coordinate_dim), MAX_SCALE * np.linspace(-1 , 1, coordinate_dim), indexing="ij")
    x, y = x.reshape(coordinate_dim, coordinate_dim, 1), y.reshape(coordinate_dim, coordinate_dim, 1)
    xy = np.concatenate((x, y), axis=2)
    xy = imutils.rotate(xy, rotation)
    center = coordinate_dim/2

    new_dim = int(center + center * scale/MAX_SCALE) - int(center - center * scale/MAX_SCALE)
    offset = 0
    if int(new_dim) != dim:
        offset = int(dim - new_dim)

    xy = xy[int(center - center * scale/MAX_SCALE) : int(center + center * scale/MAX_SCALE) + offset, \
         int(center - center * scale/MAX_SCALE):int(center + center * scale/MAX_SCALE) + offset, :]

    return xy
    

# @profile
def transform(img, rotation: float, scale: float, dim: int, constant_positional_encoding: bool = False):


    #to adjust code
    rotation *= -1 

    img = transforms.functional.resize(img, (dim, dim))
    img = transforms.functional.rotate(img, rotation, interpolation=transforms.InterpolationMode.BILINEAR)
    
    if scale < 1:
        img = transforms.functional.center_crop(img, (int(dim * scale), int(dim * scale)))
        img = transforms.functional.resize(img, (dim, dim))
    else:
        zeros = torch.zeros((img.shape[0], dim, dim), device=img.device)
        img = transforms.functional.resize(img, (int(dim/scale), int(dim/scale)))

        end = zeros.shape[-1]//2 + int(img.shape[-1])//2
        begin = zeros.shape[-1]//2 - int(img.shape[-1])//2
        if end - begin < img.shape[-1]:
            end += 1        
        
        zeros[..., begin:end, begin:end] = img
        img = zeros

    coordinate_map = torch.tensor(generate_coordinate_map(dim, rotation, scale, normalize=constant_positional_encoding))
    coordinate_map = coordinate_map.permute(2, 0, 1)
    
    img = torch.cat([img, coordinate_map], axis=0)

    return img

# transform_async = ray.remote(transform)

@lru_cache
def generate_positional_encodings(transformations, dim, type="linear"):

    maps = []

    for scale, rotation in transformations:
        maps.append(torch.tensor(generate_positional_encoding(scale, dim, type)))
    maps = torch.stack(maps)
    # maps = maps.repeat(1, 1, 1)
    maps = torch.unsqueeze(maps, 1)
    return maps

def prepare_image(img, transformations, dim: int,
                  parallelize=False, log=False, orientation_net=None, nocs_mode=None, constant_positional_enc = False, inter_dim=256):

    assert nocs_mode == "collapsed" or nocs_mode == "distribution"

    if orientation_net is not None:

        mask = torch.sum(img[:3,], axis=0) > 0
        mask = torch.unsqueeze(mask, 0)

        #resize to network input shape
        input_img = transforms.functional.resize(img, (128, 128))


        with torch.no_grad():
            prepped_img = torch.unsqueeze(input_img[:3, :, :], 0).cpu()
            #print the type of prepped img
            out = ray.get(orientation_net.forward.remote(prepped_img))[0]
            # out = orientation_net.forward(torch.unsqueeze(input_img[:3, :, :], 0))[0]

        nocs_x_bins = out[:, 0, :, :]
        nocs_y_bins = out[:, 1, :, :]
        n_bins = out.shape[0]

        #out shape: 32, 2, 128, 128
        if nocs_mode == "collapsed":
            # mask = torch.cat(2*[torch.unsqueeze(mask, 0)], dim=0)
            #32 bins
            nocs_x = torch.unsqueeze(torch.argmax(nocs_x_bins, dim=0).type(torch.float32)/(n_bins-1), 0)
            nocs_y = torch.unsqueeze(torch.argmax(nocs_y_bins, dim=0).type(torch.float32)/(n_bins-1), 0)
            #mask out bg
            nocs = torch.cat([nocs_x, nocs_y], dim=0)

        elif nocs_mode == "distribution":
            # mask = torch.cat((n_bins * 2)*[torch.unsqueeze(mask, 0)], dim=0)

            nocs_x = torch.nn.functional.softmax(nocs_x_bins, dim=0)
            nocs_y = torch.nn.functional.softmax(nocs_y_bins, dim=0)

            nocs = torch.cat([nocs_x, nocs_y], dim=0)

            nocs = nocs[::2] + nocs[1::2]
        else:
            raise NotImplementedError 

        #to make things more computationally tractable
        # print("NOCS shape", nocs.shape)
        nocs = transforms.functional.resize(nocs, (img.shape[-1], img.shape[-2])).to(img.device)
        nocs = nocs * mask.int() + (1 - mask.int()) * 0.0


        img = torch.cat([img, nocs], dim=0)

    log = False
    if log:
        start = time()
        # print('Preparing images')
    # if parallelize:
    #     imgs = ray.get([transform_async.remote(img, *t, dim=dim)
    #                     for t in transformations])
    # else:
    img = img.cpu()
    img = transforms.functional.resize(img, (inter_dim, inter_dim))
    imgs = torch.stack([transform(img, *t, dim=dim, constant_positional_encoding=constant_positional_enc) for t in transformations])

    if log:
        print(f'\r prepare_image took {float(time()-start):.02f}s with parallelization {parallelize}')

    return imgs.float()
