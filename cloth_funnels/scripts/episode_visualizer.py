#%%

import copy
import datetime
import os
os.chdir('/local/crv/acanberk/cloth-funnels/cloth_funnels')
import pathlib
import pickle as pkl
import sys
from itertools import product
from typing import Dict, Optional, OrderedDict, Tuple

import cv2
import h5py
import imutils
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm

import open3d as o3d
from cloth_funnels.learning.utils import deformable_distance
from cloth_funnels.pc_vis import *
import pandas as pd

from skimage import draw
from cloth_funnels.notebooks.visualization_utils import *
from filelock import FileLock

def get_edges(im):
    im = np.ascontiguousarray(im.copy() * 255).astype(np.uint8) 
    contours, hierarchy = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    out = np.zeros((480, 480)).astype(np.uint8)
    out = cv2.drawContours(out, contours, -1, 255, 3)
    return np.array(out).astype(float)/255

def pick_nth_step(steps, n):
    num_steps = len(steps)
    if n >= num_steps:
        return steps.iloc[-1]
    return steps.iloc[n]

def visualize_episode(k, dataset_path, steps=8, 
                        vis_index=None, 
                        visualize_metrics=True, 
                        title=None,
                        custom_vmap_index=None, 
                        visualize_actions=True,
                        visualize_goal=True):
    # episode = episode[:100]
    with FileLock(dataset_path + ".lock"):
        with h5py.File(dataset_path, "r") as dataset:

            keys = get_episode_keys(k, steps)

            visualize_pointclouds = 4 in vis_index or 5 in vis_index

            if visualize_pointclouds:
                renderer = Renderer(480, 480)

            fig, axs = None, None
            original_observations = []
            original_value_maps = []

            for i, k in enumerate(keys):

                if k not in dataset:
                    continue

                group = dataset[k]
                init_verts, preaction_verts, postaction_verts = group['init_verts'], group['preaction_verts'], group['postaction_verts']
                preaction_weighted_distance, preaction_l2_distance, \
                            preaction_icp_cost, _, preaction_clouds = deformable_distance(np.array(init_verts), np.array(preaction_verts), group.attrs['max_coverage'])
                postaction_weighted_distance, postaction_l2_distance, \
                    postaction_icp_cost, _, postaction_clouds = deformable_distance(np.array(init_verts), np.array(postaction_verts), group.attrs['max_coverage'])

                init_vert_cloud = preaction_clouds['init_vert_cloud']
                preaction_verts_cloud = preaction_clouds['verts_cloud']
                postaction_verts_cloud = postaction_clouds['verts_cloud']
                icp_preaction_verts_cloud = preaction_clouds['icp_verts_cloud']
                icp_postaction_verts_cloud = postaction_clouds['icp_verts_cloud']

                preaction_reverse_init_verts_cloud = preaction_clouds['reverse_init_verts_cloud']
                postaction_reverse_init_verts_cloud = postaction_clouds['reverse_init_verts_cloud']

                pretransform_obs = np.array(group['pretransform_observations'])[:3, :, :].\
                    transpose(1, 2, 0)

                if visualize_goal:
                    binary_mask = np.array(group['preaction_init_mask'])
                    edges = np.stack(3 * [get_edges(binary_mask)]).transpose(1, 2, 0)
                    cloth_mask = np.stack(3 * [np.sum(pretransform_obs, axis=-1) > 0]).transpose(1, 2, 0)
                    pretransform_obs = pretransform_obs + ((edges * ~cloth_mask) * 0.5)


                workspace_mask = np.array(group[f'workspace_mask'])

                observations = np.array(group['observations']).transpose(1, 2, 0)
                rgb_obs = observations[:, :, :3]
                # if custom_vmap_index:
                #     rgb_obs = observations[custom_vmap_index, :, :, :3]

                direction_obs = observations[:, :, (-2, -1)]
                obs_mask = np.sum(rgb_obs, axis=-1) > 0

                action_vis = group['action_visualization']
                value_map = np.array(group['value_map'])
                if custom_vmap_index:
                    custom_index = custom_vmap_index[0]
                    custom_primitive = custom_vmap_index[1]
                    if f"{custom_primitive}_value_maps" in group:
                        value_map = np.array(group[f'{custom_primitive}_value_maps'])[custom_index]
                    else:
                        value_map = np.array(group[f'value_maps'])[custom_index]
                        
                masked_value_map = value_map
            
                action_coordinates = np.array(np.where(np.array(group['actions']) > 0))[:, 0]
                action_coordinates[1], action_coordinates[0] = action_coordinates[0], action_coordinates[1]
            
                rotation = group.attrs['rotation']
                scale = group.attrs['scale']

                dim_factor = 480/128

                circle_thickness = 1
                circle_radius = 3

                if visualize_pointclouds:
                    rigid_img = np.rot90(renderer.render_point_cloud(preaction_reverse_init_verts_cloud + init_vert_cloud, 
                            camera_loc=[1,5,0], gaze_loc=[0,0,0.05]), 1)
                    deformable_img = np.rot90(renderer.render_point_cloud(preaction_reverse_init_verts_cloud + preaction_verts_cloud,
                            camera_loc=[1,5,0], gaze_loc=[0,0,0.05]), 1)
                else:
                    rigid_img = None
                    deformable_img = None

                if 'last' not in k and visualize_actions:
                    if group.attrs['action_primitive'] == 'fling':
                        p1, p2 = action_coordinates + np.array([0, 16]), action_coordinates + np.array([0, -16])
                        rgb_obs = draw_fling(rgb_obs, p1, p2, thickness=circle_thickness, radius=circle_radius)
                        p1_original, p2_original = transform_coords(p1, rotation, scale, 128, 480), transform_coords(p2, rotation, scale, 128, 480)
                        pretransform_obs = draw_fling(pretransform_obs, p1_original, p2_original, thickness=int(circle_thickness * dim_factor), radius=int(circle_radius * dim_factor))
                    elif group.attrs['action_primitive'] == 'place':
                        p1, p2 = action_coordinates, action_coordinates + np.array([0, 10])
                        rgb_obs = draw_place(rgb_obs, p1, p2, thickness=circle_thickness, radius=circle_radius)
                        p1_original, p2_original = transform_coords(p1, rotation, scale, 128, 480), transform_coords(p2, rotation, scale, 128, 480)
                        pretransform_obs = draw_place(pretransform_obs, p1_original, p2_original, thickness=int(circle_thickness * dim_factor), radius=int(circle_radius * dim_factor))
                    
                if visualize_metrics:
                    pretransform_obs = draw_text(pretransform_obs, text=f'Rigid: {group.attrs["preaction_icp_distance"]:.3f}', org=(20, 45), font_scale=1)
                    pretransform_obs = draw_text(pretransform_obs, text=f'Deformable: {group.attrs["preaction_l2_distance"]:.3f}', org=(20, 70), font_scale=1)
                    pretransform_obs = draw_text(pretransform_obs, text=f'L2: {group.attrs["preaction_pointwise_distance"]:.3f}', org=(20, 95), font_scale=1)
                    pretransform_obs = draw_text(pretransform_obs, text=f'IoU: {group.attrs["preaction_iou"]:.3f}', org=(20, 130), font_scale=1)

                    
                vis_list = [(pretransform_obs, 'Original Observation', None), 
                            (rgb_obs, 'Network Input', None), 
                            (workspace_mask, None, None), 
                            (masked_value_map, 'Value Map', 'jet'), 
                            (rigid_img, None, None), 
                            (deformable_img, None, None)]

                if vis_index is None:
                    plot_indices = range(len(vis_list))
                else:
                    plot_indices = vis_index

                if fig is None:
                    fig, axs = plt.subplots(len(plot_indices), steps, figsize=(steps*4, len(plot_indices)*4))

                original_observations.append(pretransform_obs)
                original_value_maps.append(masked_value_map)

                for plot_idx, list_idx in enumerate(plot_indices):
                    # print(f"Plotting for {i}")
                    # if plot_idx == 0:
                    #     axs[plot_idx, i].text(s=f"t={i}", size=20, horizontalalignment='center', y=0, x=0)
                    # if i == 0:
                    #     axs[plot_idx, i].text(x=-20,y=0, s=vis_list[list_idx][1], size=12,
                    #             verticalalignment='center', rotation=90)
                    axs[plot_idx, i].imshow(vis_list[list_idx][0], cmap=vis_list[list_idx][2])
                    axs[plot_idx, i].axis('off')
                        
            #set all axes off

            if fig is not None:
                fig.tight_layout()
                plt.tight_layout()

            if title is not None:
                fig.suptitle(title)

                # plt.savefig(f"logs/log_images/{'_'.join(dataset_path.split('/'))}_{k}.png")
            if visualize_pointclouds:
                del renderer

    return fig, axs, original_observations, original_value_maps



# %%

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--steps', type=int, default=10)

    args = parser.parse_args()
    replay_buffer_path = os.path.join(args.input_path, 'replay_buffer.hdf5')

    print("[Visualize] Visualizing episodes from replay buffer: ", replay_buffer_path)

    step_df = pd.DataFrame()

    steps_dict = []
    with h5py.File(replay_buffer_path, 'r') as dataset:
        keys = list(dataset.keys())
        print(f'Evaluating {replay_buffer_path} with {len(keys)} keys')

        metrics = [
                    'deformable_distance',
                    'rigid_distance',
                    'weighted_distance',
                    'l2_distance',
                    'iou',
                    'coverage'
        ]

        stat_keys = [
            'episode_length',
            'out_of_frame',
            'nonadaptive_scale',
            'rotation',
            'scale',
            'percent_fling',
            'predicted_value',
            'predicted_deformable_value',
            'predicted_rigid_value',
            'deformable_weight',
        ]
        
        difficulties = ['easy', 'hard', 'none', 'flat', 'pick']
        rewards = ['deformable', 'rigid', 'weighted']

        step_df = []
        for k in tqdm(keys, desc='Reading keys...'):

            group_data = {}
            group = dataset.get(k)

            episode = int(k.split('_')[0])
            step = int(k.split('_')[1][4:])

            level = str(group.attrs['task_difficulty'])     
            
            if level != 'hard':
                continue

            group_data['episode'] = episode
            group_data['step'] = step
            group_data['level'] = level
            group_data['key'] = str(k)
            group_data['init_verts'] = np.sum(group['init_verts'])

            for key, value in group.attrs.items():
                group_data[key] = value

            step_df.append(group_data)

        
        step_df = pd.DataFrame(step_df)

    max_step = step_df['step'].max() + 1

    episode_df = pd.DataFrame()

    unique_episodes = step_df['episode'].unique()
    for episode in unique_episodes:
        #get nth step or the latest step
        picked_step = pick_nth_step(step_df[step_df.episode == episode], n=1)
        episode_df = episode_df.append(picked_step, ignore_index=True)

    os.makedirs(os.path.join(args.input_path, 'visualizations'), exist_ok=True)
    for idx, row in tqdm(episode_df.iterrows(), total=len(episode_df), desc='Visualizing episodes...'):
        key = row['key']
        fig, axs, _, _ = visualize_episode(key, replay_buffer_path, steps=max_step, vis_index=(0, 1, 2, 3))
        fig.savefig(os.path.join(args.input_path, 'visualizations', f'vis_{idx}.png'))
        plt.close(fig)



