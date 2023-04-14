from utils import (
    setup_envs,
    seed_all, setup_network, get_loader, 
    get_pretrain_loaders,
    get_dataset_size, collect_stats, get_img_from_fig,
    step_env, visualize_value_pred)
import ray
from time import time, strftime
from copy import copy
import torch
from tensorboardX import SummaryWriter
from filelock import FileLock
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb 
# from itertools import cycle
from nocs_model.network.orientation_deeplab import OrientationDeeplab
from learning.optimization import optimize_fb

from omegaconf import OmegaConf
import hydra
import yaml
import pathlib
import glob
import imageio.v3 as iio
import cv2
import shutil
import h5py
import sys
from notebooks.episode_visualizer import visualize_episode
from scipy.spatial.transform import Rotation as R


def step_fold_env(envs, ready_envs, ready_actions, remaining_observations):
    remaining_observations.extend([e.step_fold.remote(a)
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
                or len(step_retval) == len(envs):
            break

    observations = []
    ready_envs = []

    for obs, env_id in ray.get(step_retval):
        observations.append(obs)
        ready_envs.append(env_id['val'])

    return ready_envs, observations, remaining_observations



def get_keypoint_pos(index, positions):
    return np.array(positions[index]).copy()

def state_machine(observations):

    print("From state machine", np.sum(observations['all_points']))

    keypoints = observations['keypoint_indices']
    all_points = np.array(observations['all_points']).copy()

    keypoint_positions = {
        key: get_keypoint_pos(indices[0], all_points) for key, indices in keypoints.items()
    }

    top_midpoint = (keypoint_positions['top_right'] + keypoint_positions['top_left'])/2
    bottom_midpoint = (keypoint_positions['bottom_right'] + keypoint_positions['bottom_left'])/2

    alpha = 0.8
    bottom_right_quarter_point = alpha * bottom_midpoint + (1-alpha) * keypoint_positions['bottom_right']
    bottom_left_quarter_point = alpha * bottom_midpoint + (1-alpha) * keypoint_positions['bottom_left']

    right_midpoint = (keypoint_positions['right_shoulder'] + keypoint_positions['bottom_right'])/2 
    left_midpoint = (keypoint_positions['left_shoulder'] + keypoint_positions['bottom_left'])/2 

    arm_length = np.linalg.norm(keypoint_positions['right_shoulder'] - keypoint_positions['top_right']) * 1.3

    right_shoulder_to_arm_fold = (bottom_right_quarter_point - keypoint_positions['right_shoulder'])
    right_arm_place_point = keypoint_positions['right_shoulder'] + (right_shoulder_to_arm_fold/np.linalg.norm(right_shoulder_to_arm_fold))*arm_length

    left_shoulder_to_arm_fold = (bottom_left_quarter_point - keypoint_positions['left_shoulder'])
    left_arm_place_point = keypoint_positions['left_shoulder'] + (left_shoulder_to_arm_fold/np.linalg.norm(left_shoulder_to_arm_fold))*arm_length

    right_should_arm_double_fold = keypoint_positions['bottom_right'] + (keypoint_positions['right_shoulder'] - right_arm_place_point) * 0.8
    left_should_arm_double_fold = keypoint_positions['bottom_left'] + (keypoint_positions['left_shoulder'] - left_arm_place_point) * 0.8

    
    is_right_arm_folded = np.cross(top_midpoint - bottom_midpoint, keypoint_positions['top_right'] - keypoint_positions['right_shoulder'])[1] > 0
    is_left_arm_folded = np.cross(top_midpoint - bottom_midpoint, keypoint_positions['top_left'] - keypoint_positions['left_shoulder'])[1] < 0
    #If the right arm is really on the right
    
    action = []
    
    if not is_right_arm_folded:
        print("Folding right arm")
        return [{"pick":keypoint_positions['top_right'], "place":right_arm_place_point}]
    elif not is_left_arm_folded:
        print("Folding left arm")
        return [{"pick":keypoint_positions['top_left'], "place":left_arm_place_point}]
    elif is_right_arm_folded and is_left_arm_folded:
        print("Folding both arms")
        # return [{"pick":keypoint_positions['bottom_left'], "place":left_should_arm_double_fold},
        #           {"pick":keypoint_positions['bottom_right'], "place":right_should_arm_double_fold}]
        return [{"place":keypoint_positions['bottom_left'], "pick":keypoint_positions['left_shoulder']},
                  {"place":keypoint_positions['bottom_right'], "pick":keypoint_positions['right_shoulder']}]

    return None
        # print("left and right arm are folded")
        # action = [{"place":keypoint_positions['left_shoulder'], "pick":keypoint_positions['bottom_left']}, 
        #     {"place":keypoint_positions['right_shoulder'], "pick":keypoint_positions['bottom_right']},]
  
    #rotation around the z axis
    # if len(action) == 0:
    #     return None


if __name__ == '__main__':

    @hydra.main(version_base=None, config_path="conf", config_name="config")
    def main(args):
        ray.init(local_mode=args.ray_local_mode)
        seed_all(args.seed)

        dataset_path = args.log + '/replay_buffer.hdf5'
        log_name = args.log
        pathlib.Path(log_name).mkdir(parents=True, exist_ok=True)
        all_config = {
            'config': OmegaConf.to_container(args, resolve=True),
            'output_dir': os.getcwd(),
            }
        yaml.dump(all_config, open(f'./{log_name}/config.yaml', 'w'), default_flow_style=False)


        envs, _ = setup_envs(dataset=dataset_path, orn_net_handle=None , **args)

        while True:
            ray.get([e.reset.remote() for e in envs])
            observations = ray.get([e.fold.remote() for e in envs])

                                
    
    main()

