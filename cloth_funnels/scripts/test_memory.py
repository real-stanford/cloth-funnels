from argparse import ArgumentParser
from environment import SimEnv, TaskLoader
from learning.nets import MaximumValuePolicy, ExpertGraspPolicy
from learning.utils import GraspDataset, rewards_from_group
from environment.utils import plot_before_after
from torch.utils.data import DataLoader
from filelock import FileLock
from time import time
import torch
import h5py
import os
import ray
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import torch.nn as nn
from nocs_model.network.orientation_deeplab import OrientationDeeplab
from utils import setup_network, config_parser, OrientationNet
from memory_profiler import profile

fp=open('memory_profiler.log','w+')

@profile(stream=fp)
def main():
    args = config_parser().parse_args()

    policy, orn_net_handle, optimizer, dataset_path = setup_network(args)

    # orn_net_handle = OrientationNet(orientation_network_path=args.orientation_network_path,
    #                                 input_channel_types=args.input_channel_types,
    
    # )

    task_loader = TaskLoader(
        hdf5_path=args.tasks,
        eval_hdf5_path=args.eval_tasks,
        repeat=not args.eval)


    obs = [torch.randn(80, 71, 128, 128)]

    env = SimEnv(
            replay_buffer_path=dataset_path,
            get_task_fn=lambda: task_loader.get_next_task(),
            orn_net_handle=None,
            **vars(args))
        
    env.reset()
    for _ in range(10):
        print("stepping...")
        env.step(list(policy.act(obs))[0])

if __name__ == '__main__':
    main()