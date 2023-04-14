

#%%
import os
os.chdir('/local/crv/acanberk/folding-unfolding/src')

import torch
from torchvision import transforms
import h5py
from tqdm import tqdm
import torchvision
import numpy as np
from typing import Dict, Tuple, Optional, OrderedDict
import os
import pathlib
import copy
from tqdm import tqdm
import sys
from itertools import product
import imutils
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from learning.nets import *

# dataset_path = None
# dataset_path = '786be-regression-single/latest_ckpt_eval_11/'
# dataset_path = '786be-regression-single/latest_ckpt_eval_2/'

# dataset_paths = ['786be-regression-single/latest_ckpt_eval_11/',
#                  '786be-regression-single/latest_ckpt_eval_2/',
#                  'adaptive_exp/pretrained/latest_ckpt_eval_0/',
#                  'adaptive_exp/pretrained_all/latest_ckpt_eval_0/'
#                  ]

dataset_path = '/local/crv/acanberk/folding-unfolding/src/final_experiments/all_distribution/'

if sys.argv[1] is not None and 'ip' not in sys.argv[1]:
    dataset_path = sys.argv[1]

stats = {
    'preaction_deformable_distance': [],
    'postaction_deformable_distance': [],
    'preaction_rigid_distance': [],
    'postaction_rigid_distance': [],
    'preaction_success': [],
    'postaction_success': [],
    'postaction_pointwise_distance':[],
    'preaction_pointwise_distance':[],
    'dataset_path': [],
    'task_id': [],
    'instance_id': [],
    'episode': [],
    'step':[],
    'key': [],
    'preaction_old_success': [],
    'postaction_old_success': [],
    'deformable_weight': [],
}

instances = OrderedDict()
tasks = OrderedDict()
# print(dataset_path)

def old_success_condition(deformable_distance, rigid_distance, coeffs=(1/0.16, 1/0.2)):
    return (coeffs[1] * deformable_distance + coeffs[0] * rigid_distance < 1) and (deformable_distance < 0.15) and (rigid_distance < 0.18)

def success_condition(deformable_distance, rigid_distance):
    return (deformable_distance < 0.10) and (rigid_distance < 0.10)

# %%

with h5py.File(dataset_path, "r") as dataset:
    for i, k in zip(range(10000), dataset):
        group = dataset[k]

        instance_key = (str(np.array(group['init_verts']).mean())[:4], str(np.array(group['init_verts']).std())[:4])
        task_key = (str(np.array(group.attrs['init_coverage']))[:4], str(np.array(group.attrs['init_direction']))[:4])

        if instance_key not in instances:
            instances[instance_key] = 0
        if task_key not in tasks:
            tasks[task_key] = 0

        instances[instance_key] += 1
        tasks[task_key] += 1

        instance_id = list(instances.keys()).index(instance_key)
        task_id = list(tasks.keys()).index(task_key)\

        episode = int(k.split("_")[0])
        step = int(k.split("_")[1][4:])

        stats['episode'].append(episode)
        stats['step'].append(step)
        stats['preaction_deformable_distance'].append(group.attrs['preaction_l2_distance'])
        stats['postaction_deformable_distance'].append(group.attrs['postaction_l2_distance'])
        stats['preaction_rigid_distance'].append(group.attrs['preaction_icp_distance'])
        stats['postaction_rigid_distance'].append(group.attrs['postaction_icp_distance'])
        stats['preaction_pointwise_distance'].append(group.attrs['preaction_pointwise_distance'])
        stats['postaction_pointwise_distance'].append(group.attrs['postaction_pointwise_distance'])
        stats['preaction_success'].append(success_condition(group.attrs['preaction_l2_distance'], group.attrs['preaction_icp_distance']))
        stats['postaction_success'].append(success_condition(group.attrs['postaction_l2_distance'], group.attrs['postaction_icp_distance']))
        stats['preaction_old_success'].append(old_success_condition(group.attrs['preaction_l2_distance'], group.attrs['preaction_icp_distance']))
        stats['postaction_old_success'].append(old_success_condition(group.attrs['postaction_l2_distance'], group.attrs['postaction_icp_distance']))
        stats['dataset_path'].append(dataset_path)
        stats['task_id'].append(task_id)
        stats['instance_id'].append(instance_id)
        stats['key'].append(k)
        stats['deformable_weight'].append(group.attrs['deformable_weight'])

# print("Number of instances", len(instances))
# print("Number of tasks", len(tasks))

df = pd.DataFrame(stats)
ep_df = df.groupby(['episode', 'dataset_path']).agg({'key':'first', \
                                                        'preaction_deformable_distance': 'first', \
                                                        'postaction_deformable_distance': 'last', \
                                                        'preaction_rigid_distance': 'first', \
                                                        'postaction_rigid_distance': 'last', \
                                                        'preaction_pointwise_distance': 'first', \
                                                        'postaction_pointwise_distance': 'last', \
                                                        'preaction_success': 'max', \
                                                        'postaction_success': 'max',
                                                        'preaction_old_success': 'max', \
                                                        'postaction_old_success': 'max',
                                                        'dataset_path': 'first',
                                                        'deformable_weight': 'first',})

ep_df['delta_deformable_distance'] = ep_df['postaction_deformable_distance'] - ep_df['preaction_deformable_distance']
ep_df['delta_rigid_distance'] = ep_df['postaction_rigid_distance'] - ep_df['preaction_rigid_distance']
ep_df['delta_pointwise_distance'] = ep_df['postaction_pointwise_distance'] - ep_df['preaction_pointwise_distance']

print("Number of episodes", len(ep_df))
print("Success rate", ep_df['postaction_success'].mean(), ep_df['postaction_success'].std()/np.sqrt(len(ep_df)))
print("Old success rate", ep_df['postaction_old_success'].mean(), ep_df['postaction_old_success'].std()/np.sqrt(len(ep_df)))
print("Mean episode delta deformable distance", ep_df['postaction_deformable_distance'].mean(), ep_df['postaction_deformable_distance'].std()/np.sqrt(len(ep_df)))
print("Mean episode delta rigid distance", ep_df['postaction_rigid_distance'].mean(), ep_df['postaction_rigid_distance'].std()/np.sqrt(len(ep_df)))
print("Deformable weight", ep_df['deformable_weight'].mean(), ep_df['deformable_weight'].std()/np.sqrt(len(ep_df)))
# ep_df.head()


# # %%
# sns.lineplot(df.drop_duplicates(subset='episode', keep='first')['cloth_size_bin'], df.drop_duplicates(subset='episode', keep='first')['episode_success'])

# %%

# print(dp)
# for x in ['episode_success', 'delta_pointwise_distance']:
#     print(x)
#     print("\n\tMean", df[x].mean(), "\n\tStdErr", (df[x].std()/np.sqrt(n)), "\n\tStdev", df[x].std())

# for primitive in 'fling', 'place':
#     for distance in 'rigid', 'deformable':
#         d = df[df.primitive == primitive]["preaction_" + distance] - df[df.primitive == primitive]["postaction_" + distance]
#         # d.head()
#         print(primitive, distance, "\n\tMean", d.mean(), "\n\tStdErr", (d.std()/np.sqrt(n)), "\n\tStdev", d.std())



# %%

# %%
