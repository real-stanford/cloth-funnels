

#%%
from re import A
import torch
from torchvision import transforms
import h5py
from tqdm import tqdm
import torchvision
import numpy as np
from typing import Dict, Tuple, Optional
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
from skimage import draw

os.chdir('/local/crv/acanberk/folding-unfolding/src')

from learning.nets import *

# dataset_path = None
dataset_path = 'overfit-place-random/replay_buffer.hdf5'
# print(dataset_path)

if dataset_path is None:
    dataset_path = sys.argv[1]


def prepare_image(img, transformations, dim: int,
                  parallelize=False, log=True):

    img = transforms.functional.resize(img, (256, 256)).cpu()
    imgs = [transform(img, *t, dim=dim) for t in transformations]
    retval = torch.stack(imgs).float()
    return retval

num_rotations = 16
rotations = [(2*i/num_rotations - 1) * \
                180 for i in range(num_rotations)]
original_scale_factors = [1.0, 1.5, 2.0, 2.5, 3.0]
num_scales = len(original_scale_factors)
pix_drag_dist = 16
pix_grasp_dist = 16
pix_place_dist = 10
obs_dim = 128
action_primitives = ['place']

ckpt = torch.load('/local/crv/acanberk/folding-unfolding/src/overfit-place-random-supervised-deformable/pretrain/16_ckpt.pth')
policy = MaximumValuePolicy(action_expl_prob=0, action_expl_decay=0, \
    value_expl_prob=0, value_expl_decay=0, num_rotations=num_rotations, pix_grasp_dist=pix_grasp_dist, pix_drag_dist=pix_drag_dist, \
        pix_place_dist = pix_place_dist, num_scales=num_scales, obs_dim=obs_dim, action_primitives=action_primitives, \
            scale_factors = original_scale_factors)
policy.load_state_dict(ckpt['net'])
val_net = policy.value_nets['place']


#%%

MAX_DATA_SIZE = 15
results = []
masks = []
observations = []
delta_reward = []
with h5py.File(dataset_path, "r") as dataset:
    for i, k in tqdm(zip(range(MAX_DATA_SIZE), dataset)):
        
        adaptive_scale = dataset[k].attrs["adaptive_scale"]
        scale_factors = [scale * adaptive_scale for scale in original_scale_factors]
        transformations = list(product(
                    rotations, scale_factors))
      
        obs = torch.tensor(dataset[k]["pretransform_observations"])

        transformed_obs = prepare_image(obs, transformations, dim=obs_dim)
        observations.append(transformed_obs.cpu().detach())

        obs_tensor = transformed_obs.clone().to(policy.device)
        mask = obs_tensor.cpu().clone().numpy()[:, (4,5,6), :, :].sum(axis=1) > 0
        masks.append(mask)

        vmaps = torch.zeros(obs_tensor.shape[0], 1, obs_tensor.shape[2], obs_tensor.shape[3])
        n_divide = 8
        for offset in range(0, obs_tensor.shape[0], obs_tensor.shape[0]//n_divide):
            out = val_net(obs_tensor[offset:offset + obs_tensor.shape[0]//n_divide])
            vmaps[offset:offset + obs_tensor.shape[0]//n_divide] = out

        results.append(vmaps.cpu().detach())
#%%

for i in range(0, MAX_DATA_SIZE):
    result = results[i]
    mask = masks[i]
    observation = observations[i]

    print("RESULT SHAPE", result.shape)
    print("MASK SHAPE", mask.shape)
    mask_tensor = torch.tensor(mask)
    mask_tensor = mask_tensor.unsqueeze(1).float()
    masked_result = result * mask_tensor + (1 - mask_tensor) * torch.tensor(0.0)
    masked_result = masked_result.detach()
    grid_img = torchvision.utils.make_grid(masked_result, nrow=5)[0, :, :]
    grid_img = grid_img.numpy()
    print("GRID SHAPE", grid_img.shape)


    max_value = np.stack(np.unravel_index(np.argsort(np.ravel(masked_result))[::-1], masked_result.shape)).T

    print("MAX INDEX", max_value[0])
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(grid_img, cmap='jet')
    ax.set_axis_off()
    fig.tight_layout()
    plt.title('Value Maps')
    plt.show()


    fig, axs = plt.subplots(num_rotations, num_scales*2, figsize=(2*num_scales, num_rotations))
    for i, ax in enumerate(axs.flat):
        if i % 2 == 0:

            vmap = result[i//2].cpu().detach().numpy()
            image_mask = mask[i//2]

            on_coordinates = np.where(image_mask == 1)
            max_x, min_x = on_coordinates[0].max(), on_coordinates[0].min()
            mid_x = (max_x + min_x) // 2
            max_y, min_y = on_coordinates[1].max(), on_coordinates[1].min()
            mid_y = (max_y + min_y) // 2
            largest_edge = max(max_x - min_x, max_y - min_y) + 5
 
            vmap = vmap * image_mask + vmap.min() * (1 - image_mask)

            max_indices = np.stack(np.unravel_index(np.argsort(np.ravel(vmap))[::-1], vmap.shape)).T
            vmap = vmap[:, mid_x - largest_edge//2:mid_x + largest_edge//2,\
                    mid_y - largest_edge//2:mid_y + largest_edge//2]

            ax.imshow(vmap.transpose(1, 2, 0), cmap='jet') 
        else:
            observation_image = observation[i//2].numpy()[:3, :, :].copy()
            max_index = max_indices[0]
            max_x = max_index[1]
            max_y = max_index[2]

            color = np.array([1, 0, 0])
            if i//2 == max_value[0][0]:
                color = np.array([0, 1, 0])

            mark_width = int(0.25*largest_edge)
            for x in range(mark_width):
                for y in range(mark_width):
                    if x == 0 or x == mark_width - 1 or y == 0 or y == mark_width - 1:
                        observation_image[:, max_x-(mark_width//2)+x, max_y-(mark_width//2)+y] = color

            observation_image = observation_image[:, mid_x - largest_edge//2:mid_x + largest_edge//2,\
                    mid_y - largest_edge//2:mid_y + largest_edge//2]
            
            ax.imshow(observation_image.transpose(1, 2, 0))

            

        ax.set_axis_off()

    fig.tight_layout()


    plt.show()

        
        # print(k, name)
        # if group.attrs['task_name'] not in task_frequencies:
            
# %

# %%

    





# %%
