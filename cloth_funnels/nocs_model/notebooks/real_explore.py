# %%
import os
import sys
proj_dir = os.path.expanduser("~/dev/folding-unfolding/src")
os.chdir(proj_dir)
sys.path.append(proj_dir)

# %%
# import
import pathlib
import yaml
import hydra
import pickle
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import AffineTransform
from torchvision.transforms.functional import affine, resize
from torchvision.transforms import InterpolationMode
from skimage.io import imsave
import cv2

from nocs_model.common.torch_util import dict_to, to_numpy
from nocs_model.dataset.orientation_dataset import OrientationDataModule
from nocs_model.network.orientation_deeplab import OrientationDeeplab
from nocs_model.pl_vis.orientation_callback import OrientationCallback

# %%
# load real data
model = 'ours' # flingbot, ours, pp
cloth = 'blue' # blue, dress, red, pink

real_raw_imgs = list()
for id in range(10):
    data = pickle.load(open(f'/proj/crv/zhenjia/cloth-real-data/final-{model}/test-real:{cloth}{id}/test_data.pkl', 'rb'))
    for step in range(5):
        final_color_img = data[step]['final_observation'][0]['color_img']
        real_raw_imgs.append(((id, step), final_color_img))
raw_imgs = np.array([x[1] for x in real_raw_imgs])

# replace background
is_bg = np.all(raw_imgs == 90, axis=-1)
raw_imgs[is_bg] = 0
imgs = torch.from_numpy(raw_imgs).moveaxis(-1,1).type(torch.float32) / 255

# shrink image
scale = 0.6
input_size = (128, 128)
input_scaled = affine(imgs, 
    angle=0, translate=[0,0], scale=scale, shear=0,
    interpolation=InterpolationMode.BILINEAR)
input_cpu = resize(input_scaled, size=input_size, 
    interpolation=InterpolationMode.BILINEAR)

# plt.imshow(input_cpu[9].moveaxis(0,-1))
# %%
device = torch.device('cuda:0')
output_dir = '/local/crv/cchi/data/folding-unfolding/outputs/2022-02-27/23-05-43'
cfg = OmegaConf.load(os.path.join(output_dir, 'config.yaml')).config
ckpt_path = os.path.join(output_dir, 'checkpoints', 'last.ckpt')

model = OrientationDeeplab.load_from_checkpoint(ckpt_path)
model = model.to(device).eval()

# %%
input = input_cpu.to(device)
with torch.no_grad():
    probs = torch.softmax(model(input), dim=1)
probs = to_numpy(probs)

# %%
def pred_to_img(pred_img):
    img_shape = pred_img.shape[-2:]
    n_bins = pred_img.shape[0]
    vis_img = np.zeros(img_shape + (3,), dtype=np.float32)
    best_bin = np.argmax(pred_img, axis=0) / n_bins
    vis_img[...,[0,2]] = np.moveaxis(best_bin,0,-1)
    return vis_img


def pred_to_axis(pred_img, vis_img, length=3, grid=8):
    img_size = pred_img.shape[-2:]
    n_bins = pred_img.shape[0]
    best_bin = np.argmax(pred_img, axis=0) / n_bins
    start = grid//2
    sampled_coords = np.indices(img_size)[:,start::grid,start::grid]
    sampled_angles = best_bin[:,start::grid,start::grid] * (np.pi * 2)

    sampled_offsets = np.stack([
        np.cos(sampled_angles),
        np.sin(sampled_angles)
    ])
    vis_img = (vis_img * 255).astype(np.uint8).copy()
    for channel_id, color in enumerate([(255,0,0),(0,0,255)]):
    # channel_id = 1
    # color = (0,0,255)
        for r, c in np.ndindex(sampled_coords.shape[-2:]):
            offset = sampled_offsets[:,channel_id,r,c]
            base = np.array([r,c], dtype=np.float32) * grid + start
            tip = base + offset * length
            cv2.line(vis_img, 
                pt1=base[::-1].astype(np.int32),
                pt2=tip[::-1].astype(np.int32),
                color=color,
                thickness=1)
    return vis_img

# %%
idx = 20
input_img = to_numpy(input_cpu[idx].moveaxis(0,-1))
# pred_img = pred_to_img(probs[idx])
# vis_img = np.concatenate([input_img, pred_img], axis=1)
pred_img = probs[idx]
vis_img = input_img
vis_img = pred_to_axis(probs[idx], input_img)
plt.imshow(vis_img)


# %%
# dump
vis_dir = '/home/cchi/dev/folding-unfolding/src/nocs_model/data/vis/direction_vis'
for idx in range(len(input_cpu)):
    id, step = real_raw_imgs[idx][0]
    input_img = to_numpy(input_cpu[idx].moveaxis(0,-1))
    vis_img = pred_to_axis(probs[idx], input_img)
    pred_img = (pred_to_img(probs[idx]) * 255).astype(np.uint8)
    out_img = np.concatenate([vis_img, pred_img], axis=1)
    fname = '{:02d}_{:02d}.png'.format(id, step)
    path = os.path.join(vis_dir, fname)
    imsave(path, out_img)


# %%
