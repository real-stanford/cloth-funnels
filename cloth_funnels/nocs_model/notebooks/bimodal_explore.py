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
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
import numpy as np
from matplotlib import pyplot as plt

from nocs_model.common.torch_util import dict_to, to_numpy
from nocs_model.dataset.orientation_dataset import OrientationDataModule
from nocs_model.network.orientation_deeplab import OrientationDeeplab
from nocs_model.pl_vis.orientation_callback import OrientationCallback

# %%
device = torch.device('cuda:0')
output_dir = '/local/crv/cchi/data/folding-unfolding/outputs/2022-02-27/23-05-43'
cfg = OmegaConf.load(os.path.join(output_dir, 'config.yaml')).config
ckpt_path = os.path.join(output_dir, 'checkpoints', 'last.ckpt')

datamodule = OrientationDataModule(**cfg.datamodule)
datamodule.prepare_data()
val_loader = datamodule.val_dataloader()

model = OrientationDeeplab.load_from_checkpoint(ckpt_path)
model = model.to(device).eval()

# %%
batch_cpu = next(iter(val_loader))
input = batch_cpu['input'].to(device)
with torch.no_grad():
    probs = torch.softmax(model(input), dim=1)
probs = to_numpy(probs)
gt = to_numpy(batch_cpu['target'])

# %%
def pred_to_img(pred_img):
    img_shape = pred_img.shape[-2:]
    n_bins = pred_img.shape[0]
    vis_img = np.zeros(img_shape + (3,), dtype=np.float32)
    best_bin = np.argmax(pred_img, axis=0) / n_bins
    vis_img[...,[0,2]] = np.moveaxis(best_bin,0,-1)
    return vis_img

def gt_to_img(gt_img, n_bins=32):
    img_shape = gt_img.shape[-2:]
    vis_img = np.zeros(img_shape + (3,), dtype=np.float32)
    best_bin = np.clip(gt_img / n_bins, 0, 1)
    vis_img[...,[0,2]] = np.moveaxis(best_bin,0,-1)
    return vis_img

# %%
idx = 32
gt_img = gt_to_img(gt[idx])
pred_img = pred_to_img(probs[idx])
vis_img = np.concatenate([pred_img, gt_img], axis=1)
plt.imshow(vis_img)

# %%
