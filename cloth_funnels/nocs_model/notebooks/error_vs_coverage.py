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
from tqdm import tqdm

from nocs_model.common.torch_util import dict_to, to_numpy
from nocs_model.dataset.orientation_dataset import OrientationDataModule
from nocs_model.network.orientation_deeplab import OrientationDeeplab
from nocs_model.pl_vis.orientation_callback import OrientationCallback

# %%
device = torch.device('cuda:0')
output_dir = '/local/crv/cchi/data/folding-unfolding/outputs/2022-02-27/23-05-43'
cfg = OmegaConf.load(os.path.join(output_dir, 'config.yaml')).config
ckpt_path = os.path.join(output_dir, 'checkpoints', 'last.ckpt')

datamodule = OrientationDataModule(**cfg.datamodule, include_coverage=True)
datamodule.prepare_data()
val_loader = datamodule.val_dataloader()

model = OrientationDeeplab.load_from_checkpoint(ckpt_path)
model = model.to(device).eval()

# %%
# predict
coverage_all = list()
error_all = list()

with torch.no_grad():
    for batch_cpu in tqdm(val_loader):
        # input
        input = batch_cpu['input'].to(device)
        target = batch_cpu['target'].to(device)
        coverage = batch_cpu['coverage']

        # prediction
        probs = torch.softmax(model(input), dim=1)

        # compute metric
        n_bins = probs.shape[1]
        target_angle = torch.clip(target/n_bins,0,1) * (np.pi * 2)
        _, pred_bins = torch.max(probs, dim=1)
        pred_angle = torch.clip(pred_bins/n_bins,0,1) * (np.pi * 2)
        abs_diff = torch.abs(target_angle - pred_angle)
        angle_diff = torch.minimum(np.pi*2 - abs_diff, abs_diff).moveaxis(1,0)

        # aggregate result
        mask = torch.all(input > 1e-7, dim=1)
        weights = mask.type(torch.float32)
        weighted_sum = torch.sum(weights * angle_diff, dim=[-1,-2])
        weights_sum = torch.maximum(
            torch.sum(weights,dim=[-1,-2]),
            torch.tensor(1e-7, dtype=weights.dtype, device=weights.device))
        weighted_avg = weighted_sum / weights_sum
        
        # save
        coverage_all.append(to_numpy(coverage))
        error_all.append(to_numpy(weighted_avg.T))

coverage_all = np.concatenate(coverage_all,axis=0)
error_all = np.concatenate(error_all, axis=0)

# %%
# plot
from scipy import stats
fig, axes = plt.subplots(1,2)
fig.set_size_inches(8,4)

ax_names = ['X', 'Z']
for i, ax in enumerate(axes):
    this_error = error_all[:,i]/np.pi*180
    bin_means, bin_edges, binnumber = stats.binned_statistic(
        coverage_all, this_error, bins=20, statistic='mean')
    print(bin_means[-1])
    ax.plot(bin_edges[:-1],bin_means)
    ax.set_xlabel('Coverage (relative)')
    ax.set_ylabel('Mean error (deg)')
    ax.set_title('NOCS {} orientation error vs coverage'.format(ax_names[i]))
fig.set_facecolor("w")

vis_dir = '/home/cchi/dev/folding-unfolding/src/nocs_model/data/vis'
plt.savefig(
    os.path.join(vis_dir, 'error_vs_coverage.png'),
    dpi=200)

# %%
