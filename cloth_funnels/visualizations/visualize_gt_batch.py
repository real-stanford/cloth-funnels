#%%

from visualize_gt_vmaps import get_gt_vmaps
import os 
os.chdir('/local/crv/acanberk/folding-unfolding/src')
os.environ['DISPLAY']=":0.0"


DELTA_WEIGHTED_REWARDS_STD = 0.072
DELTA_POINTWISE_REWARDS_STD = 0.12881897698788683


import numpy as np
import matplotlib.pyplot as plt
import glob 
import os
import h5py
from scipy.ndimage import rotate, zoom
from scipy.interpolate import interp2d
import skimage.filters
from notebooks.episode_visualizer import visualize_episode, get_edges
import cv2

input_dir = '/local/crv/acanberk/folding-unfolding/src/quantitative_comparisons/pull_arm_gts_offset_2/grid_search'
paths = glob.glob(os.path.join(input_dir, '*/replay_buffer.hdf5'))


primitive = 'place'
scales = [3]
rotations = range(0, 16, 2)

num_rotations = 17
num_scales = 6


rotation_values = np.linspace(-180, 180, num_rotations)
scale_values = np.array([0.75, 1.0, 1.5, 2.0, 2.5, 3.0])

inverse_rotation_values = -1 * rotation_values
inverse_scale_values = 1/scale_values

gt_vmaps = {}

metrics = ['Weighted Distance', 'L2 Distance']

mins = {}
maxes = {}
preaction_obs = {}
postaction_obs = {}

for path in paths:
    with h5py.File(path, 'r') as f:
       
 
        vmap_idx = path.split("/")[-2].split("_")[0]
        primitive = path.split("/")[-2].split("_")[1]
        if primitive not in gt_vmaps:
            gt_vmaps[(int(vmap_idx), primitive)] = {}

        gt_vmaps[(int(vmap_idx), primitive)]['vmap'] = get_gt_vmaps(path)
        gt_vmaps[(int(vmap_idx), primitive)]['vmap']['Weighted Distance'] /= DELTA_WEIGHTED_REWARDS_STD
        gt_vmaps[(int(vmap_idx), primitive)]['vmap']['L2 Distance'] /= DELTA_POINTWISE_REWARDS_STD

        gt_vmaps[(int(vmap_idx), primitive)]['preaction'] = {key: None for key in metrics}
        gt_vmaps[(int(vmap_idx), primitive)]['postaction'] = {key: None for key in metrics}

        keys = list(f.keys())
        METRICS_MAP = {"Weighted Distance":'weighted_distance', "L2 Distance":'pointwise_distance'}
        best = {key:[] for key in metrics}
        for key in keys:
            if 'step00' not in key:
                continue
            for metric in metrics:
                delta = f[key].attrs[f"preaction_{METRICS_MAP[metric]}"] - f[key].attrs[f"postaction_{METRICS_MAP[metric]}"]
             
                if metric == 'Weighted Distance':
                    delta /= DELTA_WEIGHTED_REWARDS_STD
                elif metric == 'L2 Distance':
                    delta /= DELTA_POINTWISE_REWARDS_STD
                
                pretransform_obs = f[key]['pretransform_observations'][:3].transpose(1, 2, 0)

                binary_mask = np.array(f[key]['preaction_init_mask'])
                edges = np.stack(3 * [get_edges(binary_mask)]).transpose(1, 2, 0)
                cloth_mask = np.stack(3 * [np.sum(pretransform_obs, axis=-1) > 0]).transpose(1, 2, 0)
                pretransform_obs = pretransform_obs + ((edges * ~cloth_mask) * 0.5)

                best[metric].append([delta, pretransform_obs, key])
        
        for metric in metrics:
            best[metric].sort()
            gt_vmaps[(int(vmap_idx), primitive)]['postaction'][metric] = best[metric][-1][1]

            fig, axs, _, _ = visualize_episode(best[metric][-1][2], path, steps=2, vis_index=(0,1), visualize_actions=False, visualize_metrics=True)
            axs[0, 0].set_title(f"Best {metric} for {primitive} {vmap_idx}")
            plt.show()
            plt.close()
            
        gt = gt_vmaps[(int(vmap_idx), primitive)]['vmap']
        for metric in gt:
            mins[metric] = min(mins.get(metric, 1e10), np.min(gt[metric]))
            maxes[metric] = max(maxes.get(metric, -1e10), np.max(gt[metric]))
            
print(mins, maxes)

#%%
for metric in metrics:
    print("Metric: ", metric)
    vmap_min = mins[metric]
    vmap_max = maxes[metric]
    fig, axs = plt.subplots(len(scales)*2, len(rotations), figsize=(len(rotations)*3, len(scales)*6))
    for i, scale in enumerate(scales):
        for j, rotation in enumerate(rotations):

            vmap = gt_vmaps[(num_scales * rotation + scale, primitive)]['vmap'][metric]

            vmap = rotate(vmap, rotation_values[rotation])
            vmap = cv2.resize(vmap, (vmap.shape[0]*3, vmap.shape[1]*3), interpolation=cv2.INTER_AREA)
            vmap = vmap[vmap.shape[0]//4: 3*vmap.shape[0]//4, vmap.shape[1]//4: 3*vmap.shape[1]//4]    


            vmap = skimage.filters.gaussian(vmap, sigma=(3.0, 3.0), truncate=10, multichannel=True)
            vmap = skimage.filters.unsharp_mask(vmap, radius=1, amount=5)

            im = axs[i*2, j].imshow(vmap, cmap='jet', vmin=vmap_min, vmax=vmap_max)
            axs[i*2+1, j].imshow(gt_vmaps[(num_scales * rotation + scale, primitive)]['postaction'][metric])
            plt.colorbar(im, ax=axs[i*2, j], cmap='jet')
            axs[i*2+1, j].set_title(f"{gt_vmaps[(num_scales * rotation + scale, primitive)]['vmap'][metric].max():.3f}", fontsize=20)

    for ax in axs.flat:
        ax.set_axis_off()

    fig.tight_layout()
    plt.show()
    #%%