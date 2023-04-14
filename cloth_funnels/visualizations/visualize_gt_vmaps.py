
import h5py
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.interpolate.rbf import Rbf  # radial basis functions

input_path = '/local/crv/acanberk/folding-unfolding/src/result1_qualitative/grid_search/replay_buffer.hdf5'
save_path = input_path.replace('replay_buffer.hdf5', 'groundtruth_vmaps.png')

def get_gt_vmaps(input_path):
    action_masks = []
    input_tuples = []

    metrics = [
        'weighted_distance',
        'l2_distance',
        'icp_distance',
        'pointwise_distance'
    ]

    zs = {}
    for metric in metrics:
        zs[metric] = []

    with h5py.File(input_path, 'r') as dataset:
        keys = list(dataset.keys())
        for i, key in enumerate(keys):
            if "step00" not in key:
                continue
            group = dataset[key]
            primitive = group.attrs['action_primitive']
            observations = np.array(group['observations']).transpose(1, 2, 0)
        
            action_mask = np.array(group['action_mask'])
            action_coords = np.where(action_mask)

            if [action_coords[0][0], action_coords[1][0]] in input_tuples:
                continue

            input_tuples.append([action_coords[0][0], action_coords[1][0]])
            
            for metric in metrics:
                rew = group.attrs[f'preaction_{metric}'] - group.attrs[f'postaction_{metric}']
                zs[metric].append(rew)

            cloth_mask = np.array(group[f'mask'])

            action_masks.append(action_mask)

            rgb_obs = observations[:, :, :3]
        

    NAMES = {
        'l2_distance':'Deformable Distance',
        'icp_distance':'Rigid Distance',
        'pointwise_distance':'L2 Distance',
        'weighted_distance':'Weighted Distance'
    }

    gt_vmaps_dict = {}
    # fig, axs = plt.subplots(1, len(metrics), figsize=(10, 5))
    for i, metric in enumerate(metrics):
        xy = np.array(input_tuples).T
        z = np.array(zs[metric]).T
        x = xy[0].astype(np.float32)
        y = xy[1].astype(np.float32)

        rbf_fun = Rbf(x, y, z, function='linear')

        x_new = np.linspace(0, 128, 128)
        y_new = np.linspace(0, 128, 128)
        x_grid, y_grid = np.meshgrid(x_new, y_new)
        z_new = rbf_fun(x_grid.ravel(), y_grid.ravel()).reshape(x_grid.shape)

        gt_vmap = z_new.T * cloth_mask

        gt_vmaps_dict[NAMES[metric]] = gt_vmap

        # axs[i].set_title(f'{NAMES[metric]}')
        # axs[i].imshow(gt_vmap, cmap='jet')
        # axs[i].show()
    return gt_vmaps_dict
