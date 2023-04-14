
from notebooks.episode_visualizer import *
from visualize_gt_vmaps import get_gt_vmaps
import h5py
import glob
import pathlib
import os

# vis_path = 'quantitative_comparisons/result1_qualitative_3'
vis_path = sys.argv[1]
vis_path = pathlib.Path(vis_path).resolve()

replay_buffers = glob.glob(f'{vis_path}/*/replay_buffer.hdf5')
replay_buffers = [buffer for buffer in replay_buffers if ('unfactorized' in buffer or 'nonocs' in buffer)]
print(replay_buffers)
text = False

FACTORIZED_VMAP_NORMALIZATION = 0.072
UNFACTORIZED_VMAP_NORMALIZATION = 0.12881897698788683

buffer_info = {}
for path in replay_buffers:
    buffer_nickname = path.split('/')[-2]
    corresponding_gt_path = glob.glob(f'{vis_path}/grid_search/{buffer_nickname}*/replay_buffer.hdf5')[0]
    vmap_index = (int(corresponding_gt_path.split('/')[-2].split('_')[-2]),
                    corresponding_gt_path.split('/')[-2].split('_')[-1])
    buffer_info[buffer_nickname] = {'gt_path': corresponding_gt_path, 'vmap_index': vmap_index, 'step_path': path, 'respective_value_maps':{}}

    


for buffer_nickname in tqdm(buffer_info.keys()):

    buffer_values = buffer_info[buffer_nickname]

    fig, axs, observations, value_maps = visualize_episode('000000000_step00', buffer_values['step_path'], \
        steps=2, vis_index=(0,1,3), visualize_metrics=False, custom_vmap_index=buffer_values['vmap_index'], visualize_actions=True)
   
    plt.savefig(f"{vis_path}/{buffer_nickname}_episode_vis.png")
    plt.close()

    buffer_info[buffer_nickname]['observations'] = observations
    buffer_info[buffer_nickname]['value_maps'] = value_maps
    buffer_info[buffer_nickname]['respective_value_maps'][buffer_nickname] = value_maps

    for other_buffer_nickname in buffer_info.keys():
        other_step_path = buffer_info[other_buffer_nickname]['step_path']
        if other_buffer_nickname == buffer_nickname:
            continue 

        fig, axs, observations, value_maps = visualize_episode('000000000_step00', other_step_path, \
          steps=2, vis_index=(0,1,3), visualize_metrics=True, custom_vmap_index=buffer_values['vmap_index'])
      
        plt.close()

        buffer_info[buffer_nickname]['respective_value_maps'][other_buffer_nickname] = value_maps

    def noninf(arr):
        return arr[arr != -np.inf]

    buffer_info[buffer_nickname]['weighted_min'] = np.min([noninf(buffer_info[buffer_nickname]['respective_value_maps'][other_buffer_nickname][0])\
        .min() \
            for other_buffer_nickname in buffer_info.keys() if 'unfactorized' not in other_buffer_nickname])
    buffer_info[buffer_nickname]['weighted_max'] = np.max([noninf(buffer_info[buffer_nickname]['respective_value_maps'][other_buffer_nickname][0])\
        .max() \
            for other_buffer_nickname in buffer_info.keys() if 'unfactorized' not in other_buffer_nickname])
    
    buffer_info[buffer_nickname]['unfactorized_min'] = np.min([noninf(buffer_info[buffer_nickname]['respective_value_maps'][other_buffer_nickname][0])\
        .min() \
            for other_buffer_nickname in buffer_info.keys() if 'unfactorized' in other_buffer_nickname])
    buffer_info[buffer_nickname]['unfactorized_max'] = np.max([noninf(buffer_info[buffer_nickname]['respective_value_maps'][other_buffer_nickname][0])\
        .max() \
            for other_buffer_nickname in buffer_info.keys() if 'unfactorized' in other_buffer_nickname])
            
        
   
fig, axs = plt.subplots(len(buffer_info.keys())+3, len(buffer_info.keys()), figsize=(len(buffer_info.keys())*3,(len(buffer_info.keys())+3)*3))

for i, (buffer_nickname, buffer_values) in enumerate(buffer_info.items()):

    gt_value_maps = get_gt_vmaps(buffer_values['gt_path'])

    l2_value_map = gt_value_maps["L2 Distance"]/FACTORIZED_VMAP_NORMALIZATION
    weighted_value_map = gt_value_maps["Weighted Distance"]/UNFACTORIZED_VMAP_NORMALIZATION

    axs[0, i].imshow(buffer_values['observations'][0])

    if text:
        axs[0, i].set_title(buffer_nickname)

    buffer_info[buffer_nickname]['weighted_min'] = np.min([ buffer_info[buffer_nickname]['weighted_min'], (weighted_value_map * (weighted_value_map > -np.inf)).min()])
    buffer_info[buffer_nickname]['weighted_max'] = np.max([ buffer_info[buffer_nickname]['weighted_max'], (weighted_value_map * (weighted_value_map > -np.inf)).max()])
    buffer_info[buffer_nickname]['unfactorized_min'] = np.min([ buffer_info[buffer_nickname]['unfactorized_min'], (l2_value_map * (l2_value_map > -np.inf)).min()])
    buffer_info[buffer_nickname]['unfactorized_max'] = np.max([ buffer_info[buffer_nickname]['unfactorized_max'], (l2_value_map * (l2_value_map > -np.inf)).max()])

    factorized_min = buffer_info[buffer_nickname]['weighted_min']
    factorized_max = buffer_info[buffer_nickname]['weighted_max']
    unfactorized_min = buffer_info[buffer_nickname]['unfactorized_min']
    unfactorized_max = buffer_info[buffer_nickname]['unfactorized_max']

    im = axs[len(replay_buffers) + 1, i].imshow(weighted_value_map, cmap='jet', vmin=-1.5, vmax=0.4)
    cbar = plt.colorbar(im, ax=axs[len(replay_buffers) + 1, i], shrink=0.6, ticks=np.linspace(-1.5, 0.4, 6), format='%.1f')
    cbar.ax.tick_params(labelsize=8)

    im = axs[len(replay_buffers) + 2, i].imshow(l2_value_map, cmap='jet', vmin=-1.5, vmax=0.4)
    cbar = plt.colorbar(im, ax=axs[len(replay_buffers) + 2, i], shrink=0.6, ticks=np.linspace(-1.5, 0.4, 6), format='%.1f')
    cbar.ax.tick_params(labelsize=8)

    if text:
        axs[len(replay_buffers) + 1, i].set_title(f"Weighted Distance")
        axs[len(replay_buffers) + 2, i].set_title(f"L2 Distance")



for i, buffer_nickname in enumerate(buffer_info.keys()):
    factorized_min = buffer_info[buffer_nickname]['weighted_min']
    factorized_max = buffer_info[buffer_nickname]['weighted_max']
    unfactorized_min = buffer_info[buffer_nickname]['unfactorized_min']
    unfactorized_max = buffer_info[buffer_nickname]['unfactorized_max']

    for j, other_buffer_nickname in enumerate(buffer_info.keys()):

        vmap_vmin = factorized_min if ('unfactorized' not in other_buffer_nickname) else unfactorized_min
        vmap_vmax = factorized_max if ('unfactorized' not in other_buffer_nickname) else unfactorized_max

        print(vmap_vmin, vmap_vmax)

        value_map = buffer_info[buffer_nickname]['respective_value_maps'][other_buffer_nickname][0]

        value_map[value_map == -np.inf] = 0

        im = axs[j+1, i].imshow(buffer_info[buffer_nickname]['respective_value_maps'][other_buffer_nickname][0], cmap='jet', 
            vmin=-1.5, vmax=0.4)

        if text:
            axs[j+1, i].set_title(f"{other_buffer_nickname}")

        cbar = plt.colorbar(im, ax=axs[j+1, i], shrink=0.6, ticks=np.linspace(-1.5, 0.4, 6), format='%.1f')
        cbar.ax.tick_params(labelsize=8)
        # print(buffer_info[buffer_nickname]['respective_value_maps'][other_buffer_nickname][0].shape)

for ax in axs.flatten():
    ax.axis('off')


plt.savefig(f"{vis_path}/all_value_maps.png", transparent=True)


