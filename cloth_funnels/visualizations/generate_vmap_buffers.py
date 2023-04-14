
import pathlib
import sys
import h5py
import glob
import numpy as np
import os
from visualize_gt_vmaps import get_gt_vmaps

# output_dir_name = 'quantitative_comparisons/result1_qualitative_3'
# recreate_buffer = 'experiments/05-2296e-unfactorized-nonocs/analysis_buffer_2.hdf5'
# recreate_key = '000000117_step00'

output_dir_name = sys.argv[1]
recreate_buffer = sys.argv[2]
recreate_key = sys.argv[3]

tasks = 'final_datasets/multi-longsleeve-eval.hdf5'
deformable_weight = 0.7
gpu_idx = "0,1,2,3"
GT_SAMPLE_STEPS=50

#(checkpoints nickname), (checkpoints path), (additional params to the command)
checkpoints = [
    ('ours_gtnocs', 'experiments/03-2296e-ours-gtnocs/latest_ckpt.pth', 
        'input_channel_types=rgb_pos_gtnocs', ),
    ('unfactorized', 'experiments/05-2296e-unfactorized-nonocs/latest_ckpt.pth', 
        'input_channel_types=rgb_pos unfactorized_rewards=True unfactorized_networks=True'),
    ('ours_nonocs', 'experiments/03-2296e-ours-nonocs-longsleeve/latest_ckpt.pth', 
        'input_channel_types=rgb_pos'),
    ('ours_prednocs', 'experiments/04-2296e-ours-prednocs/latest_ckpt.pth',
        'input_channel_types=rgb_pos_nocs'),
]

print(pathlib.Path(output_dir_name))
if pathlib.Path(output_dir_name).exists():
    print(f"Directory {output_dir_name} exists, recreating")
else:
    print(f"Directory {output_dir_name} does not exist, creating")
    pathlib.Path(output_dir_name).mkdir(parents=True)

commands = []

for checkpoint_nickname, checkpoint_path, additional_params in checkpoints:
    command_str = f"CUDA_VISIBLE_DEVICES={gpu_idx} python run_sim.py wandb=offline "\
    f"log={output_dir_name}/{checkpoint_nickname} num_processes=1 tasks={tasks} "\
    f"ray_local_mode=True recreate_buffer={recreate_buffer} recreate_key={recreate_key} "\
    f"deformable_weight={deformable_weight} {additional_params} max_steps=1 episode_length=2 "\
    f"load={checkpoint_path}"

    commands.append(command_str)

generate_steps_command = " && ".join(commands)
print("\n Generating steps: \n\n", generate_steps_command)

# os.system(generate_steps_command)


print(f"Pulling buffers from {output_dir_name}")
buffer_paths = [f"{output_dir_name}/{v[0]}/replay_buffer.hdf5" for v in checkpoints]

for path in buffer_paths:
    if 'grid_search' in path:
        buffer_paths.remove(path)

picked_vmap_tuples = []

for path in buffer_paths:
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        picked_vmap_tuples.append((np.array(f[keys[0]]['max_indices'])[0], f[keys[0]].attrs['action_primitive']))

print(f"Picked vmaps: {picked_vmap_tuples}")

grid_search_dir = f"{output_dir_name}/grid_search"
pathlib.Path(grid_search_dir).mkdir(parents=True, exist_ok=True)
generate_gt_commands = []

for (checkpoint_nickname, checkpoint_path, additional_params), (vmap_idx, primitive) in zip(checkpoints, picked_vmap_tuples):
    log_name = f"{grid_search_dir}/{checkpoint_nickname}"
    command_str = f"DISPLAY=:0.0 CUDA_VISIBLE_DEVICES={gpu_idx} python run_sim.py " + \
    f"wandb=offline log={log_name}_{vmap_idx}_{primitive} num_processes=6 " + \
    f"tasks={tasks} input_channel_types=rgb_pos " + \
    f"recreate_buffer={recreate_buffer} " + \
    f"recreate_key={recreate_key} episode_length=2 deformable_weight=0.7 " +\
    f"grid_search=True wandb=offline max_steps={GT_SAMPLE_STEPS} grid_search_vmap_idx={vmap_idx} grid_search_primitive={primitive} "

    generate_gt_commands.append(command_str)

generate_gt_command = " && ".join(generate_gt_commands)

print("\n Generating groundtruth buffers (this may take a bit longer): \n\n", generate_gt_command)






# os.system(generate_gt_command)


# "CUDA_VISIBLE_DEVICES=3 python run_sim.py"
# wandb=offline log=result1_qualitative/case_0_ours_gtnocs 
# num_processes=1 tasks=final_datasets/multi-longsleeve-eval.hdf5 
# input_channel_types=rgb_pos_gtnocs ray_local_mode=True 
# recreate_buffer=experiments/05-2296e-unfactorized-nonocs/analysis_buffer_2.hdf5 
# recreate_key=000000117_step00 load=experiments/03-2296e-ours-gtnocs/latest_ckpt.pth 
# deformable_weight=0.7 episode_length=2 max_steps=1


# INPUT:
# - Recreate buffer:
# - Recreate key:
# - Checkpoints:
# - Representative transform:


