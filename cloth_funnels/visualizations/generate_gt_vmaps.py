import pathlib
import sys
import h5py
import glob
import numpy as np
import os
from visualize_gt_vmaps import get_gt_vmaps
import itertools

# output_dir_name = 'quantitative_comparisons/result1_qualitative_3'
# recreate_buffer = 'experiments/05-2296e-unfactorized-nonocs/analysis_buffer_2.hdf5'
# recreate_key = '000000117_step00'

#python visualizations/generate_gt_vmaps.py quantitative_comparisons/pull_arm_gts experiments/05-2296e-unfactorized-nonocs/analysis_buffer_2.hdf5 000000117_step00"

output_dir_name = sys.argv[1]
recreate_buffer = sys.argv[2]
recreate_key = sys.argv[3]

tasks = 'final_datasets/multi-longsleeve-eval.hdf5'
deformable_weight = 0.7
gpu_idx = "2,3"
GT_SAMPLE_STEPS=60
recreate_x_offset = 0.015


num_scales = 6
num_rotations = 17
primitive = 'place'

scales = [3]
rotations = range(0, 16, 2)

vmap_indices = []
for scale in scales:
    for rotation in rotations:
        vmap_indices.append((num_scales * rotation + scale, primitive))
print(vmap_indices)
grid_search_dir = f"{output_dir_name}/grid_search"
pathlib.Path(grid_search_dir).mkdir(parents=True, exist_ok=True)
generate_gt_commands = []
for vmap_idx, primitive in vmap_indices:
    command_str = f"DISPLAY=:0.0 CUDA_VISIBLE_DEVICES={gpu_idx} python run_sim.py " + \
    f"wandb=offline log={output_dir_name}/grid_search/{vmap_idx}_{primitive} num_processes=24 " + \
    f"tasks={tasks} input_channel_types=rgb_pos " + \
    f"recreate_buffer={recreate_buffer} recreate_x_offset={recreate_x_offset} " + \
    f"recreate_key={recreate_key} episode_length=2 deformable_weight=0.7 " +\
    f"grid_search=True wandb=offline max_steps={GT_SAMPLE_STEPS} grid_search_vmap_idx={vmap_idx} grid_search_primitive={primitive} "

    generate_gt_commands.append(command_str)

generate_gt_command = " && ".join(generate_gt_commands)
print(generate_gt_command)

os.system(generate_gt_command)




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


