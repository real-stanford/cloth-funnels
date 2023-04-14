from argparse import ArgumentParser
from typing import Any, Dict, List, MutableMapping, Tuple
from cloth_funnels.environment import SimEnv
from cloth_funnels.tasks.generate_tasks import TaskLoader
from cloth_funnels.learning.nets import MaximumValuePolicy, ExpertGraspPolicy
from cloth_funnels.learning.utils import GraspDataset, rewards_from_group
from cloth_funnels.utils.env_utils import plot_before_after
from torch.utils.data import DataLoader
from filelock import FileLock
from time import time
import torch
import h5py
import os
import ray
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import torch.nn as nn
import pandas as pd
import itertools
import hydra

def seed_all(seed):
    print(f"SEEDING WITH {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_network(args, gpu=0):
    if(args.grid_search):
        with h5py.File(args.recreate_buffer, "r") as dataset:
            grid_search_params = {
                'vmap_idx':args.grid_search_vmap_idx,
                'primitive':args.grid_search_primitive,
            }
            print("[Policy] Grid search with params:", grid_search_params)
        policy = MaximumValuePolicy(**args, gpu=gpu, grid_search_params=grid_search_params)
    else:
        policy = MaximumValuePolicy(**args, gpu=gpu)

    optimizer = torch.optim.Adam(
        policy.value_net.parameters(), lr=args.lr,
        weight_decay=args.weight_decay)

    dataset_path = args.dataset_path

    if os.path.exists(f'{args.cont}/latest_ckpt.pth'):
        checkpoint_path = f'{args.cont}/latest_ckpt.pth'
        print("[Network Setup] Load checkpoint specified", checkpoint_path)
    elif args.load is not None:
        print("[Network Setup] Load checkpoint specified", args.load)
        checkpoint_path = args.load

    ckpt = torch.load(checkpoint_path, map_location=policy.device)
    policy.load_state_dict(ckpt['net'])
    optimizer.load_state_dict(ckpt[f'optimizer'])

    print(f'\t[Network Setup] Action Exploration Probability: {policy.action_expl_prob.item():.4e}')
    print(f'\t[Network Setup] Value Exploration Probability: {policy.value_expl_prob.item():.4e}')
    print(f'\t[Network Setup] Train Steps: {policy.train_steps.item()}')

    dataset_path = f'{args.log}/replay_buffer.hdf5'
    print(f'Replay Buffer path: {dataset_path}')

    return policy, optimizer, dataset_path

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
    
def visualize_value_pred(observation, value_pred_dense, writer, step, epoch, global_step, validation=False, value_net_key=None):
    #tensorboard log value_pred dense
    obs = torch.clone(observation[0]).detach().cpu().numpy()
    obs = np.transpose(obs, (1, 2, 0))

    pred = torch.clone(value_pred_dense)
    pred = pred.detach().cpu().numpy()
    pred = pred.transpose(1,2,0)
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.imshow(np.clip(obs[:, :, :3], 0, 1))
    # print("RGB", np.max(obs[:, :, :3]), np.min(obs[:, :, :3]), "\n")
    ax2.imshow(obs[:, :, -1])
    imshow = ax3.imshow(pred, cmap='jet')
    fig.colorbar(mappable=imshow, ax=ax3)
    #title fig epoch and step
    train_or_valid = 'train' if not validation else 'valid'
    writer.add_figure(f'{value_net_key}/{train_or_valid}/value_pred_dense', fig, global_step)

def setup_envs(dataset, num_processes=16, **kwargs):
    # original_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
    # os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(original_visible_devices.split(',')[1:])
    # print("CUDA VISIBLE DEVICES BEFORE RAY SETUP", os.environ["CUDA_VISIBLE_DEVICES"])
    recreate_replay_buffer = kwargs.get('recreate_buffer', None)
    recreate_key = kwargs.get('recreate_key', None)
    recreate_primitive = kwargs.get('recreate_primitive', None)
    grid_search = kwargs.get('grid_search', False)

    recreate_task_query = None
    recreate_verts = None 

    if recreate_replay_buffer is not None:
        print(f"[Setup] Recreating {recreate_key} replay buffer {recreate_replay_buffer}")
        with h5py.File(recreate_replay_buffer, 'r') as ds:
            group = ds[recreate_key]
            recreate_verts = np.array(group['preaction_verts'])
            recreate_task_query = float(np.sum(group['init_verts']))
        x_offset = float(kwargs.get('recreate_x_offset', 0))
        y_offset = float(kwargs.get('recreate_y_offset', 0))
        recreate_verts[:, 0] += x_offset
        recreate_verts[:, 2] += y_offset
    

    task_loader = ray.remote(TaskLoader).remote(
        hdf5_path=kwargs['tasks'],
        eval_hdf5_path=kwargs['eval_tasks'],
        repeat=kwargs['eval'],
        eval=kwargs['eval'],
        recreate_task_query=recreate_task_query,
        grid_search = grid_search,
    )

    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    gpu_per_process = torch.cuda.device_count() / num_processes
    envs = [ray.remote(SimEnv).options(
        num_gpus=gpu_per_process,
        num_cpus=0.2).remote(
        replay_buffer_path=dataset,
        get_task_fn=lambda: ray.get(task_loader.get_next_task.remote()),
        recreate_verts = recreate_verts,
        **kwargs)
        for _ in range(num_processes)]
    
    ray.get([e.setup_ray.remote(e) for e in envs])

    # os.environ['CUDA_VISIBLE_DEVICES'] = original_visible_devices
    return envs, task_loader

def old_success_condition(deformable_distance, rigid_distance, coeffs=(1/0.16, 1/0.2)):
    return (coeffs[1] * deformable_distance + coeffs[0] * rigid_distance < 1) and (deformable_distance < 0.15) and (rigid_distance < 0.18)

def success_condition(deformable_distance, rigid_distance):
    return (deformable_distance < 0.10) and (rigid_distance < 0.10)

def get_loader(batch_size=256,
               num_workers=16,
               **kwargs):

    dataset = GraspDataset(**kwargs)    

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers)


def get_pretrain_loaders(batch_size=256,
               num_workers=4,
               **kwargs):

    dataset = GraspDataset(**kwargs, replay_buffer_size=100000)
    print("dataset size:", len(dataset))
    #split dataset 80 20
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    return DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers), DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers)


def get_dataset_size(path, pbar=None):
    if not os.path.exists(path):
        return 0
    with FileLock(path + ".lock"):
        return len(h5py.File(path, "r"))

# @profile
def collect_stats(dataset_path, num_points=512, label=None,
                  action_primitives=['fling', 'place'],
                  pad_episode=False, filter_fn=None, evaluate=False):
    with FileLock(dataset_path + ".lock"):
        with h5py.File(dataset_path, "r") as dataset:

            retval = {}

            def filter_fn(group):
                condition = 'preaction_pointwise_distance' in group.attrs
                return condition

            # keys = [k for k in dataset]
            # if filter_fn is not None:
            keys = [k for k in dataset if filter_fn(dataset[k])]

            if evaluate:
                print("[Collect Stats] Collecting stats for evaluation")
                num_points = min(num_points, len(keys))

            keys = keys[-num_points:]
        
            if len(keys) == 0:
                return {}

            num_points = len(keys)

            metrics = {
                'deformable_distance',
                'rigid_distance',
                'weighted_distance',
                'l2_distance',
                'iou',
                'coverage'
            }

            stat_keys = [
                'episode_length',
                'out_of_frame',
                'nonadaptive_scale',
                'rotation',
                'scale',
                'percent_fling',
                'predicted_value',
                'predicted_deformable_value',
                'predicted_rigid_value',
                # 'prediction_distance',
                # 'abs_prediction_distance',
                'deformable_weight',
            ]
            
            difficulties = ['easy', 'hard', 'none', 'flat', 'pick']

            episodes = {}

            step_df = []
            for k in keys:

                group_data = {}
                group = dataset.get(k)

                episode = int(k.split('_')[0])
                step = int(k.split('_')[1][4:])

                level = str(group.attrs['task_difficulty'])                
                group_data['episode'] = episode
                group_data['step'] = step
                group_data['level'] = level

                # print(level)

                for key, value in group.attrs.items():
                    group_data[key] = value

                step_df.append(group_data)

            
            step_df = pd.DataFrame(step_df)
            max_step = step_df['step'].max()

            retval['deformable_weight'] = step_df.deformable_weight.mean()

            DELTA_WEIGHTED_REWARDS_STD = 0.072
            DELTA_POINTWISE_REWARDS_STD = 0.12881897698788683


            for primitive in action_primitives:
                retval[f'percent_{primitive}'] = step_df[step_df.action_primitive == primitive].shape[0] / step_df.shape[0]
                retval[f'{primitive}/nonadaptive_scale_distribution'] = step_df[step_df.action_primitive == primitive]['nonadaptive_scale']
                retval[f'{primitive}/rotation_distribution'] = step_df[step_df.action_primitive == primitive]['rotation']
                retval[f'{primitive}/scale_distribution'] = step_df[step_df.action_primitive == primitive]['scale']



                gt_deformable_value = step_df[step_df.action_primitive == primitive]['preaction_l2_distance'] - step_df[step_df.action_primitive == primitive]['postaction_l2_distance']
                gt_deformable_value /= DELTA_WEIGHTED_REWARDS_STD

                gt_rigid_value = step_df[step_df.action_primitive == primitive]['preaction_icp_distance'] - step_df[step_df.action_primitive == primitive]['postaction_icp_distance']
                gt_rigid_value /= DELTA_WEIGHTED_REWARDS_STD

                gt_pointwise_value = step_df[step_df.action_primitive == primitive]['preaction_pointwise_distance'] - step_df[step_df.action_primitive == primitive]['postaction_pointwise_distance']
                gt_pointwise_value /= DELTA_POINTWISE_REWARDS_STD

                deformable_weight = step_df[step_df.action_primitive == primitive]['deformable_weight']
                gt_weighted_value = deformable_weight * gt_deformable_value + (1-deformable_weight) * gt_rigid_value


                primitive_steps = step_df[step_df.action_primitive == primitive]

                # metrics = ['gt_value', 'pred_value', 'prediction_error', 'percent_prediction_error']
                rewards = ['deformable', 'rigid', 'weighted']

         
                
                for difficulty in difficulties:
                    gt_values = {
                        'deformable': gt_deformable_value[primitive_steps.task_difficulty == difficulty],
                        'rigid': gt_rigid_value[primitive_steps.task_difficulty == difficulty],
                        'weighted': gt_weighted_value[primitive_steps.task_difficulty == difficulty],
                    }
                    primitive_difficulty_steps = primitive_steps[primitive_steps.task_difficulty == difficulty]
                    if len(primitive_difficulty_steps) == 0:
                        continue
                    for rew in rewards:
                        retval[f'{primitive}/{difficulty}/{rew}_gt_value'] = np.mean(gt_values[rew])
                        retval[f'{primitive}/{difficulty}/{rew}_predicted_value'] = np.mean(primitive_difficulty_steps[f'predicted_{rew}_value'])
                        retval[f'{primitive}/{difficulty}/{rew}_correlation'] = np.corrcoef(gt_values[rew], primitive_difficulty_steps[f'predicted_{rew}_value'])[0, 1]
                        retval[f'{primitive}/{difficulty}/{rew}_prediction_error'] = np.mean(np.abs(gt_values[rew] - primitive_steps[f'predicted_{rew}_value']))
                        retval[f'{primitive}/{difficulty}/{rew}_percent_prediction_error'] = np.mean(np.abs((gt_values[rew] - primitive_steps[f'predicted_{rew}_value']) / gt_values[rew]))

            last_step_df = step_df.groupby('episode').last()
            
            best_step_df = step_df.groupby('episode').agg({ 'task_difficulty': 'first',
                                                            'action_primitive': 'last',        
                                                            'step':'last', \
                                                            'episode':'last', \
                                                            'preaction_iou':'max',
                                                            'preaction_l2_distance':'min',
                                                            'preaction_icp_distance':'min',
                                                            'preaction_pointwise_distance':'min',
                                                            'postaction_iou':'max',
                                                            'postaction_l2_distance':'min',
                                                            'postaction_icp_distance':'min',
                                                            'postaction_pointwise_distance':'min',
                                                            'preaction_weighted_distance':'min',
                                                            'postaction_weighted_distance':'min',
                                                            'preaction_coverage':'max',
                                                            'postaction_coverage':'max' })

            last_step_df = last_step_df[last_step_df.step == max_step]

            LEGACY_METRICS_MAP = {
                'deformable_distance': 'l2_distance',
                'rigid_distance': 'icp_distance',
                'weighted_distance': 'weighted_distance',
                'l2_distance': 'pointwise_distance',
                'coverage': 'coverage',
                'iou': 'iou',
            }

            # if unfactorized_networks:
            #     retval['predicted_']


            for metric in metrics:
                for step in range(max_step):
                    retval[f'combined/step/mean_step_{step}_{metric}'] = step_df[step_df.step == step]['postaction_' + LEGACY_METRICS_MAP[metric]].mean()

            for action_primitive in action_primitives + ['combined']:
                for difficulty in difficulties:

                

                    if action_primitive == 'combined':
                        step_data = step_df[step_df.task_difficulty == difficulty]
                        last_step_data = last_step_df[last_step_df.task_difficulty == difficulty]
                        best_step_data = best_step_df[best_step_df.task_difficulty == difficulty]

                    else:
                        step_data = step_df[step_df.action_primitive == action_primitive]\
                                [step_df.task_difficulty == difficulty]
                        last_step_data = last_step_df[last_step_df.action_primitive == action_primitive]\
                                [last_step_df.task_difficulty == difficulty]
                        best_step_data = best_step_df[best_step_df.action_primitive == action_primitive]\
                                [best_step_df.task_difficulty == difficulty]
                    
                    if len(step_data) == 0:
                        continue
             
                    for metric in metrics:

                        legacy_metric = LEGACY_METRICS_MAP[metric]

                        #delta
                        key = f'{action_primitive}/{difficulty}/delta_{metric}'

                        preaction_step_data = step_data[f'preaction_{legacy_metric}']
                        postaction_step_data = step_data[f'postaction_{legacy_metric}']

                        delta_step_data = postaction_step_data - preaction_step_data

                        init_step_data = last_step_data[f'init_{legacy_metric}']
                        final_step_data = last_step_data[f'postaction_{legacy_metric}']

                        best_data = best_step_data[f'postaction_{legacy_metric}']


                        episode_delta_step_data = final_step_data - init_step_data

                        retval[f'{key}/mean'] = delta_step_data.mean()
                        retval[f'{key}/min'] = delta_step_data.min()
                        retval[f'{key}/max'] = delta_step_data.max()
                            
                        if action_primitive == 'combined':
   
                            #episode_delta
                            key = f'{action_primitive}/{difficulty}/episode_delta_{metric}'

                            if evaluate:
                                retval[f'{key}/std'] = episode_delta_step_data.std()

                                final_key = f'{action_primitive}/{difficulty}/final_{metric}'
                                retval[f'{final_key}/mean'] = final_step_data.mean()
                                retval[f'{final_key}/min'] = final_step_data.min()
                                retval[f'{final_key}/max'] = final_step_data.max()
                                retval[f'{final_key}/stderr'] = final_step_data.std()/np.sqrt(len(final_step_data))

                                # max_key = f'{action_primitive}/{difficulty}/max_{metric}'
                                # min_key = f'{action_primitive}/{difficulty}/min_{metric}'
                                best_key = f'{action_primitive}/{difficulty}/best_{metric}'
                                retval[f'{best_key}/mean'] = best_data.mean()
                                retval[f'{best_key}/stderr'] = best_data.std()/np.sqrt(len(best_data))

                            retval[f'{key}/mean'] = episode_delta_step_data.mean()
                            retval[f'{key}/min'] = episode_delta_step_data.min()
                            retval[f'{key}/max'] = episode_delta_step_data.max()


            key = random.choice(keys)
            group = dataset.get(key)
            level = str(group.attrs['task_difficulty'])
            retval.update({
                f'img_before_after/{level}':
                np.swapaxes(np.swapaxes(
                    np.array(plot_before_after(group=group, step=key.split('step')[1].split('_')[0])),
                    -1, 0), 1, 2),
                f'img_action_visualization/{level}':
                torch.tensor(
                    np.array(group['action_visualization'])).permute(2, 0, 1),
                })
            retval['vis_key'] = key

            return retval


def step_env(all_envs, ready_envs, ready_actions, remaining_observations, deterministic):
    remaining_observations.extend([e.step.remote(a)
                                   for e, a in zip(ready_envs, ready_actions)])
    step_retval = []
    start = time()
    total_time = 0
    while True:

        if deterministic:
            ready = ray.get(
                remaining_observations)
        else:
            ready, remaining_observations = ray.wait(
                remaining_observations, num_returns=1)

        if len(ready) == 0:
            continue
        step_retval.extend(ready)
        total_time = time() - start
        if (total_time > 0.01 and len(step_retval) > 0)\
                or len(step_retval) == len(all_envs):
            break

    observations = []
    ready_envs = []

    for obs, env_id in ray.get(step_retval):
        observations.append(obs)
        ready_envs.append(env_id['val'])

    return ready_envs, observations, remaining_observations


def shift_tensor(tensor, offset):
    new_tensor = torch.zeros_like(tensor).bool()
    #shifted up
    if offset > 0:
        new_tensor[:, :-offset, :] = tensor[:, offset:, :]
    #shifted down
    elif offset < 0:
        offset *= -1
        new_tensor[:, offset:, :] = tensor[:, :-offset, :]
    return new_tensor

def generate_workspace_mask(
                            render_dim : int, 
                            reach_distance_limit : float,
                            table_width: float,
                            pix_place_dist: float,
                            pix_grasp_dist: float,
                            action_primitives: dict,
                            **kwargs):
                                
    pix_radius = int(render_dim * (reach_distance_limit/table_width))

    left_arm_reach = np.zeros((render_dim, render_dim))
    left_arm_reach = cv2.circle(left_arm_reach, (render_dim//2, 0), pix_radius, (255, 255, 255), -1)

    right_arm_reach = np.zeros((render_dim, render_dim))
    right_arm_reach = cv2.circle(right_arm_reach, (render_dim//2, render_dim), pix_radius, (255, 255, 255), -1)

    left_mask = torch.tensor(right_arm_reach).bool().unsqueeze(0)
    right_mask = torch.tensor(right_arm_reach).bool().unsqueeze(0)

    

    workspace_masks = {}
    for primitive in action_primitives:
        if primitive == 'place':

            lowered_left_primitive_mask = shift_tensor(left_mask, -pix_place_dist)
            lowered_right_primitive_mask = shift_tensor(right_mask, -pix_place_dist)
            #WORKSPACE CONSTRAINTS (ensures that both the pickpoint and the place points are located within the workspace)
            left_primitive_mask = torch.logical_and(left_mask, lowered_left_primitive_mask)
            right_primitive_mask = torch.logical_and(right_mask, lowered_right_primitive_mask)
            primitive_workspace_mask = torch.logical_or(left_primitive_mask, right_primitive_mask)

        elif primitive == 'fling' or primitive == 'drag' or primitive == 'stretchdrag':

            raised_left_primitive_mask = shift_tensor(left_mask, pix_grasp_dist)
            lowered_left_primitive_mask = shift_tensor(left_mask, -pix_grasp_dist)
            raised_right_primitive_mask = shift_tensor(right_mask, pix_grasp_dist)
            lowered_right_primitive_mask = shift_tensor(right_mask, -pix_grasp_dist)
            #WORKSPACE CONSTRAINTS
            aligned_workspace_mask = torch.logical_and(raised_left_primitive_mask, lowered_right_primitive_mask)
            opposite_workspace_mask = torch.logical_and(raised_right_primitive_mask, lowered_left_primitive_mask)
            primitive_workspace_mask = torch.logical_or(aligned_workspace_mask, opposite_workspace_mask)
        
        workspace_masks[primitive] = primitive_workspace_mask

    return workspace_masks 

def flatten_dict(
    d: MutableMapping, parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, List) and isinstance(v[0], MutableMapping):
            for idx in range(len(v)):
                items.extend(flatten_dict(v[idx], f"{new_key}/{idx}", sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


if __name__ == '__main__':
    collect_stats('./experiments/test/replay_buffer.hdf5')