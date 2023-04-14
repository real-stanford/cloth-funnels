from multiprocessing.managers import ValueProxy
from re import U
from statistics import StatisticsError
from pyrsistent import s
import ray
from time import time, strftime
import shutil
from copy import copy
import torch
from filelock import FileLock
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb 
import itertools

LOSS_NORMALIZATIONS = {
            'rigid':{'fling':{'manipulation':5, 'nocs':3}, 'place':{'manipulation':1.2, 'nocs':3}},
            'deformable':{'fling':{'manipulation':0.8, 'nocs':3}, 'place':{'manipulation':0.1, 'nocs':3}}
        }

# @profile
def optimize_fb(
             policy, 
             optimizer, 
             loader,
             criterion, 
             num_updates, 
             dataset_size,
             deformable_weight=None, 
             action_primitives=None, 
             unfactorized_networks=False, 
             coverage_reward=False,
             pretrain_dataset_path=None, 
             verbose=True, 
             unfactorized_rewards=False,
             **kwargs):

    value_net = policy.value_net
 
    print("[Network] >> Optimizing value network, with reward factorization:", not unfactorized_rewards)
    if coverage_reward:
        print("[Network] Using coverage reward")

    distances = ['rigid', 'deformable']
   
    if loader is None or optimizer is None:
        print(">> No loader or optimizer provided, skipping training")
        return

    device = value_net.device
    value_net.train()
    mean_update_stats = {}

    visualizations = {distance:{'fling':{'manipulation':None, 'nocs':None, 'obs':None, 'distribution':None}, 'place':{'manipulation':None, 'nocs':None, 'obs':None, 'distribution':None}} for distance in distances}
    
    update_id = 0
    while update_id < num_updates: 


        ### ensure the sample from the dataset is valid
        while True:
            # if the loader is not exhausted, get the next sample
            try:
                sample = next(loader)
            except StopIteration:
                print(">> Loader exhausted")
                break 
            
            is_valid = True
            for primitive_id in range(len(action_primitives)):
                for key, value in sample[primitive_id].items():
                    if torch.isnan(value).any() or torch.isinf(value).any():
                        print("NaN or Inf detected in sample: ", key)
                        is_valid = False
            if is_valid: 
                break
        ###

        losses = {distance:
                {'fling':{'manipulation':0}, 
                'place':{'manipulation':0}} 
                for distance in distances}
        l2_error = {distance:{'fling':{'manipulation':0}, 
                              'place':{'manipulation':0}} 
                    for distance in distances}

        unfactorized_losses = {'fling':0, 'place':0}
        
        visualizations = {distance:{'fling':{'manipulation':None, 'nocs':None, 'obs':None, 'distribution':None}, 'place':{'manipulation':None, 'nocs':None, 'obs':None, 'distribution':None}} for distance in distances}
        stats = dict()

        #COMPUTE LOSSES
        for primitive_id in range(len(action_primitives)):

            action_primitive = action_primitives[primitive_id]

            in_dict = sample[primitive_id]
            obs = in_dict['obs']
            action_mask = in_dict['action']
            weighted_reward = in_dict['weighted_reward']
            deformable_reward = in_dict['deformable_reward']
            rigid_reward = in_dict['rigid_reward']
            l2_reward = in_dict['l2_reward']
            cov_reward = in_dict['coverage_reward']
            is_terminal = in_dict['is_terminal'].to(device)

            action_mask = action_mask.unsqueeze(1)
            
            rewards = {'rigid': rigid_reward, 'deformable': deformable_reward}

            #preprocess here so that we can log the obs
            obs = value_net.preprocess_obs(obs.to(device, non_blocking=True))
            out = value_net.forward_for_optimize(obs, action_primitive, preprocess=False)

            action_mask = torch.cat([action_mask], dim=1)

            unfactorized_value_pred_dense = (1-deformable_weight) * out['rigid'][action_primitive] + \
                 deformable_weight * out['deformable'][action_primitive]
            unfactorized_value_pred = torch.masked_select(unfactorized_value_pred_dense, action_mask.to(device, non_blocking=True))

            if unfactorized_rewards:
                if coverage_reward:
                    print("[Network] Using coverage reward")
                    unfactorized_reward = cov_reward.to(device)
                else:
                    print("[Network] Using unfactorized reward")
                    unfactorized_reward = l2_reward.to(device)
            else:
                print("[Network] Using factorized reward")
                unfactorized_reward = weighted_reward.to(device)

          
            unfactorized_losses[action_primitive] = torch.nn.functional.smooth_l1_loss(unfactorized_value_pred, unfactorized_reward.to(device, non_blocking=True))

            for distance in distances:

                value_pred_dense = out[distance][action_primitive]
                value_pred = torch.masked_select(
                    value_pred_dense,
                    action_mask.to(device, non_blocking=True))
    
                reward = rewards[distance].to(device)

                manipulation_loss = torch.nn.functional.smooth_l1_loss(value_pred, reward)

                losses[distance][action_primitive]['manipulation'] = manipulation_loss / LOSS_NORMALIZATIONS[distance][action_primitive]['manipulation']
             
                l2_error[distance][action_primitive]['manipulation'] = manipulation_loss

                log_idx = 0
                visualizations[distance][action_primitive]['manipulation'] = value_pred_dense[log_idx].detach().cpu().numpy()
                visualizations[distance][action_primitive]['obs'] = obs[log_idx].detach().cpu().numpy()

        #OPTIMIZE
        loss = 0

        for distance in distances:
            for primitive in action_primitives:
                stats[f'loss/{primitive}/unfactorized']= unfactorized_losses[primitive] / len(action_primitives)
                stats[f'loss/{primitive}/{distance}/factorized'] = losses[distance][primitive]['manipulation'] / len(action_primitives)
                stats[f'l2_error/{primitive}/{distance}/factorized'] = l2_error[distance][primitive]['manipulation'] / len(action_primitives)

        optimizer.zero_grad()
        if unfactorized_networks:
            loss = sum(v for k,v in stats.items() if 'loss/' in k and '/unfactorized' in k)
        else:
            loss = sum(v for k,v in stats.items() if 'loss/' in k and '/factorized' in k)
        loss.backward()
        optimizer.step()

        if verbose:
            print(f"Update {update_id+1}/{num_updates} - Loss: {loss.item():.4f}")

        policy.train_steps += 1
        for k,v in stats.items():
            if k not in mean_update_stats:
                mean_update_stats[k] = []
            mean_update_stats[k].append(float(v))

        update_id += 1


    #VISUALIZATIONS
    pairings = itertools.product(action_primitives, distances)
    sample_obs = visualizations[distance][action_primitive]['obs']
    num_channels = sample_obs.shape[0]

    num_with_rgb = num_channels - 2
    fig, axs = plt.subplots(num_with_rgb, 2, figsize=(4, num_with_rgb*2))

    for i, action_primitive in enumerate(action_primitives):
        chosen_obs = visualizations[distance][action_primitive]['obs']
        axs[0, i].set_title(f"({chosen_obs[:3].min():.2f}, {chosen_obs[:3].max():.2f})", fontsize=8)
        axs[0, i].imshow((chosen_obs[:3].transpose(1, 2, 0) * 0.5) + 0.5)
        for j in range(3, num_channels):
            axs[j-2, i].set_title(f"({chosen_obs[j].min():.2f}, {chosen_obs[j].max():.2f})", fontsize=8)
            axs[j-2, i].imshow(chosen_obs[j])
    for ax in axs.flat:
        ax.set_axis_off()
    fig.tight_layout()
    wandb.log({"network_input": wandb.Image(fig)}, step=dataset_size)
    # plt.savefig("logs/log_images/network_input.png")
    fig.clear()
    plt.close()

    fig, axs = plt.subplots(4, 1, figsize=(2, 8))
    for i, (primitive, distance) in enumerate(list(pairings)):
        axs[i].set_title(f"{primitive}_{distance} {visualizations[distance][primitive]['manipulation'][0].min():.2f},{visualizations[distance][primitive]['manipulation'][0].max():.2f}", fontsize=8)
        axs[i].imshow(visualizations[distance][primitive]['manipulation'][0], cmap='jet')
        axs[i].set_axis_off()
    fig.tight_layout()
    wandb.log({"network_output": wandb.Image(fig)}, step=dataset_size)
    # plt.savefig("logs/log_images/network_output.png")
    fig.clear()
    plt.close()

    #### NETWORK LOGGING ####
    for k,v in mean_update_stats.items():
        wandb.log({k:np.mean(v), "network_step": float(policy.train_steps.item())}, step=dataset_size)

    
    value_net.eval()
    print("[Network] << Optimized value network")



def nocs_pred_to_img(nocs_pred, obs, n_bins=32):
    # print("Obs shape: ", obs.shape)
    mask = torch.sum(obs[(4,5,6),], axis=0) > 0.01
    mask = mask.to(nocs_pred.device)

    nocs_x = (torch.argmax(nocs_pred[:, 0, :, :], dim=0).type(torch.float32))/(n_bins-1)
    nocs_y = (torch.argmax(nocs_pred[:, 1, :, :], dim=0).type(torch.float32))/(n_bins-1)

    nocs_x = nocs_x * mask.int() + (1 - mask.int()) * 0.0
    nocs_y = nocs_y * mask.int() + (1 - mask.int()) * 0.0

    return nocs_x, nocs_y

def parse_nocs_output(nocs_pred, obs, n_bins=32, n_orientations=2, nocs_indices=(0, 2), ignore_index=-1):
    
    nocs_pred = nocs_pred.view(nocs_pred.shape[0], n_bins, n_orientations, nocs_pred.shape[-2], nocs_pred.shape[-1])

    device = nocs_pred.device
    # nocs_pred_2 = out['deformable']['nocs']
    # print("NOCS PRED SHAPE: ", nocs_pred.shape)
    #b, 2, 32, 128, 128

    nocs_gt = obs[:, (4,5,6), :, :]
    nocs_gt = nocs_gt.to(device, non_blocking=True)
    mask = torch.sum(nocs_gt, dim=1, keepdim=True)< 1e-7
    mask = torch.repeat_interleave(mask, n_orientations, dim=1).to(device, non_blocking=True)
    #b, 2, 128, 128


    orientations = nocs_gt[:, nocs_indices, :, :]
    orientations[:, 0] = torch.abs(orientations[:, 0] - 0.5) * 2
    labels = torch.clip((orientations * n_bins).type(torch.int64),0, n_bins-1)
    labels[mask] = ignore_index

    return nocs_pred, labels

def get_predicted_nocs(img, orientation_net):
    with torch.no_grad():
        mask = torch.sum(img[:, (4,5,6),], axis=1, keepdim=True) > 0.01
        mask = torch.cat(2*[mask], dim=1).to(orientation_net.device)
        out = orientation_net(img[:, :3, :, :].to(orientation_net.device))
        n_bins = out.shape[0]

        nocs_x_bins = out[:, :, 0, :, :]
        nocs_x = torch.unsqueeze(torch.argmax(nocs_x_bins, dim=1).type(torch.float32)/(n_bins-1), 1)
        nocs_y_bins = out[:, :, 1, :, :]
        nocs_y = torch.unsqueeze(torch.argmax(nocs_y_bins, dim=1).type(torch.float32)/(n_bins-1), 1)
        nocs = torch.cat([nocs_x, nocs_y], dim=1)

        # #mask out bg
        nocs = nocs * mask.int() + (1 - mask.int()) * 0.0
        img = torch.cat([img[:, :-2], nocs.to(img.device), img[:, -2:]], dim=1)
    return img.detach()
