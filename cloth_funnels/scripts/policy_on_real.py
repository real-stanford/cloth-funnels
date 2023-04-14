

import numpy as np
from cair_robot.cameras.kinect_client import KinectClient
from cair_robot.scenes.dual_arm_table_scene import DualArmTableScene
from cair_robot.robots.ur5_robot import UR5RTDE
from cair_robot.robots.grippers import WSG50
from cair_robot.envs.transformed_view_env import TransformedViewEnv, ImageStackTransformer
import pickle
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from real.utils import *

import os
import sys
sys.path.append('/home/alper/folding-unfolding/src/PyFlex/bindings/build')
os.environ['PYFLEXROOT'] = '/home/alper/folding-unfolding/src/PyFlex'
os.environ['PYTHONPATH'] = '/home/alper/folding-unfolding/src/PyFlex/bindings/build'
os.environ['LD_LIBRARY_PATH'] = '/home/alper/folding-unfolding/src/PyFlex/external/SDL2-2.0.4/lib/x64'

# from learning.nets import 
from utils import OrientationNet
from learning.nets import Factorized_UNet, MaximumValuePolicy, generate_coordinate_map, transform
import skimage.transform as st
from itertools import product

def main():

    tx_table_camera = np.loadtxt('cam_pose/cam2table_pose.txt')
    tx_left_camera = np.loadtxt('cam_pose/cam2left_pose.txt')
    tx_right_camera = np.loadtxt('cam_pose/cam2right_pose.txt')
    tx_camera_view = np.loadtxt('cam_pose/view2cam.txt')

    camera = KinectClient('128.59.23.32', '8080', fielt_bg=False)
    # scene = PyBulletDualArmTableScene(tx_table_camera, tx_left_camera, tx_right_camera)
    env = TransformedViewEnv(kinect_camera=camera, 
        tx_camera_img=tx_camera_view, tx_world_camera=tx_table_camera,
        tx_left_camera=tx_left_camera, tx_right_camera=tx_right_camera)

    print("WSG50 initializing")
    wsg50 = WSG50('192.168.0.231', 1006)
    print('WSG50 ready')
    left_ur5 = UR5RTDE('192.168.0.139', wsg50) # latte
    print('Latte ready')
    right_ur5 = UR5RTDE('192.168.0.204', 'rg2') # oolong
    print('Oolong ready')
    wsg50.home()
    wsg50.open()
    scene = DualArmTableScene(
        tx_table_camera=tx_table_camera,
        tx_left_camera=tx_left_camera,
        tx_right_camera=tx_right_camera,
        left_robot=left_ur5,
        right_robot=right_ur5
    )
    scene.home(speed=0.5)

    env.initialize_fling_primitive()
    rs = np.random.RandomState(0)


    checkpoint_path = "./manipulation_policies/latest_ckpt.pth"
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    policy = MaximumValuePolicy(
        action_primitives=['place', 'fling'],
        num_rotations = 16,
        scale_factors = [1.0, 1.5, 2.0, 2.5, 3.0],
        obs_dim = 128,
        pix_grasp_dist = 16,
        pix_drag_dist = 16,
        pix_place_dist = 10,
        deformable_weight = 0.65,
        nocs_mode = "collapsed",
        network_gpus = [0, 0],
        action_expl_prob = 0,
        action_expl_decay = 0,
        value_expl_prob = 0,
        value_expl_decay = 0,
        dual_networks = None,
        input_channel_types = 'rgb_pos_nocs',
        deformable_pos = True,
    )
    policy.load_state_dict(ckpt['net'])
    orn_net = OrientationNet('./nocs_classification.ckpt', 'rgb_pos_nocs', device='cpu')
    #For obtaining NOCS
    COORDINATE_MAP_NORMALIZER=np.array([-0.5, 0.5]).reshape(1, 1, 2)
    try: 
        while True:
            
#%%

            color, depth, obj_mask = env.get_obs()
            middle_dim = color.shape[1]//2 - color.shape[0]//2
            pretransform_observation = color[:, middle_dim:-middle_dim, :].transpose(2, 0, 1)
            pretransform_mask = obj_mask[:, 280:-280]

            pretransform_observation *= pretransform_mask

            old_pretransform_obs = pretransform_observation.copy()
            pretransform_observation = sharpen_edges(pretransform_observation, 0, 5)

            # plt.imshow(pretransform_observation)
            # old_nocs_x, old_nocs_y = nocs_from_rgb(old_pretransform_obs, orn_net)
            nocs_x, nocs_y = nocs_from_rgb(pretransform_observation, orn_net)

            nocs = np.stack([nocs_x, nocs_y])
            env_input = env.get_input()
            
            # fig, axs = plt.subplots(2, 3, figsize=(12, 8))
            # axs[0, 0].imshow(old_pretransform_obs.transpose(1, 2, 0))
            # axs[0, 1].imshow(old_nocs_x)
            # axs[0, 2].imshow(old_nocs_y)

            # axs[1, 0].imshow(pretransform_observation.transpose(1, 2, 0))
            # axs[1, 1].imshow(nocs_x)
            # axs[1, 2].imshow(nocs_y)
            # plt.imshow(nocs_x)

            primitive_info = {"fling":{}, "place":{}}

            for primitive in ['fling', 'place']:

                transformer = env_input[f'pick_and_{primitive}']['info']['transformer']
                transform_tuples = transformer.transform_tuples
                valid_pixels = env_input[f'pick_and_{primitive}']['is_valid']

                if np.max(valid_pixels) == 0:
                    primitive_info[primitive]['value'] = -torch.inf
                    primitive_info[primitive]['index'] = (0, 0, 0)
                    primitive_info[primitive]['info'] =  env_input[f'pick_and_{primitive}']['info']
                    print("ERROR: No valid fling found")
                    continue
                    # raise RuntimeError(f"No valid {primitive} found")

                rgb_input = (env_input[f'pick_and_{primitive}']['obs'].astype(np.float32))/255
                transformed_nocs_input = transformer.forward_img((nocs.transpose(1, 2, 0)*255).astype(np.uint8))
                transformed_nocs_input = transformed_nocs_input.astype(np.float32)/255
                positional_encoding_input = np.stack(\
                    [generate_coordinate_map(128, -1*rotation*(360/(2*np.pi)), 1/scale)/COORDINATE_MAP_NORMALIZER for rotation, scale in transform_tuples])

                rgb_input = rgb_input.transpose(0, 3, 1, 2)
                transformed_nocs_input = transformed_nocs_input.transpose(0, 3, 1, 2)
                positional_encoding_input = positional_encoding_input.transpose(0, 3, 1, 2)

                extra_channels = np.zeros((rgb_input.shape[0], 4, rgb_input.shape[-2], rgb_input.shape[-1]))
                # policy_input = np.concatenate([rgb_input, extra_channels, transformed_nocs_input, positional_encoding_input], axis=1)
                policy_input = np.concatenate([rgb_input, extra_channels, positional_encoding_input], axis=1)

                
                visualize_input(policy_input, transform_tuples)
                
                value_maps, masks = policy.get_action_single(torch.tensor(policy_input).float())

                rigid_vmap = value_maps['rigid'][primitive].squeeze(1).detach().cpu().numpy()
                deformable_vmap = value_maps['deformable'][primitive].squeeze(1).detach().cpu().numpy()

                mask = masks[primitive].detach().cpu().numpy()

                DEFORMABLE_WEIGHT = 0.65
            
                sum_vmap = (1-DEFORMABLE_WEIGHT) * rigid_vmap + DEFORMABLE_WEIGHT * deformable_vmap
             
                # sum_vmap[~mask] = -np.inf
                sum_vmap[~valid_pixels] = -np.inf
                rigid_vmap[~valid_pixels] = -np.inf 
                deformable_vmap[~valid_pixels] = -np.inf

                visualize_vmaps(rigid_vmap, valid_pixels)
                visualize_vmaps(deformable_vmap, valid_pixels)
                visualize_vmaps(sum_vmap, valid_pixels)

                max_value_index = np.unravel_index(np.argmax(sum_vmap), shape=sum_vmap.shape)
                print(primitive, "picking index", max_value_index, " with value", sum_vmap[max_value_index])

                primitive_info[primitive]['value'] = sum_vmap[max_value_index]
                primitive_info[primitive]['index'] = max_value_index
                primitive_info[primitive]['info'] =  env_input[f'pick_and_{primitive}']['info']

                rgb_img = policy_input[max_value_index[0], :3, :, :].transpose(1, 2, 0)
                p = max_value_index[1:][::-1]
                if primitive == "fling":
                    p1, p2 = p + np.array([0, 16]),  p + np.array([0, -16])
                    rgb_img = draw_fling(rgb_img, p1, p2)
                elif primitive == "place":
                    p1, p2 = p, p  + np.array([0, 10])
                    rgb_img = draw_place(rgb_img, p1, p2)
                
                fig, axs = plt.subplots(1, 2)
                for ax in axs:
                    ax.set_axis_off()
                fig.tight_layout()
                axs[0].imshow(sum_vmap[max_value_index[0]], cmap='jet')
                axs[1].imshow(rgb_img)

#%% 
            
            if primitive_info['fling']['value'] > primitive_info['place']['value']:
                env.pick_and_fling_coord(scene, primitive_info['fling']['index'], primitive_info['fling']['info'])
                print("Using fling")
            else:
                print("Using pick and place")
                env.pick_and_place_coord(scene, primitive_info['place']['index'], primitive_info['place']['info'])

    except Exception as e:
        scene.disconnect()
        raise e 
        

if __name__ == '__main__':
    main()

# %%
