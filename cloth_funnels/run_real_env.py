
#%%

import torch
from torchmetrics import StatScores
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from real.utils import *
from argparse import ArgumentParser


import os
import sys
sys.path.append('/home/alper/folding-unfolding/src/PyFlex/bindings/build')
os.environ['PYFLEXROOT'] = '/home/alper/folding-unfolding/src/PyFlex'
os.environ['PYTHONPATH'] = '/home/alper/folding-unfolding/src/PyFlex/bindings/build'
os.environ['LD_LIBRARY_PATH'] = '/home/alper/folding-unfolding/src/PyFlex/external/SDL2-2.0.4/lib/x64'
os.chdir('/home/zhenjia/dev/folding-unfolding/src')
# from learning.nets import 
from utils import OrientationNet
from learning.nets import MaximumValuePolicy
from itertools import product

from real.setup import *
from learning.Memory import Memory

import time
import hashlib
import imageio
from tqdm import tqdm
from functools import partial
from cair_robot.envs.transformed_view_env import ImageStackTransformer
from real.realEnv import RealEnv

#%%
if __name__ == "__main__":

    parser = ArgumentParser("Dynamic Cloth Manipulation")
    parser.add_argument('--log', type=str, default='0309/buffer3')
    parser.add_argument('--orientation_network_path', type=str, default='./nocs_classification.ckpt')
    parser.add_argument('--orientation_network_device', type=str, default='cuda')
    parser.add_argument('--episode_length', type=int, default=4)
    parser.add_argument('--confirm_actions', action='store_true', default=False)
    parser.add_argument('--visualize_online', action='store_true', default=False)
    parser.add_argument('--mat_thickness', type=float, default=0.25)
    parser.add_argument('--left_picker_offset', type=float, default=0.00)

    parser.add_argument('--num_fling_rotations', type=int, default=17)
    parser.add_argument('--num_place_rotations', type=int, default=17)
    parser.add_argument('--scales', default=[1, 1.5, 2.0, 2.5, 3.0], nargs='+', type=float)
    parser.add_argument('--deformable_weight', type=float, default=0.7)
    parser.add_argument('--place_y', type=float, default=0.1)


    args = parser.parse_args()

    if not os.path.exists(args.log):
        os.makedirs(args.log)

    replay_buffer_path = args.log + '/replay_buffer.hdf5'

    input = 'rgb_pos'
    
    keypoint_model_path = './keypoint_detector/keypoint_model_dir'
    
    orn_net = OrientationNet('rgb_pos_nocs', args.orientation_network_path, device=args.orientation_network_device)
    env = RealEnv(replay_buffer_path=replay_buffer_path,
                        orn_net_handle=orn_net, 
                        episode_length=args.episode_length,
                        mat_thickness = args.mat_thickness,
                        left_picker_offset = args.left_picker_offset,
                        confirm_actions = args.confirm_actions,
                        visualize_online = args.visualize_online,
                        input = input,
                        deformable_weight=args.deformable_weight,
                        fling_only=False,
                        num_fling_rotations=args.num_fling_rotations,
                        num_place_rotations=args.num_place_rotations,
                        scales=args.scales,
                        place_y=args.place_y,
                        keypoint_model_path=keypoint_model_path)
                        
    
    checkpoint_path = "./manipulation_policies/pants_unfold_rgb.pth"
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    policy = MaximumValuePolicy(
        action_primitives=['place', 'fling'],
        num_rotations = 16,
        scale_factors = [1.0, 1.5, 2.0, 2.5, 3.0],
        obs_dim = 128,
        pix_grasp_dist = 16,
        pix_drag_dist = 16,
        pix_place_dist = 10,
        deformable_weight = args.deformable_weight,
        nocs_mode = "collapsed",
        network_gpus = [0, 0],
        action_expl_prob = 0,
        action_expl_decay = 0,
        value_expl_prob = 0,
        value_expl_decay = 0,
        dual_networks = None,
        input_channel_types = input,
        deformable_pos = True,
        dump_network_inputs=False,
    )
    policy.load_state_dict(ckpt['net'])
#%%
if __name__ == "__main__":
    try:
        state = env.reset()
        while env.terminate == False:
            # vmaps_dict = {'rigid': {'fling':None, 'place':None}, 'deformable': {'fling':None, 'place':None}}
            # masks_dict = {'fling':None, 'place':None}
            action_tuple = policy.get_action_single(state, explore=False)
            keypoints = state['keypoints']
            average_keypoint_confidence = keypoints[:, 2].mean()
            print("Average keypoint confidence: {}".format(average_keypoint_confidence))
            if False and ( action_tuple['best_value'] < 0.15 and average_keypoint_confidence > 0.9):
                print("Folding stage started")
               
                # print("Depth shape: {}".format(state['pretransform_depth'].shape))
            

                world_image_stack = ImageStackTransformer(
                    img_shape=(128, 128),
                    rotations=[0], scales=[1])
                world_camera = env.cameras['tx_table_camera']
                camera_view = env.cameras['tx_camera_view']
                intrinsic = env.kinect_camera.get_intr()
                # depth_map = state['pretransform_depth']
                world_coords = world_image_stack.get_world_coords_stack(
                                                        env.env.get_obs()[1],
                                                        camera_view, 
                                                        world_camera, 
                                                        intrinsic)
                scale_factor = 128/720
                # world_coords = TF.resize(torch.tensor(world_coords), 720).numpy()

                SHIRT_KEYPOINTS = {
                    'left_collar':0,
                    'right_collar':1,
                    'left_shoulder':2,
                    'right_shoulder':3,
                    'left_arm_top':4,
                    'right_arm_top':5,
                    'left_arm_bottom':6,
                    'right_arm_bottom':7,
                    'left_bottom_corner':8,
                    'right_bottom_corner':9,
                }
                keypoint_world_coords = \
                    [world_coords[0, int(keypoint[1] * scale_factor), int(keypoint[0] * scale_factor), :] for keypoint in keypoints]
                
                img = state['pretransform_observations']
                print("Img shape: {}".format(img.shape))
                image = np.ascontiguousarray(img).astype(np.uint8)
                mask_image = image.copy()

                for coordinate in keypoints:
                    x, y, z = coordinate
                    confidence = z
                    green = confidence * 255
                    red = (1-confidence) * 255  
                    cv2.circle(img, (int(x), int(y)), 5, (red, green, 0), -1)

                plt.imshow(img)
                plt.show()
                plt.savefig(f"./logs/log_images/keypoints_{np.random.randint(0, 100)}.png")
                plt.close()


                SHIRT_FOLD = [  
                                ['left_collar', 'left_collar'], 
                                ['left_shoulder', 'left_shoulder'],
                                ['left_arm_top', 'left_arm_top'], 
                                ['left_arm_bottom', 'left_arm_bottom'],
                                ['left_bottom_corner', 'left_bottom_corner'],
                                ['right_bottom_corner', 'right_bottom_corner'],
                                ['right_arm_bottom', 'right_arm_bottom'],
                                ['right_arm_top', 'right_arm_top'],
                                ['right_shoulder', 'right_shoulder'],
                                ['right_collar', 'right_collar'],
                            ]
                while True:
                    for begin_pt_name, end_pt_name in SHIRT_FOLD:
                        begin_pt = keypoint_world_coords[SHIRT_KEYPOINTS[begin_pt_name]]
                        end_pt = keypoint_world_coords[SHIRT_KEYPOINTS[end_pt_name]]
                        path = np.stack([begin_pt, end_pt])

                        is_left = None
                        right_valid = True
                        left_valid = True
                        if not(env.env.is_coord_valid_robot(path, is_left=True).all()):
                            left_valid = False
                        if not(env.env.is_coord_valid_robot(path, is_left=False).all()):
                            right_valid = False
                        if not left_valid and not right_valid:
                            print("No valid path")
                        else:
                            if left_valid:
                                print("Left valid")
                                is_left = True 
                            elif right_valid:
                                print("Right valid")
                                is_left = False

                        print("is_left=True", env.env.is_coord_valid_robot(path, is_left=True))
                        print("is_left=False", env.env.is_coord_valid_robot(path, is_left=False))
                        print("Is left: {}".format(is_left))
                        # r = input("ready?")
                        if is_left is not None:
                            env.env.pick_and_place(env.scene, is_left, begin_pt, end_pt)

                        print("GEtting new state")
                        state = env.get_transformed_obs()
                        keypoints = state['keypoints']
                        keypoint_world_coords = \
                            [world_coords[0, int(keypoint[1] * scale_factor), int(keypoint[0] * scale_factor), :] for keypoint in keypoints]
                        
                        ###VISUALIZE
                        img = state['pretransform_observations']
                        print("Img shape: {}".format(img.shape))
                        image = np.ascontiguousarray(img).astype(np.uint8)
                        mask_image = image.copy()

                        for coordinate in keypoints:
                            x, y, z = coordinate
                            confidence = z
                            green = confidence * 255
                            red = (1-confidence) * 255  
                            cv2.circle(img, (int(x), int(y)), 5, (red, green, 0), -1)

                        plt.imshow(img)
                        plt.show()
                        plt.savefig(f"./logs/log_images/keypoints_{np.random.randint(0, 1000)}.png")
                        plt.close()
                        ####VISUALIZE
                    # exit(1)
                exit(1)
                # right_valid_coords = env.is_coord_valid_robot(grid, is_left=False)


                print(keypoints)
                state = env.reset()
                continue
                # for distance in 'rigid', 'deformable':
                #     vmaps_dict[distance][primitive] = vmaps[distance][primitive]
                # masks_dict[primitive] = masks[primitive]
            state = env.step(action_tuple)
            # obs = env.step(obs)

    except Exception as e:
        env.scene.disconnect()
        raise(e)


# %%
