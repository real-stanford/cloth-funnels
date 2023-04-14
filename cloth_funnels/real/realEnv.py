
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


# from learning.nets import 
from utils import OrientationNet, visualize_value_pred
from learning.nets import Factorized_UNet, MaximumValuePolicy
from learning.utils import generate_coordinate_map
import skimage.transform as st
from itertools import product

from real.setup import *

from learning.Memory import Memory

import time
import hashlib
import imageio
from tqdm import tqdm
from functools import partial

from keypoint_detector.keypoint_inference import KeypointModel

class RealEnv():
    
    def __init__(self, orn_net_handle, 
                    replay_buffer_path,
                    wsg_port=1006, 
                    left_picker_offset=0.03, 
                    mat_thickness=0.045,
                    action_primitives=['place', 'fling'],
                    dump_visualizations=False,
                    episode_length=8,
                    confirm_actions=False,
                    visualize_online=False,
                    input='rgb_pos_nocs',
                    deformable_weight=0.7,
                    fling_only=False,
                    num_fling_rotations=17,
                    num_place_rotations=17,
                    scales=[1.0, 1.5, 2.0, 2.5, 3.0],
                    place_y=1.5,
                    keypoint_model_path=None,
                    observation_only=True,
                    **kwargs):

        self.env = None
        self.scene = None
        self.cameras = None
        self.confirm_actions = confirm_actions
        self.visualize_online = visualize_online
        self.deformable_weight = deformable_weight
        self.fling_only = fling_only

        self.orn_net_handle = orn_net_handle
        assert orn_net_handle is not None
        
        self.action_primitives = action_primitives

        self.replay_buffer_path = replay_buffer_path
        self.log_dir = os.path.dirname(self.replay_buffer_path)
        self.dump_visualizations = dump_visualizations
        self.recording = False

        self.episode_length = episode_length

        self.terminate = False

        self.input = input

        if self.dump_visualizations:
            print("Initializing cameras")
            self.top_cam = get_top_cam()
            self.front_cam = get_front_cam()
            print("Initializing environment")

        tx_table_camera = np.loadtxt('real/cam_pose/cam2table_pose.txt')
        tx_left_camera = np.loadtxt('real/cam_pose/cam2left_pose.txt')
        tx_right_camera = np.loadtxt('real/cam_pose/cam2right_pose.txt')
        tx_camera_view = np.loadtxt('real/cam_pose/view2cam.txt')


        camera = KinectClient('128.59.23.32', '8080', fielt_bg=False)
        self.kinect_camera = camera
        # scene = PyBulletDualArmTableScene(tx_table_camera, tx_left_camera, tx_right_camera)
        self.env  = TransformedViewEnv(kinect_camera=camera, 
            tx_camera_img=tx_camera_view, tx_world_camera=tx_table_camera,
            tx_left_camera=tx_left_camera, tx_right_camera=tx_right_camera, 
            mat_thickness=mat_thickness, left_picker_offset=left_picker_offset,
            pick_place_rotations=np.linspace(-np.pi, np.pi, num_place_rotations),
            fling_rotations=np.linspace(-np.pi, np.pi, num_fling_rotations),
            scales=scales,
            place_y=place_y)
        self.env.initialize_fling_primitive()

        self.cameras = {
            'tx_table_camera': tx_table_camera,
            'tx_left_camera': tx_left_camera,
            'tx_right_camera': tx_right_camera,
            'tx_camera_view': tx_camera_view
        }
        print("Initializing robots")

        self.init_robots()

        self.fling_action = partial(self.env.pick_and_fling_coord, self.scene)
        self.place_action = partial(self.env.pick_and_place_coord, self.scene)

        self.action_handlers = {
            'place': self.place_action,
            'fling': self.fling_action
        }

        self.recording_timer = 0

        self.keypoint_detector = KeypointModel(keypoint_model_path)
        self.observation_only = observation_only

    def reset(self):
        self.episode_memory = Memory()
        self.terminate = False 
        self.env_video_frames = {}
        self.scene.home(speed=0.5)
        self.current_timestep = 0

        self.transformed_obs_dict = self.get_transformed_obs()
        for k,v in self.transformed_obs_dict.items():
            self.episode_memory.add_value(k, v)

        return self.transformed_obs_dict


    def init_robots(self):
        print("WSG50 initializing")
        wsg50 = WSG50('192.168.0.231', 1005)
        print('WSG50 ready')
        left_ur5 = UR5RTDE('192.168.0.139', wsg50) # latte
        print('Latte ready')
        right_ur5 = UR5RTDE('192.168.0.204', 'rg2') # oolong
        print('Oolong ready')
        wsg50.home()
        wsg50.open()
        scene = DualArmTableScene(
            tx_table_camera=self.cameras['tx_table_camera'],
            tx_left_camera=self.cameras['tx_left_camera'],
            tx_right_camera=self.cameras['tx_right_camera'],
            left_robot=left_ur5,
            right_robot=right_ur5
        )
        scene.home(speed=0.5)
        self.scene = scene

    def get_nocs(self, color, obj_mask):
        print("Getting NOCS input")
        middle_dim = color.shape[1]//2 - color.shape[0]//2
        pretransform_observation = color[:, middle_dim:-middle_dim, :].transpose(2, 0, 1)
        pretransform_mask = obj_mask[:, 280:-280]

        pretransform_observation *= pretransform_mask

        old_pretransform_obs = pretransform_observation.copy()
        pretransform_observation = sharpen_edges(pretransform_observation, 0, 5)

        nocs_x, nocs_y = nocs_from_rgb(pretransform_observation, self.orn_net_handle)

        nocs = np.stack([nocs_x, nocs_y])
        return nocs

    def get_transformed_obs(self):
        print("Getting transformed observations")

        COORDINATE_MAP_NORMALIZER=1
        
        if self.observation_only:
            r = input("Ready for obs?")


        color, depth, obj_mask = self.env.get_obs()

        highres_transformer = ImageStackTransformer(
            img_shape=(720, 720),
            rotations=[0],
            scales=[1]
        )
        tx_camera_view = self.cameras['tx_camera_view']
        tx = np.eye(3)
        tx[:2, :2] *= (720/128) 
        highres_tx_camera_view = tx @ tx_camera_view

        highres_observations = highres_transformer.forward_raw(color, highres_tx_camera_view)[0]
        highres_obj_mask = highres_transformer.forward_raw(obj_mask, highres_tx_camera_view)[0]
        highres_obj_mask = np.expand_dims(highres_obj_mask, axis=-1)
        highres_depth = highres_transformer.forward_raw(depth, highres_tx_camera_view)[0]
    
        # data = env.transform_obs(highres_transformer, 
        #     depth=depth, obs=color, obj_mask=obj_mask)

        # pretransform_observation = color[:, middle_dim:-middle_dim, :].transpose(2, 0, 1)
        # pretransform_mask = obj_mask[:, middle_dim:-middle_dim]

        env_input = self.env.get_input()

        transformed_obs_dict = {
                                'fling_mask':torch.tensor(env_input['pick_and_fling']['is_valid']).bool(),
                                'place_mask':torch.tensor(env_input['pick_and_place']['is_valid']).bool(),
                                'pretransform_observations':highres_observations, 
                                'pretransform_mask':highres_obj_mask,
                                'pretransform_depth':highres_depth,
                                'pretransform_obj_mask':highres_obj_mask,
                                'fling_info':env_input['pick_and_fling']['info'],
                                'place_info':env_input['pick_and_place']['info'],
                                'keypoints':self.keypoint_detector.get_keypoints((highres_observations * highres_obj_mask).transpose(2, 0, 1)
                                                                                , mask=highres_obj_mask.squeeze(-1)),
                               }

        transformed_rgb = env_input['pick_and_place']['obs']

        transformed_rgb = transformed_rgb.transpose(0, 3, 1, 2).astype(np.float32)/255


        transformer = env_input[f'pick_and_place']['info']['transformer']
        transform_tuples = transformer.transform_tuples


        positional_encoding_input = np.stack(\
                        [generate_coordinate_map(128, -1*rotation*(360/(2*np.pi)), 1/scale)*COORDINATE_MAP_NORMALIZER for rotation, scale in transform_tuples])
        positional_encoding_input = positional_encoding_input.transpose(0, 3, 1, 2)

        extra_channels = torch.zeros(positional_encoding_input.shape[0], 1, positional_encoding_input.shape[2], positional_encoding_input.shape[3])
        transformed_obs = torch.tensor(np.concatenate([transformed_rgb, extra_channels, positional_encoding_input], axis=1)).float()
        transformed_obs_dict['transformed_obs'] = transformed_obs

        return transformed_obs_dict

    def execute_and_log_action(self, action):


        info = {}
        # if self.visualize_online:
        for primitive in self.action_primitives:
            info[primitive] = {}

            primitive_max_index = action[primitive]['max_index']
            # print(f"{primitive}_max_index", primitive_max_index)
            if primitive_max_index == None: 
                print(f"{primitive} not available") 
                continue

            primitive_observation = self.transformed_obs_dict['transformed_obs'][primitive_max_index[0]]
            primitive_observation_rgb = primitive_observation[:3, :, :]
            primitive_observation_pos = primitive_observation[(4,5),: :, :]
            rgb_img = np.array(primitive_observation_rgb).transpose(1, 2, 0)
            p = primitive_max_index[1:][::-1]
            if primitive == "fling":
                p1, p2 = p + np.array([0, 16]),  p + np.array([0, -16])
                rgb_img = draw_fling(rgb_img, p1, p2)
            elif primitive == "place":
                p1, p2 = p, p  + np.array([0, 10])
                rgb_img = draw_place(rgb_img, p1, p2)

            info = {
                'value_maps':action[primitive]['all_value_maps'].cpu(),
                'max_index':primitive_max_index,
                'max_value':action[primitive]['max_value'],
                'visualization':rgb_img
            }

            for key, value in info.items():
                self.episode_memory.add_value(
                    key=f'{primitive}_{key}',
                    value=value)

            # fig, ax = plt.subplots(1, 4, figsize=(20, 5))
            # ax[0].imshow(rgb_img)
            # ax[1].imshow(primitive_observation_pos[0])
            # ax[2].imshow(primitive_observation_pos[1])
            # ax[3].imshow(self.transformed_obs_dict[f"{primitive}_mask"][primitive_max_index[0]])
            # plt.show()

        primitive = action['best_primitive']
        max_value_index = action['best_index']
        max_value = action['best_value']
        masks = self.transformed_obs_dict[f"{primitive}_mask"]
        
        if self.confirm_actions:
            r = input('ready?')

        print("Executing action:", primitive)

        if not self.observation_only:
            self.action_handlers[primitive](max_value_index, \
                                                self.transformed_obs_dict[f"{primitive}_info"])

        x, y, z = max_value_index
        value_maps = action[primitive]['all_value_maps']
        chosen_value_map = value_maps[x, :, :]
        mask = masks[x, :, :]
        action_mask = np.zeros_like(chosen_value_map.cpu().numpy())
        action_mask[y, z] = 1

        action = {
            'observation': self.transformed_obs_dict['transformed_obs'][x],
            'action_primitive': primitive,
            'action_mask': action_mask,
            'value_map': chosen_value_map,
            'mask': mask,
            'all_value_maps': value_maps,
            'all_obs': self.transformed_obs_dict['transformed_obs'],
            'all_masks': masks,
            'info': None
        }

        self.episode_memory.add_observation(action['observation'])
        self.episode_memory.add_action(action['action_mask'])
        self.episode_memory.add_value(
            key='value_map',
            value=action['value_map'].cpu())
        self.episode_memory.add_value(
            key='action_mask',
            value=action['action_mask'])
        self.episode_memory.add_value(
            key='action_primitive',
            value=action['action_primitive'])


        if self.dump_visualizations:
            if action['all_value_maps'] is not None:
                self.episode_memory.add_value(
                    key='value_maps',
                    value=action['all_value_maps'])
            self.episode_memory.add_value(
                key='all_obs',
                value=action['all_obs'])

    def step(self, action):

        if self.dump_visualizations:
            self.start_recording()

        self.execute_and_log_action(action)

        if self.dump_visualizations:
            self.stop_recording()

        self.is_frame_empty = False
        self.current_timestep += 1
        self.terminate = self.terminate or self.current_timestep >= self.episode_length

        self.episode_memory.add_rewards_and_termination(
            0, self.terminate)

        if self.terminate:
            self.on_episode_end()
            exit(1)
            return self.reset()
        else:
            self.transformed_obs_dict = self.get_transformed_obs()
            for k,v in self.transformed_obs_dict.items():
                self.episode_memory.add_value(k, v)

        return self.transformed_obs_dict
      


    def on_episode_end(self, log=False):
        if self.dump_visualizations and len(self.episode_memory) > 0:
            while True:
                hashstring = hashlib.sha1()
                hashstring.update(str(time.time()).encode('utf-8'))
                vis_dir = self.log_dir + '/' + hashstring.hexdigest()[:10]
                if not os.path.exists(vis_dir):
                    break
            os.mkdir(vis_dir)
            for key, frames in self.env_video_frames.items():
                if len(frames) == 0:
                    continue
                path = f'{vis_dir}/{key}.mp4'
                with imageio.get_writer(path, mode='I', fps=24) as writer:
                    for frame in (
                        frames if not log
                            else tqdm(frames, desc=f'Dumping {key} frames')):
                        writer.append_data(frame)
            self.episode_memory.add_value(
                key='visualization_dir',
                value=vis_dir)
        self.env_video_frames.clear()
        #time to dump memory
        self.episode_memory.dump(
            self.replay_buffer_path)
        del self.episode_memory
        self.episode_memory = Memory()

    def start_recording(self):
        if self.recording:
            return
        self.recording = True
        self.recording_daemon = setup_thread(
            target=self.record_video_daemon_fn)

        self.recording_timer = time.time()

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.recording_daemon.join()
        self.recording_daemon = None

        # print("FPS:", len(self.env_video_frames['top']) / (time.time() - self.recording_timer))
        


    def record_video_daemon_fn(self):
        while self.recording:
            # NOTE: negative current_timestep's
            # act as error codes
            text = f'step {self.current_timestep}'
            if 'top' not in self.env_video_frames:
                self.env_video_frames['top'] = []
            if 'front' not in self.env_video_frames:
                self.env_video_frames['front'] = []

            top_view = get_workspace_crop(
                    self.top_cam.get_rgb())
            front_view = get_workspace_crop(
                    self.front_cam.get_rgb())

            # self.env_video_frames['front'].append(front_view)

            if not(len(self.env_video_frames['front']) and \
                (self.env_video_frames['front'][-1] == front_view).all()):
                    self.env_video_frames['front'].append(front_view)

            if not(len(self.env_video_frames['top']) and \
                (self.env_video_frames['top'][-1] == top_view).all()):
                    self.env_video_frames['top'].append(top_view)
            
            if len(self.env_video_frames['top']) > 50000:
                print("Robot stuck, terminating")
                exit()



        

if __name__ == '__main__':
    orn_net = OrientationNet('rgb_pos_nocs', './nocs_classification.ckpt', device='cpu')
    env = realEnv(replay_buffer_path="real_replay_buffer/buf.hdf5",
                     orn_net_handle=orn_net, 
                     episode_length=1,
                     mat_thickness_offset = 0.010)
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
        dump_network_inputs=False,
    )
    policy.load_state_dict(ckpt['net'])
    try:
        obs = env.reset()
        for _ in range(1):
            vmaps_dict = {'rigid': {'fling':None, 'place':None}, 'deformable': {'fling':None, 'place':None}}
            masks_dict = {'fling':None, 'place':None}
            for primitive in 'place', 'fling':
                action_tuple = policy.get_action_single()
                for distance in 'rigid', 'deformable':
                    vmaps_dict[distance][primitive] = vmaps[distance][primitive]
                masks_dict[primitive] = masks[primitive]
            obs = env.step(vmaps_dict, masks_dict)
            # obs = env.step(obs)

    except Exception as e:
        env.scene.disconnect()
        raise(e)

