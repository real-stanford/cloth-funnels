
import torch.nn as nn
import torch
from scipy import ndimage as nd
import cv2
from typing import List
import random
from time import time
import ray
import numpy as np
from cloth_funnels.learning.utils import generate_positional_encoding, shift_tensor
from cloth_funnels.learning.gui_utils import ExpertDemonstrationWindow
from cloth_funnels.learning.unet_parts import *
import imutils
import matplotlib.pyplot as plt 
import torchvision.transforms as transforms
import math

torch.inf = math.inf

class Factorized_UNet(nn.Module):
    def __init__(self, 
                n_channels, 
                bilinear=False, 
                action_primitives : List[str] = ['place', 'fling'] ,
                primitive_vmap_indices=None, 
                n_bins=1, 
                deformable_pos=True, 
                unfactorized_networks=False,
                coverage_reward=False,
                 **kwargs):
        super(Factorized_UNet, self).__init__()

        if unfactorized_networks:
            print("[Network] Initializing unfactorized network")
        else:
            print("[Network] Initializing factorized network")

        if not deformable_pos:
            print("[Network] Not giving deformable network positional encoding")
                # deformable network does not obtain positional encodings
            self.n_channels = {
                'rigid':n_channels,
                'deformable':n_channels-2
            }
        else:
            print("[Network] Giving deformable network positional encoding")
            self.n_channels = {
                'rigid':n_channels,
                'deformable':n_channels
            }

        if coverage_reward:
            print("[Network] Giving coverage reward")
            self.n_channels = {
                'rigid':n_channels - 2,
                'deformable':n_channels - 2
            }

        self.n_bins = n_bins

        self.action_primitives = action_primitives
        DISTANCES = ['rigid', 'deformable']

        self.DISTANCES = DISTANCES

        self.bilinear = bilinear

        FACTOR = 1/2
        factor = 2 if bilinear else 1

        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.nocs_decoders = nn.ModuleDict()

        self.kwargs = kwargs
        
        for distance in DISTANCES:
            self.decoders[distance] = nn.ModuleDict()
            for action_primitive in self.action_primitives:
                    if unfactorized_networks and distance != 'rigid':
                        self.encoders[distance] = self.encoders['rigid']
                        self.decoders[distance][action_primitive] = self.decoders['rigid'][action_primitive]
                        continue
                    #one encoder and nocs decoder per distance type
                    self.encoders[distance] = nn.ModuleList([
                        DoubleConv(self.n_channels[distance], int(64 * FACTOR)),
                        Down(int(64 * FACTOR), int(128 * FACTOR)),
                        Down(int(128 * FACTOR), int(256 * FACTOR)),
                        Down(int(256 * FACTOR), int(512 * FACTOR)),
                        Down(int(512 * FACTOR), int(1024 * FACTOR) // factor),
                    ])
                    self.decoders[distance][action_primitive] = nn.ModuleList([
                        Up(int(1024 * FACTOR), int(512 * FACTOR) // factor, bilinear),
                        Up(int(512 * FACTOR), int(256 * FACTOR) // factor, bilinear),
                        Up(int(256 * FACTOR), int(128 * FACTOR) // factor, bilinear),
                        Up(int(128 * FACTOR), int(64 * FACTOR), bilinear),
                        OutConv(int(FACTOR *64), n_bins)
                    ])

    def decode(self, encodings, decoder):
        encodings = encodings[::-1]
        cur = encodings[0]
        for i in range(1, len(encodings)):
            cur = decoder[i-1](cur, encodings[i])
        cur = decoder[-1](cur)
        return cur

    def encode(self, encoders, x, distance):
        encoding = x[:, :self.n_channels[distance]]
        encodings = []
        for encoder in encoders:
            encoding = encoder(encoding)
            encodings.append(encoding)
        return encodings

    def forward_for_optimize(self, x, action_primitive):


        # visualization
        # if self.kwargs['dump_network_inputs']:
        #     print("Dumping network inputs")
        #     obs = x[0]
        #     fig, axs = plt.subplots(1, obs.shape[0], figsize=(obs.shape[0], 1))
        #     for i in range(obs.shape[0]):
        #         axs[i].imshow(obs[i,:,:].clone().detach().cpu().numpy())
        #     plt.savefig(f"logs/log_images/network_input/{'_'.join(self.kwargs['log'].split('/'))}_input.png")
        ###

        out = {}
        for distance in self.DISTANCES:
            out[distance] = {}
            encodings = self.encode(self.encoders[distance], x, distance)
            for primitive in self.action_primitives:
                if primitive == action_primitive:
                    out[distance][primitive] =\
                         self.decode(encodings, self.decoders[distance][primitive])

        return out

    def forward(self, x, use_random_value=False):

        with torch.no_grad():
            out = {}
            for distance in self.DISTANCES:
                out[distance] = {}
                encodings = self.encode(self.encoders[distance], x, distance)
                for primitive in self.action_primitives:
                    if use_random_value:
                        out[distance][primitive] = torch.rand((x.shape[0], 1, x.shape[2], x.shape[3]), device=torch.device('cuda'))
                    else: 
                        out[distance][primitive] = self.decode(encodings, self.decoders[distance][primitive])
        return out
        
class SpatialValueNet(nn.Module):
    def __init__(self, input_channel_types=None,
                 steps=None, device='cuda', primitive_vmap_indices=None, **kwargs):
        super().__init__()

        self.device = device
        self.kwargs = kwargs

        assert(input_channel_types is not None)

        channel_ids_dict = { 
            'rgb': (0, 1, 2),
            'rgb_pos': (0, 1, 2, 4, 5),
            'rgb_pos_gtnocs': (0, 1, 2, 4, 5, 6, 7),
            'rgb_pos_nocs': (0, 1, 2, 4, 5, 6, 7),
            'rgb_pos_fullgtnocs': (0, 1, 2, 4, 5, 6, 7, 8),
        }

        self.mean = torch.tensor([0.5, 0.5, 0.5, 1.99, 0, 0, 0, 0, 0])
        self.std = torch.tensor([0.5, 0.5, 0.5, 0.006, 1, 1, 1, 1, 1])

        print(f"[Network] Initializing with inputs: {input_channel_types}")

        self.channels_tuple = channel_ids_dict[input_channel_types]
        self.n_input_channels = len(self.channels_tuple)
        self.net = self.setup_net()

    def setup_net(self):
        return Factorized_UNet(n_channels=self.n_input_channels, bilinear=True, **self.kwargs)

    def extract_obs(self, obs):
        assert len(obs.size()) == 4
        obs = obs[:, self.channels_tuple, :, :]
        return obs

    def preprocess_obs(self, obs):

        mean = self.mean[self.channels_tuple,].reshape(1, -1, 1, 1)
        std = self.std[self.channels_tuple,].reshape(1, -1, 1, 1)
        obs = self.extract_obs(obs)

        obs = (obs - mean.to(obs.device)) / std.to(obs.device)

        return obs.to(self.device)

    def forward(self, x, use_random_value):
        x = self.preprocess_obs(x)
        out = self.net(x, use_random_value)
        return out

    def forward_for_optimize(self, x, action_primitive, preprocess=True):
        if preprocess:
            x = self.preprocess_obs(x)
        out = self.net.forward_for_optimize(x, action_primitive)
        return out


class Policy:
    def __init__(self,
                 action_primitives: List[str],
                 num_rotations: int,
                 scale_factors: List[float],
                 obs_dim: int,
                 pix_grasp_dist: int,
                 pix_drag_dist: int,
                 pix_place_dist: int,
                 deformable_weight: float,
                 network_gpu: List[int],
                 **kwargs):
        assert len(action_primitives) > 0
        self.action_primitives = action_primitives
        self.scale_factors = scale_factors
        # self.num_transforms = len(self.rotations) * len(self.scale_factors)
        self.obs_dim = obs_dim
        self.pix_grasp_dist = pix_grasp_dist
        self.pix_drag_dist = pix_drag_dist
        self.pix_place_dist = pix_place_dist

        self.deformable_weight = deformable_weight
        self.network_gpus = network_gpu

        self.workspace_mask = None

    def set_workspace_mask(self, workspace_mask):
        self.workspace_mask = workspace_mask

    def  get_action_single(self, obs):
        raise NotImplementedError()

    def act(self, obs):
        return (self.get_action_single(o) for o in obs)


class MaximumValuePolicy(nn.Module, Policy):
    def __init__(self,
                 action_expl_prob: float,
                 action_expl_decay: float,
                 value_expl_prob: float,
                 value_expl_decay: float,
                 train_steps: int = 0,
                 device=None,
                 action_forward_size=4,
                 gpu=0,
                 grid_search_params=None,
                 dump_visualizations=False,
                 **kwargs):
        super().__init__()
        Policy.__init__(self, **kwargs)
        
        self.device = torch.device('cuda:' + str(gpu))

        self.action_expl_prob = nn.parameter.Parameter(
            torch.tensor(action_expl_prob).float(), requires_grad=False)
        self.action_expl_decay = nn.parameter.Parameter(
            torch.tensor(action_expl_decay).float(), requires_grad=False)
        self.value_expl_prob = nn.parameter.Parameter(
            torch.tensor(value_expl_prob).float(), requires_grad=False)
        self.value_expl_decay = nn.parameter.Parameter(
            torch.tensor(value_expl_decay).float(), requires_grad=False)
        self.train_steps = nn.parameter.Parameter(
            torch.tensor(train_steps).int(), requires_grad=False)
        # one value net per action primitive
        self.value_net = SpatialValueNet(
            device=self.device, num_primitives=len(self.action_primitives), **kwargs).to(self.device)

        self.should_explore_action = lambda: \
            self.action_expl_prob > random.random()
        self.should_explore_value = lambda: \
            self.value_expl_prob > random.random()

        self.pix_grasp_dist = kwargs['pix_grasp_dist']
        self.action_forward_size =  action_forward_size
        self.eval()

        #for grid search mode
        self.grid_search_params = grid_search_params
        self.dump_visualizations = dump_visualizations


    def decay_exploration(self, value_expl_halflife, action_expl_halflife, dataset_size, pts_per_update, is_eval=False):

        value_expl_decay = torch.tensor(np.exp(np.log(0.5)/(value_expl_halflife)))
        action_expl_decay = torch.tensor(np.exp(np.log(0.5)/(action_expl_halflife)))
        update_step = torch.tensor(int(dataset_size/pts_per_update) * pts_per_update)

        if not is_eval:
            self.action_expl_prob = torch.nn.Parameter(torch.pow(action_expl_decay, update_step), requires_grad=False)
            self.value_expl_prob = torch.nn.Parameter(torch.pow(value_expl_decay, update_step), requires_grad=False)
        else:
            self.action_expl_prob = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
            self.value_expl_prob = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

    # @profile
    def get_action_single(self, state, explore=True):

        if explore:
            use_random_action = self.should_explore_action()
            use_random_value = self.should_explore_value()
        else:
            use_random_action = False
            use_random_value = False

        if self.grid_search_params is not None:
            use_random_action = True
            use_random_value = True

        obs = state['transformed_obs']

        with torch.no_grad():
            value_maps = self.value_net(obs, use_random_value=use_random_value)

        primitive_retval = {}

        valid_action_dict = {}
        vmap_dict = {}

        for primitive in self.action_primitives:

            primitive_retval[primitive] = {}

            mask = state[f'{primitive}_mask']

            vmap = (1-self.deformable_weight) * value_maps['rigid'][primitive].squeeze(1) + \
                self.deformable_weight * value_maps['deformable'][primitive].squeeze(1)

            if self.dump_visualizations:
                raw_vmap = vmap.clone()
                primitive_retval[primitive]['raw_value_maps'] = raw_vmap

                        
            valid_actions = mask.sum(-1).sum(-1).bool()
            valid_action_dict[primitive] = valid_actions
            vmap_dict[primitive] = vmap

            if mask.any():
                vmap[~mask] = -torch.inf
                max_flat_index = torch.argmax(vmap).item()
                max_index = np.unravel_index(max_flat_index, vmap.shape)
                max_value = vmap[max_index]
            else:
                max_index = None 
                max_value = -torch.inf
        
            primitive_retval[primitive]['max_index'] = max_index
            primitive_retval[primitive]['max_value'] = max_value
            primitive_retval[primitive]['all_value_maps'] = vmap
            primitive_retval[primitive]['max_deformable_value'] = (value_maps['deformable'][primitive].squeeze(1))[max_index]
            primitive_retval[primitive]['max_rigid_value'] = (value_maps['rigid'][primitive].squeeze(1))[max_index]

        valid_primitives = [primitive for primitive in self.action_primitives if primitive_retval[primitive]['max_index'] is not None]
      
        if len(valid_primitives) == 0:
            print("[Policy] Error: no valid actions")
            return None

        if use_random_action:
            primitive_retval['random_action'] = np.random.choice(valid_primitives)
        else:
            primitive_retval['random_action'] = None

        if self.grid_search_params is not None:
            primitive_retval['random_action'] = self.grid_search_params['primitive']

        if use_random_value:
            for primitive in valid_primitives:
                try:
                    random_value_transform_idx = np.random.choice(valid_action_dict[primitive].nonzero().squeeze(1))
                    if self.grid_search_params is not None and primitive == self.grid_search_params['primitive']:
                        random_value_transform_idx = self.grid_search_params['vmap_idx']
                except Exception as e:
                    print("[Policy] Error: no valid actions", e)
                    return None
                chosen_vmap = vmap_dict[primitive][random_value_transform_idx].unsqueeze(0)
                random_index = np.unravel_index(torch.argmax(chosen_vmap).item(), chosen_vmap.shape)
                real_index = (random_value_transform_idx, random_index[1], random_index[2])
                primitive_retval[primitive]['max_index'] = real_index
                primitive_retval[primitive]['max_value'] = chosen_vmap[random_index]

        best_val = (-torch.inf, None, None)
        for primitive in self.action_primitives:
            if primitive_retval[primitive]['max_value'] > best_val[0]:
                best_val = (primitive_retval[primitive]['max_value'], primitive, primitive_retval[primitive]['max_index'])
        
        primitive_retval['best_value'] = best_val[0]
        primitive_retval['best_primitive'] = best_val[1]
        primitive_retval['best_index'] = best_val[2]
        primitive_retval['best_deformable_value'] = primitive_retval[best_val[1]]['max_deformable_value']
        primitive_retval['best_rigid_value'] = primitive_retval[best_val[1]]['max_rigid_value']

        return primitive_retval

    def get_action(self, states, explore=True):
        raise NotImplementedError()

    def steps(self):
        return sum([net.steps for net in self.value_nets.values()])

    def forward(self, obs):
        return self.act(obs)
    
    def act(self, obs, batch=False, explore=True):
        batch=False
        start = time()
        print("Starting policy.act()")
        # if batch:
        #     r = self.get_action(obs, explore=explore)
        # else:
        r = [self.get_action_single(o, explore=explore) for o in obs]
        end = time()
        print("[Policy] Forward took: ", end - start, "with #obs: ", len(obs))
        return r
        # return [self.get_action_single(o) for o in obs]


class ExpertGraspPolicy(Policy):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

        self.devices = [torch.device('cuda:' + str(self.network_gpu))]

        self.rotation_dict = {
            primitive: self.rotations[self.rotation_indices[primitive]] for primitive in self.action_primitives
        }

        self.value_net = SpatialValueNet(
            device=self.devices[0], num_primitives=len(self.action_primitives) **kwargs).to(self.devices[0])

    def get_action_single(self, obs):
        print("Obs shape", obs.shape)
        window = ExpertDemonstrationWindow(
            rotations=self.rotation_dict,
            scales=self.scale_factors,
            images=obs,
            pix_grasp_dist=self.pix_grasp_dist,
            pix_place_dist=self.pix_place_dist,
            action_primitives=self.action_primitives,)
        image_idx, center_grasp, left_grasp, right_grasp, scale_idx, rotation_idx, action_primitive = window.run()
        del window
        action_mask_dict = {primitive: torch.zeros(obs.shape[2:]) for primitive in self.action_primitives}
        x, y = center_grasp
        action_mask_dict[action_primitive][x, y] = 1
        num_scales = len(self.scale_factors)
        value_maps = \
            {distance:{primitive: \
                torch.zeros((obs.shape[0], \
                obs.shape[2], \
                obs.shape[3])) for primitive in self.action_primitives} for distance in ['rigid', 'deformable']}

        for distance in 'rigid','deformable':
            value_maps[distance][action_primitive][num_scales * rotation_idx + scale_idx + self.primitive_vmap_indices[action_primitive][0],:, :] = action_mask_dict[action_primitive]

        value_maps['random_action'] = None

        return value_maps, {key: self.get_mask(obs, key) for key in self.action_primitives}
