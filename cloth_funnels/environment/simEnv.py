
from ssl import ALERT_DESCRIPTION_UNSUPPORTED_EXTENSION
from threading import active_count
from cv2 import Mat_DEPTH_MASK, Mat_MAGIC_MASK

from .Memory import Memory
import numpy as np
import pathlib
from cloth_funnels.utils.env_utils import (
    generate_workspace_mask,
    preprocess_obs,
    blender_render_cloth,
    get_cloth_mesh,
    shirt_folding_heuristic,
    visualize_action,
    compute_pose,
    get_largest_component,
    pixels_to_3d_positions,
    pixel_to_3d,
    get_pointcloud,
    find_max_indices,
    shirt_keypoints)
from cloth_funnels.learning.utils import deformable_distance
import torch
from .exceptions import MoveJointsException
from cloth_funnels.learning.utils import prepare_image
from typing import List, Callable
from itertools import product
import math
from cloth_funnels.environment.flex_utils import (
    set_scene,
    get_image,
    get_current_covered_area,
    wait_until_stable,
    get_camera_matrix,
    PickerPickPlace)
from cloth_funnels.utils.env_utils import (
    generate_workspace_mask,
    generate_primitive_cloth_mask
)
from cloth_funnels.tasks.generate_tasks import Task
from tqdm import tqdm
import time
import hashlib
import imageio
import os
import pyflex
import cv2
import matplotlib.pyplot as plt
import pickle
import random
import pathlib
from memory_profiler import profile
import wandb
from cloth_funnels.keypoint_detector.keypoint_inference import KeypointDetector

def seed_all(seed):
    # print(f"SEEDING WITH {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SimEnv:
    def __init__(self,
                 replay_buffer_path: str,
                 obs_dim: int,
                 num_rotations: int,
                 scale_factors: List[float],
                 get_task_fn: Callable[[None], Task],
                 action_primitives: List[str],
                 pix_grasp_dist: int,
                 pix_drag_dist: int,
                 pix_place_dist: int,
                 stretchdrag_dist: float,
                 reach_distance_limit: float,
                 fixed_fling_height: float,
                 conservative_grasp_radius: int = 4,
                 use_adaptive_scaling=False,
                 dump_visualizations=False,
                 parallelize_prepare_image=False,
                 gui=False,
                 grasp_height=0.03,
                 fling_speed=8e-3,
                 episode_length=10,
                 render_dim=480,
                 particle_radius=0.00625,
                 render_engine='opengl',
                 alpha=0,
                 expert_demonstration=False,
                 deformable_weight=None,
                 adaptive_fling_momentum=1,
                 orn_net_handle = None,
                 nocs_mode = None,
                 input_channel_types = None,
                 dump_video_interval=29,
                 constant_positional_enc=False,
                 seed=0,
                 recreate_verts = None,
                 recreate_primitive = None,
                 grid_search_params = None,
                 fold_finish = False,
                 keypoint_model_path = None,
                 record_task_config = False,
                 fling_only=False,
                 place_only=False,
                 **kwargs):

        seed_all(seed)

        # environment state variables
        self.grasp_states = [False, False]
        self.ray_handle = None
        self.particle_radius = particle_radius
        self.replay_buffer_path = replay_buffer_path
        self.log_dir = os.path.dirname(self.replay_buffer_path)
        self.image_dim = render_dim  # what to render blender with
        self.obs_dim = obs_dim  # what to resize to fit in net
        self.episode_length = episode_length
        self.render_engine = render_engine
        self.alpha=alpha
        self.expert_demonstration = expert_demonstration
        self.action_primitives = action_primitives
        self.adaptive_fling_momentum = adaptive_fling_momentum
        self.nocs_mode = nocs_mode
        self.render_dim = render_dim
        self.constant_positional_enc = constant_positional_enc
        self.input_channel_types = input_channel_types
        self.fold_finish = fold_finish

        self.fling_only = fling_only
        self.place_only = place_only
        # print("Initializing SimEnv with alpha:", self.alpha)

        self.conservative_grasp_radius = conservative_grasp_radius
        #place -180 to 167.5
        #fling -90 to 90
        assert num_rotations % 4 == 0
        self.rotations = np.linspace(-180, 180, num_rotations + 1)
        # print("ALL ROTATIONS", self.rotations)
        self.rotation_indices = {
            'fling': np.where(np.logical_and(self.rotations >= -90, self.rotations <= 90)),
            'place': np.where(np.logical_and(self.rotations >= -180, self.rotations <= 167.5)),
        }

        self.orn_net_handle = orn_net_handle
        assert not(self.orn_net_handle is None and ('nocs' in input_channel_types and 'gt' not in input_channel_types))


        self.scale_factors = np.array(scale_factors)
        self.use_adaptive_scaling = use_adaptive_scaling
        self.adaptive_scale_factors = self.scale_factors.copy()

        # primitives parameters
        self.grasp_height = grasp_height
        self.pix_grasp_dist = pix_grasp_dist
        self.pix_drag_dist = pix_drag_dist
        self.pix_place_dist = pix_place_dist
        self.stretchdrag_dist = stretchdrag_dist
        self.fling_speed = fling_speed
        self.default_speed = 1e-2
        self.fixed_fling_height = fixed_fling_height

        self.deformable_weight = deformable_weight
        
        self.grid_search_params = grid_search_params
        self.recreate_primitive = recreate_primitive

        # visualizations
        self.dump_all_visualizations = dump_visualizations
        self.dump_visualizations = dump_visualizations

        self.parallelize_prepare_image = parallelize_prepare_image
        if gui:
            self.parallelize_prepare_image = True
        self.gui = gui
        self.gif_freq = 24
        self.env_video_frames = {}

        self.env_end_effector_positions = []
        self.env_mesh_vertices = []



        # physical limit of dual arm system
        self.TABLE_WIDTH = 0.765 * 2
        self.left_arm_base = np.array([0.765, 0, 0])
        self.right_arm_base = np.array([-0.765, 0, 0])
        self.reach_distance_limit = reach_distance_limit

        pix_radius = int(render_dim * (reach_distance_limit/self.TABLE_WIDTH))

        left_arm_reach = np.zeros((render_dim, render_dim))
        left_arm_reach = cv2.circle(left_arm_reach, (render_dim//2, 0), pix_radius, (255, 255, 255), -1)

        right_arm_reach = np.zeros((render_dim, render_dim))
        right_arm_reach = cv2.circle(right_arm_reach, (render_dim//2, render_dim), pix_radius, (255, 255, 255), -1)

        self.left_arm_mask = torch.tensor(left_arm_reach).bool()
        self.right_arm_mask = torch.tensor(right_arm_reach).bool()

        self.recreate_verts = recreate_verts
        self.record_task_config = record_task_config
    
        # tasks
        self.current_task = None
        self.tri_v, self.tri_f = None, None
        self.get_task_fn = get_task_fn
        self.gui_render_freq = 2
        self.gui_step = 0
        self.setup_env()
        self.action_handlers = {
            'fling': self.pick_and_fling_primitive,
            'stretchdrag': self.pick_stretch_drag_primitive,
            'drag': self.pick_and_drag_primitive,
            'place': self.pick_and_place_primitive
        }

        self.num_particles = 0
        self.id = None
        self.is_frame_empty = False

        self.temp_deformable_data = None
        self.init_deformable_data = None

        self.cloth_mask = None

        self.dump_video_interval = dump_video_interval
        self.dump_video_counter = np.random.randint(0, self.dump_video_interval)

        if self.dump_visualizations:
            self.dump_video_interval = 1

        # rotation angle in degrees, counter-clockwise
        self.rotations = np.linspace(-180, 180, num_rotations + 1)
        self.num_rotations = len(self.rotations)
        self.rotation_indices = {
            'fling': np.where(np.logical_and(self.rotations >= -90, self.rotations <= 90))[0],
            'place': np.where(np.logical_and(self.rotations >= -180, self.rotations <= 167.5))[0],
        }
        self.primitive_vmap_indices = {}
        for primitive, indices in self.rotation_indices.items():
            self.primitive_vmap_indices[primitive] = [None, None]
            self.primitive_vmap_indices[primitive][0] = indices[0] * len(scale_factors)
            self.primitive_vmap_indices[primitive][1] = (indices[-1]+1) * len(scale_factors)
        # print("[SimEnv] Primitive vmap indices:", self.primitive_vmap_indices)  

        if keypoint_model_path is not None and self.fold_finish:
            self.keypoint_detector = KeypointDetector(keypoint_model_path)

    def step_simulation(self):

        if self.record_task_config:
            self.env_end_effector_positions.append(self.action_tool._get_pos()[0])
            self.env_mesh_vertices.append(pyflex.get_positions().reshape((-1, 4))[:self.num_particles, :3])

        pyflex.step()
        self.gui_step += 1

    def setup_env(self):
        pyflex.init(
            not self.gui,  # headless: bool.
            True,  # render: bool
            480, # camera dimensions: int x int
            480,
            0 #msaa samples, hardcoded
            )  
        self.action_tool = PickerPickPlace(
            num_picker=2,
            particle_radius=self.particle_radius,
            picker_radius=self.grasp_height,
            picker_low=(-5, 0, -5),
            picker_high=(5, 5, 5))

    def get_transformations(self, rotations):
        return list(product(
            rotations, self.adaptive_scale_factors))

    def stretch_cloth_regular(self,
                      grasp_dist: float,
                      fling_height: float = 0.7,
                      max_grasp_dist: float = 0.7,
                      increment_step=0.02):
        # keep stretching until cloth is tight
        # i.e.: the midpoint of the grasped region
        # stops moving
        left, right = self.action_tool._get_pos()[0]
        left[1] = fling_height
        right[1] = fling_height
        midpoint = (left + right)/2
        direction = left - right
        direction = direction/np.linalg.norm(direction)
        self.movep([left, right], speed=8e-4, min_steps=20)
        stable_steps = 0
        cloth_midpoint = 1e2
        while True:
            positions = pyflex.get_positions().reshape((-1, 4))[:, :3]
            # get midpoints
            high_positions = positions[positions[:, 1] > fling_height-0.1, ...]
            if (high_positions[:, 0] < 0).all() or \
                    (high_positions[:, 0] > 0).all():
                # single grasp
                return grasp_dist
            positions = [p for p in positions]
            positions.sort(
                key=lambda pos: np.linalg.norm(pos[[0, 2]]-midpoint[[0, 2]]))
            new_cloth_midpoint = positions[0]
            stable = np.linalg.norm(
                new_cloth_midpoint - cloth_midpoint) < 3e-2
            if stable:
                stable_steps += 1
            else:
                stable_steps = 0
            stretched = stable_steps > 2
            if stretched:
                return grasp_dist
            cloth_midpoint = new_cloth_midpoint
            grasp_dist += increment_step
            left = midpoint + direction*grasp_dist/2
            right = midpoint - direction*grasp_dist/2
            self.movep([left, right], speed=5e-4)
            if grasp_dist > max_grasp_dist:
                return max_grasp_dist

    def stretch_cloth(self, grasp_dist: float, fling_height: float = 0.7, max_grasp_dist: float = 0.5, increment_step=0.02):
            # Option1: get GT init position
            picked_particles = self.action_tool.picked_particles
            # try:
            #     grasp_dist = igl.exact_geodesic(v=self.tri_v, f=self.tri_f, vs=np.array([picked_particles[0]]), vt=np.array([picked_particles[1]]))
            # except:
            #     print(">>> Error in exact_geodesic")
            return self.stretch_cloth_regular(grasp_dist, fling_height, max_grasp_dist, increment_step)
            # print(picked_particles[0], picked_particles[1])
            hack_scale = 1
            grasp_dist_scaling = 1
            grasp_dist *= grasp_dist_scaling * hack_scale
            
            grasp_dist = min(grasp_dist, max_grasp_dist)

            left, right = self.action_tool._get_pos()[0]
            pre_left, pre_right = left, right
            left[1] = fling_height
            right[1] = fling_height
            midpoint = (left + right) / 2
            direction = left - right
            direction = direction/np.linalg.norm(direction)
            left = midpoint + direction * grasp_dist/2
            right = midpoint - direction * grasp_dist/2
            self.movep([left , right ], speed=2e-3)
            return grasp_dist

    def lift_cloth(self,
                   grasp_dist: float,
                   fling_height: float = 0.7,
                   increment_step: float = 0.03,
                   max_height=0.7,
                   height_offset : float = 0.1):
        while True:
            positions = pyflex.get_positions().reshape((-1, 4))[:, :3]
            heights = positions[:, 1][:self.num_particles]

            if heights.min() > height_offset + 0.05:
                fling_height -= increment_step
            elif heights.min() < height_offset - 0.05:
                fling_height += increment_step 

            self.movep([[grasp_dist/2, fling_height, -0.3],
                        [-grasp_dist/2, fling_height, -0.3]], speed=1e-3)

            return fling_height

    def check_action(self, action_primitive, pixels,
                     transformed_depth, transformed_rgb,
                     scale, rotation,
                     value_map=None, all_value_maps=None,
                     **kwargs):
        args = {
            'pretransform_depth': self.pretransform_depth.copy(),
            'pretransform_rgb': self.pretransform_rgb.copy(),
            'transformed_depth': transformed_depth,
            'transformed_rgb': transformed_rgb,
            'scale': scale,
            'rotation': rotation
        }
        retval = pixels_to_3d_positions(
            transform_pixels=pixels,
            pose_matrix=compute_pose(
                pos=[0, 2, 0],
                lookat=[0, 0, 0],
                up=[0, 0, 1]), **args)

        def get_action_visualization():
            return visualize_action(
                action_primitive=action_primitive,
                transformed_pixels=pixels,
                pretransform_pixels=retval['pretransform_pixels'],
                value_map=value_map,
                all_value_maps=all_value_maps,
                mask=kwargs['mask'],
                **args)

        retval.update({
            'get_action_visualization_fn': get_action_visualization
        })

        # cloth_mask = self.cloth_mask
        # pix_1, pix_2 = retval['pretransform_pixels']

        # if action_primitive in ['fling','drag','stretchdrag']:
        #     if self.conservative_grasp_radius > 0:

        #         grasp_mask_1 = np.zeros(cloth_mask.shape)
        #         grasp_mask_1 = cv2.circle(
        #             img=grasp_mask_1,
        #             center=(pix_1[1], pix_1[0]),
        #             radius=self.conservative_grasp_radius,
        #             color=1, thickness=-1).astype(bool)

        #         grasp_mask_2 = np.zeros(cloth_mask.shape)
        #         grasp_mask_2 = cv2.circle(
        #             img=grasp_mask_2,
        #             center=(pix_2[1], pix_2[0]),
        #             radius=self.conservative_grasp_radius,
        #             color=1, thickness=-1).astype(bool)

        #         # p1_grasp_cloth = cloth_mask[grasp_mask_1].all()
        #         # p2_grasp_cloth = cloth_mask[grasp_mask_2].all()
    
        #         retval.update({
        #             'p1_grasp_cloth': cloth_mask[grasp_mask_1].all(),
        #             'p2_grasp_cloth': cloth_mask[grasp_mask_2].all(),
        #         })
        #     else:
        #         retval.update({
        #             'p1_grasp_cloth': True,
        #             'p2_grasp_cloth': True,
        #         })

        # elif action_primitive == 'place':
        #     midpoint = (pix_1).astype(int)
        #     grasp_mask = np.zeros(cloth_mask.shape)
        #     grasp_mask = cv2.circle(
        #         img=grasp_mask,
        #         center=(midpoint[1], midpoint[0]),
        #         radius=self.conservative_grasp_radius,
        #         color=1, thickness=-1).astype(bool)

        #     retval.update({
        #         'p1_grasp_cloth': cloth_mask[grasp_mask].all(),
        #         'p2_grasp_cloth': True,
        #     })
        # else:
        #     raise NotImplementedError


        # TODO can probably refactor so args to primitives have better variable names
        return retval

    def fling_primitive(self, dist, fling_height, fling_speed, cloth_height):
    
        x = cloth_height/2

        x_release = x * 0.9 * self.adaptive_fling_momentum
        x_drag = x * self.adaptive_fling_momentum
        # fling
        self.movep([[dist/2, fling_height, -x],
                    [-dist/2, fling_height, -x]], speed=fling_speed)
        self.movep([[dist/2, fling_height, x],
                    [-dist/2, fling_height, x]], speed=fling_speed)
        # self.movep([[dist/2, fling_height, x],
        #             [-dist/2, fling_height, x]], speed=1e-2, min_steps=4)
        # lower
        self.movep([[dist/2, self.grasp_height*2, -x_release],
                    [-dist/2, self.grasp_height*2, -x_release]], speed=1e-2)
        self.movep([[dist/2, self.grasp_height*2, -x_drag],
                    [-dist/2, self.grasp_height*2, -x_drag]], speed=5e-3)
        # release
        self.set_grasp(False)
        if self.dump_visualizations:
            self.movep(
                [[dist/2, self.grasp_height*2, -x_drag],
                 [-dist/2, self.grasp_height*2, -x_drag]], min_steps=10)
        self.reset_end_effectors()

    def pick_and_fling_primitive(
            self, p1, p2):

        left_grasp_pos, right_grasp_pos = p1, p2

        left_grasp_pos[1] += self.grasp_height 
        right_grasp_pos[1] += self.grasp_height

        # grasp distance
        dist = np.linalg.norm(
            np.array(left_grasp_pos) - np.array(right_grasp_pos))
        
        APPROACH_HEIGHT = 0.3
        pre_left_grasp_pos = (left_grasp_pos[0], APPROACH_HEIGHT, left_grasp_pos[2])
        pre_right_grasp_pos = (right_grasp_pos[0], APPROACH_HEIGHT, right_grasp_pos[2])
        
        #approach from the top (to prevent collisions)
        self.movep([pre_left_grasp_pos, pre_right_grasp_pos], speed=0.06)
        self.movep([left_grasp_pos, right_grasp_pos], speed=0.005)

        # only grasp points on cloth
        self.grasp_states = [True, True]
        # if self.dump_visualizations:
        #     self.movep([left_grasp_pos, right_grasp_pos], min_steps=10)

        PRE_FLING_HEIGHT = 0.7
        #lift up cloth
        self.movep([[left_grasp_pos[0], PRE_FLING_HEIGHT, left_grasp_pos[2]],\
             [right_grasp_pos[0], PRE_FLING_HEIGHT, right_grasp_pos[2]]], speed=9e-3)
        # lift to prefling
        self.movep([[dist/2, PRE_FLING_HEIGHT, -0.3], \
            [-dist/2, PRE_FLING_HEIGHT, -0.3]], speed=6e-3)

        # if not self.is_cloth_grasped():
        #     self.terminate = True
        #     return

        # if self.fixed_fling_height == -1:
        fling_height = self.lift_cloth(
            grasp_dist=dist, fling_height=PRE_FLING_HEIGHT)
        # else:
        #     fling_height = self.fixed_fling_height

        dist = self.stretch_cloth(grasp_dist=dist, fling_height=fling_height)

        wait_until_stable(100, tolerance=0.005)

        positions = pyflex.get_positions().reshape((-1, 4))[:, :3]
        heights = positions[:self.num_particles][:, 1]
        cloth_height = heights.max() - heights.min()

        self.fling_primitive(
            dist=dist,
            fling_height=fling_height,
            fling_speed=self.fling_speed,
            cloth_height=cloth_height,
            )

    def pick_and_drag_primitive(
            self, p1, p2):

        left_grasp_pos, right_grasp_pos = p1, p2

        orthogonal_vec = np.cross(p1-p2, [0, 1, 0])
        orthogonal_vec = orthogonal_vec / np.linalg.norm(orthogonal_vec)
        orthogonal_vec *= 0.2
        #rotate between vec 90 degrees


        left_grasp_pos[1] += self.grasp_height
        right_grasp_pos[1] += self.grasp_height

        # grasp distance
        dist = np.linalg.norm(
            np.array(left_grasp_pos) - np.array(right_grasp_pos))
        
        APPROACH_HEIGHT = 0.3
        pre_left_grasp_pos = (left_grasp_pos[0], APPROACH_HEIGHT, left_grasp_pos[2])
        pre_right_grasp_pos = (right_grasp_pos[0], APPROACH_HEIGHT, right_grasp_pos[2])
        
        #approach from the top (to prevent collisions)
        self.movep([pre_left_grasp_pos, pre_right_grasp_pos], speed=0.015)
        self.movep([left_grasp_pos, right_grasp_pos], speed=0.005)

        # only grasp points on cloth
        # self.grasp_states = [p1_grasp_cloth, p2_grasp_cloth]

        self.movep([left_grasp_pos + orthogonal_vec, right_grasp_pos + orthogonal_vec], speed=0.005)
        
        self.set_grasp(False)


        self.reset_end_effectors()

    def pick_and_place_primitive(
        self, p1, p2, lift_height=0.25):
        # prepare primitive params
        pick_pos, place_pos = p1.copy(), p2.copy()
        pick_pos[1] += self.grasp_height
        place_pos[1] += self.grasp_height + 0.05

        prepick_pos = pick_pos.copy()
        prepick_pos[1] = lift_height
        preplace_pos = place_pos.copy()
        preplace_pos[1] = lift_height

        # execute action
        self.movep([prepick_pos, [-0.2, 0.5, -0.5]], speed=8e-2)
        self.movep([pick_pos, [-0.2, 0.3, -0.5]], speed=6e-3)
        self.set_grasp(True)
        self.movep([prepick_pos, [-0.2, 0.3, -0.5]], speed=1e-2)
        self.movep([preplace_pos, [-0.2, 0.3, -0.5]], speed=2e-3)
        self.movep([place_pos, [-0.2, 0.3, -0.5]], speed=2e-3)
        self.set_grasp(False)
        self.movep([preplace_pos, [-0.2, 0.3, -0.5]], speed=8e-3)
        self.reset_end_effectors()

    def double_arm_pick_and_place_primitive(
        self, right_p1, right_p2, left_p1, left_p2, lift_height=0.25):

        left_pick_pos, left_place_pos = left_p1.copy(), left_p2.copy()
        left_pick_pos += self.grasp_height
        left_place_pos += self.grasp_height + 0.05

        right_pick_pos, right_place_pos = right_p1.copy(), right_p2.copy()
        right_pick_pos += self.grasp_height
        right_place_pos += self.grasp_height + 0.05

        left_prepick_pos = left_pick_pos.copy()
        left_prepick_pos[1] = lift_height
        left_preplace_pos = left_place_pos.copy()
        left_preplace_pos[1] = lift_height

        right_prepick_pos = right_pick_pos.copy()
        right_prepick_pos[1] = lift_height
        right_preplace_pos = right_place_pos.copy()
        right_preplace_pos[1] = lift_height

        # execute action
        self.movep([left_prepick_pos, right_prepick_pos], speed=8e-2)
        self.movep([left_pick_pos, right_pick_pos], speed=6e-3)
        self.set_grasp(True)
        self.movep([left_prepick_pos, right_prepick_pos], speed=1e-2)
        self.movep([left_preplace_pos, right_preplace_pos], speed=2e-3)
        self.movep([left_place_pos, right_place_pos], speed=2e-3)
        self.set_grasp(False)
        self.movep([left_preplace_pos, right_preplace_pos], speed=8e-3)
        self.reset_end_effectors()

    def pick_stretch_drag_primitive(
            self, p1, p2,
            p1_grasp_cloth: bool,
            p2_grasp_cloth: bool):
        if not (p1_grasp_cloth or p2_grasp_cloth):
            # both points not on cloth
            return
        left_start_drag_pos, right_start_drag_pos = p1, p2
        left_start_drag_pos[1] = self.grasp_height
        right_start_drag_pos[1] = self.grasp_height

        left_prestart_drag_pos = left_start_drag_pos.copy()
        left_prestart_drag_pos[1] = 0.3
        right_prestart_drag_pos = right_start_drag_pos.copy()
        right_prestart_drag_pos[1] = 0.3

        self.movep([left_prestart_drag_pos, right_prestart_drag_pos])
        self.movep([left_start_drag_pos, right_start_drag_pos], speed=2e-3)
        # only grasp points on cloth
        self.set_grasp([p1_grasp_cloth,
                        p2_grasp_cloth])
        if self.dump_visualizations:
            self.movep(
                [left_start_drag_pos, right_start_drag_pos], min_steps=10)

        # grasp distance
        dist = np.linalg.norm(
            np.array(left_start_drag_pos) - np.array(right_start_drag_pos))

        self.grasp_states = [p1_grasp_cloth, p2_grasp_cloth]

        self.set_grasp(True)

        # stretch if cloth is grasped by both
        # if all(self.grasp_states):
        #     dist = self.stretch_cloth(
        #         grasp_dist=dist, fling_height=self.grasp_height)

        # compute drag direction
        drag_direction = np.cross(
            left_start_drag_pos - right_start_drag_pos, np.array([0, 1, 0]))
        drag_direction = self.stretchdrag_dist * \
            drag_direction / np.linalg.norm(drag_direction)
        left_start_drag_pos, right_start_drag_pos = \
            self.action_tool._get_pos()[0]
        left_end_drag_pos = left_start_drag_pos + drag_direction
        right_end_drag_pos = right_start_drag_pos + drag_direction
        # prevent ee go under cloth
        left_end_drag_pos[1] += 0.1
        right_end_drag_pos[1] += 0.1

        left_postend_drag_pos = left_end_drag_pos.copy()
        left_postend_drag_pos[1] = 0.3
        right_postend_drag_pos = right_end_drag_pos.copy()
        right_postend_drag_pos[1] = 0.3

        self.movep([left_end_drag_pos, right_end_drag_pos], speed=2e-3)
        self.set_grasp(False)
        self.movep([left_postend_drag_pos, right_postend_drag_pos])
        self.reset_end_effectors()

    def compute_coverage(self):
        return get_current_covered_area(self.num_particles, self.particle_radius)

    def compute_percent_coverage(self):
        return self.compute_coverage()/self.current_task.get_config()["flatten_area"]

    def normalize_correspondence(self, correspondence):
        return correspondence

    def set_particle_pos(self, pos):
        current_pos = pyflex.get_positions().reshape((-1, 4))
        current_pos[:self.num_particles, :3] = pos
        #flatten current pos and put it back to pyflex
        current_pos = current_pos.reshape((-1,))
        pyflex.set_positions(current_pos)


    def deformable_reward(self, log_prefix=None):
        self.episode_memory.add_value(
            key='init_verts', value=self.init_pos
        )
        self.current_pos = pyflex.get_positions().reshape((-1, 4))[:self.num_particles, :3]
        self.episode_memory.add_value(
            key=f"{log_prefix}_verts", value=self.current_pos)

        weighted_distance, l2_distance, icp_distance, real_l2_distance, _ = \
            deformable_distance(self.init_pos, self.current_pos, self.current_task.get_config()["cloth_area"], self.deformable_weight)

        task_image_dict = self.current_task.get_images()

        init_img = task_image_dict["init_rgb"]
        curr_img = np.array(self.pretransform_rgb).transpose(2, 0, 1)[:3]
   
        try:
            init_mask = self.get_cloth_mask(init_img.transpose(2, 0, 1))
        except:
            print("[SimEnv] Init mask failed, using curr_img.")
            init_mask = self.get_cloth_mask(curr_img)
            
        curr_mask = self.get_cloth_mask(curr_img)

        intersection = np.logical_and(curr_mask, init_mask)
        union = np.logical_or(curr_mask, init_mask)
        iou = np.sum(intersection) / np.sum(union)
        coverage = np.sum(curr_mask) / np.sum(init_mask)

        deformable_dict = {
            "weighted_distance": weighted_distance,
            "l2_distance": l2_distance,
            "icp_distance": icp_distance,
            "pointwise_distance": real_l2_distance,
            "iou": iou,
            "coverage": coverage
        }

        self.episode_memory.add_value(
                        log_prefix + "_init_mask", init_mask)
        self.episode_memory.add_value(
                        log_prefix + "_curr_mask", curr_mask)
        
        if log_prefix is not None:
            for k, v in deformable_dict.items():
                if type(v) == float or type(v) == int \
                    or type(v) == np.float64 or type(v) == np.float32:
                    self.episode_memory.add_value(
                        log_prefix + "_" + k, float(v))
                elif type(v) == np.ndarray:
                    self.episode_memory.add_value(
                        log_prefix + "_" + k, v)

        if self.init_deformable_data is None:
            self.init_deformable_data = deformable_dict

        for k, v in self.init_deformable_data.items():
            if type(v) == float or type(v) == int or \
                type(v) == np.float64 or type(v) == np.float32:
                self.episode_memory.add_value(
                    "init_" + k, float(v))

        # print("Coverage in memory", self.episode_memory.data['init_coverage'])
        # print("IoU in memory", self.episode_memory.data['init_iou'])


        return l2_distance, icp_distance, weighted_distance

    
    def log_step_stats(self, action):
        # print("adding observation", action['observation'])
        self.episode_memory.add_observation(action['observation'])
        self.episode_memory.add_action(action['action_mask'])
        self.episode_memory.add_value(key='mask', value=action['mask'])
        self.episode_memory.add_value(key='workspace_mask', value=action['workspace_mask'])
        self.episode_memory.add_value(key='max_indices', value=action['max_indices'])

        self.episode_memory.add_value(key='predicted_deformable_value', value=action['predicted_deformable_value'])
        self.episode_memory.add_value(key='predicted_rigid_value', value=action['predicted_rigid_value'])
        self.episode_memory.add_value(key='predicted_weighted_value', value=action['predicted_weighted_value'])
        self.episode_memory.add_value(key='task_pkl_path', value=str(self.current_task.pkl_path))

        self.episode_memory.add_value(
            key='action_visualization',
            value=action['action_visualization'])
        self.episode_memory.add_value(
            key='rotation', value=float(action['rotation']))
        self.episode_memory.add_value(
            key='scale', value=float(action['scale']))
        self.episode_memory.add_value(
            'nonadaptive_scale', value=float(action['nonadaptive_scale']))
        self.episode_memory.add_value(
            key='value_map',
            value=action['value_map'].cpu())
        self.episode_memory.add_value(
            key='action_mask',
            value=action['action_mask'])
        self.episode_memory.add_value(
            key='action_primitive',
            value=action['action_primitive'])
        self.episode_memory.add_value(
            key='deformable_weight',
            value=float(self.deformable_weight)
        )
        for key, value in self.transformed_obs.items():
            self.episode_memory.add_value(key, value)

        for key, value in self.current_task.get_stats().items():
            self.episode_memory.add_value(key=key, value=value)

        if self.record_task_config:
            print("Recording task config")
            for key, value in self.current_task.get_config().items():
                if 'mesh' in key and not ('flip' in key):
                    self.episode_memory.add_value(key=f"task_{key}", value=value)

            print("DUMPING SHAPE", np.array(self.env_mesh_vertices).shape)
            # print("DUMPING SHAPE", np.array(self.env_mesh_vertices.copy().shape)
            self.episode_memory.add_value(key='end_effector_positions', value=np.array(self.env_end_effector_positions.copy()))
            self.episode_memory.add_value(key='env_mesh_vertices', value=np.array(self.env_mesh_vertices.copy()))

        if self.dump_visualizations or (self.recreate_verts is not None):
            if action['all_value_maps'] is not None:
                self.episode_memory.add_value(
                    key='value_maps',
                    value=action['all_value_maps'].cpu())
                
            
            if self.recreate_verts is not None:
                for primitive in self.action_primitives:

                    self.episode_memory.add_value(
                        key=f"{primitive}_value_maps",
                        value=action[f"{primitive}_value_maps"].cpu())

                    self.episode_memory.add_value(
                        key=f"{primitive}_raw_value_maps",
                        value=action[f"{primitive}_raw_value_maps"].cpu())
                
                

          
    def preaction(self):
        self.preaction_positions = pyflex.get_positions().reshape(-1, 4)[:, :3]

    def postaction(self):
        self.reset_end_effectors()
        wait_until_stable(gui=self.gui)
        postaction_positions = pyflex.get_positions().reshape(-1, 4)[:, :3]
        deltas = np.linalg.norm(
            np.abs(postaction_positions - self.preaction_positions), axis=1)
        if deltas.max() < 5e-2:
            # if didn't really move cloth then end early
            self.terminate = True
    
    def terminate_if_empty(self, obs):
        assert len(obs.shape) == 3
        self.is_frame_empty = obs[:3, :, :].sum() == 0
        if self.is_frame_empty:
            print("Frame is empty! This shouldn't happen too frequently")
            raise ValueError("Frame is empty")

    #only for the downstream folding purposes
    def fold(self):

        # print("Cloth mask shape", self.cloth_mask.shape)
        # pixel_keypoints = shirt_keypoints(self.cloth_mask)
        keypoint_names = ['left_shoulder', 'right_shoulder', 'top_left', 'top_right', 'bottom_left', 'bottom_right']
        keypoint_coords = self.keypoint_detector.get_keypoints(self.pretransform_rgb.astype(np.float32)/255)

        pixel_keypoints = {k:v for k,v in zip(keypoint_names, keypoint_coords)}

        # exit(0)
        pose_matrix = compute_pose(
                pos=[0, 2, 0],
                lookat=[0, 0, 0],
                up=[0, 0, 1])
        keypoints_3d = {key: pixel_to_3d(self.pretransform_depth, coord[1], coord[0], pose_matrix=pose_matrix) for key, coord in pixel_keypoints.items()}

        # gt_keypoint_indices = self.current_task.get_keypoint_data()
        # all_positions = pyflex.get_positions().reshape((-1, 4))[:self.num_particles, :3]
        # keypoints_3d = {key: all_positions[value[0]] for key, value in gt_keypoint_indices.items()}
        # print(keypoints_3d)
        # keypoints_3d = {key: value[0] for value in self.current_task.get_k  }

        actions = shirt_folding_heuristic(keypoints_3d)
        # print(actions)
        # exit(0)
        
        for action in actions:

            if len(action) == 1:
                a = action[0]
                pick = a['pick']
                place = a['place']
                self.pick_and_place_primitive(pick, place, lift_height=0.25)
            
            if len(action) == 2:
                left_a = action[0]
                right_a = action[1]

                left_pick = left_a['pick']
                left_place = left_a['place']

                right_pick = right_a['pick']
                right_place = right_a['place']

                self.double_arm_pick_and_place_primitive(left_pick, left_place, right_pick, right_place, lift_height=0.25)

        print("[SimEnv] Folding done")

        # print("Terminating")
        # log = True
        # while True:
        #     hashstring = hashlib.sha1()
        #     hashstring.update(str(time.time()).encode('utf-8'))
        #     vis_dir = f"{self.log_dir}/videos/{hashstring.hexdigest()[:10]}"
        #     if not os.path.exists(vis_dir):
        #         break
        # pathlib.Path(vis_dir).mkdir(parents=True, exist_ok=True)
        # for key, frames in self.env_video_frames.items():
        #     if len(frames) == 0:
        #         continue
        #     path = f'{vis_dir}/{key}_{len(frames)}.mp4'
        #     with imageio.get_writer(path, mode='I', fps=24) as writer:
        #         for frame in tqdm(frames, desc=f'Dumping {key} frames'):
        #             writer.append_data(frame)
                    
        return 

        # return (retval, self.ray_handle)

    def step(self, value_maps):

        #No valid actions
        if value_maps is None:
            return self.reset()

        self.preaction()

        prev_deformable_reward, prev_rigid_reward, _ = self.deformable_reward(log_prefix="preaction")

        if self.expert_demonstration:
            print("-"*10, "Expert Demonstration", "-"*10)
            print("[ICP] Preaction Deformable Reward:", prev_deformable_reward)
            print("[ICP] Preaction Rigid Reward:", prev_rigid_reward)
            print("[ICP] Preaction Weighted Reward (0.7):", 0.7 * prev_deformable_reward + 0.3 * prev_rigid_reward)
            print("-"*42)

        #time to perform actions
        start = time.time()

        try:
            action_primitive, action = self.get_max_value_valid_action(value_maps)       
        except Exception as e:
            print("Exception in get_max_value_valid_action:", e)
            return self.reset() 

        if action_primitive is not None and action is not None:
            # self.episode_memory.add_value(
            #     key='grasp_success', value=int(action['p1_grasp_cloth'] or action['p2_grasp_cloth']))
            if (self.current_timestep >= self.episode_length - 1) and self.fold_finish:
                print("[SimEnv] Folding cloth")
                self.fold()
            else:
                self.action_handlers[action_primitive](**action)
        self.postaction()

        obs = self.get_obs()
        #check if the arm has thrown the cloth out of the scene
        
        try:
            self.terminate_if_empty(obs)
        except Exception as e:
            print("Exception in terminate_if_empty:", e)
            return self.reset()

        post_deformable_reward, post_rigid_reward, _ = self.deformable_reward(log_prefix="postaction")

        self.current_timestep += 1
        self.terminate = self.terminate or \
            self.current_timestep >= self.episode_length or\
                self.is_frame_empty

        # try:
        self.episode_memory.add_rewards_and_termination(
            0, self.terminate)
        # except:
        #     print("Error adding rewards")
        #     self.reset()

        if self.expert_demonstration:
                print("-"*10, "Expert Demonstration", "-"*10)
                print("[ICP] Delta Deformable Reward:", post_deformable_reward - prev_deformable_reward)
                print("[ICP] Delta Rigid Reward", post_rigid_reward - prev_rigid_reward)
                print("[ICP] Delta Weighted Reward", (0.7 * post_deformable_reward + 0.3 * post_rigid_reward) - (0.7 * prev_deformable_reward + 0.3 * prev_rigid_reward))
                print("-"*42)

        self.episode_memory.add_value(
            key='next_observations', value=obs)
 
        if self.terminate:
            self.on_episode_end()
            return self.reset()
        else:
            self.episode_memory.add_value(
                key='pretransform_observations', value=obs)

           
        self.transformed_obs = self.generate_transformed_obs(obs)

        return (self.transformed_obs, self.ray_handle)
        
    # @profile(stream=fp)
    def generate_transformed_obs(self, obs):

        """
        Generates transformed observations and masks
        """

        retval = {}

        ##GENERATE OBSERVATION

        retval['transformed_obs'] = prepare_image(
                        obs, 
                        self.get_transformations(self.rotations), 
                        self.obs_dim,
                        orientation_net = self.orn_net_handle,
                        parallelize=self.parallelize_prepare_image,
                        nocs_mode=self.nocs_mode,
                        inter_dim=256,
                        constant_positional_enc=self.constant_positional_enc,)   

        ##GENERATE MASKS
        pretransform_cloth_mask = self.get_cloth_mask(obs[:3])
        pretransform_left_arm_mask = self.left_arm_mask
        pretransform_right_arm_mask = self.right_arm_mask 

        pretransform_mask = torch.stack([pretransform_cloth_mask, 
                                        pretransform_left_arm_mask, 
                                        pretransform_right_arm_mask], 
                                        dim=0)

        transformed_mask = prepare_image(
                        pretransform_mask, 
                        self.get_transformations(self.rotations), 
                        self.obs_dim,
                        parallelize=self.parallelize_prepare_image,
                        nocs_mode=self.nocs_mode,
                        inter_dim=128,
                        constant_positional_enc=self.constant_positional_enc,)   

        cloth_mask = transformed_mask[:, 0]
        left_arm_mask = transformed_mask[:, 1]
        right_arm_mask = transformed_mask[:, 2]

        workspace_mask = generate_workspace_mask(left_arm_mask, 
                                                right_arm_mask, 
                                                self.action_primitives, 
                                                self.pix_place_dist, 
                                                self.pix_grasp_dist)
        cloth_mask = generate_primitive_cloth_mask(
                                cloth_mask,
                                self.action_primitives,
                                self.pix_place_dist,
                                self.pix_grasp_dist)

        for primitive in self.action_primitives:

            GUARANTEE_OFFSET=6
            offset = self.pix_grasp_dist if primitive == 'fling' else self.pix_place_dist + GUARANTEE_OFFSET
            primitive_vmap_indices = self.primitive_vmap_indices[primitive]

            valid_transforms_mask = torch.zeros_like(cloth_mask[primitive]).bool()
            valid_transforms_mask[primitive_vmap_indices[0]:primitive_vmap_indices[1], 
                        offset:-offset,
                         offset:-offset] = True

                
            table_mask = retval['transformed_obs'][:, 3] > 0
            offset_table_mask_up = torch.zeros_like(table_mask).bool()
            offset_table_mask_down = torch.zeros_like(table_mask).bool()
            offset_table_mask_up[:, :-offset, :] = table_mask[:, offset:]
            offset_table_mask_down[:, offset:, :] = table_mask[:, :-offset]
            table_mask = offset_table_mask_up & offset_table_mask_down & table_mask

            primitive_workspace_mask = torch.logical_and(workspace_mask[primitive], table_mask)
            primitive_workspace_mask = torch.logical_and(primitive_workspace_mask, valid_transforms_mask)

            retval[f"{primitive}_cloth_mask"] = cloth_mask[primitive]
            retval[f"{primitive}_workspace_mask"] = primitive_workspace_mask
            retval[f"{primitive}_mask"] = torch.logical_and(cloth_mask[primitive], primitive_workspace_mask)

        return retval


    def get_action_params(self, action_primitive, max_indices, cloth_mask=None):
        x, y, z = max_indices
        if action_primitive == 'fling' or\
                action_primitive == 'stretchdrag':
            center = np.array([x, y, z])
            p1 = center[1:].copy()
            p2 = center[1:].copy()

            p1[0] = p1[0] + self.pix_grasp_dist
            p2[0] = p2[0] - self.pix_grasp_dist

        elif action_primitive == 'drag':
            p1 = np.array([y, z])
            p2 = p1.copy()
            p2[0] += self.pix_drag_dist
        elif action_primitive == 'place':
            p1 = np.array([y, z])
            p2 = p1.copy()
            p2[0] += self.pix_place_dist
        else:
            raise Exception(
                f'Action Primitive not supported: {action_primitive}')
        if (p1 is None) or (p2 is None):
            raise Exception(
                f'None reach points: {action_primitive}')
        return p1, p2

    def check_arm_reachability(self, arm_base, reach_pos):
        try:
            return np.linalg.norm(arm_base - reach_pos) < self.reach_distance_limit
        except Exception as e:
            print(e)
            print("[Check arm] Reachability error")
            print("arm_base:", arm_base)
            print("reach_pos:", reach_pos)
            return False, None

    def check_action_reachability(
            self, action: str, p1: np.array, p2: np.array):
        if (p1 is None) or (p2 is None):
           raise ValueError(f'[Invalid action] {action} reach points are None')
        if action in ['fling','drag','stretchdrag']:
            # right and left must reach each point respectively
            return self.check_arm_reachability(self.left_arm_base, p1) \
                and self.check_arm_reachability(self.right_arm_base, p2), None
        elif action == 'drag' or action == 'place':
            # either right can reach both or left can reach both
            if self.check_arm_reachability(self.left_arm_base, p1) and\
                    self.check_arm_reachability(self.left_arm_base, p2):
                return True, 'left'
            elif self.check_arm_reachability(self.right_arm_base, p1) and \
                    self.check_arm_reachability(self.right_arm_base, p2):
                return True, 'right'
            else:
                return False, None
        raise NotImplementedError()

    # @profile
    def get_max_value_valid_action(self, action_tuple) -> dict:

        # try:
        if action_tuple['random_action'] is not None:
            primitive = action_tuple['random_action']
        else:
            primitive = action_tuple['best_primitive']

        if self.fling_only:
            primitive = 'fling'
        elif self.place_only:
            primitive = 'place'

        if self.recreate_primitive:
            primitive = self.recreate_primitive
        # except Exception as e:
        #     print("\b[Invalid action] No valid actions\n", e)
        #     raise ValueError("Invalid action")
        # print("Action tuple keys:", action_tuple.keys())
        max_indices = action_tuple[primitive]['max_index']
        # print(f"Max indices: {max_indices}")

        if self.fling_only:
            if max_indices is None:
                self.reset()

        max_deformable_value = action_tuple[primitive]['max_deformable_value']
        max_rigid_value = action_tuple[primitive]['max_rigid_value']
        max_value = action_tuple[primitive]['max_value']

        x, y, z = max_indices
        all_value_maps = action_tuple[primitive]['all_value_maps']
        value_map = all_value_maps[x]
        action_mask = torch.zeros(value_map.size())

        try:
            action_mask[y, z] = 1 
        except:
            print("Indices", max_indices)
            exit(1)

        num_scales = len(self.adaptive_scale_factors)
        rotation_idx = torch.div(x, num_scales, rounding_mode='floor')
        scale_idx = x - rotation_idx * num_scales
        scale = self.adaptive_scale_factors[scale_idx]

        rotation = self.rotations[rotation_idx]

        reach_points = np.array(self.get_action_params(
            action_primitive=primitive,
            max_indices=(x, y, z),
            # cloth_mask = transform_cloth_mask
            ))

        p1, p2 = reach_points[:2]

        if (p1 is None) or (p2 is None):
            print("\n [SimEnv] Invalid pickpoints \n", primitive, p1, p2)
            raise ValueError("Invalid pickpoints")

        action_kwargs = {
            'observation': self.transformed_obs['transformed_obs'][x],
            'mask': self.transformed_obs[f'{primitive}_mask'][x],
            'workspace_mask': self.transformed_obs[f'{primitive}_workspace_mask'][x],
            'action_primitive': str(primitive),
            'primitive': primitive,
            'p1': p1,
            'p2': p2,
            'scale': scale,
            'nonadaptive_scale': self.scale_factors[scale_idx],
            'rotation': rotation,
            'predicted_deformable_value': float(max_deformable_value),
            'predicted_rigid_value': float(max_rigid_value),
            'predicted_weighted_value': float(max_value),
            'max_indices': np.array(max_indices),
            'action_mask': action_mask,
            'value_map': value_map,
            'info': None,
            'all_value_maps': all_value_maps,
        }

        if action_tuple[primitive].get('raw_value_map') is not None:
            action_kwargs['raw_value_maps'] = action_tuple[primitive]['raw_value_maps']


        assert ((action_kwargs['p1'] is not None) and (action_kwargs['p2'] is not None))

        action_kwargs.update({
            'transformed_depth':
            action_kwargs['observation'][3, :, :],
            'transformed_rgb':
            action_kwargs['observation'][:3, :, :],
        })

        if self.recreate_verts is not None:
            for primitive in self.action_primitives:
                action_kwargs[f'{primitive}_value_maps'] = action_tuple[primitive]['all_value_maps']
                action_kwargs[f'{primitive}_raw_value_maps'] = action_tuple[primitive]['raw_value_maps']


        action_params = self.check_action(
            pixels=np.array([p1, p2]),
            **action_kwargs)

        #enforce interaction with the cloth
        # if not (action_params['p1_grasp_cloth'] and action_params['p2_grasp_cloth']):
        #     fig, axs = plt.subplots(1, 3)
        #     axs[0].imshow(mask[x, :, :])
        #     axs[1].imshow(mask[x, :, :].bool() & ~action_mask.bool())
        #     axs[2].imshow(action_mask.bool())
        #     axs[0].set_title(indices)
        #     axs[1].set_title(action)
        #     #create grasp cltoh images directory
        #     self.grasp_clothes_dir = "/local/crv/acanberk/folding-unfolding/src/failed_grasp"
        #     if not os.path.exists(self.grasp_clothes_dir):
        #         os.makedirs(self.grasp_clothes_dir)
        #     plt.savefig(f"{self.grasp_clothes_dir}/grasp_cloth{np.random.randint(0, 100)}.png")
        #     print("Cloth not grasped")
        #     exit(1)
        #     continue
        # continue

        # if not action_params['valid_action']:
        #     print("Invalid action")
        #     continue
        try:
            reachable, left_or_right = self.check_action_reachability(
                action=primitive,
                p1=action_params['p1'],
                p2=action_params['p2'])
        except ValueError as e:
            raise ValueError("Reach pos none")
            # print(" \n [SimEnv] Invalid reachability, resetting \n")
            # self.reset()
            

        # if not reachable:
        #     print(" \n [SimEnv] Invalid reachability \n")
        #     raise ValueError("Invalid reachability")

        if primitive == 'place':
            action_kwargs['left_or_right'] = left_or_right

        action_kwargs['action_visualization'] =\
            action_params['get_action_visualization_fn']()
        self.log_step_stats(action_kwargs)

        for k in ['valid_action',
                    'pretransform_pixels',
                    'get_action_visualization_fn']:
            del action_params[k]

        return action_kwargs['action_primitive'], action_params


    def record_mesh_data(self):
        # mesh = get_cloth_mesh(*self.current_task.cloth_size)
        self.current_pos = pyflex.get_positions().reshape((-1, 4))[:self.num_particles, :3]
        return np.mean(np.linalg.norm(self.current_pos - self.init_pos, axis=1))


    # @profile
    def reset(self):
        
        self.episode_memory = Memory()
        self.episode_reward_sum = 0
        self.current_timestep = 0
        self.terminate = False
        self.is_frame_empty = False
        self.current_task = self.get_task_fn()

        self.init_direction = None
        self.init_coverage = None
        self.cloth_mask = None

        # self.dump_visualizations = self.dump_all_visualizations or (self.dump_video_counter % self.dump_video_interval == 0)
        self.dump_visualizations = (self.dump_video_counter % self.dump_video_interval == 0)

        self.tri_v, self.tri_f =  self.current_task.get_config()['mesh_verts'].reshape(-1, 3), \
        self.current_task.get_config()['mesh_faces'].reshape(-1, 3)

        self.num_particles = self.current_task.get_config()['mesh_verts'].shape[0] // 3

        self.init_pos = self.current_task.get_state()['init_particle_pos'].reshape(-1, 4)[:self.num_particles, :3]

        try:
            cloth_instance = self.current_task.get_config()['cloth_instance']
            # print("Cloth instance: ", cloth_instance)
            if '.pkl' not in cloth_instance:
                raise Exception("[TaskLoader] Cloth instance must be a pickle file")
            self.episode_memory.add_value(
                key='cloth_instance', value=cloth_instance)
        except:
            print("[TaskLoader] Cloth instance name could not be found")
            cloth_instance = 0

        self.episode_memory.add_value(
            key='cloth_instance', value=cloth_instance)

        if self.gui:
            print(self.current_task)
        set_scene(
            config=self.current_task.get_config(),
            state=self.current_task.get_state())
        
        #find the number of

        if self.recreate_verts is not None:
            print("[SimEnv] Recreating verts, shape: ", self.recreate_verts.shape)
            self.set_particle_pos(self.recreate_verts)

        
        self.action_tool.reset([0.2, 0.5, 0.0])
        self.reset_end_effectors()

        self.step_simulation()
        self.set_grasp(False)
        self.env_video_frames = {}

        obs = self.get_obs()
        #create an assets folder for the images if it does not exist
        # if not os.path.exists("../assets/images"):
        #     os.makedirs("../assets/images")
        # plt.savefig(f"../assets/images/obs{np.random.randint(0, 100)}.png")

        try:
            self.terminate_if_empty(obs)
        except Exception as e:
            print("Exception in terminate_if_empty:", e)
            return self.reset()
      
        self.episode_memory.add_value(
            key='pretransform_observations', value=obs)     

        self.transformed_obs = self.generate_transformed_obs(obs)
        
        return (self.transformed_obs, self.ray_handle)

    def render_cloth(self):
        if self.render_engine == 'blender':
            mesh = get_cloth_mesh(*self.current_task.cloth_size)
            return blender_render_cloth(mesh, self.image_dim)
        elif self.render_engine == 'opengl':

            #for debugging
            # mesh = get_cloth_mesh(*self.current_task.cloth_size)
            # color, depth = blender_render_cloth(mesh, self.image_dim)
            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(1, 4)
            # axs[0].imshow(color)
            # axs[1].imshow(depth)
            # axs[2].imshow(get_image(self.image_dim)[0])
            # axs[3].imshow(get_image(self.image_dim)[1])

            # fig.tight_layout(pad=0)
            # plt.show()
            #for debugging

            return get_image(self.image_dim, self.image_dim)
        else:
            raise NotImplementedError()

    def get_cloth_mask(self, rgb):
        return rgb.sum(axis=0) > 0

    # @profile
    def get_obs(self):

        self.hide_end_effectors()

        rgb, d = self.render_cloth()
     
        self.pretransform_depth = d
        self.pretransform_rgb = rgb
        # cloths are closer than 2.0 meters from camera plane
        self.cloth_mask = self.get_cloth_mask(rgb.transpose(2, 0, 1))
        x, y = np.where(self.cloth_mask)
        dimx, dimy = self.pretransform_depth.shape
        self.adaptive_scale_factors = self.scale_factors.copy()
        if self.use_adaptive_scaling:
            try:
                # Minimum square crop
                cropx = max(dimx - 2*x.min(), dimx - 2*(dimx-x.max()))
                cropy = max(dimy - 2*y.min(), dimy - 2*(dimy-y.max()))
                crop = max(cropx, cropy)
                # Some breathing room
                crop = int(crop*1.5)
                if crop < dimx:
                    self.adaptive_scale_factors *= crop/dimx
                    # self.episode_memory.add_value(
                    #     key='adaptive_scale',
                    #     value=float(crop/dimx))

            except Exception as e:
                print(e)
                print(self.current_task)
        
        obs = preprocess_obs(rgb.copy(), d.copy())

        #combine obs and dot product map tensors
        _, _, nocs_img = pyflex.render(uv=True)

        if '_gtnocs' in self.input_channel_types:
            nocs_img = nocs_img.reshape([480, 480, 3])
            nocs_x = np.abs(nocs_img[..., 0] - 0.5) * (np.sum(nocs_img, axis=2) > 0)
            nocs_z = nocs_img[..., 2]
            nocs_img = np.stack([nocs_x, nocs_z], axis=2)
            nocs_img = cv2.flip(cv2.resize(nocs_img, (self.image_dim, self.image_dim)), 0)
            obs = torch.cat((obs, torch.tensor(nocs_img.transpose(2,0,1))), dim=0)

        if '_fullgtnocs' in self.input_channel_types:
            nocs_img = nocs_img.reshape([480, 480, 3])
            nocs_img = cv2.flip(cv2.resize(nocs_img, (self.image_dim, self.image_dim)), 0)
            obs = torch.cat((obs, torch.tensor(nocs_img.transpose(2,0,1))), dim=0)

        self.reset_end_effectors()

        return obs

    def movep(self, pos, speed=None, limit=1000, min_steps=None, eps=1e-4):
        if speed is None:
            if self.dump_visualizations:
                speed = self.default_speed
            else:
                speed = 0.1
        target_pos = np.array(pos)
        for step in range(limit):
            curr_pos = self.action_tool._get_pos()[0]
            deltas = [(targ - curr)
                      for targ, curr in zip(target_pos, curr_pos)]
            dists = [np.linalg.norm(delta) for delta in deltas]
            if all([dist < eps for dist in dists]) and\
                    (min_steps is None or step > min_steps):
                return
            action = []
            for targ, curr, delta, dist, gs in zip(
                    target_pos, curr_pos, deltas, dists, self.grasp_states):
                if dist < speed:
                    action.extend([*targ, float(gs)])
                else:
                    delta = delta/dist
                    action.extend([*(curr+delta*speed), float(gs)])
            action = np.array(action)
            self.action_tool.step(action, step_sim_fn=self.step_simulation)

            if step % 4 == 0 and self.dump_visualizations:
                if 'top' not in self.env_video_frames:
                    self.env_video_frames['top'] = []

                self.env_video_frames['top'].append(
                    np.squeeze(np.array(get_image()[0])))
                

        raise MoveJointsException

    def reset_end_effectors(self):
        self.movep([[0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]], speed=8e-2)

    def hide_end_effectors(self):
        self.movep([[0.5, 0.5, -1], [-0.5, 0.5, -1]], speed=5e-2)


    def set_grasp(self, grasp):
        if type(grasp) == bool:
            self.grasp_states = [grasp] * len(self.grasp_states)
        elif len(grasp) == len(self.grasp_states):
            self.grasp_states = grasp
        else:
            raise Exception()

    # @profile
    def on_episode_end(self, log=False):

        ####
        if self.dump_visualizations:
            while True:
                hashstring = hashlib.sha1()
                hashstring.update(str(time.time()).encode('utf-8'))
                vis_dir = f"{self.log_dir}/videos/{hashstring.hexdigest()[:10]}"
                if not os.path.exists(vis_dir):
                    break
            pathlib.Path(vis_dir).mkdir(parents=True, exist_ok=True)
            for key, frames in self.env_video_frames.items():
                if len(frames) == 0:
                    continue
                path = f'{vis_dir}/{key}_{len(frames)}.mp4'
                with imageio.get_writer(path, mode='I', fps=24) as writer:
                    for frame in (
                        frames if not log
                            else tqdm(frames, desc=f'Dumping {key} frames')):
                        writer.append_data(frame)
                self.episode_memory.add_value(
                    key='visualization_dir',
                    value=vis_dir)
            print("[Episode video dumped]")

            # if self.dump_video_counter % self.dump_video_interval == 0:
            #     print("Dumping video to wandb")
            #     wandb.log({"video": wandb.Video(
            #         f"{vis_dir}/top.mp4")})
        ###


        self.env_video_frames.clear()
        #time to dump memory
        start = time.time()
        self.episode_memory.dump(
            self.replay_buffer_path)
        self.episode_memory.clear()
        del self.episode_memory
        self.episode_memory = Memory()
        self.dump_video_counter += 1

        if self.recreate_verts is not None:
            print("Recreation done!")

    def is_cloth_grasped(self):
        positions = pyflex.get_positions().reshape((-1, 4))
        positions = positions[:self.num_particles, :3]
        heights = positions[:, 1]
        return heights.max() > 0.2

    def setup_ray(self, id):
        self.id = id
        self.ray_handle = {"val": id}
