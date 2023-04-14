import numpy as np
import skimage.transform as st
import scipy.interpolate as si
import scipy.ndimage as sni
import itertools
from cair_robot.common.geometry import get_center_affine, pixel_to_3d, transform_points, transform_pose, get_pointcloud
from cair_robot.common.primitive_util import get_base_fling_poses, center_to_action_frame
from skimage.color import rgb2hsv
import cv2

def camera_image_to_view(cam_img, tx_camera_view, img_shape=(128,128)):
    tf_camera_view = st.AffineTransform(matrix=tx_camera_view)
    result_img = st.warp(cam_img, tf_camera_view.inverse, output_shape=img_shape)
    return result_img

class ImageStackTransformer:
    """
    Note: this class follows skimage.transform coordinate convention
    (x,y) x right, y down
    The rest of skimage uses (col, row) convention
    """
    def __init__(self, img_shape=(128,128), 
            rotations=np.linspace(-np.pi, np.pi, 17), 
            scales=[1.0, 1.5, 2.0, 2.5, 3.0]):
        """
        Create a stack of rotations * scales images.
        Rotation: counter-clockwise
        Scales: >1 object appear bigger
        """
        assert len(img_shape) == 2
        stack_shape = (len(rotations) * len(scales),) + tuple(img_shape)

        transforms = list()
        self.transform_tuples = list(itertools.product(rotations, scales))

        for rot, scale in itertools.product(rotations, scales):
            # both skimage and torchvision use
            tf = get_center_affine(
                img_shape=img_shape, 
                rotation=rot, scale=scale)
            tf.params = tf.params.astype(np.float32)
            transforms.append(tf)

        self.shape = stack_shape
        self.transforms = transforms
        self.rotations = rotations
        self.scales = scales
    
    def forward_img(self, img, mode='constant'):
        results = [st.warp(img, tf.inverse, mode=mode, preserve_range=True) for tf in self.transforms]
        stack = np.stack(results).astype(np.uint8)
        return stack
    
    def forward_raw(self, raw, tx_camera_view):
        img_shape = self.shape[1:]
        stack = np.empty(
            (len(self.transforms),) + img_shape + raw.shape[2:], 
            dtype=raw.dtype)
        for i, tf in enumerate(self.transforms):
            ntf = st.AffineTransform(tf.params @ tx_camera_view)
            stack[i] = st.warp(raw, ntf.inverse, 
                order=1,
                output_shape=img_shape,
                preserve_range=True)
        return stack
    
    def inverse_coord(self, stack_coord):
        """
        Convert 3d stack coordinate integers to
        float coordinate in the original image
        """
        return self.transforms[stack_coord[0]].inverse(stack_coord[1:])

    def get_inverse_coord_map(self):
        identity_map = np.moveaxis(
            np.indices(self.shape[1:], dtype=np.float32)[::-1],0,-1
            )

        maps = list()
        for tf in self.transforms:
            tx = np.linalg.inv(tf.params)
            r = transform_points(
                identity_map.reshape(-1,2), 
                tx).reshape(identity_map.shape)
            maps.append(r)
        coord_stack = np.stack(maps)
        return coord_stack

    def get_world_coords_stack(self, depth, tx_camera_view, tx_world_camera, cam_intr):
        img_coords_stack = self.get_inverse_coord_map()
        raw_img_coords_stack = transform_points(
            img_coords_stack.reshape(-1,2), 
            np.linalg.inv(tx_camera_view)).reshape(
                img_coords_stack.shape)

        # x,y
        # transform to world coord
        world_coords_stack = np.empty(
            img_coords_stack.shape[:-1]+(3,), 
            dtype=np.float32)
        for i in range(len(img_coords_stack)):
            img_coords = raw_img_coords_stack[i]
            # skimage uses (x,y) coordinate, pixel_to_3d uses (y,x)
            coords_3d = pixel_to_3d(depth, img_coords.reshape(-1,2)[:,::-1], 
                cam_pose=tx_world_camera, cam_intr=cam_intr)
            img_coords_3d = coords_3d.reshape(img_coords.shape[:-1] + (3,))
            world_coords_stack[i] = img_coords_3d
        return world_coords_stack


def is_coord_valid_robot(coords, tx_robot_world, 
        reach_radius=0.93, near_radius=0.0755):
    """
                    max     recommended             
    reach_radius:   0.946   0.85
    near_radius:    0       0.0755

    Reference:
    https://www.universal-robots.com/articles/ur/application-installation/what-is-a-singularity/
    """
    coords_robot = transform_points(coords, tx_robot_world)
    dist_3d = np.linalg.norm(coords_robot, axis=-1)
    dist_xy = np.linalg.norm(coords_robot[...,:2], axis=-1)
    is_valid = (dist_3d < reach_radius) & (dist_xy > near_radius)
    return is_valid

def is_coord_valid_table(coords, table_low=(-0.58,-0.88,-0.05), table_high=(0.58,0.87,0.2)):
    is_valid = np.ones(coords.shape[:-1], dtype=bool)
    for i in range(3):
        this_valid = (table_low[i] < coords[...,i]) & (coords[...,i] < table_high[i])
        is_valid = is_valid & this_valid
    return is_valid


def fill_nearest(depth_im, mask):
    coords = np.moveaxis(np.indices(depth_im.shape),0,-1)
    interp = si.NearestNDInterpolator(coords[~mask], depth_im[~mask])
    out_im = depth_im.copy()
    out_im[mask] = interp(coords[mask])
    return out_im

def get_offset_stack(stack, offset=16):
    """
    Assuming (N,H,W,D)
    up: move up offset pixels
    down: move down offset pixels
    """
    value = np.nan
    if stack.dtype is np.dtype('bool'):
        value = False
    up_stack = np.full(stack.shape, value, dtype=stack.dtype)
    down_stack = np.full(stack.shape, value, dtype=stack.dtype)
    up_stack[:,offset:,...] = stack[:,:-offset,...]
    down_stack[:,:-offset:,...] = stack[:,offset:,...]
    return up_stack, down_stack


def check_line_validity(stack, offset=16, axis=1, eps=1e-7):
    length = offset*2+1
    weights = np.full((length,),1/length, dtype=np.float32)
    result = sni.convolve1d(stack.astype(np.float32), 
        weights, axis=axis, mode='constant', cval=0)
    out = result > (1-eps)
    return out

class FlingPathGenerator:
    def __init__(self, center, to, 
        env,
        fling_widths = np.linspace(0.1, 0.7, 100),
        strokes = np.linspace(0.2, 0.6, 100),
        lift_height=0.4, place_y=0.3, 
        place_height=0.02, swing_angle=np.pi/4,
        angle_range_bonus=0.14):

        fling_widths = np.sort(fling_widths)
        strokes = np.sort(strokes)[::-1]
        tx_world_action = center_to_action_frame(center, to)

        valid_widths = list()
        valid_strokes = list()
        valid_actions = list()
        for fling_width in fling_widths:
            for stroke in strokes:
                base_fling = get_base_fling_poses(
                    place_y=place_y, 
                    stroke=stroke,
                    lift_height=lift_height,
                    swing_angle=swing_angle,
                    place_height=place_height)
                left_path_local = base_fling.copy()
                right_path_local = base_fling.copy()
                left_path_local[:,0] -= fling_width/2
                right_path_local[:,0] += fling_width/2
                left_path = transform_pose(tx_world_action, left_path_local)
                right_path = transform_pose(tx_world_action, right_path_local)
                left_valid = np.all(env.is_coord_valid_robot(
                    left_path[:,:3], is_left=True,
                    robot_far=env.robot_far + angle_range_bonus))
                right_valid = np.all(env.is_coord_valid_robot(
                    right_path[:,:3], is_left=False,
                    robot_far=env.robot_far + angle_range_bonus))
                is_valid = left_valid and right_valid
                if is_valid:
                    valid_widths.append(fling_width)
                    valid_strokes.append(stroke)
                    valid_actions.append([left_path, right_path])
                    break
        if len(valid_actions) == 0:
            raise RuntimeError("No valid fling possible")
        
        self.widths = np.array(valid_widths)
        self.strokes = np.array(valid_strokes)
        self.action_paths = np.array(valid_actions)
    
    def get_action_paths(self, width):
        nn_idx = np.argmin(np.abs(self.widths - width))
        action_paths = self.action_paths[nn_idx]
        return action_paths
    
    def get_max_width(self):
        return np.max(self.widths)


class PickPointGenerator:
    def __init__(self, offset=-0.01, min_height=0.05):
        self.offset = offset
        self.min_height = min_height
    
    def __call__(self, obs_point):
        pick_point = obs_point.copy()
        pick_point[2] = max(obs_point[2] + self.offset, self.min_height)
        return pick_point


class TransformedViewEnv:
    """
    All parameters in image space.
    """
    def __init__(self, kinect_camera, 
            tx_camera_img, tx_world_camera, 
            tx_left_camera, tx_right_camera,
            img_size=128,
            fling_rotations=np.linspace(-np.pi/2,np.pi/2,9),
            pick_place_rotations=np.linspace(-np.pi,np.pi,17),
            scales=[1.0, 1.5, 2.0, 2.5, 3.0],
            # fling
            lift_height = 0.4,
            place_y = 0.2,
            pick_place_lift_height=0.1,
            # safety
            robot_far = 0.80,
            robot_near = 0.25,
            tcp_offset = 0.2,
            # pick
            mat_thickness=0.055,
            left_picker_offset = 0.0,
            must_pick_on_obj=True,
            ):
        """
        tx_camera_img -> (3,3)
        """
        print("[TransformedViewEnv]", fling_rotations, pick_place_rotations)
        self.camera = kinect_camera
        self.tx_camera_img = tx_camera_img.astype(np.float32)
        self.tx_world_camera = tx_world_camera.astype(np.float32)
        self.tx_left_camera = tx_left_camera.astype(np.float32)
        self.tx_right_camera = tx_right_camera.astype(np.float32)
        self.img_size = img_size
        self.fling_rotations = fling_rotations
        self.pick_place_rotations = pick_place_rotations
        self.scales = scales
        self.cam_intr = self.camera.get_intr().astype(np.float32)
        self.robot_far = robot_far
        self.robot_near = robot_near
        self.lift_height = lift_height
        self.place_y = place_y
        self.fling_generator = None
        self.left_pick_geneator = PickPointGenerator(
            offset=-0.008,
            min_height=mat_thickness+left_picker_offset)
        self.right_pick_geneator = PickPointGenerator(
            offset=-0.01,
            min_height=mat_thickness)
        self.left_place_geneator = PickPointGenerator(
            offset=0.05,
            min_height=mat_thickness+0.02)
        self.right_place_geneator = PickPointGenerator(
            offset=0.05,
            min_height=mat_thickness+0.02)
        self.must_pick_on_obj = must_pick_on_obj
        self.tcp_offset = tcp_offset
        self.pick_place_lift_height = pick_place_lift_height
        self.mat_thickness = mat_thickness

    def is_coord_valid_robot(self, coords, is_left, robot_far=None, robot_near=None):
        if robot_far is None:
            robot_far = self.robot_far
        if robot_near is None:
            robot_near = self.robot_near
        tx = self.tx_left_camera if is_left else self.tx_right_camera
        tx_robot_world = tx @ np.linalg.inv(self.tx_world_camera)
        r = is_coord_valid_robot(coords, tx_robot_world, 
            reach_radius=robot_far, near_radius=robot_near)
        return r

    def get_obs(self):
        color, depth = self.camera.get_camera_data()

        # fill 0 in depth with reasonable stuff
        depth = cv2.inpaint(
            depth.astype(np.float32), 
            (depth==0).astype(np.uint8), 
            inpaintRadius=0, flags=cv2.INPAINT_NS)
        
        pcloud = get_pointcloud(depth_img=depth, 
            cam_intr=self.cam_intr, 
            cam_pose=self.tx_world_camera)

        is_table = is_coord_valid_table(pcloud)
        # color filter on the table
        hsv = rgb2hsv(color)
        table_mask = np.logical_and(hsv[..., 2] < 0.5, hsv[:, :, 1] < 0.5)
        obj_mask = ~((~is_table) | table_mask)
        return color, depth, obj_mask
    
    def transform_obs(self,transformer, depth, 
            obs=None, obj_mask=None):
        
        world_coords_stack = transformer.get_world_coords_stack(
            depth=depth, 
            tx_camera_view=self.tx_camera_img, 
            tx_world_camera=self.tx_world_camera,
            cam_intr=self.cam_intr)
        data = {'world_coord': world_coords_stack}
        if obs is not None:
            obs_stack = transformer.forward_raw(obs, self.tx_camera_img)
            data['obs'] = obs_stack
        if obj_mask is not None:
            mask_stack = transformer.forward_raw(
                obj_mask.astype(np.float32), self.tx_camera_img) > 0.5
            data['obj_mask'] = mask_stack
        return data
    
    def render(self):
        camera_img = self.scene.render()
        tf = st.AffineTransform(matrix=self.tx_world_img)
        tf_img = tf(camera_img)
        img = tf_img[:self.img_size,:self.img_size]
        return img
        
    def get_pick_and_place_input(self, color, depth, obj_mask):        
        offset = 10 # move distance in pixels
        # calculating adaptive scaling
        view_mask = camera_image_to_view(obj_mask, self.tx_camera_img, 
            img_shape=(self.img_size, self.img_size))
        r,c = np.nonzero(view_mask)

        try:
            max_width = max(r.max() - r.min(), c.max() - c.min())
        except:
            print("No max width calculated")
            max_width = 0.5

        adaptive_scale_factor = max_width * 1.5 / self.img_size
        # compute image and coordiante stack
        scales = 1/(np.array(self.scales) * adaptive_scale_factor)
        transformer = ImageStackTransformer(
            img_shape=(self.img_size, self.img_size),
            rotations=self.pick_place_rotations,
            scales=scales
        )
        
        data = self.transform_obs(transformer, 
            depth=depth, obs=color, obj_mask=obj_mask)
        obs_stack = data['obs']
        world_coords_stack = data['world_coord']
        obj_mask_stack = data['obj_mask']

        data = self.transform_obs(ImageStackTransformer(
            img_shape=(self.img_size, self.img_size),
            rotations=[0], scales=[1]),
            depth=depth,
            obs=color,
            obj_mask=obj_mask
            )
        center_obs = data['obs'][0]
        center_obj_mask = data['obj_mask'][0]
        
        # raw validity
        is_table_valid = is_coord_valid_table(world_coords_stack)
        # lift point above pick point much be reachable
        coords = world_coords_stack.copy()
        coords[...,2] += (self.tcp_offset + self.pick_place_lift_height)
        is_left_valid = is_table_valid & self.is_coord_valid_robot(coords, is_left=True)
        is_right_valid = is_table_valid & self.is_coord_valid_robot(coords, is_left=False)
        # since valid set is mostly convex, checking two endpoints are sufficient

        _, is_left_end_valid = get_offset_stack(is_left_valid, offset=offset)
        _, is_right_end_valid = get_offset_stack(is_right_valid, offset=offset)
        is_left_action_valid = is_left_valid & is_left_end_valid
        is_right_action_valid = is_right_valid & is_right_end_valid
        is_start_on_obj = obj_mask_stack

        start_coord = world_coords_stack
        _, end_coord = get_offset_stack(start_coord, offset=offset)

        is_any_valid = is_left_action_valid | is_right_action_valid
        use_left = is_left_action_valid
        obs = obs_stack
        is_valid = is_any_valid
        if self.must_pick_on_obj:
            is_valid = is_valid & is_start_on_obj

        info = {
            'pick_on_obj': is_start_on_obj,
            'use_left': use_left,
            'start_coord': start_coord,
            'end_coord': end_coord,
            'center_obs': center_obs,
            'center_obj_mask': center_obj_mask,
            'transformer': transformer
        }
        return obs, is_valid, info
    
    def get_pick_and_fling_input(self, color, depth, obj_mask):
        offset = 16 # half width in pixels
        # calculating adaptive scaling
        view_mask = camera_image_to_view(obj_mask, self.tx_camera_img, 
            img_shape=(self.img_size, self.img_size))
        r,c = np.nonzero(view_mask)

        try:
            max_width = max(r.max() - r.min(), c.max() - c.min())
        except:
            print("No max width calculated")
            max_width = 0.5

        adaptive_scale_factor = max_width * 1.5 / self.img_size
        # compute image and coordiante stack
        scales = 1/(np.array(self.scales) * adaptive_scale_factor)
        transformer = ImageStackTransformer(
            img_shape=(self.img_size, self.img_size),
            rotations=self.fling_rotations,
            scales=scales
        )
        data = self.transform_obs(transformer, 
            depth=depth, obs=color, obj_mask=obj_mask)
        obs_stack = data['obs']
        world_coords_stack = data['world_coord']
        obj_mask_stack = data['obj_mask']

        data = self.transform_obs(ImageStackTransformer(
            img_shape=(self.img_size, self.img_size),
            rotations=[0], scales=[1]),
            depth=depth,
            obs=color,
            obj_mask=obj_mask
            )
        center_obs = data['obs'][0]
        center_obj_mask = data['obj_mask'][0]

        # raw validty maps
        is_table_valid = is_coord_valid_table(world_coords_stack)
        coords = world_coords_stack.copy()
        # lift point above pick point much be reachable
        coords[...,2] += (self.tcp_offset + self.lift_height)
        is_left_valid = is_table_valid & self.is_coord_valid_robot(coords, is_left=True)
        is_right_valid = is_table_valid & self.is_coord_valid_robot(coords, is_left=False)

        is_left_action_valid, _ = get_offset_stack(is_left_valid, offset=offset)
        _, is_right_action_valid = get_offset_stack(is_right_valid, offset=offset)
        is_action_valid = is_left_action_valid & is_right_action_valid
        is_left_on_obj, is_right_on_obj = get_offset_stack(obj_mask_stack, offset=offset)
        is_action_on_obj = is_left_on_obj & is_right_on_obj

        left_coord, right_coord = get_offset_stack(world_coords_stack, offset=offset)
        obs = obs_stack
        is_valid = is_action_valid
        if self.must_pick_on_obj:
            is_valid = is_valid & is_action_on_obj

        info = {
            'pick_on_obj': is_action_on_obj,
            'left_coord': left_coord,
            'right_coord': right_coord,
            'center_obs': center_obs,
            'center_obj_mask': center_obj_mask,
            'transformer': transformer,
        }
        return obs, is_valid, info

    def get_input(self, pick_and_place=True, pick_and_fling=True, obs_in=None):

        if obs_in is None:
            color, depth, obj_mask = self.get_obs()
        else:
            color, depth, obj_mask = obs_in

        color[~obj_mask] = 0
        
        data = dict()
        if pick_and_place:
            obs, is_valid, info = self.get_pick_and_place_input(
                color, depth, obj_mask)
            pp_data = {
                'obs': obs,
                'is_valid': is_valid,
                'info': info
            }
            data['pick_and_place'] = pp_data

        if pick_and_fling:
            obs, is_valid, info = self.get_pick_and_fling_input(
                color, depth, obj_mask)
            pf_data = {
                'obs': obs,
                'is_valid': is_valid,
                'info': info
            }
            data['pick_and_fling'] = pf_data
        return data

    def get_valid_fling_params(self):
        color, depth, _ = self.get_obs()
        data = self.transform_obs(
            transformer=ImageStackTransformer(
                rotations=[0], scales=[1]),
            depth=depth)
        world_coords_stack = data['world_coord']
        world_coord = world_coords_stack[0]
        center = world_coord[64,64]
        right = world_coord[64,74]
        fling_param_generator = FlingPathGenerator(
            center, right, env=self,
            place_y = self.place_y,
            lift_height = self.lift_height + self.mat_thickness)
        return fling_param_generator
    
    def initialize_fling_primitive(self):
        self.fling_generator = self.get_valid_fling_params()
    
    def pick_and_place(self, scene, 
            is_left, start_point, end_point):
        pick_gen = self.left_pick_geneator if is_left else self.right_pick_geneator
        place_gen = self.left_place_geneator if is_left else self.right_place_geneator

        r = scene.single_arm_pick_and_place(
            is_left, 
            pick_gen(start_point), 
            place_gen(end_point), 
            min_pick_height=0.0,
            lift_height=self.pick_place_lift_height + 
                self.mat_thickness)
        if not r : return False
        return scene.home()
    
    def pick_and_place_coord(self, scene, map_coord, info):
        """
        map_coord: coordinate index into the spatial action map tensor
        """
        use_left = info['use_left']
        start_coord = info['start_coord']
        end_coord = info['end_coord']

        is_left = use_left[tuple(map_coord)]
        start_point = start_coord[tuple(map_coord)]
        end_point = end_coord[tuple(map_coord)]
        return self.pick_and_place(scene, is_left, start_point, end_point)

    def center_pick_and_place_coord(self, scene, map_coord, info):
        """
        center_pick_and_place_coord: for resetting the cloth to the center
        """
        use_left = info['use_left']
        start_coord = info['start_coord']

        is_left = use_left[tuple(map_coord)]
        start_point = start_coord[tuple(map_coord)]
        end_point = np.array([0.0, 0.0, 0.2])
        return self.pick_and_place(scene, is_left, start_point, end_point)

    def pick_and_fling(self, scene, left_point, right_point):
        if self.fling_generator is None:
            raise RuntimeError("Fling primitive not initialized.")
        fling_generator = self.fling_generator
        width = np.linalg.norm((left_point-right_point)[:2])
        a, b = fling_generator.get_action_paths(width)
        strech_l, strech_r = a[0], b[0]
        max_width = fling_generator.get_max_width()


        r = scene.dual_arm_pick(
            self.left_pick_geneator(left_point), 
            self.right_pick_geneator(right_point),
            min_pick_height=0.0,
            lift_height=self.lift_height + self.mat_thickness)
        # assert(r)
        r = scene.dual_arm_strech(strech_l, strech_r, 
            max_width=max_width)
        assert(r)
        post_width = scene.get_tcp_distance()
        left_path, right_path = fling_generator.get_action_paths(post_width)
        r = scene.dual_arm_movel(left_path[0], right_path[0])
        r = scene.dual_arm_fling(left_path, right_path, 
            # speed=0.2, acceleration=1)
            speed=1.3, acceleration=5)
        assert(r)
        scene.open_grippers()
        scene.lift_grippers(0.2)
        scene.home()

    def pick_and_fling_coord(self, scene, map_coord, info):
        """
        map_coord: coordinate index into the spatial action map tensor
        """
        left_coord = info['left_coord']
        right_coord = info['right_coord']
        left_point = left_coord[tuple(map_coord)]
        right_point = right_coord[tuple(map_coord)]
        return self.pick_and_fling(scene, left_point, right_point)
    
    def random_pick_and_place(self, scene, rs=None):
        if rs is None:
            rs = np.random.RandomState()
        data = self.get_input(pick_and_place=True, pick_and_fling=False
            )['pick_and_place']
        info = data['info']
        is_valid = data['is_valid']
        if self.must_pick_on_obj:
            is_valid = is_valid & info['pick_on_obj']
        if np.max(is_valid) == False:
            return False
        valid_coords = np.stack(np.nonzero(is_valid)).T
        idx = rs.choice(len(valid_coords))
        coord = tuple(valid_coords[idx])
        return self.pick_and_place_coord(scene, coord, info)
    
    def center_pick_and_place(self, scene, rs=None):
        """
        For resetting the cloth to the center
        """
        if rs is None:
            rs = np.random.RandomState()
        data = self.get_input(pick_and_place=True, pick_and_fling=False
            )['pick_and_place']
        info = data['info']
        is_valid = data['is_valid']
        if self.must_pick_on_obj:
            is_valid = is_valid & info['pick_on_obj']
        if np.max(is_valid) == False:
            return False
        valid_coords = np.stack(np.nonzero(is_valid)).T
        idx = rs.choice(len(valid_coords))
        coord = tuple(valid_coords[idx])
        return self.center_pick_and_place_coord(scene, coord, info)
    
    def random_pick_and_fling(self, scene, rs=None):
        if rs is None:
            rs = np.random.RandomState()
        data = self.get_input(pick_and_place=False, pick_and_fling=True
            )['pick_and_fling']
        info = data['info']
        is_valid = data['is_valid']
        if self.must_pick_on_obj:
            is_valid = is_valid & info['pick_on_obj']
        if np.max(is_valid) == False:
            return False
        valid_coords = np.stack(np.nonzero(is_valid)).T
        idx = rs.choice(len(valid_coords))
        coord = tuple(valid_coords[idx])
        return self.pick_and_fling_coord(scene, coord, info)

def test():
    from cair_robot.cameras.kinect_client import KinectClient
    from cair_robot.scenes.pybullet_dual_arm_table_scene import PyBulletDualArmTableScene

    tx_table_camera = np.loadtxt('cam_pose/cam2table_pose.txt')
    tx_left_camera = np.loadtxt('cam_pose/cam2left_pose.txt')
    tx_right_camera = np.loadtxt('cam_pose/cam2right_pose.txt')
    tx_camera_view = np.loadtxt('cam_pose/view2cam.txt')

    camera = KinectClient('128.59.23.32', '8080', fielt_bg=False)
    # scene = PyBulletDualArmTableScene(tx_table_camera, tx_left_camera, tx_right_camera)
    env = TransformedViewEnv(kinect_camera=camera, 
        tx_camera_img=tx_camera_view, tx_world_camera=tx_table_camera,
        tx_left_camera=tx_left_camera, tx_right_camera=tx_right_camera)

    import time
    s = time.perf_counter()
    data = env.get_input()
    print(time.perf_counter() - s)

def test_real_fling():
    from cair_robot.cameras.kinect_client import KinectClient
    from cair_robot.scenes.dual_arm_table_scene import DualArmTableScene
    from cair_robot.robots.ur5_robot import UR5RTDE
    from cair_robot.robots.grippers import WSG50

    tx_table_camera = np.loadtxt('cam_pose/cam2table_pose.txt')
    tx_left_camera = np.loadtxt('cam_pose/cam2left_pose.txt')
    tx_right_camera = np.loadtxt('cam_pose/cam2right_pose.txt')
    tx_camera_view = np.loadtxt('cam_pose/view2cam.txt')

    camera = KinectClient('128.59.23.32', '8080', fielt_bg=False)
    # scene = PyBulletDualArmTableScene(tx_table_camera, tx_left_camera, tx_right_camera)
    env = TransformedViewEnv(kinect_camera=camera, 
        tx_camera_img=tx_camera_view, tx_world_camera=tx_table_camera,
        tx_left_camera=tx_left_camera, tx_right_camera=tx_right_camera)
    fling_generator = env.get_valid_fling_params()

    wsg50 = WSG50('192.168.0.231', 1002)
    left_ur5 = UR5RTDE('192.168.0.139', wsg50) # latte
    right_ur5 = UR5RTDE('192.168.0.204', 'rg2') # oolong
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

    import time
    s = time.perf_counter()
    data = env.get_input()
    print(time.perf_counter() - s)

    pp_data = data['pick_and_fling']
    obs = pp_data['obs']
    is_valid = pp_data['is_valid']
    info = pp_data['info']
    left_coord = info['left_coord']
    right_coord = info['right_coord']

    valid_coords = np.stack(np.nonzero(is_valid)).T
    rs = np.random.RandomState()
    idxs = rs.choice(len(valid_coords), size=1000)
    for i, idx in enumerate(idxs):
        print(i, idx)
        coord = valid_coords[idx]
        left = left_coord[tuple(coord)]
        right = right_coord[tuple(coord)]
        width = np.linalg.norm((left-right)[:2])
        a, b = fling_generator.get_action_paths(width)
        strech_l, strech_r = a[0], b[0]
        max_width = fling_generator.get_max_width()

        r = scene.dual_arm_pick(left, right,
            pick_height=0.02,
            lift_height=0.3)
        # assert(r)
        r = scene.dual_arm_strech(strech_l, strech_r, 
            max_width=max_width)
        assert(r)
        post_width = scene.get_tcp_distance()
        left_path, right_path = fling_generator.get_action_paths(post_width)
        r = scene.dual_arm_movel(left_path[0], right_path[0])
        r = scene.dual_arm_fling(left_path, right_path, 
            # speed=0.2, acceleration=1)
            speed=1.3, acceleration=5)
        assert(r)
        scene.open_grippers()
        scene.lift_grippers(0.2)
        scene.home()


def test_real_pick_place():
    from cair_robot.cameras.kinect_client import KinectClient
    from cair_robot.scenes.dual_arm_table_scene import DualArmTableScene
    from cair_robot.robots.ur5_robot import UR5RTDE
    from cair_robot.robots.grippers import WSG50

    tx_table_camera = np.loadtxt('cam_pose/cam2table_pose.txt')
    tx_left_camera = np.loadtxt('cam_pose/cam2left_pose.txt')
    tx_right_camera = np.loadtxt('cam_pose/cam2right_pose.txt')
    tx_camera_view = np.loadtxt('cam_pose/view2cam.txt')

    camera = KinectClient('128.59.23.32', '8080', fielt_bg=False)
    # scene = PyBulletDualArmTableScene(tx_table_camera, tx_left_camera, tx_right_camera)
    env = TransformedViewEnv(kinect_camera=camera, 
        tx_camera_img=tx_camera_view, tx_world_camera=tx_table_camera,
        tx_left_camera=tx_left_camera, tx_right_camera=tx_right_camera)
    
    wsg50 = WSG50('192.168.0.231', 1002)
    left_ur5 = UR5RTDE('192.168.0.139', wsg50) # latte
    right_ur5 = UR5RTDE('192.168.0.204', 'rg2') # oolong
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


    # random pick and place
    while True:
        transformer = ImageStackTransformer(
            rotations=[0], scales=[1])
        color, depth, obj_mask = env.get_obs()
        data = env.transform_obs(transformer=transformer, 
            depth=depth, obj_mask=obj_mask)
        mask = data['obj_mask']
        coords = data['world_coord'][0]

        # is_left = np.random.random() > 0.5
        is_left = True
        robot_valid = env.is_coord_valid_robot(coords, is_left=is_left)
        valid_mask = robot_valid & mask
        valid_idxs = np.stack(np.nonzero(valid_mask)).T
        idx = valid_idxs[np.random.choice(len(valid_idxs), size=2)]
        start = coords[tuple(idx[0])]
        end = coords[tuple(idx[1])]

        start_z = max(start[2] - 0.01, 0.05)
        print(start_z)
        start[2] = start_z
        end_z = max(end[2] + 0.02, 0.08)
        end[2] = end_z

        r = scene.single_arm_pick_and_place(
            is_left, start, end, 
            min_pick_height=0,
            lift_height=0.2)
        assert(r)
        scene.home()



    import time
    s = time.perf_counter()
    data = env.get_input()
    print(time.perf_counter() - s)

    pp_data = data['pick_and_place']
    obs = pp_data['obs']
    is_valid = pp_data['is_valid']
    info = pp_data['info']
    use_left = info['use_left']
    start_coord = info['start_coord']
    end_coord = info['end_coord']

    valid_coords = np.stack(np.nonzero(is_valid)).T
    rs = np.random.RandomState()
    idxs = rs.choice(len(valid_coords), size=1000)
    for i, idx in enumerate(idxs):
        print(i, idx)
        coord = valid_coords[idx]
        is_left = use_left[tuple(coord)]
        start = start_coord[tuple(coord)]
        end = end_coord[tuple(coord)]
        r = scene.single_arm_pick_and_place(
            is_left, start, end, 
            min_pick_height=0.01,
            lift_height=0.2)
        assert(r)
        scene.home()



def test_real_random():
    from cair_robot.cameras.kinect_client import KinectClient
    from cair_robot.scenes.dual_arm_table_scene import DualArmTableScene
    from cair_robot.robots.ur5_robot import UR5RTDE
    from cair_robot.robots.grippers import WSG50

    tx_table_camera = np.loadtxt('cam_pose/cam2table_pose.txt')
    tx_left_camera = np.loadtxt('cam_pose/cam2left_pose.txt')
    tx_right_camera = np.loadtxt('cam_pose/cam2right_pose.txt')
    tx_camera_view = np.loadtxt('cam_pose/view2cam.txt')

    camera = KinectClient('128.59.23.32', '8080', fielt_bg=False)
    # scene = PyBulletDualArmTableScene(tx_table_camera, tx_left_camera, tx_right_camera)
    env = TransformedViewEnv(kinect_camera=camera, 
        tx_camera_img=tx_camera_view, tx_world_camera=tx_table_camera,
        tx_left_camera=tx_left_camera, tx_right_camera=tx_right_camera)
    
    wsg50 = WSG50('192.168.0.231', 1002)
    left_ur5 = UR5RTDE('192.168.0.139', wsg50) # latte
    right_ur5 = UR5RTDE('192.168.0.204', 'rg2') # oolong
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
    while True:
        env.random_pick_and_place(scene=scene, rs=rs)
        # env.random_pick_and_fling(scene=scene, rs=rs)

