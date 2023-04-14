import numpy as np
import open3d as o3d


def get_extrinsic_look_at(location, focus_point):
    assert(not np.allclose(focus_point, location))
    focus_point_local = focus_point - location
    focus_distance = np.linalg.norm(focus_point_local)

    z_local = focus_point_local / focus_distance
    z_global = np.array([0, 0, 1])

    x_local = np.array([1, 0, 0])
    if not np.allclose(np.abs(z_local[2]), 1):
        x_local = np.cross(z_local, z_global)
        x_local /= np.linalg.norm(x_local)

    y_local = np.cross(z_local, x_local)

    Tx_world_cv = np.eye(4)
    Tx_world_cv[:3, 0] = x_local
    Tx_world_cv[:3, 1] = y_local
    Tx_world_cv[:3, 2] = z_local
    Tx_world_cv[:3, 3] = location

    Tx_cv_world = np.linalg.inv(Tx_world_cv)
    return Tx_cv_world


class Renderer:
    def __init__(self,
            width=1024, 
            height=1024):
        self.width = width
        self.height = height
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height)
        self.vis = vis
        # set render option
        self.set_rendering_options()
        self.closed = False
        
    
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        self.close()
    
    def __del__(self):
        self.close()
    
    def close(self):
        if not self.closed:
            self.vis.destroy_window()
            self.vis.close()
            self.closed = True
    
    def set_rendering_options(self, 
            light_on=True,
            mesh_show_back_face=True,
            mesh_show_wireframe=False,
            point_size=4.0):
        vis = self.vis
        render_opt = vis.get_render_option()
        render_opt.light_on = light_on
        render_opt.mesh_show_back_face = mesh_show_back_face
        render_opt.mesh_show_wireframe = mesh_show_wireframe
        render_opt.show_coordinate_frame = False
        render_opt.point_size = point_size
    
    def set_camera(self, 
            focal_length=2048,
            camera_loc=[0,0,1], 
            gaze_loc=[0.5,0.5,0.5]):
        width = self.width
        height = self.height
        vis = self.vis
        ctr = vis.get_view_control()
        pinhole_param = ctr.convert_to_pinhole_camera_parameters()
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, 
            focal_length, focal_length,
            (width-1)/2, (height-1)/2)
        pinhole_param.intrinsic = intrinsic
        extrinsic = get_extrinsic_look_at(
            np.array(camera_loc), np.array(gaze_loc))
        pinhole_param.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(pinhole_param)
    
    def render_mesh_sphere(self, 
            verts, faces, vert_colors, 
            sphere_loc, sphere_color, sphere_radius=0.05,
            focal_length=2048,
            camera_loc=[0,0,1], 
            gaze_loc=[0.5,0.5,0.5],
            **kwargs):
        vis = self.vis
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        if vert_colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors)
        mesh.compute_vertex_normals()

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere.translate(sphere_loc)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(sphere_color)

        vis.add_geometry(sphere, reset_bounding_box=True)
        vis.add_geometry(mesh, reset_bounding_box=True)
        self.set_camera(
            focal_length=focal_length, 
            camera_loc=camera_loc, 
            gaze_loc=gaze_loc)
        image = self.render_rgba_image()
        vis.clear_geometries()
        return image
    
    def render_point_cloud_sphere(
            self, points, point_colors,
            sphere_loc, sphere_color, sphere_radius=0.05,
            focal_length=2048,
            camera_loc=[0,0,1], 
            gaze_loc=[0.5,0.5,0.5],
            **kwargs):
        vis = self.vis
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(point_colors)

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere.translate(sphere_loc)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(sphere_color)

        vis.add_geometry(sphere, reset_bounding_box=True)
        vis.add_geometry(pc, reset_bounding_box=True)
        self.set_camera(
            focal_length=focal_length, 
            camera_loc=camera_loc, 
            gaze_loc=gaze_loc)
        image = self.render_rgba_image()
        vis.clear_geometries()
        return image

    def render_mesh(self, verts, faces, vert_colors=None, 
            focal_length=2048,
            camera_loc=[0,0,1], 
            gaze_loc=[0.5,0.5,0.5],
            **kwargs):
        vis = self.vis
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        if vert_colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors)
        mesh.compute_vertex_normals()

        vis.add_geometry(mesh, reset_bounding_box=True)
        self.set_camera(
            focal_length=focal_length, 
            camera_loc=camera_loc, 
            gaze_loc=gaze_loc)
        image = self.render_rgba_image()
        vis.clear_geometries()
        return image
    
    def render_point_cloud(self, pc,
            focal_length=2048,
            camera_loc=[0,1,0], 
            gaze_loc=[0.5,0.5,0.5],
            **kwargs):
        vis = self.vis
        # pc = o3d.geometry.PointCloud()
        # pc.points = o3d.utility.Vector3dVector(points)
        # pc.colors = o3d.utility.Vector3dVector(point_colors)

        vis.add_geometry(pc, reset_bounding_box=True)
        self.set_camera(
            focal_length=focal_length, 
            camera_loc=camera_loc, 
            gaze_loc=gaze_loc)
        image = self.render_rgba_image()
        vis.clear_geometries()

        return image

    def render_rgba_image(self):
        vis = self.vis

        vis.poll_events()
        vis.update_renderer()
        o3d_rgb = vis.capture_screen_float_buffer(do_render=False)
        o3d_d = vis.capture_depth_float_buffer(do_render=False)

        rgb_image = np.asarray(o3d_rgb)
        d_image = np.asarray(o3d_d)
        alpha_image = (d_image != 0).astype(np.float32)
        rgba_image = np.concatenate([rgb_image, np.expand_dims(alpha_image, axis=-1)], axis=-1)
        return rgba_image
    
    def generate_rotating_camera_configs(self,
            focal_length=2048,
            camera_loc=[0,0,1], 
            gaze_loc=[0.5,0.5,0.5],
            num_views=64):
        
        camera_loc = np.array(camera_loc)
        gaze_loc = np.array(gaze_loc)
        camera_gaze_frame = camera_loc - gaze_loc

        cam_heigh = camera_gaze_frame[2]
        radius = np.linalg.norm(camera_gaze_frame[:2])
        theta = np.arctan2(camera_gaze_frame[1], camera_gaze_frame[0])

        delta = np.pi * 2 / num_views
        view_thetas = np.arange(num_views) * delta + theta
        view_x = np.cos(view_thetas) * radius
        view_y = np.sin(view_thetas) * radius

        rows = list()
        for i in range(num_views):
            view_camera_gaze_frame = np.array([
                view_x[i], view_y[i], cam_heigh])
            view_camera = view_camera_gaze_frame + gaze_loc
            rows.append({
                'focal_length': focal_length,
                'camera_loc': view_camera,
                'gaze_loc': gaze_loc
            })
        return rows
    
    def render_mesh_rotating(self, 
            verts, faces, vert_colors=None, 
            sphere_loc=None, sphere_color=None, sphere_radius=0.05,
            focal_length=2048,
            camera_loc=[0,0,1], 
            gaze_loc=[0.5,0.5,0.5],
            num_views=16,
            **kwargs):
        
        # generate camera configurations
        vis = self.vis
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        if vert_colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors)
        mesh.compute_vertex_normals()

        if sphere_loc is not None:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
            sphere.translate(sphere_loc)
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color(sphere_color)
            vis.add_geometry(sphere, reset_bounding_box=True)

        vis.add_geometry(mesh, reset_bounding_box=True)
        camera_configs = self.generate_rotating_camera_configs(
            focal_length=focal_length, 
            camera_loc=camera_loc,
            gaze_loc=gaze_loc,
            num_views=num_views)
        images = list()
        for camera_config in camera_configs:
            self.set_camera(**camera_config)
            image = self.render_rgba_image()
            images.append(image)
        vis.clear_geometries()
        return images

    def render_point_cloud_rotating(
            self, pc,
            sphere_loc=None, sphere_color=None, sphere_radius=0.05,
            focal_length=2048,
            camera_loc=[0,0,1], 
            gaze_loc=[0.5,0.5,0.5],
            num_views=16,
            **kwargs):
    
        vis = self.vis
        # pc = o3d.geometry.PointCloud()
        # pc.points = o3d.utility.Vector3dVector(points)
        # pc.colors = o3d.utility.Vector3dVector(point_colors)

        # if sphere_loc is not None:
        #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        #     sphere.translate(sphere_loc)
        #     sphere.compute_vertex_normals()
        #     sphere.paint_uniform_color(sphere_color)
        #     vis.add_geometry(sphere, reset_bounding_box=True)

        vis.add_geometry(pc, reset_bounding_box=True)
        camera_configs = self.generate_rotating_camera_configs(
            focal_length=focal_length, 
            camera_loc=camera_loc,
            gaze_loc=gaze_loc,
            num_views=num_views)
        images = list()
        for camera_config in camera_configs:
            self.set_camera(**camera_config)
            image = self.render_rgba_image()
            images.append(image)
        vis.clear_geometries()
        return images

    def render_mesh_interpolated(self, 
            faces,
            verts_pair,
            vert_colors_pair,
            interp_values,
            focal_length=2048,
            camera_loc=[0,0,1], 
            gaze_loc=[0.5,0.5,0.5],
            **kwargs):
        
        # generate camera configurations
        vis = self.vis
        
        images = list()
        for interp_value in interp_values:
            verts = verts_pair[0] * (1-interp_value) + verts_pair[1] * interp_value
            vert_colors = vert_colors_pair[0] * (1-interp_value) + vert_colors_pair[1] * interp_value

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors)
            mesh.compute_vertex_normals()

            vis.add_geometry(mesh, reset_bounding_box=True)
            self.set_camera(
                focal_length=focal_length, 
                camera_loc=camera_loc, 
                gaze_loc=gaze_loc)
            image = self.render_rgba_image()
            images.append(image)
            vis.clear_geometries()
        return images

    def render_point_cloud_interpolated(
        self, 
        points_pair, 
        point_colors_pair, 
        interp_values,
        focal_length=2048,
        camera_loc=[0,0,1], 
        gaze_loc=[0.5,0.5,0.5],
        **kwargs):
        vis = self.vis

        images = list()
        for interp_value in interp_values:
            points = points_pair[0] * (1-interp_value) + points_pair[1] * interp_value
            point_colors = point_colors_pair[0] * (1-interp_value) + point_colors_pair[1] * interp_value

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points)
            pc.colors = o3d.utility.Vector3dVector(point_colors)

            vis.add_geometry(pc, reset_bounding_box=True)
            self.set_camera(
                focal_length=focal_length, 
                camera_loc=camera_loc, 
                gaze_loc=gaze_loc)
            image = self.render_rgba_image()
            vis.clear_geometries()
            images.append(image)
        return images