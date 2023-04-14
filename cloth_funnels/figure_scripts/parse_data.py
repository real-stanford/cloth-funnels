import h5py
import numpy as np
import trimesh
import os
import pickle

mesh_output_dir = 'cloth_meshes/'

if __name__ == "__main__":
    assert os.path.exists(mesh_output_dir)
    with h5py.File('replay_buffer.hdf5', 'r') as file:
        for step_key, episode_step in map(lambda k: (k, file[k]), file.keys()):
            num_substeps = len(episode_step['self.env_mesh_vertices'])
            # print(episode_step['end_effector_positions'].shape)
            assert num_substeps == len(
                episode_step['end_effector_positions'])
            faces = np.array(episode_step['task_mesh_faces'])
            pickle.dump(np.array(episode_step['end_effector_positions']), open(
                f'{mesh_output_dir}/postaction_{step_key}_gripper.pkl', 'wb'))
            for substep_idx in range(num_substeps):
                trimesh.Trimesh(
                    vertices=np.array(
                        episode_step['self.env_mesh_vertices'][substep_idx]),
                    faces=faces.reshape(-1, 3)).export(
                    f'{mesh_output_dir}/postaction_{step_key}_{substep_idx:03d}.obj')
