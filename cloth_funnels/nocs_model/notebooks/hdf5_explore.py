# %%
import os
import sys
proj_dir = os.path.expanduser("~/dev/folding-unfolding/src")
os.chdir(proj_dir)
sys.path.append(proj_dir)

# %%
import h5py
from nocs_model.common.canonical import nocs_to_direction, direction_to_angle

# %%
# buffer_path = '/proj/crv/cchi/data/folding-unfolding/new-fling-align-hard/replay_buffer.hdf5'
buffer_path = '/local/crv/cchi/data/folding-unfolding/garmentnet-dataset/hires_replay_buffer.hdf5'
f = h5py.File(buffer_path, 'r')

all_keys = list(f.keys())

# %%
data = f[all_keys[0]]

input = data['garmentnets_supervision_input'][:]
output = data['garmentnets_supervision_target'][:]

rgb = input[...,:3]/255
nocs = output
angles = direction_to_angle(nocs_to_direction(nocs))
# %%
hires = data['pretransform_observations'][:]
