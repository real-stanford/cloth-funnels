

import os
import h5py
import matplotlib.pyplot as plt
import random
import numpy as np
import pickle as pkl
import cv2
import torch
import torchvision
import pytorch_lightning as pl
from cloth_funnels.learning.utils import rgbd_to_tensor
from scipy.spatial import KDTree

network_checkpoint = "/local/crv/acanberk/folding-unfolding/src/learning/nocs_unet_pretrained_2.ckpt"
target_nocs_path = "/local/crv/acanberk/folding-unfolding/src/learning/target_nocs_1.pkl"

#load nocs unet from checkpoint
# nocs_unet = NOCSUNet.load_from_checkpoint(network_checkpoint, n_channels=1, n_classes=64*3)

#finds the symmetrical nocs image
def flip_nocs(nocs_a):
    nocs_b = nocs_a.copy()
    mask = np.all(nocs_b != np.array([0, 0, 0]), axis=2)
    #the shirt is symmetric along yz and xz planes
    (nocs_b[:, :, :2])[mask] = (-1* nocs_b[:, :, :2] + 1)[mask]
    return nocs_b

def nocs_pred(rgbd, model):

    network_input = rgbd_to_tensor(rgbd)

    out = model(network_input).cpu().detach()
    out = out.reshape(-1, 64, 3, out.shape[2], out.shape[3])
    out = np.squeeze(np.argmax(out, axis=1) / 63)
    out = out.permute(1, 2, 0).numpy()

    return out

def correspondence_loss_from_rgbd(rgbd, nocs_gt=None, model=None):

    if nocs_gt == None:
        nocs_gt = pickle_to_rgbd_nocs(target_nocs_path)[1]

    assert rgbd.shape[-1] == 4
    assert nocs_gt.shape[-1] == 3
    assert np.sum(nocs_gt**2) > 0

    pred = nocs_pred(rgbd)

    nocs_target_a = nocs_gt
    nocs_target_b = flip_nocs(nocs_gt)

    correspondence_loss_a = get_correspondence_loss(pred, nocs_target_a)
    correspondence_loss_b = get_correspondence_loss(pred, nocs_target_b)

    return np.min([correspondence_loss_a, correspondence_loss_b])


def correspondence_loss_from_nocs(nocs, nocs_gt=None, model=None):

    if nocs_gt == None:
        nocs_gt = pickle_to_rgbd_nocs(target_nocs_path)[1]

    assert nocs.shape[-1] == 3
    assert nocs_gt.shape[-1] == 3
    assert np.sum(nocs_gt**2) > 0

    nocs_target_a = nocs_gt
    nocs_target_b = flip_nocs(nocs_gt)

    correspondence_loss_a = get_correspondence_loss(nocs, nocs_target_a)
    correspondence_loss_b = get_correspondence_loss(nocs, nocs_target_b)

    return np.min([correspondence_loss_a, correspondence_loss_b])

# creates a (N, 3) array of NOCS coordinates along with a
# (N, 2) array of their 2d coordinates on the image
def get_nocs_and_cartesian(nocs_image):
    nonzero_target = np.all(nocs_image != 0, axis=2)
    nonzero_nocs = nocs_image[nonzero_target]
    nonzero_nocs_array = np.where(nonzero_target == True)
    coordinates = np.concatenate([nonzero_nocs_array[0].reshape(-1, 1), \
        nonzero_nocs_array[1].reshape(-1, 1)], axis=1)
    return nonzero_nocs, coordinates.astype(np.float32)

# finds the symmetrical nocs image
def flip_nocs(nocs_a):
    nocs_b = nocs_a.copy()
    mask = np.all(nocs_b != np.array([0, 0, 0]), axis=2)
    #the shirt is symmetric along yz and xz planes
    (nocs_b[:, :, :2])[mask] = (-1* nocs_b[:, :, :2] + 1)[mask]
    return nocs_b

def get_correspondence_loss(nocs_input, nocs_target):

    target_nocs, target_cartesian = get_nocs_and_cartesian(nocs_target)
    input_nocs, input_cartesian = get_nocs_and_cartesian(nocs_input)
 
    #only use x and z dimensions
    nocs_correspondence_indices = [0, 1, 2]
    tree = KDTree(input_nocs[:, nocs_correspondence_indices])
    _, indices = tree.query(target_nocs[:, nocs_correspondence_indices], k=1)

    correspondence_loss = np.mean(np.linalg.norm(input_cartesian[indices] - target_cartesian, axis=1))

    return correspondence_loss

def pickle_to_rgbd_nocs(path):
    data = pkl.load(open(path, 'rb'))
    rgbd = data[0]
    nocs_gt = data[1]
    return rgbd, nocs_gt

def get_flow_correspondence(nocs, coverage, u=[1, 0], \
    alpha=1, dx=2, dy=2, x_interval=2, y_interval=2, \
        visualize=False):

    # assert coverage >= 0 and coverage <= 1
    coverage = np.clip(coverage, 0, 1)
    assert alpha >= 0 and alpha <= 1

    if(np.sum(nocs**2) == 0):
        return 0, -1 * np.ones((nocs.shape[0], nocs.shape[1]))
    # assert np.sum(nocs) != 0
    #use the nocs z coordinate
    def flow(axis):
        surface = np.pad(nocs[:, :, axis], [(dx, 0), (dy, 0)], mode="constant")
        mask = (surface != 0)[dx:, dy:]
        image = np.repeat(np.expand_dims(surface.copy(), 2), 3, axis=2)
        f_xs = (surface[dx:, :] - surface[:-dx, :])/dx
        f_ys = (surface[:, dy:] - surface[:, :-dy])/dy
        f_ys = f_ys[:f_xs.shape[0], :]
        f_xs = f_xs[:, :f_ys.shape[1]]
        mags = np.sqrt(f_xs**2 + f_ys**2) + 1e-5
        f_xs, f_ys = f_xs/mags, f_ys/mags
        vecs = np.stack([f_xs, f_ys])

        if axis == 2:
            dot_product_map = f_xs*u[1] + f_ys*u[0]
        elif axis == 0:
            dot_product_map = -1*f_xs*u[0] + f_ys*u[1]
            if np.mean(dot_product_map[np.where(mask == True)]) < np.mean(-1 * dot_product_map[np.where(mask == True)]):
                dot_product_map = -1 * dot_product_map


        # dot_product_map = dot_product_map * mask + -1 * np.ones(dot_product_map.shape) * (1 - mask)
        dot_product_map = (dot_product_map + 1)/2

        u_dot_gradients = dot_product_map[np.where(mask == True)].reshape(-1, 1)

        flow_correspondence =  np.mean(np.array(u_dot_gradients)) * coverage

        return (flow_correspondence, np.mean(np.array(u_dot_gradients)), np.clip(dot_product_map, 0, 1), vecs, mask)

    z_flow_correspondence, z_direction, z_dot_product_map, z_vecs, mask = flow(2)
    x_flow_correspondence, x_direction, x_dot_product_map, x_vecs, _ = flow(0)

    dot_product_map = 0.5 * z_dot_product_map + 0.5 * x_dot_product_map
    flow_correspondence = 0.5 * z_flow_correspondence + 0.5 * x_flow_correspondence
    direction_score = 0.5 * z_direction + 0.5 * x_direction

    weighted_flow_correspondence = flow_correspondence * alpha + (1 - alpha) * coverage
   
    if visualize:
        print("visualizing")
        plt.subplot(1, 4, 1)
        plt.imshow(nocs)
        plt.subplot(1, 4, 2)
        plt.title("X Flow: {:.2f}".format(x_flow_correspondence))
        plt.imshow(x_dot_product_map)
        plt.subplot(1, 4, 3)
        plt.title("Z Flow: {:.2f}".format(z_flow_correspondence))
        plt.imshow(z_dot_product_map)
        plt.subplot(1, 4, 4)
        plt.title("Combined Flow: {:.2f}".format(flow_correspondence))
        plt.imshow(dot_product_map)
        print("X Flow Correspondence: {:.2f}".format(x_flow_correspondence))
        print("Z Flow Correspondence: {:.2f}".format(z_flow_correspondence))
        print("Coverage {:.2f}".format(coverage))
        print("Z Direction {:.2f}".format(z_direction))
        print("X Direction {:.2f}".format(x_direction))
        
        plt.show()


    retval = {
        "x_flow_correspondence": x_flow_correspondence,
        "z_flow_correspondence": z_flow_correspondence,
        "x_direction": x_direction,
        "z_direction": z_direction,
        "coverage": coverage,
        "x_dot_product_map": x_dot_product_map,
        "z_dot_product_map": z_dot_product_map,
        "dot_product_map": dot_product_map,
        "flow_correspondence": flow_correspondence,
        "dot_product_map": dot_product_map,
        "direction": direction_score,
        "weighted_flow_correspondence": weighted_flow_correspondence,
        "x_vecs":x_vecs,
        "z_vecs":z_vecs,
        "mask":mask
    }

    return retval



# %%


if __name__ == "__main__":
    import sys 
    sys.path.append("/local/crv/acanberk/folding-unfolding/src/")

    from learning.nocs_unet_inference import flip_nocs, nocs_pred, pickle_to_rgbd_nocs, get_flow_correspondence
    import matplotlib.pyplot as plt 
    import numpy as np
    import torch
    import torchvision

    replay_buffer_path = "/local/crv/acanberk/folding-unfolding/src/correspondence_images_"
    target_nocs_path = "/local/crv/acanberk/folding-unfolding/src/learning/target_nocs_1.pkl"
    num_figs = 10

    _, target_nocs_a = pickle_to_rgbd_nocs(target_nocs_path)
    target_nocs_b = flip_nocs(target_nocs_a)

    for file in os.listdir(replay_buffer_path):
        #load numpy array from .npy and plot
        print(file)
        env_path = os.path.join(replay_buffer_path, file)
        steps = os.listdir(env_path)
        steps.sort()

        for step_n in range(5):

        
            step_path = os.path.join(env_path, steps[step_n])
            
            #create a 5,2 plot
            # fig, axs = plt.subplots(num_figs, 6, figsize=(12, num_figs*2.5))

            rgbd_instance, gt_nocs_instance = pickle_to_rgbd_nocs(step_path)
            floor_depth = np.max(rgbd_instance[:,:,3])
            # print("Floor depth ", floor_depth)

            for i in range(num_figs):
                ##

                degrees = 360
                translate = (0.25, 0.25)
                scale = (1, 1)
                shear = 0.0

                rgbd, gt_nocs = rgbd_instance.copy(), gt_nocs_instance.copy()

                transformation = torchvision.transforms.RandomAffine(degrees=degrees, \
                    translate=translate, scale=scale, shear=shear, \
                        interpolation=torchvision.transforms.InterpolationMode.NEAREST)
                
                rgbd = torch.tensor(rgbd.transpose(2, 0, 1))
                gt_nocs = torch.tensor(gt_nocs.transpose(2, 0, 1))
                rgbd_nocs = torch.cat([rgbd, gt_nocs], dim=0)
                rgbd_nocs = transformation(rgbd_nocs)
                rgbd, gt_nocs = rgbd_nocs[:4, :, :], rgbd_nocs[4:, :, :]
                rgbd = rgbd.permute(1, 2, 0).numpy()
                gt_nocs = gt_nocs.permute(1, 2, 0).numpy()

                d = rgbd[:,:,3]
                # plt.imshow(d)
                # plt.show()
                zero_depth_indices = np.where(d < 0.2)
                d[zero_depth_indices] = floor_depth
                # print(np.min(d))
                # plt.imshow(d)
                # plt.show()

                rgbd[:, :, 3] = d

                coverage = np.sum((gt_nocs[:, :, 2] != 0))
                if step_n == 0:
                    max_coverage = coverage

                rew, map, dir, flow  = get_flow_correspondence(gt_nocs, coverage/max_coverage, visualize=True)


# %%

# %%
