import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from socket import socket, AF_INET, SOCK_STREAM
from threading import Thread

def setup_thread(target):
    thread = Thread(target=target)
    thread.daemon = True
    thread.start()
    return thread

def visualize_vmaps(vmaps, masks=None, mask_fill=None, num_scales=5, cmap='jet', title="", ret=False):

    print("num_scales", num_scales)

    if masks is not None and mask_fill is None:
        mask_fill = vmaps[masks].min()
    
    num_rotations = vmaps.shape[0]//num_scales
    # print(vmaps.shape[0])
    # print("Num scales", num_scales, "Num rotations", num_rotations)

    full_img_shape = (num_rotations * 128, num_scales * 128)

    full_img = np.zeros(full_img_shape)
    mask = np.zeros(full_img_shape).astype(np.int32)

    for i in range(num_rotations):
        for j in range(num_scales):
            full_img[i * 128: i * 128 + 128, j * 128 : j * 128 + 128] = vmaps[i * num_scales + j]
            if masks is not None:
                mask[i * 128: i * 128 + 128, j * 128 : j * 128 + 128] = masks[i * num_scales + j]
    
    if masks is not None:
        full_img[~mask] = mask_fill
    full_img = full_img * mask + (1 - mask) * mask_fill

    if ret:
        return full_img
    else:
        fig, ax = plt.subplots(1 ,1, figsize=(num_scales, num_rotations))
        ax.imshow(full_img, cmap=cmap)
        ax.set_title(title + f"_max:{full_img.max():.2f}")
        ax.set_axis_off()
        fig.tight_layout()
        plt.show()

def sharpen_edges(rgb, threshold1=0, threshold2=100):
    # kernel = np.ones((2,2), np.uint8)
    edges = cv2.Canny(rgb.transpose(1, 2, 0), threshold1=0, threshold2=100)
    new_image = rgb * np.stack([(255 - edges)/255]*3).astype(np.int32)
    # print("new image shape", new_image.shape)
    return new_image

    # old_nocs_x, old_nocs_y = nocs_from_rgb(pretransform_observation, orn_net)
    # new_nocs_x, new_nocs_y = nocs_from_rgb(new_image.transpose(2, 0, 1), orientation_network=orn_net)

    # fig, axs = plt.subplots(2, 3)

    # axs[0, 0].imshow(pretransform_observation.transpose(1, 2, 0))
    # axs[0, 1].imshow(old_nocs_x)
    # axs[0, 2].imshow(old_nocs_y)

    # axs[1, 0].imshow(new_image)
    # axs[1, 1].imshow(new_nocs_x)
    # axs[1, 2].imshow(new_nocs_y)



def pltitle(arr):
    return f"max:{arr.max():.2f},min:{arr.min():.2f}"


def visualize_input(policy_input):
    random_plots = 10
    random_indices = np.random.choice(np.arange(policy_input.shape[0]), size=random_plots)
    random_indices=np.arange(random_plots)
    fig, axs = plt.subplots(5, random_plots, figsize=(40, 10))
    for ax in axs.flatten():
        ax.set_axis_off()
    fig.tight_layout()
    for i in range(random_plots):
        inp = policy_input[random_indices[i]]
        axs[0, i].set_title(label=pltitle(inp[:3]))
        axs[0, i].imshow(inp[:3].transpose(1, 2, 0))
        axs[1, i].imshow(inp[7])        
        axs[1, i].set_title(label=pltitle(inp[7]))
        axs[2, i].imshow(inp[8])
        axs[2, i].set_title(label=pltitle(inp[8]))
        axs[3, i].imshow(inp[-2])
        axs[3, i].set_title(f"min:{inp[-2].min():.2f}, max:{inp[-2].max():.2f}")
        axs[4, i].imshow(inp[-1])
        axs[4, i].set_title(f"min:{inp[-1].min():.2f}, max:{inp[-1].max():.2f}")
    plt.show()

def nocs_from_rgb(rgb, orientation_network, network_in_dim=128, n_bins=32):
    """
    Takes in a square rgb image shaped (3, x, x) typed uint8,
    returns tuple of (nocs_x, nocs_y) with shape (128, 128)
    """
    if rgb.shape[:2] != (network_in_dim, network_in_dim):
        rgb = TF.resize(torch.tensor(rgb), network_in_dim)
    network_input_rgb = rgb.unsqueeze(0).float()/255
    out = orientation_network.forward(network_input_rgb).detach() 
    nocs_x_bins = out[0, :, 0, :, :]
    nocs_y_bins = out[0, :, 1, :, :]
    mask = (network_input_rgb[0] > 0).sum(dim=0).bool().float()
    nocs_x = torch.argmax(nocs_x_bins, 0) * mask
    nocs_x /= n_bins - 1
    nocs_y = torch.argmax(nocs_y_bins, 0) * mask 
    nocs_y /= n_bins - 1

    return nocs_x.cpu().numpy(), nocs_y.cpu().numpy()

def draw_circle(img, center, radius, color, thickness=3):
    canvas = np.zeros(img.shape, np.uint8)
    canvas = cv2.circle(canvas, center, radius, color, thickness)
    canvas = (canvas/255).astype(np.float32)
    canvas_mask = np.expand_dims(np.sum(canvas, axis=-1) == 0, -1)
    img = img * canvas_mask + canvas
    return img

def draw_line(img, begin, end, color, thickness=3):
    canvas = np.zeros(img.shape, np.uint8)
    canvas = cv2.line(canvas, begin, end, color, thickness)
    canvas = (canvas/255).astype(np.float32)
    canvas_mask = np.expand_dims(np.sum(canvas, axis=-1) == 0, -1)
    img = img * canvas_mask + canvas
    return img

def draw_text(img, text, org, font_scale, color=(255, 255, 255), thickness=3):
    canvas = np.zeros(img.shape, np.uint8)
    canvas = cv2.putText(canvas, 
                         text, 
                         org, 
                         cv2.FONT_HERSHEY_SIMPLEX, 
                         font_scale, 
                         color, 
                         thickness)
    canvas = (canvas/255).astype(np.float32)
    canvas_mask = np.expand_dims(np.sum(canvas, axis=-1) == 0, -1)
    img = img * canvas_mask + canvas
    return img

def draw_triangle(img, center, size, angle, color, thickness=3):

    canvas = np.zeros(img.shape, np.uint8)

    p1 = center + np.array([np.cos(angle) * size, np.sin(angle) * size])
    p2 = center + np.array([np.cos(angle + np.pi/2) * size/2, np.sin(angle + np.pi/2) * size/2])
    p3 = center + np.array([np.cos(angle - np.pi/2) * size/2, np.sin(angle - np.pi/2) * size/2])

    p1, p2, p3 = p1.astype(int), p2.astype(int), p3.astype(int)

    cv2.line(canvas, p1, p2, color, 3)
    cv2.line(canvas, p2, p3, color, 3)
    cv2.line(canvas, p1, p3, color, 3)

    canvas = (canvas/255).astype(np.float32)

    canvas_mask = np.expand_dims(np.sum(canvas, axis=-1) == 0, -1)
    img = img * canvas_mask + canvas
    return img

def transform_coords(coords, rotation, scale, source_dim, target_dim):
    rotation *= (2*np.pi)/360
    from_center = coords - source_dim // 2

    angle = np.arccos(np.dot(from_center/np.linalg.norm(from_center), [0, 1]))
    if from_center[0] < 0:
        angle = 2*np.pi - angle
    rotation += angle

    len_from_center = np.linalg.norm(from_center)
    scale_ratio = target_dim * scale / source_dim
    new_coords = np.array([np.sin(rotation), np.cos(rotation)]) * len_from_center * scale_ratio
    new_coords = new_coords.astype(np.int32) + target_dim//2
    return new_coords


def draw_fling(img, p1, p2, thickness=2, radius=5):
        COLOR = (0, 255, 100)
        img = draw_circle(img, p1, radius, COLOR, thickness)
        img = draw_circle(img, p2, radius, COLOR, thickness)
        img = draw_line(img, p1, p2, COLOR, thickness)
        return img
            
def draw_place(img, p1, p2, thickness=2, radius=5):
    action_angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    COLOR = (255, 50, 0)
    img = draw_circle(img, p1, radius, COLOR, thickness)
    img = draw_triangle(img, p2, radius, action_angle, COLOR, thickness)
    img = draw_line(img, p1, p2, COLOR, thickness)
    return img

def get_workspace_crop(img):
    return img[:, img.shape[1]//2 - img.shape[0]//2: img.shape[1]//2 + img.shape[0]//2, :]