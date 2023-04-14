import numpy as np
from turtle import position
import h5py
import hashlib
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import pyflex
import torch
import random
from time import sleep
from typing import List
from tqdm import tqdm
from argparse import ArgumentParser
from filelock import FileLock
from pathlib import Path
import trimesh
import ray
import os
import pickle
import open3d as o3d
from cloth_funnels.utils.flex_utils import (
    set_scene,
    get_default_config,
    center_object,
    set_to_flatten,
    wait_until_stable,
    get_current_covered_area,
    PickerPickPlace
)


def get_keypoint_groups(xzy : np.ndarray):
    x = xzy[:, 0]
    y = xzy[:, 2]

    cloth_height = float(np.max(y) - np.min(y))
    cloth_width = float(np.max(x) - np.min(x))
    
    max_ys, min_ys = [], []
    num_bins = 40
    x_min, x_max = np.min(x),  np.max(x)
    mid = (x_min + x_max)/2
    lin = np.linspace(mid, x_max, num=num_bins)
    for xleft, xright in zip(lin[:-1], lin[1:]):
        max_ys.append(-1 * y[np.where((xleft < x) & (x < xright))].min())
        min_ys.append(-1 * y[np.where((xleft < x) & (x < xright))].max())

    #plot the rate of change of the shirt height wrt x
    diff = np.array(max_ys) - np.array(min_ys)
    roc = diff[1:] - diff[:-1]

    #pad beginning and end
    begin_offset = num_bins//5
    end_offset = num_bins//10
    roc[:begin_offset] = np.max(roc[:begin_offset])
    roc[-end_offset:] = np.max(roc[-end_offset:])
    
    #find where the rate of change in height dips, it corresponds to the x coordinate of the right shoulder
    right_x = (x_max - mid) * (np.argmin(roc)/num_bins) + mid

    #find where the two shoulders are and their respective indices
    xzy_copy = xzy.copy()
    xzy_copy[np.where(np.abs(xzy[:, 0] - right_x) > 0.01), 2] = 10
    right_pickpoint_shoulder = np.argmin(xzy_copy[:, 2])
    right_pickpoint_shoulder_pos = xzy[right_pickpoint_shoulder, :]

    left_shoulder_query = np.array([-right_pickpoint_shoulder_pos[0], right_pickpoint_shoulder_pos[1], right_pickpoint_shoulder_pos[2]])
    left_pickpoint_shoulder = (np.linalg.norm(xzy - left_shoulder_query, axis=1)).argmin()
    left_pickpoint_shoulder_pos = xzy[left_pickpoint_shoulder, :]

    #top left and right points are easy to find
    pickpoint_top_right = np.argmax(x - y)
    pickpoint_top_left = np.argmax(-x - y)

    #to find the bottom right and bottom left points, we need to first make sure that these points are
    #near the bottom of the cloth
    pickpoint_bottom = np.argmax(y)
    diff = xzy[pickpoint_bottom, 2] - xzy[:, 2]
    idx = diff < 0.1
    locations = np.where(diff < 0.1)
    points_near_bottom = xzy[idx, :]
    x_bot = points_near_bottom[:, 0]
    y_bot = points_near_bottom[:, 2]

    #after filtering out far points, we can find the argmax as usual
    pickpoint_bottom_right = locations[0][np.argmax(x_bot + y_bot)]
    pickpoint_bottom_left = locations[0][np.argmax(-x_bot + y_bot)]


    all_keypoints = {
        'bottom_right': pickpoint_bottom_right,
        'bottom_left': pickpoint_bottom_left,
        'top_right': pickpoint_top_right,
        'top_left': pickpoint_top_left,
        'right_shoulder': right_pickpoint_shoulder,
        'left_shoulder': left_pickpoint_shoulder,
    }
    def return_close_indices(keypoint_index, xzy, threshold=0.03, num_pts=10):
        # print(np.linalg.norm(xzy[np.where(np.linalg.norm(xzy - xzy[keypoint_index, :], axis=1) < threshold)], axis=-1))
        queries = np.where(np.linalg.norm(xzy - xzy[keypoint_index, :], axis=1) < threshold)[0]
        distances = np.linalg.norm(xzy[queries] - xzy[keypoint_index], axis=1)
        sorted_distances = np.argsort(distances)
        # print(keypoint_index)
        return queries[sorted_distances[:num_pts]]

    #query
    all_keypoint_groups = np.array([np.array(return_close_indices(all_keypoints[key], xzy)) for key in all_keypoints]).astype(int)

    return all_keypoint_groups


def get_vertices(pyflex_positions):
    position = pyflex_positions.reshape(-1, 4)
    mask = np.all(position != np.zeros(4),axis=-1)
    return position[mask,:]

def get_rotation_matrix(rotationVector, angle):
    angle = float(angle)
    axis = rotationVector/np.sqrt(np.dot(rotationVector , rotationVector))
    a = np.cos(angle/2)
    b,c,d = -axis*np.sin(angle/2.)
    return np.array( [ [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                       [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                       [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c] ]) 

def load_cloth(path):
    vertices, faces = [], []
    nocs = pickle.load(open(path,'rb'))
    vertices = nocs['verts']
    faces = nocs['faces']
    uvs = nocs['nocs']
 
    triangle_faces = list()
    for face in faces:
        triangle_faces.append(face[[0,1,2]])
        triangle_faces.append(face[[0,2,3]])

    stretch_edges, shear_edges, bend_edges = set(), set(), set()

    # Stretch & Shear
    for face in faces:
        stretch_edges.add(tuple(sorted([face[0], face[1]])))
        stretch_edges.add(tuple(sorted([face[1], face[2]])))
        stretch_edges.add(tuple(sorted([face[2], face[3]])))
        stretch_edges.add(tuple(sorted([face[3], face[0]])))

        shear_edges.add(tuple(sorted([face[0], face[2]])))
        shear_edges.add(tuple(sorted([face[1], face[3]])))

    # Bend
    neighbours = dict()
    for vid in range(len(vertices)):
        neighbours[vid] = set()
    for edge in stretch_edges:
        neighbours[edge[0]].add(edge[1])
        neighbours[edge[1]].add(edge[0])
    for vid in range(len(vertices)):
        neighbour_list = list(neighbours[vid])
        N = len(neighbour_list)
        for i in range(N - 1):
            for j in range(i+1, N):
                bend_edge = tuple(sorted([neighbour_list[i], neighbour_list[j]]))
                if bend_edge not in shear_edges:
                    bend_edges.add(bend_edge)
    
    vertices = np.array(vertices)
        
    vertices = vertices.dot(get_rotation_matrix(np.array([0, 1, 0]),np.pi))

    return vertices, \
        np.array(triangle_faces), \
        np.array(list(stretch_edges)), \
        np.array(list(bend_edges)), \
        np.array(list(shear_edges)),\
        np.array(uvs).astype(np.float32)

def edge_lengths(mesh : trimesh.Trimesh):
    #compute edge lengths
    edge_vertices = mesh.vertices[mesh.edges.reshape(-1), :].reshape((-1, 2, 3))
    edge_lengths = np.linalg.norm(edge_vertices[:, 0, :] - edge_vertices[:, 1, :], axis=-1)
    return edge_lengths

def is_eval(pkl_file_name):
    # print("IS EVAL", (pkl_file_name[:2] == '06') and (pkl_file_name[:2] == '07'))
    # print(pkl_file_name[:2] )
    return pkl_file_name[:2] in ['05', '06', '07']


def get_mass(pickpoint, curr_pos):
        return curr_pos[4 * pickpoint + 3]


def grasp_point(pickpoint):
    curr_pos = pyflex.get_positions()

    mass = curr_pos[pickpoint * 4 + 3]
    position = curr_pos[pickpoint * 4: pickpoint * 4 + 3]
    # original_inv_mass = curr_pos[pickpoint * 4 + 3]
    # Set the mass of the pickup point to infinity so that
    # it generates enough force to the rest of the cloth
    curr_pos[pickpoint * 4 + 3] = 0

    pyflex.set_positions(curr_pos)
    
    return mass, position

def release_point(pickpoint, mass):
    curr_pos = pyflex.get_positions()
    curr_pos[pickpoint * 4 + 3] = mass
    pyflex.set_positions(curr_pos)
    
def move(pickpoint, final_point, speed=0.05, cond=lambda : True, step_fn=None):
    curr_pos = pyflex.get_positions()
    init_point = curr_pos[pickpoint * 4: pickpoint * 4 + 3].copy()
    for j in range(int(1/speed)):
        if not cond():
            break

        curr_pos = pyflex.get_positions()
        curr_vel = pyflex.get_velocities()
        pickpoint_pos = (final_point-init_point)*(j*speed) + init_point
        curr_pos[pickpoint * 4: pickpoint * 4 + 3] = pickpoint_pos
        curr_pos[pickpoint * 4 + 3] = 0
        curr_vel[pickpoint * 3: pickpoint * 3 + 3] = [0, 0, 0]

        pyflex.set_positions(curr_pos)
        pyflex.set_velocities(curr_vel)

        if step_fn is not None:
            step_fn()

def move_multiple(pickpoints, final_points, speed=0.05, cond=lambda : True, step_fn=None):

    curr_pos = pyflex.get_positions()
    init_points = [curr_pos[pickpoint * 4: pickpoint * 4 + 3].copy() for pickpoint in pickpoints]

    for j in range(int(1/speed)):

        if not cond():
            break

        curr_pos = pyflex.get_positions()
        curr_vel = pyflex.get_velocities()

        pickpoint_poses = [((final_point-init_point)*(j*speed) + init_point) for init_point, final_point in zip(init_points, final_points)]
        for pickpoint, pickpoint_pos in zip(pickpoints, pickpoint_poses):
            curr_pos[pickpoint * 4: pickpoint * 4 + 3] = pickpoint_pos
            curr_pos[pickpoint * 4 + 3] = 0
            curr_vel[pickpoint * 3: pickpoint * 3 + 3] = [0, 0, 0]

        pyflex.set_positions(curr_pos)
        pyflex.set_velocities(curr_vel)
        step_fn()

def fold(pick, hold, target, gui, max_height=0.5, height_thresh=0.07, move_thresh=0.07, speed=0.005, step_fn=None):
        xzy = pyflex.get_positions().reshape(-1, 4)[:, :3]
        hold_point = xzy[hold, :]
        
        init_point = xzy[pick].copy()
        target_point = xzy[target].copy()

        init_point_top = init_point + np.array([0, max_height, 0])

        mass, pos = grasp_point(pick)

        lifting_condition = lambda : np.linalg.norm((pyflex.get_positions()[hold * 4 : hold * 4 + 3] - hold_point)) < height_thresh
        move(pick, init_point_top, speed=speed, cond=lifting_condition, step_fn=step_fn)
        wait_until_stable(gui=gui)

        lifting_condition = lambda : np.linalg.norm((pyflex.get_positions()[hold * 4 : hold * 4 + 3][0] - hold_point[0])) < height_thresh
        move(pick, target_point, speed=speed, cond=lifting_condition, step_fn=step_fn)

        release_point(pick, mass)

def fold_multiple(pick, hold, target, gui, max_height=0.5, height_thresh=0.07, move_thresh=0.07, speed=0.005, step_fn=None):

        xzy = pyflex.get_positions().reshape(-1, 4)[:, :3]
        hold_points = [xzy[h, :] for h in hold]
        
        init_points = [xzy[p].copy() for p in pick]
        target_points = [xzy[p].copy() for p in target]

        init_points_top = [init_point + np.array([0, max_height, 0]) for init_point in init_points]

        masses, posses = [], []
        for p in pick:
            mass, pos = grasp_point(p)
            masses.append(mass)
            posses.append(pos)

        lifting_conditions = [lambda : np.linalg.norm((pyflex.get_positions()[hold[i] * 4 : hold[i] * 4 + 3] - hold_points[i])) < height_thresh for i in range(len(hold))]
        move_multiple(pick, init_points_top, speed=speed, cond=lambda : all([l() for l in lifting_conditions], step_fn=step_fn))
        wait_until_stable(gui=gui)

        lifting_conditions = [lambda : np.linalg.norm((pyflex.get_positions()[hold[i] * 4 : hold[i] * 4 + 3][0] - hold_points[i][0])) < move_thresh for i in range(len(hold))]
        move_multiple(pick, target_points, speed=speed, cond=lambda : all([l() for l in lifting_conditions], step_fn=step_fn))

        for p, m in zip(pick, masses):
            release_point(p, m)