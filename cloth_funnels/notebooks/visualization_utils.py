import numpy as np
import torch
import cv2

def get_episode_keys(k, steps=8):
    episode = int(k.split("_")[0])
    step = int(k.split("_")[1][4:])
    keys = []
    for i in range(steps):
        key = f"{episode:09d}_step{i:02d}"
        if i == int(steps) - 1:
            key += "_last"
        keys.append(key)
    return keys
        

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

    angle = np.arccos(np.dot(from_center/(np.linalg.norm(from_center) + 1e-6), [0, 1]))
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