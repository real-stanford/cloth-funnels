import pickle
import requests
import cv2
import numpy as np

class KinectClient:
    def __init__(self, ip, port, fielt_bg=False):
        self.ip = ip
        self.port = port
        self.fielt_bg = fielt_bg

    def get_intr(self):
        return pickle.loads(requests.get(f'http://{self.ip}:{self.port}/intr').content)

    def get_camera_data(self, n=1, fielt_bg=None):
        cam_data = pickle.loads(requests.get(f'http://{self.ip}:{self.port}/pickle/{n}').content)
        color_img = cam_data['color_img']
        depth_img = cam_data['depth_img']
        depth_img *= 0.973 # camera's depth offset
        if fielt_bg is None:
            fielt_bg = self.fielt_bg
        if fielt_bg:
            mask = (cv2.cvtColor(color_img, cv2.COLOR_RGB2HSV)[:, :, 2] > 150)
            color_img = color_img * mask[:, :, np.newaxis] + (1 - mask[:, :, np.newaxis]) * np.array([90, 89, 89])
            color_img = color_img.astype(np.uint8)
        
        return color_img, depth_img