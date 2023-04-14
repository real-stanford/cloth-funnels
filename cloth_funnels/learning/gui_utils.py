import tkinter as tk
from PIL import ImageTk, Image
import cv2
import numpy as np
from itertools import product

def draw_circled_lines(pixels, shape=None, img=None, thickness=1):
    if shape is not None:
        shape = list(shape) + [4]
        img = np.zeros(shape)
    else:
        assert img is not None
    left, right = pixels
    img = cv2.circle(
        img=img, center=(int(left[1]), int(left[0])),
        radius=thickness*2, color=(0, 1, 0, 1), thickness=thickness)
    img = cv2.line(
        img=img,
        pt1=(int(left[1]), int(left[0])),
        pt2=(int(right[1]), int(right[0])),
        color=(1, 1, 0, 1), thickness=thickness)
    img = cv2.circle(
        img=img, center=(int(right[1]), int(right[0])),
        radius=thickness*2, color=(1, 0, 0, 1), thickness=thickness)
    return img

def draw_arrow(pixels, shape=None, img=None, thickness=1):
    if shape is not None:
        shape = list(shape) + [4]
        img = np.zeros(shape)
    else:
        assert img is not None
    left, right = pixels
    img = cv2.circle(
        img=img, center=(int(left[1]), int(left[0])),
        radius=thickness*2, color=(0.5, 1, 0.5, 1), thickness=thickness)
    img = cv2.line(
        img=img,
        pt2=(int((left[1] + right[1])/2), int((left[0] + right[0])/2)),
        pt1=(int(left[1]), int(left[0])),
        color=(1, 1, 0, 1), thickness=thickness)
    # img = cv2.circle(
    #     img=img, center=(int(right[1]), int(right[0])),
    #     radius=thickness*2, color=(1, 0, 0, 1), thickness=thickness)
    return img

def draw_circle_arrows(pixels, shape=None, img=None, thickness=1):
    if shape is not None:
        shape = list(shape) + [4]
        img = np.zeros(shape)
    else:
        assert img is not None
    left, right = pixels

    orth = np.array(left)-np.array(right) 
    orth[0], orth[1] = -orth[1], orth[0]
    mid = ((np.array(left) + np.array(right))/2).astype(int)
    mid_end = mid - 0.1 * orth

    img = cv2.circle(
        img=img, center=(int(left[1]), int(left[0])),
        radius=thickness*2, color=(0, 1, 0, 1), thickness=thickness)
    img = cv2.line(
        img=img,
        pt1=(int(left[1]), int(left[0])),
        pt2=(int(right[1]), int(right[0])),
        color=(1, 1, 0, 1), thickness=thickness)
    img = cv2.line(
        img=img,
        pt1=(int(mid[1]), int(mid[0])),
        pt2=(int(mid_end[1]), int(mid_end[0])),
        color=(1, 1, 0, 1), thickness=thickness)
    img = cv2.circle(
        img=img, center=(int(right[1]), int(right[0])),
        radius=thickness*2, color=(1, 0, 0, 1), thickness=thickness)
    return img
    # img = cv2.circle(
    #     img=img, center=(int(right[1]), int(right[0])),
    #     radius=thickness*2, color=(1, 0, 0, 1), thickness=thickness)
    return img


class ExpertDemonstrationWindow:
    def __init__(self,
                 rotations,
                 scales,
                 images,
                 pix_grasp_dist,
                 pix_place_dist,
                 action_primitives,
                 primitive_vmap_indices,
                 ):


        self.action_primitives = action_primitives
        self.action_primitive = action_primitives[0]
        self.primitive_id = 0

        self.scale_idx = 0
        self.rotation_idx = 0
        self.rotation_dict = rotations
        self.transformations_dict = {primitive: list(product(
            rotations, scales)) for primitive, rotations in self.rotation_dict.items()}

        print("Transformations dict information:")
        for k,v in self.transformations_dict.items():
            print(f"{k}: {len(v)}")

        self.pix_grasp_dist = pix_grasp_dist
        self.image_dict = {primitive: images[primitive_vmap_indices[primitive][0]:primitive_vmap_indices[primitive][1]].numpy() for primitive in self.action_primitives}

        print("Image dict information:")
        for k,v in self.image_dict.items():
            print(f"{k}: {len(v)}")


        print("Image dict shapes")
        for primitive in self.action_primitives:
            print(f"{primitive}: {self.image_dict[primitive].shape}")

        dim = images.shape[-1]
        self.center_grasp = np.array([dim//2, dim//2])
        self.scales = scales
        self.window = tk.Tk()
        title = tk.Label(text="Expert Demonstration",
                         font=("Ubuntu", 32))
        title.pack()
        self.action_text = tk.Label(
            text=self.get_action_text(), font=("Ubuntu", 20))
        self.action_text.pack()
        self.canvas = tk.Canvas(
            self.window,
            width=400, height=400)
        self.canvas.pack()
        self.canvas.create_image(
            0, 0, anchor=tk.NW, image=self.get_obs())
        self.window.bind('<KeyPress>', self.on_key_press)


    def update_action_primitive(self):
        self.rotation_idx = self.rotation_idx % len(self.rotation_dict[self.action_primitive])
        self.action_primitive = \
            self.action_primitives[self.primitive_id % len(self.action_primitives)]

    def get_action_text(self):
        return f"{self.action_primitive} Scale: {self.scales[self.scale_idx]}|" +\
            f" Rotation: {self.rotation_dict[self.action_primitive][self.rotation_idx]}"

    def get_obs(self):
        item = (self.rotation_dict[self.action_primitive][self.rotation_idx],
                self.scales[self.scale_idx])
        self.image_idx = self.transformations_dict[self.action_primitive].index(item)
        self.left_grasp = self.center_grasp.copy()
        self.left_grasp[0] = self.left_grasp[0] + self.pix_grasp_dist
        self.right_grasp = self.center_grasp.copy()
        self.right_grasp[0] = self.right_grasp[0] - self.pix_grasp_dist
        img = self.image_dict[self.action_primitive][self.image_idx][:3, :, :].transpose(1,2,0).copy()
        img = (img - img.min())/(img.max() - img.min())
        img = cv2.cvtColor(
            np.float32(img*255),
            cv2.COLOR_RGB2GRAY)

        if self.action_primitive in ['fling','drag']:
            img = draw_circled_lines(
                img=img,
                pixels=[self.left_grasp, self.right_grasp])
        elif self.action_primitive == 'place':
            img = draw_arrow(
                img=img,
                pixels=[self.left_grasp, self.right_grasp])
        elif self.action_primitive == 'stretchdrag':
            img = draw_circle_arrows(
                img=img,
                pixels=[self.left_grasp, self.right_grasp])
        else:
            raise NotImplementedError


        one = ImageTk.PhotoImage(
            Image.fromarray(
                cv2.resize(img, (400, 400))))
        self.window.one = one
        return one

    def update_action_text(self):
        self.action_text.configure(
            text=f"{self.action_primitive} Scale: {self.scales[self.scale_idx]}|" +
            f" Rotation: {self.rotation_dict[self.action_primitive][self.rotation_idx]:.01f}",
            font=("Ubuntu", 20))

    def update_image(self):
        self.canvas.create_image(
            0, 0, anchor=tk.NW, image=self.get_obs())

    def on_key_press(self, event):
        x, y = self.center_grasp
        if event.keysym == 'Right':
            self.rotation_idx = min(self.rotation_idx + 1,
                                    len(self.rotation_dict[self.action_primitive]) - 1)
        elif event.keysym == 'Left':
            self.rotation_idx = max(self.rotation_idx - 1, 0)
        elif event.keysym == 'Up':
            self.scale_idx = min(self.scale_idx + 1,
                                 len(self.scales) - 1)
        elif event.keysym == 'Down':
            self.scale_idx = max(self.scale_idx - 1, 0)
        elif event.keysym == 'w':
            x -= 1
            x = max(x, 0)
        elif event.keysym == 'a':
            y -= 1
            y = max(y, 0)
        elif event.keysym == 's':
            x += 1
            x = min(x, self.image_dict[self.action_primitive].shape[-1]-1)
        elif event.keysym == 'd':
            y += 1
            y = min(y,self.image_dict[self.action_primitive].shape[-1]-1)
        elif event.keysym == 'Return':
            self.window.destroy()
            self.window.quit()
            return
        elif event.keysym == 'q':
            self.primitive_id -= 1
        elif event.keysym == 'e':
            self.primitive_id += 1

        self.center_grasp = np.array([x, y])
        self.update_action_primitive()
        self.update_action_text()
        self.update_image()

    def run(self):
        self.window.mainloop()
        return self.image_idx, self.center_grasp,\
            self.left_grasp, self.right_grasp, self.scale_idx, self.rotation_idx, self.action_primitive