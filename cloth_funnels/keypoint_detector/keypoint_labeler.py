# %%
import os
import pathlib
import json
import argparse
import h5py
import numpy as np
import cv2
from datetime import datetime
import hashlib
import matplotlib.pyplot as plt
import skvideo.io
from utils.keypoint_util import shirt_keypoints

# %%
class VideoLabeler:
    def __init__(self, input_path, label_dir, num_keypoints=10, min_step=6, min_coverage=0.6):

        self.input_path = input_path
        self.input_path_hash = str(hashlib.sha256(input_path.encode('utf-8')).hexdigest())[:4]

        self.frames = skvideo.io.vread(input_path)
        self.keys = []
        for i, key in enumerate(self.frames):
            key_name = f"{self.input_path_hash}_frame_{i}"
            self.keys.append(key_name)

        
        self.num_keypoints = num_keypoints
        self.image_dim = None

        self.curr_keypoint = 0
        self.set_keypoint(self.curr_keypoint)

        self.keypoint_id_to_name = {
            # 0: 'left_collar',
            # 1: 'right_collar',
            0: 'left_shoulder',
            1: 'right_shoulder',
            2: 'top_left',
            3: 'top_right',
            # 6: 'left_bottom_arm_tip',
            # 7: 'right_bottom_arm_tip',
            4: 'bottom_left',
            5: 'bottom_right',
        }
        self.name_to_keypoint_id = {
            key:value for value, key in self.keypoint_id_to_name.items()
        }

        self.keypoint_id_to_color = dict()
        for i in range(self.num_keypoints):
            if i % 2 == 0:
                self.keypoint_id_to_color[i] = (255, 0, 0)
            else:
                self.keypoint_id_to_color[i] = (0, 0, 255)

        # frames = skvideo.io.vread(video_path)
        # self.frames = frames
        self.i = 0

        # load label state
        pathlib.Path(label_dir).mkdir(parents=True, exist_ok=True)
        self.label_dir = label_dir
        self.label_dict = dict()

        if os.path.isfile(self.json_path):
            self.label_dict = json.load(open(self.json_path, 'r'))
      
        #if a key contains 'reflected' delete it frmo the dictionary 
        delete_reflected_keys = []
        for key in self.label_dict.keys():
            if 'reflected' in key:
                delete_reflected_keys.append(key)
        for key in delete_reflected_keys:
            del self.label_dict[key]

        self.image_buffer = dict()
        self.mask_buffer = dict()
    
    @property
    def curr_index(self):
        return self.i

    @property
    def curr_key(self):
        return self.keys[self.i]

    def __len__(self):
        return len(self.keys)
    
    @property
    def json_path(self):
        return os.path.join(self.label_dir, 'labels.json')

    def save_labels(self):

        # generating reflected data
        reflected_label_dict = dict()
        for key in self.label_dict.keys():
            new_key = f"{key}_reflected"
            reflected_label_dict[new_key] = [None] * self.num_keypoints
            for i in range(self.num_keypoints):
                if i % 2 == 0:
                    coord = self.label_dict[key][i+1]
                else:
                    coord = self.label_dict[key][i-1]
                flipped_coord = (coord[0], (self.image_dim - coord[1])) if coord is not None else None
                reflected_label_dict[new_key][i] = flipped_coord
        ###########################################
        combined_dict = {**self.label_dict, **reflected_label_dict}

        pathlib.Path(self.label_dir).mkdir(parents=True, exist_ok=True)
        json.dump(combined_dict, open(self.json_path, 'w'), indent=2)

    def save_images(self):
        pathlib.Path(self.label_dir).mkdir(parents=True, exist_ok=True)
        # glob
        files = pathlib.Path(self.label_dir).glob('*.jpg')
        file_path_map = dict()
        for file in files:
            key = file.stem
            path = str(file.absolute())
            file_path_map[key] = path
        
        # delete unlabeled images
        # for key, path in file_path_map.items():
        #     file_key = "_".join(key.split('_')[:3])
        #     if all([item is None for item in self.label_dict[file_key]]):
        #         os.remove(path)
 
        # save unsaved images
        for key in self.image_buffer.keys():
            img = self.image_buffer[key]

            img_path = os.path.join(self.label_dir, key + '.jpg')
            reflected_img_path = os.path.join(self.label_dir, key + '_reflected.jpg')

            # print("Writing image to:", path)
            cv2.imwrite(img_path, img)
            cv2.imwrite(reflected_img_path, cv2.flip(img, 0))

        self.image_buffer = dict()

    def get_image(self):
        return self.frames[self.i]

    def add_label(self, coord):
        key = self.curr_key
        if key not in self.label_dict:
            self.label_dict[key] = [None] * self.num_keypoints
        self.label_dict[key][self.curr_keypoint] = coord

    def delete_label(self):
        key = str(self.curr_key)
        self.label_dict[key][self.curr_keypoint] = None

        self.image_buffer.pop(key, None)
        self.mask_buffer.pop(key, None)

    def set_keypoint(self, keypoint):
        self.curr_keypoint = keypoint

    def next_frame(self):
        self.i = min(self.i + 1, len(self.keys) - 1)
        return self.i
    
    def prev_frame(self):
        self.i = max(self.i - 1, 0)
        return self.i
    
    def get_curr_img(self, key=None, input_path=None):
   
        if key is None:
            key = self.curr_key
        if input_path is None:
            input_path = self.input_path

        self.image_buffer[key] = self.get_image()
        self.image_dim = self.image_buffer[key].shape[1]

        if key not in self.label_dict:
            print("Heuristic keypoints")
            mask = self.get_image().sum(axis=-1) > 1e-2
            heuristic_keypoints = shirt_keypoints(mask)
            self.label_dict[key] = [None] * self.num_keypoints
            for keypoint_name, coord in heuristic_keypoints.items():
                self.label_dict[key][self.name_to_keypoint_id[keypoint_name]] = (int(coord[1]), int(coord[0]))

            
        vis_img = np.array(self.get_image(), dtype=np.uint8)
        if key in self.label_dict:
            for i, coord in enumerate(self.label_dict[key]):
                if coord is None: continue
                cv2.drawMarker(vis_img, coord, 
                    color=self.keypoint_id_to_color[i], markerType=cv2.MARKER_CROSS,
                    markerSize=20, thickness=1)
                cv2.putText(vis_img, f"{i}_{self.keypoint_id_to_name[i]}", coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        #put hello world on the top left corner of the screen
        cv2.putText(vis_img, f"Keypoints: {sum([(1 if item is not None else 0) for item in self.label_dict[key]]) if key in self.label_dict else 0}/{self.num_keypoints}", 
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        

        return vis_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-k', '--num_keypoints', type=int, default=6)
    args = parser.parse_args()

    state = VideoLabeler(args.input, args.output, args.num_keypoints)

    # state = VideoLabeler("/home/zhenjia/dev/folding-unfolding/src/1906-real-fold/buffer0/replay_buffer.hdf5", "labels_1")
    # img = state.get_curr_img("712440_000000000_step00")
    # labels = state.label_dict["712440_000000000_step00"]
    # middle = img.shape[0]//2
    # labels = state.label_dict["712440_000000000_step00"]
    # reflected_labels = [([tup[0], -tup[1] + 2*middle] if tup is not None else None) for tup in labels]
    # reflected_img = cv2.flip(img, 0)
    # print(reflected_labels)
    # print(labels)
    # fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    # ax[0].imshow(img)
    # ax[1].imshow(reflected_img)
    # plt.show()

    def callback(action, x, y, flags, *userdata):
        if action == cv2.EVENT_LBUTTONDOWN:
            coord = (x,y)
            state.add_label(coord=coord)
    
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("img", callback)
   

    while True:
        cv2.imshow("img", state.get_curr_img())
        key = cv2.waitKey(17)


        if key == ord('q'):
            print('exit')
            break
        elif key == ord('a'):
            print('prev')
            frame = state.prev_frame()
            print(f'{frame}/{len(state)}')
        elif key == ord('d'):
            print('next')
            frame = state.next_frame()
            print(f'{frame}/{len(state)}')
        elif key in [ord(str(i)) for i in range(args.num_keypoints)]:
            state.set_keypoint(int(chr(key)))
        elif key == ord('x'):
            state.delete_label()
        elif key == ord('s'):
            print("Saving")
            state.save_labels()
            state.save_images()
        # if key != -1:
            # print("key:", key)

# %%
if __name__ == '__main__':
    main()
# %%
