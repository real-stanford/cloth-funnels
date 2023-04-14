# %%
from tkinter import N
import os
import pathlib
import json
import argparse

import skvideo.io
import h5py
import numpy as np
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

#%%
def buffer_to_video(input_path, output_path, min_step=0, min_coverage=0.6, max_images=200):
    """
    Converts a list of replay buffers into a video.
    """
    all_images = []
    with h5py.File(input_path, 'r') as dataset:
        image = None
        for key in tqdm(dataset.keys(), desc='Loading images'):
            group = dataset[key]

            step = key.split('step')[1][:2]
            episode = int(key.split('step')[0][:-2])

            if group.attrs['task_difficulty'] != 'hard':
                continue

            if np.random.random() < 0.8:
                continue

            #filter if early step or low coverage
            if int(step) < min_step or group.attrs['preaction_coverage'] < min_coverage:
                continue

            image = np.array(group['pretransform_observations'][:3])

            try:
                #probably in real
                image = image * np.array(group['pretransform_mask'])
            except:
                #probably in sim, background should already be filtered
                image = (image * 255).transpose(1, 2, 0)
                pass

            assert image.shape[-1] == 3

            image = np.ascontiguousarray(image, dtype=np.uint8)
            all_images.append(image)

            if len(all_images) > max_images:
                break

    print(f"{len(all_images)} images found for labeling.")
    print("Combining images into video, saving to:", output_path)
    skvideo.io.vwrite(output_path, all_images)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()

    buffer_to_video(args.input, args.output)

    
#%%

if __name__ == '__main__':
    main()

# %%
