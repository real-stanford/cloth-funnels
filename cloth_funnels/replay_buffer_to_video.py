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
def buffer_to_video(input_path, output_path):
    """
    Converts a list of replay buffers into a video.
    """
    all_images = []
    with h5py.File(input_path, 'r') as dataset:
        for key in tqdm(dataset.keys(), desc='Loading images'):
            group = dataset[key]
            image = np.array(group['pretransform_observations'])
            mask = np.array(group['pretransform_mask'])
            image = image * mask
            image = np.ascontiguousarray(image.transpose(1, 2, 0), dtype=np.uint8)
            all_images.append(image)
    
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
