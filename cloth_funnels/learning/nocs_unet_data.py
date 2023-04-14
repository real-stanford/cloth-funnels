
 # %%

import os
from typing import Optional
import numpy as np
from numpy.core.fromnumeric import _nonzero_dispatcher
import h5py
import matplotlib.pyplot as plt
import torch
import pandas as pd
import random
import cv2
import torchvision.transforms as transforms
from tqdm import tqdm
import pickle
import hashlib
from filelock import FileLock
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

hashing_func = hashlib.md5
def str2int(s): return int(hashing_func(s.encode()).hexdigest(), 16)


class NOCSUNet_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, percent_coverage_threshold,
                 max_size=None, nonzero_pixel_threshold=0.05,  \
                     image_transforms=None, all_transforms=None, find_val_steps=False):
        print("Loading data...")

        self.data_dir = data_dir
        self.percent_coverage_threshold = percent_coverage_threshold

        self.transform = image_transforms
        self.all_transforms = all_transforms

        self.valid_steps = []

        filename = os.path.join(data_dir, 'valid_steps.pkl')

        try:
            with open(filename, 'rb') as f:
                self.valid_steps = pickle.load(f)
        except:
            pass

        if(len(self.valid_steps) == 0 or find_val_steps):
            keys = os.listdir(os.path.join(data_dir, "data"))
            num_keys = len(keys)
            print("Finding valid points...")
            pbar = tqdm(range(num_keys))
            for i in pbar:
                step = keys[i]
                if(max_size and len(self.valid_steps) >= max_size):
                    print("Maximum size reached")
                    break
                f = pickle.load(
                    open(os.path.join(data_dir, "data", step), 'rb'))
                percent_coverage = f['percent_coverage']
                nonzero_ratio = f['nonzero_ratio']
                if(percent_coverage > percent_coverage_threshold and nonzero_ratio > nonzero_pixel_threshold):
                    self.valid_steps.append(step)
                pbar.set_postfix({"valid_points": len(self.valid_steps)})

            with open(filename, 'wb') as f:
                pickle.dump(self.valid_steps, f)
        else:
            print(">> Validated steps found, initializing dataset")
            print("Number of valid points:", len(self.valid_steps))

    def __getitem__(self, idx):
        # as of now, some of the preprocessing happens here because I'm prototyping
        step = self.valid_steps[idx]

        f = pickle.load(open(os.path.join(self.data_dir, "data", step), 'rb'))

        input = f['input']
        target = f['target']
        mask = f['mask']

        train_input = torch.empty(
            input.shape[2], input.shape[0], input.shape[1])

        train_target = torch.empty(
            target.shape[2], target.shape[0], target.shape[1])

        if self.transform:
            train_input[:3, :, :] = self.transform(
                input[:, :, :3].astype(np.uint8))
            train_target = self.transform(target)

        depth = input[:, :, 3]
        floor_depth = np.max(depth) 
        train_input[3, :, :] = torch.tensor(depth)

        if self.all_transforms:
            all_data = self.all_transforms(
                torch.cat((train_input, train_target, mask), dim=0))
            train_input, train_target, mask = all_data[:4, :,
                                                       :], all_data[4:7, :, :], all_data[None, 7, :, :]

        #set background back to floor depth
        d = np.array(train_input[3, :, :])
        d[np.where(np.squeeze(mask) == False)] = floor_depth

        d = (d - d.mean())/(d.std() + 1e-8)
        train_input[3, :, :] = torch.tensor(d)

        # multiply by 64 and round down to get the labels
        train_target = (train_target * 64 - 1e-3).type(torch.int32)

        return train_input, train_target, mask

    def __len__(self):
        return len(self.valid_steps)

   

def load_datasets(data_dir, max_dataset_size, image_transforms, all_transforms, find_val_steps, train_val_split, percent_coverage_threshold, nonzero_pixel_threshold):
    dataset = NOCSUNet_Dataset(data_dir,
                               max_size=max_dataset_size,
                               percent_coverage_threshold=percent_coverage_threshold,
                               nonzero_pixel_threshold=nonzero_pixel_threshold,
                               image_transforms=image_transforms,
                               all_transforms=all_transforms,
                               find_val_steps=find_val_steps)

    train_set_size = int(len(dataset) * train_val_split)
    val_set_size = len(dataset) - train_set_size

    train_set, valid_set = random_split(
        dataset, [train_set_size, val_set_size])

    return train_set, valid_set


class NOCSUNetDataModule(pl.LightningDataModule):

    def __init__(self, replay_buffer_path, download_dir, max_dataset_size=None, find_val_steps=False, train_val_split=0.95, batch_size=32):
        super().__init__()

        self.train = None
        self.val = None

        self.max_dataset_size = max_dataset_size
        self.find_val_steps = find_val_steps
        self.train_val_split = train_val_split
        self.replay_buffer_path = replay_buffer_path
        self.batch_size = batch_size
        self.download_dir = download_dir

        self.image_transforms = transforms.Compose(
            [transforms.ToTensor()]
        )

        self.all_transforms = transforms.Compose(
            [transforms.RandomAffine(degrees=(0, 360), translate=(0.4, 0.4),
                                     scale=(0.5, 1.6), \
                                    interpolation=transforms.InterpolationMode.BILINEAR)]
        )

        self.replay_buffer_name = str(str2int(self.replay_buffer_path))[:10]

    def prepare_data(self, rewrite=False):
        with FileLock(self.replay_buffer_path + '.lock'):
            with h5py.File(self.replay_buffer_path, 'r') as f:
                dir_name = os.path.join(
                    self.download_dir, self.replay_buffer_name, "data")

                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)

                if(len(os.listdir(dir_name)) == 0 or rewrite):
                    keys = list(f.keys())
                    num_keys = len(keys)
                    print("Preparing data")
                    pbar = tqdm(range(num_keys))
                    for i in pbar:
                        step = keys[i]
                        input = np.array(f[step]
                                         ['garmentnets_supervision_input'])
                        input = cv2.resize(input, dsize=(128, 128))

                        target = np.array(f[step]
                                          ['garmentnets_supervision_target'])
                        target = cv2.resize(target, dsize=(128, 128))

                        mask = torch.tensor(
                            1 - np.all(target == 0, axis=-1).astype(np.float32).reshape(1, 128, 128))

                        percent_coverage = f[step].attrs['percent_coverage']
                        nonzero_ratio = np.mean(
                            ((np.sum(target, axis=-1) == 0) == 0).astype(np.float32))
                        data = {"input": input,
                                "target": target,
                                "mask": mask,
                                "percent_coverage": percent_coverage,
                                "nonzero_ratio": nonzero_ratio}
                        pickle.dump(data, open(
                            os.path.join(dir_name, str(step)), 'wb'))
                else:
                    print("Data already prepared")

    def setup(self, nonzero_pixel_threshold=0.05, percent_coverage_threshold=0.8, stage: Optional[str] = None):
        # called on every GPU
        self.train, self.val = load_datasets(
            nonzero_pixel_threshold=nonzero_pixel_threshold,
            percent_coverage_threshold=percent_coverage_threshold,
            data_dir=os.path.join(self.download_dir, self.replay_buffer_name),
            max_dataset_size=self.max_dataset_size,
            image_transforms=self.image_transforms,
            all_transforms=self.all_transforms,
            find_val_steps=self.find_val_steps,
            train_val_split=self.train_val_split)

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True )


if __name__ == "__main__":

    # replay_buffer_path="/local/crv/acanberk/folding-unfolding/src/test-sim/replay_buffer.hdf5"
    # with FileLock(replay_buffer_path + '.lock'):
    #     transforms=transforms.Compose(
    #         [transforms.ToTensor()]
    #     )
    #     dataset=NOCSUNet_Dataset(
    #         replay_buffer_path, percent_coverage_threshold=0.5, transform=transforms)
    #     print("dataset size", len(dataset))
    #     for data in dataset:
    #         print(data[0].shape, data[1].shape, data[2].shape)
    #         break
    # ds = NOCSUNet_Dataset("/local/crv/acanberk/folding-unfolding/src/flingbot_eval_2/replay_buffer.hdf5", percent_coverage_threshold=0.5)
    # print(ds[0])

    dm = NOCSUNetDataModule("/local/crv/acanberk/folding-unfolding/src/flingbot_eval_2/replay_buffer.hdf5",
                            download_dir="/local/crv/acanberk/folding-unfolding/src/learning/replay_buffer_data")
    dm.prepare_data()
    dm.setup(nonzero_pixel_threshold=0.05, percent_coverage_threshold=0.2)

    train_dataloader = dm.train_dataloader()

    for input, target, mask in train_dataloader:
        print("target shape", target.shape)
        print("mask shape", mask.shape)

        target_b = target.clone()
        target_b[:, :2, :, :] = (-1 * target_b[:, :2, :, :] + 64)*mask

        # (target_b[np.where(mask == True)])[:, :2, :, :] = \
        #     -1 * (target_b[np.where(mask == True)])[:, :2, :, :] + 1
            
        # target_b[0]
        plt.imshow(target[0].numpy().transpose(1, 2, 0) / 63)
        plt.show()
        plt.imshow(target_b[0].numpy().transpose(1, 2, 0) / 63)
        plt.show()

        break
        
        

# %%
