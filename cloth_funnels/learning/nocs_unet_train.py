# %% 

from torchvision import transforms, utils
import torch
import matplotlib.pyplot as plt
from nocs_unet_data import NOCSUNet_Dataset, NOCSUNetDataModule
from torch.utils.data import DataLoader
from nocs_unet import NOCSUNet
from filelock import FileLock
import numpy as np
from torch import nn
from datetime import datetime
import wandb
import os
from argparse import ArgumentParser
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


if __name__ == "__main__":
    parser = ArgumentParser('Training NOCS UNet')
    parser.add_argument("--wandb", type=str, default="disabled",
                        help="Run on wandb")
    parser.add_argument("--save_params", type=bool, default=True)
    parser.add_argument("--save_dir", type=str, default="/local/crv/acanberk/folding-unfolding/src/learning/unet_models")
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_only", type=bool, default=False)
    parser.add_argument("--from_checkpoint", type=str, default=None)
    parser.add_argument("--replay_buffer_path", type=str,  \
        default='/local/crv/acanberk/folding-unfolding/src/learning/4-shirt-finetune/replay_buffer.hdf5')
    parser.add_argument("--download_dir", type=str,  \
        default='/local/crv/acanberk/folding-unfolding/src/learning/nocs_unet_data')
    parser.add_argument("--max_dataset_size", type=int, default=None)
    parser.add_argument("--find_val_steps", action="store_true")
    parser.add_argument("--train_val_split", type=bool, default=0.95)
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Unknown arguments:", unknown)
    #number of input channels and the number of output channels

    now = datetime.now()
    training_instance = now.strftime("%m%d-%H%M%S")
    models_dir = "/local/crv/acanberk/folding-unfolding/src/unet_models"
    save_path = os.path.join(models_dir, training_instance)
    os.makedirs(save_path)
    print("Starting training instance: ", training_instance)
    wandb.init(project="nocs_unet", name=training_instance, mode=args.wandb, sync_tensorboard=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.save_dir,
        filename=str(training_instance) + "-nocs-unet-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    datamodule = NOCSUNetDataModule(args.replay_buffer_path, 
                                    args.download_dir,
                                    args.max_dataset_size, 
                                    args.find_val_steps, 
                                    args.train_val_split, 
                                    args.batch_size)
    datamodule.prepare_data()
    datamodule.setup(nonzero_pixel_threshold=0.05, percent_coverage_threshold=0.2)

    tb_logger = pl_loggers.TensorBoardLogger("lightning_logs/")
    model = NOCSUNet(n_channels=1, n_classes=64*3)
    trainer = Trainer(gpus=[0, 1], max_epochs=args.num_epochs, logger=tb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule)






# %%
