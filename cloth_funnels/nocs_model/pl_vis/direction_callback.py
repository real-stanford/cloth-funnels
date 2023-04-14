import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import Subset, DataLoader
import wandb
import numpy as np
from nocs_model.common.torch_util import dict_to, to_numpy
import cv2

def pred_to_best_bin(pred_img):
    """
    pred_img: (bin,C,H,W) float32
    return: (C,H,W) float32
    """
    n_bins = pred_img.shape[0]
    best_bin = np.argmax(pred_img, axis=0) / n_bins
    return best_bin

def best_bin_to_axis(best_bin, vis_img, length=3, grid=8):
    """
    pred_img: (bin,C,H,W) float32
    vis_img: (H,W,3) float32
    return: (H,W,3) uint8
    """
    img_size = best_bin.shape[-2:]
    start = grid//2
    sampled_coords = np.indices(img_size)[:,start::grid,start::grid]
    sampled_angles = best_bin[:,start::grid,start::grid] * (np.pi * 2)

    sampled_offsets = np.stack([
        np.cos(sampled_angles),
        np.sin(sampled_angles)
    ])
    vis_img = (vis_img * 255).astype(np.uint8).copy()
    for channel_id, color in enumerate([(255,0,0),(0,0,255)]):
        for r, c in np.ndindex(sampled_coords.shape[-2:]):
            offset = sampled_offsets[:,channel_id,r,c]
            base = np.array([r,c], dtype=np.float32) * grid + start
            tip = base + offset * length
            cv2.line(vis_img, 
                pt1=base[::-1].astype(np.int32),
                pt2=tip[::-1].astype(np.int32),
                color=color,
                thickness=1)
    return vis_img


class DirectionCallback(pl.Callback):
    def __init__(self, 
            val_dataset, 
            num_samples=16, 
            seed=0):
        super().__init__()
        rs = np.random.RandomState(seed=seed)
        vis_idxs = rs.choice(len(val_dataset), num_samples)
        vis_subset = Subset(val_dataset, vis_idxs)
        vis_dataloader = DataLoader(
            vis_subset, batch_size=num_samples)
        vis_batch = next(iter(vis_dataloader))
        self.vis_batch = vis_batch
        self.vis_idxs = vis_idxs
        
    def on_validation_epoch_end(self, 
            trainer: pl.Trainer, 
            pl_module: pl.LightningModule) -> None: 
        input = self.vis_batch['input'].to(pl_module.device)
        logits = pl_module(input)
        n_bins = logits.shape[1]
        _, pred_bin = torch.max(logits, dim=1)
        pred_value = pred_bin.type(torch.float32) / n_bins
        gt_value = torch.clip(self.vis_batch['target'].type(torch.float32) / n_bins, 0, 1)
        input = input.detach().to('cpu')
        pred_value = pred_value.detach().to('cpu')
        img_shape = input.shape[-2:]

        imgs = list()
        for i in range(len(self.vis_idxs)):
            vis_idx = self.vis_idxs[i]
            input_img = torch.moveaxis(input[i],0,-1).numpy()
            pred_img = torch.zeros(img_shape+(3,), dtype=torch.float32)
            gt_img = pred_img.clone()
            pred_img[...,[0,2]] = torch.moveaxis(pred_value[i],0,-1)
            gt_img[...,[0,2]] = torch.moveaxis(gt_value[i],0,-1)
            pred_img = (pred_img.numpy() * 255).astype(np.uint8)
            gt_img = (gt_img.numpy() * 255).astype(np.uint8)

            pred_best_bin = pred_value[i].numpy()
            gt_best_bin = gt_value[i].numpy()

            pred_vis = torch.from_numpy(best_bin_to_axis(pred_best_bin, input_img))
            gt_vis = torch.from_numpy(best_bin_to_axis(gt_best_bin, input_img))
            rows = [
                np.concatenate([pred_vis, gt_vis],axis=1),
                np.concatenate([pred_img, gt_img],axis=1)
            ]
            vis_img = np.concatenate(rows,axis=0)
            imgs.append(wandb.Image(
                vis_img, caption=f"val-{vis_idx}"
            ))
        
        trainer.logger.experiment.log({
            "val/vis": imgs,
            "global_step": trainer.global_step
        })
