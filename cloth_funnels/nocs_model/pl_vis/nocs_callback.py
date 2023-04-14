import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import Subset, DataLoader
import wandb
import numpy as np
from nocs_model.common.torch_util import dict_to, to_numpy


class NOCSCallback(pl.Callback):
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
        pred_value = pl_module(input)
        gt_value = self.vis_batch['target'].type(torch.float32)
        input = input.detach().to('cpu')
        pred_value = torch.clip(pred_value.detach().to('cpu'), 0, 1)
        img_shape = input.shape[-2:]

        imgs = list()
        for i in range(len(self.vis_idxs)):
            vis_idx = self.vis_idxs[i]
            input_img = torch.moveaxis(input[i],0,-1)
            pred_img = torch.zeros(img_shape+(3,), dtype=torch.float32)
            gt_img = pred_img.clone()
            pred_img[...,[0,2]] = torch.moveaxis(pred_value[i],0,-1)
            gt_img[...,[0,2]] = torch.moveaxis(gt_value[i],0,-1)
            vis_img = torch.cat([input_img, pred_img, gt_img], dim=1).numpy()
            imgs.append(wandb.Image(
                vis_img, caption=f"val-{vis_idx}"
            ))
        
        trainer.logger.experiment.log({
            "val/vis": imgs,
            "global_step": trainer.global_step
        })
