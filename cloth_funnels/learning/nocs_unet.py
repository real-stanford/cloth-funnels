#%%
import torch.nn as nn
from scipy import ndimage as nd
import cv2
from typing import List
import random
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from cloth_funnels.learning.unet_parts import *

#import 
class NOCSUNet(pl.LightningModule):

    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.steps = 0
        self.has_logged_val = True

    def forward(self, x):
        #depth only
        x = x[:, (3,), :, :]
        #depth only
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out

    def general_step(self, batch, batch_idx, name):

        input, target, mask = batch
        pred = self(input)
        
        out = pred.clone()

        pred = torch.reshape(pred, (pred.shape[0], 64, 3, pred.shape[2], pred.shape[3]))
        pred = pred.permute(0, 2, 1, 3, 4)
        
        pred = torch.reshape(pred, (pred.shape[0] * 3, 64, pred.shape[3], pred.shape[4]))

        #min loss only
        target_a = target.clone()
        target_b = target.clone()
        target_b[:, :2, :, :] = (-1 * target_b[:, :2, :, :] + 63)*mask

        target_a = torch.reshape(target_a, (target_a.shape[0] * 3, target_a.shape[2], target_a.shape[3]))
        target_b = torch.reshape(target_b, (target_b.shape[0] * 3, target_b.shape[2], target_b.shape[3]))

        criterion = torch.nn.CrossEntropyLoss()

        loss_a = criterion(pred, target_a.long())
        loss_b = criterion(pred, target_b.long())
        loss = torch.min(loss_a, loss_b)
        #min loss only

        self.steps += 1

        num_plots = 5
        if(self.steps % 200 == 0):
            input, target, mask, out = input[:num_plots].cpu(), target[:num_plots].cpu(), mask[:num_plots].cpu(), out[:num_plots].cpu().detach()
           
            target = torch.reshape(target, (num_plots, target.shape[1]//3, 3, target.shape[2], target.shape[3]))
            out = np.array(out.reshape(-1, 64, 3, out.shape[2], out.shape[3]))
            out = np.argmax(out, axis=1) / 63

            fig, axs = plt.subplots(num_plots, 4, figsize= (6, 8))
            # fig.suptitle("Input, Input Depth, Target NOCS, Mask")
            for i in range(num_plots):
                axs[i, 0].imshow(np.array(input[i][:3, :, :]).transpose(1, 2, 0))
                axs[i, 0].set_axis_off()
                axs[i, 1].imshow(np.array(input[i][3, :, :]))
                axs[i, 1].set_axis_off()
                axs[i, 2].imshow(np.array(target[i][0]).transpose(1, 2, 0) / 63)
                axs[i, 2].set_axis_off()
                axs[i, 3].imshow(np.array(out[i]).transpose(1, 2, 0))
                axs[i, 3].set_axis_off()
            fig.tight_layout(pad=0)
            

            self.logger.experiment.add_figure(f"{name}_images", fig, self.steps)

        self.log(f"{name}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "val")

    def on_epoch_end(self):
        self.has_logged_val = False

   
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    
    

if __name__ == "__main__":
    net = NOCSUNet(n_channels=1, n_classes=3*64, bilinear=True)
    x = (torch.zeros((8, 4, 720, 720)), torch.zeros((8, 3, 720, 720)),torch.zeros((8, 1, 720, 720)))
    pred = net(x[0])
    net.general_step(x, 9, "test")
    print(pred.shape)


# %%
