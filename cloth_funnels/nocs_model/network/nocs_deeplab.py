import torch
import torch.nn as nn
import pytorch_lightning as pl

from nocs_model.network.deeplab_v3_plus import DeepLabv3_feature, OutConv

class NOCSDeeplab(pl.LightningModule):
    def __init__(self,
            n_features=128,
            n_dims=2,
            learning_rate=1e-3,
            weight_decay=None,
            **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.net = DeepLabv3_feature(n_channels=3, n_features=n_features, os=8)
        self.outc = OutConv(in_channels=n_features, out_channels=n_dims)

        self.n_dims = n_dims
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        learning_rate = self.learning_rate
        weight_decay = self.weight_decay
        optimizer = None
        if weight_decay is None:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=learning_rate, 
                weight_decay=weight_decay)
        return optimizer

    def forward(self, input):
        features = self.net(input)
        out = self.outc(features)
        return out

    def step(self, batch, batch_idx, step_type='train'):
        input = batch['input']
        target = batch['target']

        output = self.forward(input)
        loss = self.criterion(output, target)

        self.log(f"{step_type}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')
