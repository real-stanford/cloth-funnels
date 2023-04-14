import torch
import numpy as np


def to_numpy(x):
    return x.detach().to('cpu').numpy()


def dict_to(x, device):
    new_x = dict()
    for key, value in x.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
        new_x[key] = value
    return 