import torch.nn as nn
import torch


conf = {
    'batch_size' : 256,
    'loss_fn' : nn.MSELoss(),
    'optim' : torch.optim.Adam,
    'epochs' : 30
    }
