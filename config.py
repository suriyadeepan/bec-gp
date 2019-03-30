import torch.nn as nn
import torch


conf = {
    'batch_size' : 256,
    'loss_fn' : nn.MSELoss(),
    'optim' : torch.optim.Adam,
    'epochs' : 100,
    'model'  : 'ffn',
    'data'   : 'bec2d',
    'num_samples' : 1024,
    'net'    : 'ffn',
    'nn' : {
      'dim_in' : 1, 'dim_hid' : 750, 'dim_out' : 512
      },
    'g_low'  : 90,
    'g_high' : 100,
    'save_every' : 16
    }
