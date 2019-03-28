import torch
import torch.nn as nn

from neuralbec import utils

import os

logger = utils.get_logger(__name__)
# set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model save path
BINPATH = 'bin/'


class SimpleRegressor(nn.Module):

  def __init__(self, dim_in, dim_out, name='sregressor'):
    super().__init__()
    self.dim_in = dim_in
    self.dim_out = dim_out
    self.linear = nn.Linear(dim_in, dim_out)
    self.name = name
    self.bin = os.path.join(BINPATH, '{}.pth'.format(self.name))

  def forward(self, x):
    return self.linear(x)

  def save(self):
    torch.save(self, self.bin)

  def load(self):
    return torch.load(self.bin)
