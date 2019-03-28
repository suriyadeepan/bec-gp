import torch
import torch.nn as nn
import torch.nn.functional as F

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
    return torch.tanh(self.linear(x))

  def save(self):
    torch.save(self, self.bin)

  def load(self):
    return torch.load(self.bin)


class FFN(nn.Module):

  def __init__(self, dim_in, dim_hid, dim_out, name='ffn'):
    super().__init__()
    # self.linear = nn.Linear(dim_in, dim_out)
    self.name = name
    self.bin = os.path.join(BINPATH, '{}.pth'.format(self.name))
    self.linear_1 = nn.Linear(dim_in, dim_hid)
    self.linear_2 = nn.Linear(dim_hid, dim_out)

  def forward(self, x):
    x = torch.tanh(self.linear_1(x))
    x = torch.tanh(self.linear_2(x))
    return x

  def save(self):
    torch.save(self, self.bin)

  def load(self):
    return torch.load(self.bin)
