from neuralbec.model.ffn import SimpleRegressor
from neuralbec import utils

import torch
import torch.nn as nn

import os

logger = utils.get_logger(__name__)
# model save path
BINPATH = 'bin/'


class SimpleGaussian(nn.Module):

  def __init__(self, dim_in=1, dim_out=512, name='simple_gaussian'):
    super().__init__()
    self.dim_in = dim_in
    self.dim_out = dim_out
    self.regressor = SimpleRegressor(dim_in, dim_out)
    self.name = name
    self.bin = os.path.join(BINPATH, '{}.pth'.format(self.name))

  def forward(self, x):
    mu, sigma = self.regressor(x)

  def save(self):
    torch.save(self, self.bin)

  def load(self):
    return torch.load(self.bin)


class FFGaussian(nn.Module):

  def __init__(self, dim_in=1, dim_out=2, name='ff_gaussian'):
    super().__init__()
    self.dim_in = dim_in
    self.dim_out = dim_out
    self.linear = SimpleRegressor(dim_in, dim_out)
    self.name = name
    self.bin = os.path.join(BINPATH, '{}.pth'.format(self.name))

  def forward(self, x):
    return torch.tanh(self.linear(x))

  def save(self):
    torch.save(self, self.bin)

  def load(self):
    return torch.load(self.bin)
