import torch
import torch.nn as nn
import os

from neuralbec import utils


logger = utils.get_logger(__name__)
# set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model save path
BINPATH = 'bin/'


class ConvNet(nn.Module):

  def __init__(self, dim_in, dim_hid, dim_out, name='convnet'):
    super().__init__()
    # self.linear = nn.Linear(dim_in, dim_out)
    self.name = name
    self.bin = os.path.join(BINPATH, '{}.pth'.format(self.name))
    self.linear_1 = nn.Linear(dim_in, dim_hid)

  def forward(self, x):
    x = torch.tanh(self.linear_1(x))
    x = torch.tanh(self.linear_2(x))
    return x

  def save(self):
    torch.save(self, self.bin)

  def load(self):
    return torch.load(self.bin)
