import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
from neuralbec import utils

logger = utils.get_logger(__name__)
# set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleRegressor(nn.Module):

  def __init__(self, dim_in, dim_out):
    super().__init__()
    self.dim_in = dim_in
    self.dim_out = dim_out
    self.linear = nn.Linear(dim_in, dim_out)

  def forward(self, x):
    return self.linear(x)
