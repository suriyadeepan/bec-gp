from gpbec.learn import gp
import torch
import math


if __name__ == '__main__':
  # Training data is 100 points in [0,1] inclusive regularly spaced
  train_x = torch.linspace(0, 1, 1000)
  # True function is sin(2*pi*x) with Gaussian noise
  train_y = torch.sin(train_x * (2 * math.pi)) +\
      torch.randn(train_x.size()) * 0.2
  # create a GP
  model, mll = gp(train_x, train_y)
  # fit GP
  model.update(mll)
