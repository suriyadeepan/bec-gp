from neuralbec import data
from neuralbec import utils
from neuralbec.model.ffn import SimpleRegressor
from neuralbec.train import train_model

import torch
import torch.nn as nn
import argparse
import logging

# setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# parse command-line arguments
parser = argparse.ArgumentParser(
    description='neuralbec : Neural Network based simulation of BEC'
    )
# -- Add options --
# parser.add_argument('command', type=str, help='')
#     help='command to run in remote system')


if __name__ == '__main__':
  """
  # Generate 1D BEC
  data.generate_varg(
      fn=lambda g : data.particle_density_BEC1D(
        dim=512, radius=24, angular_momentum=1,
        time_step=1e-4, coupling=g,
        iterations=10000
        ),
      num_samples=10, filename='10samples.data'
      )
  """
  inputs, psi, reference = data.load('bec1d.data')
  model = SimpleRegressor(1, 512)
  hparams = {
      'batch_size' : 64,
      'loss_fn' : nn.MSELoss(),
      'optim' : torch.optim.Adam(model.parameters())
      }
  # train/test split
  trainset, testset = utils.split_dataset((inputs, psi))
  # train
  train_model(model, ((inputs, psi), None), hparams, epochs=100)
