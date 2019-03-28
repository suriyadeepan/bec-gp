from neuralbec import data
from neuralbec import utils
from neuralbec.model.ffn import SimpleRegressor
from neuralbec.train import fit
from config import conf

import torch
import argparse
import logging
import random

# setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# parse command-line arguments
parser = argparse.ArgumentParser(
    description='neuralbec : Neural Network based simulation of BEC'
    )
# predict mode
parser.add_argument('--predict', default=False, action='store_true',
    help='Run Prediction')
# generate mode
parser.add_argument('--generate', default=False, action='store_true',
    help='Generate Data')
parser.add_argument('--train', default=False, action='store_true',
    help='Train model on data')
# model name
parser.add_argument('--model', nargs='?', default='',
    help='model name to load from file')
# data name
parser.add_argument('--data', nargs='?', default='',
    help='name of the dataset to load form file')
# select a data point in validation set
parser.add_argument('--idx', nargs='?', default='',
    help='index to select data point from validation set')
parser.add_argument('--name', nargs='?', default='',
    help='Name of data/model')
parser.add_argument('--dims', nargs='?', default='1',
    help='Dimensions')
args = parser.parse_args()


def generate_one_dim(name=None):
  name = 'data1d' if not name else name
  data.generate_varg(
      fn=lambda g : data.particle_density_BEC1D(
        dim=512, radius=24, angular_momentum=1,
        time_step=1e-4, coupling=g,
        iterations=10000
        ),
      num_samples=10, filename='{}.data'.format(name)
      )


def train_sregressor(model_name, data_name):
  inputs, psi, reference = data.load('{}.data'.format(data_name))
  model = SimpleRegressor(1, 512)
  # train/test split
  trainset, testset = utils.split_dataset(  # train/test split
      utils.shuffle((inputs, psi)),         # shuffle dataset
      ratio=0.7
      )
  # test/valid split
  testset, validset = utils.split_dataset(testset, ratio=0.5)
  # fit model on trainset
  fit(model, (trainset, testset), conf, epochs=100)
  # save model
  model.save()

  return model


def predict(model_name, data_name, idx=None):
  # load model
  model = torch.load('bin/{}.pth'.format(model_name))
  # typecast idx
  if idx and not isinstance(idx, type(42)):
    idx = int(idx)
  # load dataset
  inputs, psi, reference = data.load('{}.data'.format(data_name))
  # train/test split
  trainset, testset = utils.split_dataset(  # train/test split
      utils.shuffle((inputs, psi)),         # shuffle dataset
      ratio=0.7
      )
  # test/valid split
  testset, validset = utils.split_dataset(testset, ratio=0.5)
  if not idx:  # random sample
    idx = random.randint(0, len(validset[0]))
  input, groundtruth = validset[0][idx], validset[-1][idx]
  # run input through model
  prediction = model(torch.tensor(input).view(1, -1))
  # get output size
  olen = groundtruth.shape[-1]
  # plot prediction vs ground truth
  data.plot_wave_function(reference['x'],
      prediction.view(olen).detach().numpy(), show=False)
  data.plot_wave_function(reference['x'], groundtruth, show=True)

  return prediction


if __name__ == '__main__':
  if args.predict:
    assert args.model
    assert args.data
    predict(args.model, args.data, args.idx)
  elif args.generate:
    if int(args.dims) == 1:
      generate_one_dim(args.name)
  elif args.train:
    assert args.model
    assert args.data
    if args.model == 'sregressor':
      train_sregressor(args.model, args.data)
