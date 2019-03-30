from neuralbec import data
from neuralbec import utils
from neuralbec.train import fit
from config import conf

from neuralbec.model.ffn import SimpleRegressor
from neuralbec.model.ffn import FFN

import numpy as np

import torch
import argparse
import logging
import random
import math
import time

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
# train mode
parser.add_argument('--train', default=False, action='store_true',
    help='Train model on data')
# plot mode
parser.add_argument('--plot', default=False, action='store_true',
    help='Plot data')
parser.add_argument('--parallel', default=False, action='store_true',
    help='Parallel Execution')
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
        potential_fn=data.harmonic_potential,
        iterations=10000
        ),
      num_samples=conf['num_samples'], filename='{}.data'.format(name)
      )


def generate_two_dim(name='bec2d'):
  start_time = time.time()
  datadict = data.generate_varg(
      fn=lambda g : data.particle_density_BEC2D(
        dim=256, radius=15, angular_momentum=0.05,
        time_step=1e-3, coupling=g,
        potential_fn=data.harmonic_potential,
        iterations=8000,
        ),
      num_samples=conf['num_samples'], filename='{}.data'.format(name),
      save_every=20
      )
  logger.info('_________ {} seconds _________'.format(
    time.time() - start_time))
  return datadict


def gen_fn(g):
 return data.particle_density_BEC2D(
        dim=256, radius=15, angular_momentum=0.05,
        time_step=1e-3, coupling=g,
        potential_fn=data.harmonic_potential,
        iterations=8000,
        )


def generate_two_dim_parallel(name='bec2d'):
  start_time = time.time()
  data.generate_parallel(
      gen_fn=gen_fn,
      inputs=np.random.uniform(conf['g_low'], conf['g_high'], conf['num_samples']),
      filename='{}.data'.format(name),
      save_every=conf['save_every']
      )
  logger.info('_________ {} seconds _________'.format(
    time.time() - start_time))


def generate_one_dim_cosine(name='bec1d_cosine'):
  # define "cosine" potential
  datadict = data.generate_varg(
      fn=lambda g : data.particle_density_BEC1D(
        dim=512, radius=24, angular_momentum=1,
        time_step=1e-4, coupling=g,
        potential_fn=lambda x, y : 0.5 * (x**2) + 24. * (math.cos(x)**2),
        iterations=10000
        ),
      num_samples=conf['num_samples'], filename='{}.data'.format(name)
      )
  return datadict


def train_sregressor(model_name, data_name):
  # get dataset from name
  trainset, testset, validset = data.make_dataset(data_name)
  # build Regressor
  nconf = conf['nn']
  model = SimpleRegressor(nconf['dim_in'], nconf['dim_out'], name=model_name)
  # fit model on trainset
  fit(model, (trainset, testset), conf, epochs=100)
  # save model
  model.save()
  return model


def train_ffn(model_name, data_name):
  # get dataset from name
  trainset, testset, validset = data.make_dataset(data_name)
  # build Regressor
  nconf = conf['nn']
  model = FFN(nconf['dim_in'], nconf['dim_hid'], nconf['dim_out'],
      name=model_name)
  # fit model on trainset
  fit(model, (trainset, testset), conf, epochs=conf['epochs'])
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
  data.plot_wave_function(reference['x'], groundtruth, show=True,
      title='psi = {}'.format(input), color='green')
  data.plot_wave_function(reference['x'],
      prediction.view(olen).detach().numpy(), show=True,
      title='psi = {}'.format(input), color='red')

  return prediction


def plot(data_name, idx=None):
  idx = int(idx) if idx and not isinstance(idx, type(42)) else idx
  logger.info(idx)
  g, psi, ref = data.load('{}.data'.format(data_name))
  # get ref points
  x, y = ref['x'], ref['y']
  if not idx:  # random sample index
    idx = random.randint(0, len(g) - 1)

  # plot 2d wave function
  data.plot_wave_function_2d(x, y, psi[idx],
      title='psi = {}'.format(g[idx])
      )


if __name__ == '__main__':
  # resolve [data/model] options
  data_name = args.data if args.data else conf['data']
  model_name = args.model if args.model else conf['model']
  if args.predict:
    predict(model_name, data_name, args.idx)
  elif args.generate:
    if conf['data'] == 'bec1d_cosine':
      generate_one_dim_cosine(data_name)
    elif conf['data'] == 'bec1d':
      generate_one_dim()
    elif '2d' in conf['data']:
      if args.parallel:
        logger.info('Running parallel')
        generate_two_dim_parallel(data_name)
      else:
        generate_two_dim(data_name)
    else:
      logger.info('Unknown Data')
  elif args.train:
    if conf['net'] == 'sregressor':
      train_sregressor(model_name, data_name)
    elif conf['net'] == 'ffn':
      train_ffn(model_name, data_name)
    else:
      logger.info('Unknown Model')
  elif args.plot:
    plot(data_name, idx=args.idx)
