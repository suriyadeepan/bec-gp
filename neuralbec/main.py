from neuralbec import data
import numpy as np

import argparse
import logging
import utils
import pickle

from tqdm import tqdm

# setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# parse command-line arguments
parser = argparse.ArgumentParser(
    description='neuralbec : Neural Network based simulation of BEC'
    )
"""
parser.add_argument('command', type=str, help='')
parser.add_argument('cmd', nargs='?', default='None',
    help='command to run in remote system')
parser.add_argument('--remote_home', nargs='?', default='/home/oni/projects/',
    help='remote projects/ directory')
"""


def generate_BEC1D(num_samples=10, filename='bec1d.data'):
  data_1d = []
  # sample `g` from uniform distribution [0, 500]
  for g in tqdm(np.random.uniform(0, 500, num_samples)):
    # generate particle density for 1-dimensional BEC
    x, psi = data.particle_density_BEC1D(
        dim=512, radius=24, angular_momentum=1,
        time_step=1e-4, coupling=g,
        iterations=10000
        )
    data_1d.append((g, psi))
  # write to disk
  utils.save({ 'x' : x, 'data' : data_1d }, filename)
  logger.info('Data written to {}'.format(filename))


if __name__ == '__main__':
  generate_BEC1D(num_samples=50000)
  # data.plot_wave_function(x, psi)
  """
  data_1d = pickle.load(open('bec1d.data', 'rb'))
  for i, (g, psi) in enumerate(data_1d['data']):
    logger.info(g)
    data.plot_wave_function(data_1d['x'], psi,
        title='g = {}'.format(g),
        save_to='psi-{}.png'.format(i)
        )
  """
