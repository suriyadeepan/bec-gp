from neuralbec.simulation import VariableCouplingBec, Bec
from neuralbec.simulation import OneDimensionalData
from neuralbec.visualize import plot, plot_from_file

from config import BasicConfig, Harmonic

import logging
import argparse
import warnings

import os

warnings.filterwarnings("ignore")
config = Harmonic  # ------- configuration goes here --------

# setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# parse command-line arguments
parser = argparse.ArgumentParser(
    description='neuralbec : Neural Network based simulation of BEC'
    )
# ---------------
# -- SIMULATE ---
# ---------------
parser.add_argument('--simulate-once', default=False, action='store_true',
    help='Run simulation once')
parser.add_argument('--simulate', default=False, action='store_true',
    help='Run simulation')
parser.add_argument('--approximate', default=False, action='store_true',
    help='Create Approximation')
parser.add_argument('--visualize', default=False, action='store_true',
    help='Visualize Wave function')

"""
# train mode
parser.add_argument('--train', default=False, action='store_true',
    help='Train model on data')
"""
args = parser.parse_args()


if __name__ == '__main__':

  if args.simulate_once:  # one-off simulation mode
    data = Bec(config)
    data.save(f'g={config.coupling}.{config.name}')
    plot_from_file(os.path.join(
      'results',
      f'bec_g={config.coupling}.{config.name}.csv'
      ))
  elif args.simulate:  # simulation mode
    exp = VariableCouplingBec(config)  # create experiment
    data = exp.run()  # run experiment; generate data
    data.save(config.name)  # save simulated data to disk
  elif args.approximate:  # appoximate mode
    data = OneDimensionalData(config.name)
    # create model
    model = config.model()
    df_sub = data.df.sample(config.sub_sample_count)
    X = df_sub[['x', 'g']]
    y = df_sub.psi
    model.fit(X, y)
    model.save()
  elif args.visualize:
    # plot_from_file(os.path.join(
    #   'results',
    #   f'bec_g={config.coupling}.{config.name}.csv'
    #   ))
    # plot_from_file(os.path.join(
    #  'results',
    #  f'bec_{config.name}.csv'
    #  ))
    plot(config)
