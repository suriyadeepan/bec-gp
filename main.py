from neuralbec.simulation import VariableCouplingBec, Bec
from neuralbec.simulation import OneDimensionalData
from neuralbec.visualize import plot, plot_from_file, plot_prediction

from config import BasicConfig, Harmonic, OpticalPot
from config import ChildConfig

from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

import logging
import argparse
import warnings

import os

warnings.filterwarnings("ignore")
config = OpticalPot  # ------- configuration goes here --------
#config = ChildConfig
# check if `config.path` exists
if not os.path.exists(config.path):
  os.mkdir(config.path)

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
parser.add_argument('--stats', default=False, action='store_true',
    help='Spit out statistics of simulation/approximation')

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
      config.path,
      f'bec_g={config.coupling}.{config.name}.csv'
      ))
  elif args.simulate:  # simulation mode
    exp = VariableCouplingBec(config)  # create experiment
    data = exp.run()  # run experiment; generate data
    data.save(config.name)  # save simulated data to disk
  elif args.approximate and not args.visualize:  # appoximate mode
    data = OneDimensionalData(config.path, config.name)
    # create model
    model = config.model(config=config)
    print(len(data.df))
    df_sub = data.df.sample(config.sub_sample_count)
    X = df_sub[['x', 'g']]
    y = df_sub.psi
    model.fit(X, y)
    # calculate error on test set
    testset = data.df.sample(config.sub_sample_count)
    error = model.evaluate(testset[['x', 'g']], testset['psi'])
    # save model to disk
    model.save()
    # plot predictions; might as well
    plot_prediction(config)
  elif args.visualize and not args.approximate:
    plot(config)
  elif args.visualize and args.approximate:
    plot_prediction(config)
  elif args.stats:
    print(config.name)
    print('-' * len(config.name))
    print('\nSimulation Stats')
    pp.pprint(OneDimensionalData(path=config.path, name=config.name).stats)
    print('\nModel Stats')
    pp.pprint(config.model(config=config).load().stats)

