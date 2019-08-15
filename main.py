from neuralbec.simulation import VariableCouplingBec, Bec
from neuralbec.simulation import OneDimensionalData
from config import ConfigHarmonic
import logging
import argparse
import warnings

warnings.filterwarnings("ignore")

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

"""
# train mode
parser.add_argument('--train', default=False, action='store_true',
    help='Train model on data')
"""
args = parser.parse_args()


if __name__ == '__main__':

  if args.simulate_once:  # one-off simulation mode
    data = Bec(ConfigHarmonic)
    data.save(f'g={ConfigHarmonic.coupling}.{ConfigHarmonic.name}')
  elif args.simulate:  # simulation mode
    exp = VariableCouplingBec(ConfigHarmonic)  # create experiment
    data = exp.run()  # run experiment; generate data
    data.save(ConfigHarmonic.name)  # save simulated data to disk
  elif args.approximate:  # appoximate mode
    data = OneDimensionalData(ConfigHarmonic.name)
    # create model
    model = ConfigHarmonic.model()
    X = data.df.loc[:500, ['x', 'g']]
    y = data.df.loc[:500, 'psi']
    model.fit(X, y)
    model.save()
