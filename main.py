from neuralbec.simulation import VariableCouplingBec
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
parser.add_argument('--simulate', default=False, action='store_true',
    help='Generate Data')
"""
# train mode
parser.add_argument('--train', default=False, action='store_true',
    help='Train model on data')
"""
args = parser.parse_args()


if __name__ == '__main__':

  if args.simulate:  # simulation mode
    exp = VariableCouplingBec(ConfigHarmonic)  # create experiment
    data = exp.run()  # run experiment; generate data
    data.save(ConfigHarmonic.name)  # save simulated data to disk
