from neuralbec import data

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
  # Generate 1D BEC
  data.generate_varg(
      fn=lambda g : data.particle_density_BEC1D(
        dim=512, radius=24, angular_momentum=1,
        time_step=1e-4, coupling=g,
        iterations=10000
        ),
      num_samples=10, filename='10samples.data'
      )
