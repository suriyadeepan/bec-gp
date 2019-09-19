import math
import os

import numpy as np

from neuralbec.approximations import GPApproximation


def setup(config):
  # setup /path/to/results
  if not os.path.exists(config.path_to_results):
    os.mkdir(config.path_to_results)
  # setup /path/to/results/config
  path = os.path.join(config.path_to_results, config.name)
  if not os.path.exists(path):
    os.mkdir(path)


class BasicConfig:

  """ Basic Configuration """

  # ----------------- General --------------- #
  name = 'working'  # name of simulation/experiment/data/model; handle to everything
  path_to_results = 'results'
  two_component = False

  # -------- Trotter Suzuki ----------------- #
  dim = 512  # dimensions of grid
  radius = 24  # radius of particle?
  angular_momentum = 1
  time_step = 1e-4
  #coupling = 200.045  # coupling coefficient (g)
  #coupling = 10.  # coupling coefficient (g)
  iterations = 10000  # number of iterations of evolution
  # coupling_vars = [0.5, 1, 10, 90, 130, 200, 240, 300]
  coupling_vars = np.random.uniform(0, 150, (500,))  # variable `g` values

  sub_sample_counts = [ 200, 400, 600, 800 ]

  def wave_function(x, y):  # a working wave function
    return np.exp(-0.5 * (x**2 + y**2)) / np.sqrt(np.pi)

  def potential_fn(x, y):  # a working potential
    return 0.5 * (x ** 2 + y ** 2)

  # -------- Approximation ------------------ #
  model = GPApproximation  # approximating model
  sub_sample_count = 250  # number of datapoints to train the mdoel

  # -------- Visualization ------------------ #
  num_plots = 9
  num_prediction_plots = 3


class ChildConfig(BasicConfig):  # demo of using your own config
  name = 'child'  # just change what needs to be changed
  coupling = 0.5
  path = 'newresults'
  # variable `g`
  coupling_vars = np.arange(0.1, 1, 0.1)
  # visuals
  num_plots = 6
  num_prediction_plots = 9


class Harmonic(BasicConfig):

  """ Basic Configuration """

  # ----------------- General --------------- #
  name = 'harmonic'  # name of simulation/experiment/data/model; handle to everything
  num_plots = 15

  # -------- Trotter Suzuki ----------------- #
  coupling_vars = np.random.uniform(0, 300, (1000,))  # variable `g` values

  def wave_function(x, y):
    return np.exp(-0.5 * (x**2 + y**2)) / np.sqrt(np.pi)

  def potential_fn(x, y):
    return 0.5 * (x ** 2 + y ** 2)


class OpticalPot(BasicConfig):

  """ OpticalPot Configuration """

  # -------- Trotter Suzuki ----------------- #
  coupling_vars = np.random.uniform(0, 50, (500,))  # variable `g` values

  # ----------------- General --------------- #
  name = 'optical_pot'

  # -------- Approximation ------------------ #
  sub_sample_count = 500

  # -------- Visualization ------------------ #
  num_plots = 15
  num_prediction_plots = 15


class TwoComponentConfig(BasicConfig):

  two_component = True
  name = 'basic_2component'

  def potential_fn(x, y):
    return 0.5 * (x ** 2 + y ** 2)

  potential_fn_1 = potential_fn
  potential_fn_2 = potential_fn

  coupling = {
      'g_1' : 1,
      'g_2' : 1,
      'g_12' : 0
      }


class SimulateOnce(BasicConfig):
  name = 'simonce'
  coupling = 0.5

  coupling_vars = np.arange(50, 500, 10).astype(float)
  #coupling_vars = np.random.uniform(0, 10, (500,))  # variable `g` values
  num_plots = 15


class DoubleWell(BasicConfig):

  name = 'double'
  coupling = 0.0001
  def potential_fn(x, y):
    return (4. - x**2) ** 2 / 2.


class OpticalLattice(BasicConfig):

  name = 'optical_lat'
  coupling = 1

  coupling_vars = np.arange(0.1, 10, 0.5).astype(float)
  num_plots = 30

  sub_sample_counts = [ 100, 200, 250, 300, 350, 400, 450, 500, 550, 600 ]

  def potential_fn(x, y):
    return (x ** 2) / ( 2 + 12 * (math.sin(4 * x) ** 2) )


# [[.]] list of configurations
configs = {
    BasicConfig.name : BasicConfig,
    OpticalLattice.name : OpticalLattice
    }
