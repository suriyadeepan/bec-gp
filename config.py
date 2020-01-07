import math
import os

import numpy as np

from neuralbec.approximations import GPApproximation
from neuralbec.potentials import *


def setup(config):
  # setup /path/to/results
  if not os.path.exists(config.path_to_results):
    os.mkdir(config.path_to_results)
  # setup /path/to/results/config
  path = os.path.join(config.path_to_results, config.name)
  if not os.path.exists(path):
    os.mkdir(path)
  # and if "two-component"
  path2 = os.path.join(path, "2")
  if not os.path.exists(path2):
    os.mkdir(path2)


class BasicConfig:

  """ Basic Configuration """

  # ----------------- General --------------- #
  name = 'working'  # name of simulation/experiment/data/model; handle to everything
  _type = 'one-component'
  path_to_results = 'results'
  two_component = False

  # -------- Trotter Suzuki ----------------- #
  dim = 512  # dimensions of grid
  radius = 24  # radius of particle?
  angular_momentum = 1
  time_step = 1e-4
  iterations = 10000  # number of iterations of evolution
  # coupling_vars = [0.5, 1, 10, 90, 130, 200, 240, 300]
  coupling_vars = np.random.uniform(0, 100, (500,))  # variable `g` values

  sub_sample_counts = [ 1500 ]

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

  coupling_vars = np.random.uniform(0, 2, (500,))  # variable `g` values


class OpticalLattice(BasicConfig):

  name = 'optical_lat'
  coupling = 1

  def potential_fn(x, y):
    return (x ** 2) / ( 2 + 12 * (math.sin(4 * x) ** 2) )

  coupling_vars = np.random.uniform(0, 2, (500,))  # variable `g` values


class BasicTwoComponentConfig:

  """ Basic Configuration """

  # ----------------- General --------------- #
  name = 'basic2'  # name of simulation/experiment/data/model; handle to everything
  _type = 'two-component'
  path_to_results = 'results'
  two_component = True

  # -------- Trotter Suzuki ----------------- #
  dim = 300  # dimensions of grid
  radius = 20.  # radius of particle?
  time_step = 1e-2
  iterations = 20000  # number of iterations of evolution

  sub_sample_counts = [ 200, 400, 600, 800 ]

  # -------- Approximation ------------------ #
  model = GPApproximation  # approximating model
  sub_sample_count = 250  # number of datapoints to train the mdoel
  omega = -1


class TwoComponentConfig(BasicTwoComponentConfig):
  # type
  _type = 'two-component'
  # name
  name = '2comp'
  # coupling coefficients
  g11 = 1 # 103
  g12 = 1 # 100
  g22 = 1 # 97
  couplings = [ g11, g12, g22 ]
  coupling_vars = np.stack(
      [np.random.uniform(0, 2, (100)),
      np.random.uniform(0, 2, (100)),
      np.random.uniform(0, 2, (100))]
      ).transpose()
  # lattice
  dim, radius = 300, 20.

  # potential
  # def potential_fn(x, y):
  #   return 0.5 * (x ** 2) + 24 * (math.cos(x)) ** 2
  def potential_fn(x, y):
    return (4. - x**2) ** 2 / 2.

  # iterations
  iterations = 20000

  # time steps
  time_step = 1e-2


class BasicTwoDimensionalConfig:

  """ Basic 2D Configuration """

  # ----------------- General --------------- #
  name = 'basic2d'  # name of simulation/experiment/data/model; handle to everything
  _type = 'two-dim'
  path_to_results = 'results'
  two_component = False
  # 256, 15, 0.05, 1e-3, 500., lambda x, y : 0.5 * (x**2 + y**2), 8000)
  # -------- Trotter Suzuki ----------------- #
  dim = 256  # dimensions of grid [ 256 x 256 ]
  radius = 15  # radius of particle?
  angular_momentum = 0.05
  time_step = 1e-3
  iterations = 10000  # number of iterations of evolution
  coupling = 99.
  coupling_vars = np.random.uniform(0., 500., (2,))  # variable `g` values

  sub_sample_counts = [ 200, 400, 600, 800, 1000, 1500, 2000, 3000, 4000 ]

  def wave_function(x, y):  # a working wave function
    return np.exp(-0.5 * (x**2 + y**2)) / np.sqrt(np.pi)

  def potential_fn(x, y):  # a working potential
    return 0.5 * (x ** 2 + y ** 2)

  # -------- Approximation ------------------ #
  model = GPApproximation  # approximating model
  sub_sample_count = 1500  # number of datapoints to train the mdoel





class TwoDimConfig(BasicTwoDimensionalConfig):
  name = '2dim'
  coupling_vars = np.random.uniform(0., 50., (500,))  # variable `g` values


"""
class StepPotentialChange(BasicConfig):  # demo of using your own config
  name = 'steppotchange'  # just change what needs to be changed
  coupling = 1
  _type = 'potential-change'
  potential_fn = get_potential_fn('step', 0.1)
  potential_fns = [ get_potential_fn('step', j)
      for j in np.random.uniform(0., 0.9, (500,)) ]


class LinearPotentialChange(BasicConfig):  # demo of using your own config
  name = 'linearpotchange'  # just change what needs to be changed
  coupling = 1
  _type = 'potential-change'
  potential_fn = get_potential_fn('linear', 0.1)
  potential_fns = [ get_potential_fn('linear', j)
      for j in np.random.uniform(0., 0.9, (500,)) ]


class FourierPotentialChange(BasicConfig):  # demo of using your own config
  name = 'fourierpotchange'  # just change what needs to be changed
  coupling = 1
  _type = 'potential-change'
  potential_fn = get_potential_fn('fourier', 0.1)
  potential_fns = [ get_potential_fn('fourier', j)
      for j in np.random.uniform(0., 0.9, (500,)) ]

class DisorderPotentialChange(BasicConfig):  # demo of using your own config
  name = 'disorderpotchange'  # just change what needs to be changed
  coupling = 1
  _type = 'potential-change'
  potential_fn = gaussian_disorder_potential(amp=0)
  wave_function = gaussian_disorder_potential()
"""

class VecBec(BasicTwoComponentConfig):

  name = 'vecbec'
  omegas = np.concatenate([np.arange(-20, -2, 0.1), np.arange(-2, 0, 0.01)])
  # omegas = [-20, -1]
  dim = 300
  radius = 20.
  time_step = 1e-4
  iterations = 100000
  # coupling coefficients
  g11 = 103
  g12 = 100
  g22 = 97
  couplings = [ g11, g12, g22 ]

  def potential_fn(x, y):
    return 0.5 * (x ** 2) + 24 * (math.cos(x)) ** 2


# [[.]] list of configurations
configs = {
  BasicConfig.name : BasicConfig,
  DoubleWell.name : DoubleWell,
  OpticalLattice.name : OpticalLattice,
  BasicTwoComponentConfig.name : BasicTwoComponentConfig,
  TwoComponentConfig.name : TwoComponentConfig,
  TwoComponentConfig.name : TwoComponentConfig,
  TwoDimConfig.name : TwoDimConfig,
  VecBec.name : VecBec,
  # DisorderPotentialChange.name : DisorderPotentialChange
  # StepPotentialChange.name : StepPotentialChange,
  # LinearPotentialChange.name : LinearPotentialChange,
  # FourierPotentialChange.name : FourierPotentialChange,
  }
