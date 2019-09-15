import numpy as np
from neuralbec.approximations import GPApproximation


class BasicConfig:

  """ Basic Configuration """

  # ----------------- General --------------- #
  name = 'working'  # name of simulation/experiment/data/model; handle to everything
  path = 'results'
  two_component = False

  # -------- Trotter Suzuki ----------------- #
  dim = 512  # dimensions of grid
  radius = 24  # radius of particle?
  angular_momentum = 1
  time_step = 1e-4
  coupling = 200.045  # coupling coefficient (g)
  iterations = 10000  # number of iterations of evolution
  # coupling_vars = [0.5, 1, 10, 90, 130, 200, 240, 300]
  coupling_vars = np.random.uniform(0, 10, (500,))  # variabel `g` values

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
  coupling_vars = np.random.uniform(0, 10, (500,))  # variable `g` values

  # ----------------- General --------------- #
  name = 'optical_pot'

  # -------- Approximation ------------------ #
  sub_sample_count = 150

  # -------- Visualization ------------------ #
  num_plots = 15
  num_prediction_plots = 9


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
