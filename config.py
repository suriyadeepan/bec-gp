import numpy as np
from neuralbec.approximations import GPApproximation


class BasicConfig:

  """ Basic Configuration """

  # ----------------- General --------------- #
  name = 'working'  # name of simulation/experiment/data/model; handle to everything

  # -------- Trotter Suzuki ----------------- #
  dim = 512  # dimensions of grid
  radius = 24  # radius of particle?
  angular_momentum = 1
  time_step = 1e-4
  coupling = 200.045  # coupling coefficient (g)
  iterations = 10000  # number of iterations of evolution
  # coupling_vars = [0.5, 1, 10, 90, 130, 200, 240, 300]
  coupling_vars = np.random.uniform(0, 200, (1000,))  # variabel `g` values

  def wave_function(x, y):  # a working wave function
    return np.exp(-0.5 * (x**2 + y**2)) / np.sqrt(np.pi)

  def potential_fn(x, y):  # a working potential
    return 0.5 * (x ** 2 + y ** 2)

  # -------- Approximation ------------------ #
  model = GPApproximation  # approximating model
  sub_sample_count = 1000  # number of datapoints to train the mdoel

  # -------- Visualization ------------------ #
  num_plots = 9


class ChildConfig(BasicConfig):  # demo of using your own config
  name = 'child'  # just change what needs to be changed
  coupling = 0.5


class Harmonic(BasicConfig):

  """ Basic Configuration """

  # ----------------- General --------------- #
  name = 'harmonic'  # name of simulation/experiment/data/model; handle to everything

  # -------- Trotter Suzuki ----------------- #
  coupling_vars = np.random.uniform(0, 300, (1000,))  # variabel `g` values

  def wave_function(x, y):
    return np.exp(-0.5 * (x**2 + y**2)) / np.sqrt(np.pi)

  def potential_fn(x, y):
    return 0.5 * (x ** 2 + y ** 2)
