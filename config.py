import numpy as np
from neuralbec.approximations import GPApproximation


class BasicConfig:
  dim = 512
  radius = 24
  angular_momentum = 1
  time_step = 1e-4
  coupling = 1
  iterations = 100000
  # coupling_vars = [0.5, 1, 10, 90, 130, 200, 240, 300]  # variable `g` values
  coupling_vars = np.random.uniform(0, 500, (2000,))
  name = 'basic'
  model = GPApproximation

  def wave_function(x, y):  # constant
    return np.exp(-0.5 * (x**2 + y**2)) / np.sqrt(np.pi)

  def potential_fn(x, y):  # harmonic potential
    return 0.5 * (x ** 2 + y ** 2)


class ConfigChild(BasicConfig):
  coupling = 0.5

  def wave_function(r):
    return 0.5 * np.exp(r**2) / np.sqrt(np.pi)
