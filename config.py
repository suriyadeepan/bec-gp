import numpy as np
from neuralbec.approximations import GPApproximation


class ConfigHarmonic:
  dim = 512
  radius = 24
  angular_momentum = 1
  time_step = 1e-4
  coupling = 1
  potential_fn = None
  iterations = 10000
  # coupling_vars = [0.5, 1, 10, 90, 130, 200, 240, 300]  # variable `g` values
  coupling_vars = np.random.uniform(0, 500, (2000,))
  name = 'harmonic'
  model = GPApproximation

  def wave_function(r):  # constant
    return 1. / np.sqrt(24)

  def potential_fn(x, y):  # harmonic potential
    return 0.5 * (x ** 2 + y ** 2)
