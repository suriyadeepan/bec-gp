import numpy as np
from neuralbec.approximations import GPApproximation


"""
class Config:
    batch_size = 256
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam
    epochs = 100
    model = 'ffn'
    data = 'bec2d'
    num_samples = 1024
    net = 'ffn'
    nn = {
      'dim_in' : 1, 'dim_hid' : 750, 'dim_out' : 51
      }
    g_low = 90
    g_high = 100
    save_every = 16
"""


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
  name = 'harmonic_2000_pts'
  model = GPApproximation

  def wave_function(r):  # constant
    return 1. / np.sqrt(24)

  def potential_fn(x, y):  # harmonic potential
    return 0.5 * (x ** 2 + y ** 2)
