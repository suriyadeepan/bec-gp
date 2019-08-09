from neuralbec import data
import time
import numpy as np


G = 62.742


if __name__ == '__main__':

  # start timer
  start_time = time.time()

  def wave_function(x, y):
    # return np.exp(-(x**2 / 4.)) / np.sqrt(np.sqrt(np.pi))
    # return np.exp(-(x**2 / 4.)) / 2 * np.pi
    # return np.exp(-(x**2 / 4.)) / np.sqrt(2 * np.pi)
    # return np.exp(-(x**2 / 4.)) / (2 * np.sqrt(np.pi))
    return 1.

  datapoint = data.particle_density_BEC1D(
      dim=8000,
      radius=0.0025 * 8000,
      angular_momentum=0,
      time_step=0.000020,
      coupling=G,
      potential_fn=lambda x, y : x**2 / 4.,
      iterations=5000,
      wave_function=wave_function
      # init_state_fn=lambda x, y : 1. / np.sqrt(0.0025 * 8000)
      # init_state_fn=lambda x, y : lambda x : np.exp(x**2 / 4.)
      )
  # end timer
  print('_________ {} seconds _________'.format(
    time.time() - start_time))

  x = datapoint['x']
  g = datapoint['g']
  psi = datapoint['psi']

  data.plot_wave_function(x, psi, title='g={}'.format(G),
      save_to='{}_2_sqrt_sqrt_pi_const_init_state.png'.format(G))
