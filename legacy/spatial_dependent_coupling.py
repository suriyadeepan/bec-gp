from neuralbec import data
import time
import numpy as np

from tqdm import tqdm

G0 = 1
B = 1
DIM = 12


if __name__ == '__main__':

  # start timer
  start_time = time.time()

  def wave_function(x, y):
    return np.exp(-(x**2 / 4.)) / np.sqrt(np.sqrt(np.pi))
    # return np.exp(-(x**2 / 4.)) / 2 * np.pi
    # return np.exp(-(x**2 / 4.)) / np.sqrt(2 * np.pi)
    # return np.exp(-(x**2 / 4.)) / (2 * np.sqrt(np.pi))
    # return 1.

  def spatial_dependence(x, y):
    return G0 * np.exp((-B * x) / 2)

  datapoint = { 'x' : [], 'g' : [], 'psi' : [] }
  for i, x in tqdm(enumerate(np.arange(0, DIM, 0.1))):
    datapoint_x = data.particle_density_BEC1D(
        dim=DIM,
        radius=0.0025 * DIM,
        angular_momentum=0,
        time_step=0.000020,
        coupling=spatial_dependence(x, 0),
        potential_fn=lambda x, y : x**2 / 4.,
        iterations=5000,
        wave_function=wave_function
        )
    print(datapoint_x)
    """
    # end timer
    print('_________ {} seconds _________'.format(
      time.time() - start_time))
    print('g = {}'.format(spatial_dependence(x, 0)))
    # copy properties
    datapoint['x'] = datapoint_x['x']
    datapoint['g'] = datapoint_x['g']
    datapoint['psi'].append(datapoint_x['psi'][i])
    """

  """
  x = datapoint['x']
  g = datapoint['g']
  psi = datapoint['psi']

  data.plot_wave_function(x, psi, title='Spatial Dependence', save_to='spatial.png')
  """
