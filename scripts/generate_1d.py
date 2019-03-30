from neuralbec import data
import math


def generate_1d_cosine_potential(name='bec1d_cosine'):
  # define "cosine" potential
  datadict = data.generate_varg(
      fn=lambda g : data.particle_density_BEC1D(
        dim=512, radius=24, angular_momentum=1,
        time_step=1e-4, coupling=g,
        potential_fn=lambda x, y : 0.5 * (x**2) + 24. * (math.cos(x)**2),
        iterations=10000
        ),
      num_samples=100, filename='{}.data'.format(name)
      )
  return datadict


if __name__ == '__main__':
  # generate data
  datadict = generate_1d_cosine_potential()
  ref = datadict['x']
  g, psi = datadict['data'][0]
  # data.plot_wave_function(ref, psi, title='g = {}'.format(g))
