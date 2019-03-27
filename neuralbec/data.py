import numpy as np
import trottersuzuki as ts
import math
from matplotlib import pyplot as plt

import logging
import pickle

# setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def harmonic_potential(x, y):
  """ Harmonic Potential

      ( x^2 + y^2 ) / 2
  """
  return 0.5 * (x**2 + y**2)


def custom_potential_1(x, y):
  """ Custom Potential

      0.5x^2 + 24cos^2(x)
  """
  return 0.5 * (x**2) + 24 * (math.cos(x) ** 2)


def particle_density_BEC1D(dim, radius, angular_momentum, time_step,
  coupling, iterations):
  """Estimate Particle Density of 1-dimensional BEC system

  Parameters
  ----------
  dim : int
    dimensions of lattice
  radius : float
    physical radius
  angular_momentum : float
    quantum number
  time_step : float
    time steps
  coupling : float
    coupling strength (g)
  iterations : int
    number of iterations of trotter-suzuki

  Returns
  -------
  numpy.ndarray
    particle density of evolved system
  """
  # Set up lattice
  grid = ts.Lattice1D(dim, radius, False, "cartesian")
  # initialize state
  state = ts.State(grid, angular_momentum)
  state.init_state(lambda r : 1. / np.sqrt(radius))  # constant state
  # init potential
  potential = ts.Potential(grid)
  potential.init_potential(harmonic_potential)  # harmonic potential
  # build hamiltonian with coupling strength `g`
  hamiltonian = ts.Hamiltonian(grid, potential, 1., coupling)
  # setup solver
  solver = ts.Solver(grid, state, hamiltonian, time_step)
  # Evolve the system
  solver.evolve(iterations, True)
  # Compare the calculated wave functions w.r.t. groundstate function
  psi = np.sqrt(state.get_particle_density()[0])
  # psi / psi_max
  psi = psi / max(psi)

  return grid.get_x_axis(), psi


def plot_wave_function(x, psi, fontsize=16, title=None, save_to=None):
  """ Plot Wave Function """
  # set title
  if title:
    plt.title(title)
  # settings
  plt.plot(x, psi, 'o', markersize=3)
  plt.xlabel('x', fontsize=fontsize)
  plt.ylabel(r'$\psi$', fontsize=fontsize)
  # save figure
  if save_to:
    plt.savefig(save_to)
    logger.info('Figure saved to {}'.format(save_to))
  # disply image
  plt.show()


def load(name, path='data/'):
  datadump = pickle.load(open('{}/{}.data'.format(path, name), 'rb'))
  inputs = [ g for g, density in datadump['data'] ]
  outputs = [ density for g, density in datadump['data'] ]
  reference = { 'x' : datadump['x'] }
  return inputs, outputs, reference
