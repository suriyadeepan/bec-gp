import pytest
import numpy as np
import trottersuzuki as ts
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# @pytest.fixture
def gaussian_disorder_fn(x, y):
  sigma_d=1.
  low=-20
  high=20
  step=0.2
  # sample Ai from uniform [-20, 20]
  Xi = np.arange(-10., 10., step)  # -20 : 20 : 2e-3
  # assert Xi.shape[0] == 1e4, Xi.shape[0]
  Ai = np.random.uniform(0, 10, Xi.shape[0])
  exp_term = np.exp( -1. * np.square(x - Xi) / ( 2 * np.square(sigma_d) ) )
  return (Ai * exp_term).sum()


# @pytest.fixture
def wave_function(x, y):  # a working wave function
  return np.exp(-0.5 * (x**2 + y**2)) / np.sqrt(np.pi)


# def test_potentialchange(gaussian_disorder_fn, wave_function):
def test_potentialchange():
  # get coupling strength
  coupling = 1000
  dim = 256
  radius = 24
  angular_momentum = 1
  time_step = 1e-4
  iterations = 100000  # number of iterations of evolution
  # Set up lattice
  grid = ts.Lattice1D(dim, radius)
  # initialize state
  state = ts.State(grid, angular_momentum)
  state.init_state(wave_function)
  # init potential
  potential = ts.Potential(grid)
  potential.init_potential(gaussian_disorder_fn)
  # build hamiltonian with coupling strength `g` and potential `u(.)`
  hamiltonian = ts.Hamiltonian(grid, potential, 1., coupling)
  # setup solver
  solver = ts.Solver(grid, state, hamiltonian, time_step)
  # Evolve the system
  solver.evolve(iterations, False)
  # Compare the calculated wave functions w.r.t. groundstate function
  # psi = np.sqrt(state.get_particle_density()[0])
  psi = state.get_particle_density()[0]
  # psi / psi_max
  psi = psi / max(psi)
  # save data
  return grid.get_x_axis(), psi


if __name__ == '__main__':
  x, psi = test_potentialchange()
  plt.xlim(-6, 6)
  plt.plot(x, psi)
  plt.show()
