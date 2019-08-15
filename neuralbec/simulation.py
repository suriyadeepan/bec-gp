import trottersuzuki as ts
import numpy as np
import pandas as pd

from neuralbec import utils
from tqdm import tqdm


class SimulatedData:

  def __init__(self):
    pass


class OneDimensionalData(SimulatedData):

  def __init__(self, name=None):
    if not name:
      self.df = utils.to_df({ 'x' : [], 'psi' : [], 'g' : [] })
    else:
      self.df = pd.read_csv('results/bec_{}.csv'.format(name), sep='\t')

  @property
  def X(self):
    return self.df.x

  @property
  def psi(self):
    return self.df.psi

  @property
  def g(self):
    return self.df.g

  def add(self, df):
    self.df = self.df.append(df, ignore_index=True)

  def save(self, name=''):
    self.df.to_csv('results/bec_{}.csv'.format(name),
        sep='\t', encoding='utf-8')


class Simulation:

  def __init__(self):
    pass


class Bec(Simulation):

  def __new__(self, config):
    data = OneDimensionalData()
    data.add(one_dimensional_bec(config))
    return data


def one_dimensional_bec(config, coupling=None, iterations=None):
    # get coupling strength
    coupling = coupling if coupling else config.coupling
    # Set up lattice
    grid = ts.Lattice1D(config.dim, config.radius)
    # initialize state
    state = ts.State(grid, config.angular_momentum)
    state.init_state(config.wave_function)
    # init potential
    potential = ts.Potential(grid)
    potential.init_potential(config.potential_fn)  # harmonic potential
    # build hamiltonian with coupling strength `g`
    hamiltonian = ts.Hamiltonian(grid, potential, 1., coupling)
    # setup solver
    solver = ts.Solver(grid, state, hamiltonian, config.time_step)

    iterations = config.iterations if not iterations else iterations
    # Evolve the system
    solver.evolve(iterations, False)
    # Compare the calculated wave functions w.r.t. groundstate function
    # psi = np.sqrt(state.get_particle_density()[0])
    psi = state.get_particle_density()[0]
    # psi / psi_max
    psi = psi / max(psi)
    # save data
    return utils.to_df({
      'x' : grid.get_x_axis(),
      'g' : np.ones(psi.shape) * coupling,
      'psi' : psi
      })


class Experiment:

  def __init__(self):
    pass


class VariableCouplingBec(Experiment):

  def __init__(self, config):
    # keep track of config
    self.config = config
    # create simulations
    # self.simulations = [ OneDimensionalBec(config, coupling=g) for g in config.coupling_vars ]
    # data holder
    self.data = OneDimensionalData()

  def run(self):
    for g in tqdm(self.config.coupling_vars):
      # run a simulation
      sdata = one_dimensional_bec(self.config, coupling=g)
      # save data
      self.data.add(sdata)

    return self.data
