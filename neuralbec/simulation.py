import trottersuzuki as ts
import numpy as np
import pandas as pd

from neuralbec import utils
from tqdm import tqdm

import os


class SimulatedData:

  def __init__(self):
    pass


class OneDimensionalData(SimulatedData):

  def __init__(self, path):
    self.path = path
    self.df = utils.to_df({ 'x' : [], 'psi' : [], 'g' : [] })

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
    # add df to data
    self.df = self.df.append(df, ignore_index=True)

  def load(self, filename='sim.csv'):
    self.df = pd.read_csv(os.path.join(self.path, filename), index_col=0)
    return self.df

  def save(self, config, filename=None):
    # check if there are more than one `g` value
    g = self.df.g.unique()
    if filename is None:
      if len(g) == 1:  # figure out file name
        filename = 'sim_g={}.csv'.format(g[0])
      else:
        filename = 'sim.csv'
    # write to file
    path = os.path.join(config.path_to_results, config.name, filename)
    self.df.to_csv(path, encoding='utf-8')
    print(f'Simulation results saved to [{path}]')

  @property
  def stats(self):
    # unique simulations
    return {
        '#Simulations' : len(self.df.g.unique()),
        '#Datapoints'  : len(self.df),
        'gMin' : self.df.g.min(),
        'gMax' : self.df.g.max(),
        }

  def __repr__(self):
    return str(self.stats)


class Simulation:

  def __init__(self):
    pass


class Bec(Simulation):

  def __new__(self, config, coupling):
    data = OneDimensionalData(
        os.path.join(config.path_to_results, config.name)
        )
    data.add(one_dimensional_bec(config, coupling=coupling))
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


def one_dimensional_bec_2_component(config, coupling=None, iterations=None):
  # get coupling strength
  coupling = coupling if coupling else config.coupling
  assert isinstance(coupling, type({ 1 : 2}))
  # Set up lattice
  grid = ts.Lattice1D(config.dim, config.radius)
  # initialize state
  state = ts.State(grid, config.angular_momentum)
  state.init_state(config.wave_function)
  # init potential
  potential_1 = ts.Potential(grid)
  potential_1.init_potential(config.potential_fn_1)  # harmonic potential
  potential_2 = ts.Potential(grid)
  potential_2.init_potential(config.potential_fn_2)  # harmonic potential
  # build hamiltonian with coupling strength `g1`, `g2`, `g12`
  hamiltonian = ts.Hamiltonian2Component(grid, potential_1, potential_2,)
  """
      coupling_1=config.coupling['g_1'],
      coupling_2=config.coupling['g_2'],
      coupling_12=config.coupling['g_12'])
  """
  # setup solver
  solver = ts.Solver(grid, state, hamiltonian, config.time_step)
  # get iterations
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
    'g_1' : np.ones(psi.shape) * coupling['g_1'],
    'g_2' : np.ones(psi.shape) * coupling['g_2'],
    'g_12' : np.ones(psi.shape) * coupling['g_12'],
    'psi' : psi
    })


class Experiment:

  def __init__(self):
    pass


class VariableCouplingBec(Experiment):

  def __init__(self, config, two_component=False):
    # keep track of config
    self.config = config
    # data holder
    self.data = OneDimensionalData(
        os.path.join(config.path_to_results, config.name)
        )

  def run(self, coupling_vars=None):
    if coupling_vars is None:
      coupling_vars = self.config.coupling_vars
    for g in tqdm(coupling_vars):
      # run a simulation
      sdata = one_dimensional_bec(self.config, coupling=g)
      # save data
      self.data.add(sdata)

    return self.data


class OneDimensionalTwoComponentData(SimulatedData):

  def __init__(self, path, name=None):
    self.path = path
    if not name:
      self.df = utils.to_df({ 'x' : [], 'psi' : [],
        'g_1' : [], 'g_2' : [], 'g_12' : [] })
    else:
      self.df = pd.read_csv(os.path.join(
          self.path,
          '/bec_{}.csv'.format(name)),
          sep='\t')

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
    self.df.to_csv(os.path.join(
      self.path, 'bec_2component_{}.csv'.format(name)),
        sep='\t', encoding='utf-8')


class TwoComponentBec(Simulation):

  def __new__(self, config):
    data = OneDimensionalTwoComponentData()
    data.add(one_dimensional_bec_2_component(config))
    return data
