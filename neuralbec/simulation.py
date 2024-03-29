import trottersuzuki as ts
import numpy as np
import pandas as pd

from neuralbec import utils
from tqdm import tqdm

import os


class SimulatedData:

  def __init__(self):
    pass


class TwoDimensionalData(SimulatedData):

  def __init__(self, path):
    self.path = path
    self.df = utils.to_df({ 'x' : [], 'y' : [], 'psi' : [], 'g' : [] })

  @property
  def X(self):
    return self.df.x

  @property
  def Y(self):
    return self.df.y

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
    return self.df


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
    return self.df

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


class PotentialChangeData(SimulatedData):

  def __init__(self, path):
    self.path = path
    self.df = pd.DataFrame({ 'x' : [], 'psi' : [], 'u' : [] })

  @property
  def X(self):
    return self.df.x

  @property
  def psi(self):
    return self.df.psi

  @property
  def u(self):
    return self.df.u

  def add(self, df):
    # add df to data
    self.df = self.df.append(df, ignore_index=True)

  def load(self, filename='sim.csv'):
    self.df = pd.read_csv(os.path.join(self.path, filename), index_col=0)
    return self.df

  def save(self, config, filename=None):
    filename = 'sim.csv'
    path = os.path.join(config.path_to_results, config.name, filename)
    self.df.to_csv(path, encoding='utf-8')
    print(f'Simulation results saved to [{path}]')
    return self.df

  @property
  def stats(self):
    # unique simulations
    return {
        '#Datapoints'  : len(self.df),
        'uMin' : self.df.u.min(),
        'uMax' : self.df.u.max(),
        }

  def __repr__(self):
    return str(self.stats)


class PotentialChangeExperiment(Simulation):

  def __new__(self, config, potential_fn):
    data = PotentialChangeData(
        os.path.join(config.path_to_results, config.name)
        )
    data.add(potential_change_bec(config, potential_fn))
    return data


def two_dimensional_bec(config, coupling=None):
  """Estimate Particle Density of 2-dimensional BEC system """
  # resolve coupling
  coupling = coupling if coupling else config.coupling
  # Set up lattice
  grid = ts.Lattice2D(config.dim, config.radius)
  # initialize state
  state = ts.GaussianState(grid, config.angular_momentum)
  # create an harmonic potential
  potential = ts.HarmonicPotential(grid, 1., 1. / np.sqrt(2.))
  # build hamiltonian with coupling strength `g`
  hamiltonian = ts.Hamiltonian(grid, potential, 1., coupling)
  # setup solver
  solver = ts.Solver(grid, state, hamiltonian, config.time_step)
  # Evolve the system
  solver.evolve(config.iterations, True)
  # psi / psi_max
  psi = state.get_particle_density()
  # psi / psi_max
  psi = psi / psi.max()

  return utils.to_df({
      'x' : np.array(grid.get_x_axis()),
      'y' : np.array(grid.get_y_axis()),
      'g' : coupling,
      'psi' : psi
      })


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


def one_dimensional_bec_2_component(config, couplings=None, iterations=None, omega=None):
  # get coupling strength
  if couplings is None:
    couplings = config.couplings
  # choose omega
  if omega is None:
    omega = config.omega
  # couplings = couplings if couplings is not None else config.couplings
  # assert isinstance(couplings, type([69, ])) or isinstance(couplings, type(np.array([10.])))
  # Set up lattice
  grid = ts.Lattice1D(config.dim, config.radius)
  # initialize state
  #state = ts.State(grid, config.angular_momentum)
  #state.init_state(config.wave_function)
  state_1 = ts.GaussianState(grid, 1.)  # Create first-component system's state
  state_2 = ts.GaussianState(grid, 1.)  # Create second-component system's state
  # init potential
  potential_1 = ts.Potential(grid)
  potential_1.init_potential(config.potential_fn)  # harmonic potential
  potential_2 = ts.Potential(grid)
  potential_2.init_potential(config.potential_fn)  # harmonic potential
  # build hamiltonian with coupling strength `g1`, `g2`, `g12`
  hamiltonian = ts.Hamiltonian2Component(grid, potential_1, potential_2,
    _coupling_a=couplings[0], coupling_ab=couplings[1],
    _coupling_b=couplings[2], _omega_r=omega)
  # setup solver
  # solver = ts.Solver(grid, state, hamiltonian, config.time_step)
  solver = ts.Solver(grid, state_1, hamiltonian, config.time_step, State2=state_2)
  # get iterations
  iterations = config.iterations if not iterations else iterations
  # Evolve the system
  solver.evolve(iterations, True)
  # Compare the calculated wave functions w.r.t. groundstate function
  # psi = np.sqrt(state.get_particle_density()[0])
  psi1 = state_1.get_particle_density()
  psi2 = state_2.get_particle_density()
  assert psi1.shape == (1, config.dim)
  assert psi2.shape == (1, config.dim)
  # psi / psi_max
  psi1 = psi1[0] / max(psi1[0])
  psi2 = psi2[0] / max(psi2[0])
  # save data
  return utils.to_df({
    'x' : grid.get_x_axis(),
    'g11' : np.ones(psi1.shape) * couplings[0],
    'g12' : np.ones(psi1.shape) * couplings[1],
    'g22' : np.ones(psi1.shape) * couplings[2],
    'psi1' : psi1,
    'psi2' : psi2,
    'omega' : np.ones(psi1.shape) * omega
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

  def __init__(self, path):
    self.path = path
    self.df = utils.to_df({
      'x' : [], 'psi1' : [], 'psi2' : [],
      'g11' : [], 'g12' : [], 'g22' : [] })

  @property
  def X(self):
    return self.df.x

  @property
  def psi(self):
    return self.df[['psi1', 'psi2']]

  @property
  def g(self):
    return self.df[['g11', 'g12', 'g22']]

  def load(self, filename='sim.csv'):
    self.df = pd.read_csv(os.path.join(self.path, filename), index_col=0)
    return self.df

  def add(self, df):
    self.df = self.df.append(df, ignore_index=True)

  def save(self, config, filename='sim.csv'):
    # path = os.path.join(config.path_to_results, config.name, '2', filename)
    path = os.path.join(self.path, filename)  # .path_to_results, config.name, '2', filename)
    self.df.to_csv(path, encoding='utf-8')
    print(f'Simulation results saved to [{path}]')
    return self.df


class TwoComponentBec(Simulation):

  def __new__(self, config, couplings=None):
    data = OneDimensionalTwoComponentData(
        os.path.join(config.path_to_results, config.name, '2')
        )
    data.add(one_dimensional_bec_2_component(config, couplings))
    return data


class TwoDimensionalBec(Simulation):

  def __new__(self, config, coupling):
    data = TwoDimensionalData(
        os.path.join(config.path_to_results, config.name)
        )
    data.add(two_dimensional_bec(config, coupling=coupling))
    return data


class VariableCouplingTwoDimBec(Experiment):

  def __init__(self, config, two_component=False):
    # keep track of config
    self.config = config
    # data holder
    self.data = TwoDimensionalData(
        os.path.join(config.path_to_results, config.name)
        )

  def run(self, coupling_vars=None):
    if coupling_vars is None:
      coupling_vars = self.config.coupling_vars
    for g in tqdm(coupling_vars):
      # run a simulation
      sdata = two_dimensional_bec(self.config, coupling=g)
      # save data
      self.data.add(sdata)

    return self.data


class VariableCouplingTwoComponentBec(Experiment):

  def __init__(self, config, two_component=False):
    # keep track of config
    self.config = config
    # data holder
    self.data = OneDimensionalTwoComponentData(
        os.path.join(config.path_to_results, config.name, '2')
        )

  def run(self, coupling_vars=None):
    if coupling_vars is None:
      coupling_vars = self.config.coupling_vars
    for g in tqdm(coupling_vars):
      # run a simulation
      sdata = one_dimensional_bec_2_component(self.config, couplings=g.squeeze())
      # save data
      self.data.add(sdata)

    return self.data


def potential_change_bec(config, potential_fn=None, iterations=None):
  # get coupling strength
  coupling = config.coupling
  # Set up lattice
  grid = ts.Lattice1D(config.dim, config.radius)
  # initialize state
  state = ts.State(grid, config.angular_momentum)
  state.init_state(config.wave_function)
  # init potential
  potential = ts.Potential(grid)
  # get potential_fn
  if potential_fn is None:
    potential_fn = config.potential_fn
  potential.init_potential(potential_fn)
  # build hamiltonian with coupling strength `g` and potential `u(.)`
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
    'u' : np.array([ potential_fn(x, None) for x in grid.get_x_axis() ]),
    'psi' : psi
    })


def potential_change_experiment(config):
  """Run simulations based on multiple arbitrary potentials"""
  df = pd.DataFrame()
  for i, fn in enumerate(tqdm(config.potential_fns)):
    data = potential_change_bec(config, fn)
    data['i'] = np.array(len(data['x']) * [ i ])
    df = df.append(data, ignore_index=True)
  filepath = os.path.join(config.path_to_results, config.name, 'sim.csv')
  df.to_csv(filepath, encoding='utf-8')
