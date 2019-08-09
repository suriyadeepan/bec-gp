import trottersuzuki as ts
import numpy as np

from neuralbec import utils
from tqdm import tqdm


class SimulatedData:

  def __init__(self):
    pass


class OneDimensionalData(SimulatedData):

  def __init__(self):
    self.df = utils.to_df({ 'x' : [], 'psi' : [], 'g' : [] })

  def add(self, df):
    self.df = self.df.append(df, ignore_index=True)


class Simulation:

  def __init__(self):
    pass


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


class OneDimensionalBec(Simulation):

  def __init__(self, config, coupling=None):
    # keep track of config
    self.config = config
    # get coupling strength
    self.coupling = coupling if coupling else config.coupling
    # Set up lattice
    self.grid = ts.Lattice1D(config.dim, config.radius)
    # initialize state
    self.state = ts.State(self.grid, config.angular_momentum)
    self.state.init_state(config.wave_function)
    # init potential
    potential = ts.Potential(self.grid)
    potential.init_potential(config.potential_fn)  # harmonic potential
    # build hamiltonian with coupling strength `g`
    hamiltonian = ts.Hamiltonian(self.grid, potential, 1., self.coupling)
    # setup solver
    self.solver = ts.Solver(self.grid, self.state, hamiltonian, config.time_step)

  def simulate(self, iterations=None):
    iterations = self.config.iterations if not iterations else iterations
    # Evolve the system
    self.solver.evolve(iterations, False)
    # Compare the calculated wave functions w.r.t. groundstate function
    # psi = np.sqrt(state.get_particle_density()[0])
    psi = self.state.get_particle_density()[0]
    # psi / psi_max
    psi = psi / max(psi)
    # save data
    self.data = utils.to_df({
      'x' : self.grid.get_x_axis(),
      'g' : np.ones(psi.shape) * self.coupling,
      'psi' : psi
      })

    return self.data


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
