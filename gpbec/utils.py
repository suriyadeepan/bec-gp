from itertools import combinations
import trottersuzuki as ts
import numpy as np
import pandas as pd


def generate_proposals(inputs):
  inputs_ = []
  for name in inputs:
    inputs_.extend([ (name, value) for value in inputs[name] ])
  # combinations
  combinations_ = list(combinations(inputs_, len(inputs)))
  # remove absurd combinations
  valid_combinations = []
  for combination_ in combinations_:
    # check if the combination is valid
    names = [ item[0] for item in combination_ ]
    if len(set(names)) == len(combination_):
      valid_combinations.append(combination_)

  return [ { name : value for name, value in combo }
      for combo in set(valid_combinations) ]


def resolve_proposal(proposal, params):
  return { name : value if name not in proposal else proposal[name]
      for name, value in params.items() }


def sim2c(params):
  # get coupling strength
  g11, g12, g22 = params['g11'], params['g12'], params['g22']
  # omega
  omega = params['omega']
  # Set up lattice
  grid = ts.Lattice1D(params['dim'], params['radius'])
  # initialize state
  # Create first-component system's state
  state_1 = ts.GaussianState(grid, 1.)
  # Create second-component system's state
  state_2 = ts.GaussianState(grid, 1.)
  # init potential
  potential_1 = ts.Potential(grid)
  potential_1.init_potential(params['potential'])
  potential_2 = ts.Potential(grid)
  potential_2.init_potential(params['potential'])
  # build hamiltonian with coupling strength `g1`, `g2`, `g12`
  hamiltonian = ts.Hamiltonian2Component(grid, potential_1, potential_2,
    _coupling_a=g11, coupling_ab=g12, _coupling_b=g22,
    _omega_r=omega)
  # setup solver
  solver = ts.Solver(grid, state_1, hamiltonian, params['time_step'],
      State2=state_2)
  # get iterations
  iterations = params['iterations']
  # Evolve the system
  solver.evolve(iterations, False)
  # Compare the calculated wave functions w.r.t. groundstate function
  # psi = np.sqrt(state.get_particle_density()[0])
  psi1 = state_1.get_particle_density()
  psi2 = state_2.get_particle_density()
  assert psi1.shape == (1, params['dim'])
  assert psi2.shape == (1, params['dim'])
  # psi / psi_max
  psi1 = psi1[0] / max(psi1[0])
  psi2 = psi2[0] / max(psi2[0])
  # save data
  return { 'x' : grid.get_x_axis(), 'psi1' : psi1, 'psi2' : psi2 }


def sim1c(params):
  # get coupling strength
  coupling = params['g']
  # Set up lattice
  grid = ts.Lattice1D(params['dim'], params['radius'])
  # initialize state
  state = ts.State(grid)  # , params['angular_momentum'])
  state.init_state(params['wave_function'])
  # init potential
  potential = ts.Potential(grid)
  potential.init_potential(params['potential'])
  # build hamiltonian with coupling strength `g`
  hamiltonian = ts.Hamiltonian(grid, potential, 1., coupling)
  # setup solver
  solver = ts.Solver(grid, state, hamiltonian, params['time_step'])
  iterations = params['iterations']
  # Evolve the system
  solver.evolve(iterations, False)
  # Compare the calculated wave functions w.r.t. groundstate function
  # psi = np.sqrt(state.get_particle_density()[0])
  psi = state.get_particle_density()[0]
  # psi / psi_max
  psi = psi / max(psi)
  # save data
  return { 'x' : grid.get_x_axis(), 'psi' : psi }


def resolve_sim_fn(params):
  if 'g' in params:
    return sim1c
  elif 'g11' in params and 'g12' in params and 'g22' in params:
    return sim2c


def postsim_proc(results, proposal):
  dframe = results
  # add information from proposal
  for name, value in proposal.items():
    dframe[name] = np.ones_like(results['x']) * value
  return pd.DataFrame(dframe)


def freeze(d):
  return frozenset(d.items())
