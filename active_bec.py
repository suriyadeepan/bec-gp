from gpbec.sim import Simulator
from gpbec.params import get_params
import numpy as np


if __name__ == '__main__':
  # make params
  params = get_params(g=10)
  # run simulation
  inputs={ 'g' : np.random.uniform(0, 10, (100,)) }
  # create a simulator
  sim = Simulator(params, inputs)
  # create a learning machine
  lm = LearningMachine(inputs)
  # fit on simulated data
  done = False
  while not done:
    simdata = sim(inputs)
