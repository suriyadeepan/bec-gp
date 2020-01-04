from gpbec.sim import Simulator
from gpbec.params import get_params
import numpy as np


if __name__ == '__main__':
  # make params
  params = get_params(g=10)
  # run simulation
  sim = Simulator(params, inputs={ 'g' : np.random.uniform(0, 10, (100,)) })
  out = sim()
