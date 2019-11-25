import numpy as np


def gaussian_disorder_potential(sigma_d=0.39, low=-20, high=20, step=0.002):

  def gaussian_disorder_fn(x, y):
    # sample Ai from uniform [-20, 20]
    Xi = np.arange(-20., 0., 0.002)  # -20 : 20 : 2e-3
    assert Xi.shape[0] == 1e4
    # Ai = np.random.uniform(-20, 20, Xi.shape[0])
    Ai = np.random.uniform(-20, 20, Xi.shape[0])
    exp_term = np.exp( - np.square(x - Xi) / ( 2 * np.square(sigma_d) ) )
    return (Ai * exp_term).sum()

  return gaussian_disorder_fn
