from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import pickle
import os


class GPApproximation:

  def __init__(self, config=None):
    kernel = C(1.0, (1e-3, 1e3)) * RBF([5, 5], (1e-2, 1e2))
    self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
    self.name = config.name
    self.config = config

  def fit(self, X, y):
    self.gp.fit(X, y)

  def evaluate(self, X, y):
    self.error = ((self.gp.predict(X) - y) ** 2).sum() / len(y)
    return self.error

  def predict(self, X):
    return self.gp.predict(X, return_std=True)

  def save(self, name=None):
    assert self.config
    name = name if name else self.name
    pickle.dump(self,
        open(os.path.join(self.config.path, f'model_{name}.sav'), 'wb'))

  def load(self, name=None):
    name = name if name else self.config.name
    with open(os.path.join(self.config.path, f'model_{name}.sav'), 'rb') as bf:
      self = pickle.load(bf)
    return self

  @property
  def stats(self):
    if self.error:
      return { 'Error' : self.error }
