from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import pickle


class GPApproximation:

  def __init__(self, name=None):
    kernel = C(1.0, (1e-3, 1e3)) * RBF([5, 5], (1e-2, 1e2))
    self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
    assert name
    self.name = name

  def fit(self, X, y):
    self.gp.fit(X, y)

  def predict(self, X):
    return self.gp.predict(X, return_std=True)

  def save(self, name=None):
    name = name if name else self.name
    pickle.dump(self, open(f'results/model_{name}.sav', 'wb'))
