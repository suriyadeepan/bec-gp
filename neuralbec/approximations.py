from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from neuralbec.simulation import OneDimensionalData

import pickle
import os


class GPApproximation:

  def __init__(self, config=None):
    kernel = C(1.0, (1e-3, 1e3)) * RBF([5, 5], (1e-2, 1e2))
    self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
    self.config = config

  def fit(self, X, y):
    self.gp.fit(X, y)

  def evaluate(self, X, y):
    self.error = ((self.gp.predict(X) - y) ** 2).sum() / len(y)
    return self.error

  def predict(self, X):
    return self.gp.predict(X, return_std=True)

  def save(self, filename=None):
    filename = filename if filename else self.filename
    pickle.dump(self, open(filename, 'wb'))
    print(f'Model saved to [{filename}]')

  def load(self, filename=None):
    with open(filename, 'rb') as bf:
      self = pickle.load(bf)
    return self

  @property
  def stats(self):
    if self.error:
      return { 'Error' : self.error }


def fit(config, ssc, testset=None):
  # read data from file
  df = OneDimensionalData(
        os.path.join(config.path_to_results, config.name)
        ).load()
  assert len(df) > 0
  dataset = df.sample(ssc * 2)
  trainset = dataset[:ssc]
  testset = dataset[ssc:] if testset is None else testset
  # instantiate model
  model = config.model(config=config)
  # fit model
  model.fit(trainset[['x', 'g']], trainset.psi)
  # evaluate on test set
  error = model.evaluate(testset[['x', 'g']], testset.psi)
  print(f'Error : [ {model.error} ]')
  # write model to disk
  model.save(
      os.path.join(config.path_to_results, config.name, f'ssc={ssc}.model')
      )
  return model


def make_testset(config, ssc):
  df =  OneDimensionalData(
      os.path.join(config.path_to_results, config.name)
      ).load()
  assert len(df) > 0
  return df.sample(ssc)
 
