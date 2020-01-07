from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from neuralbec.simulation import OneDimensionalData
from neuralbec.simulation import OneDimensionalTwoComponentData
from neuralbec.simulation import TwoDimensionalData

from neuralbec.sampling import weighted_sample, random_sample

import pickle
import os


class GPApproximation:

  def __init__(self, config=None):
    if config._type == 'one-component':
      print('Choosing Kernel for one-component BEC')
      kernel = C(1.0, (1e-3, 1e3)) * RBF([5, 5], (1e-2, 1e2))
    elif config._type == 'two-component':
      print('Choosing Kernel for two-component BEC')
      kernel = C(1.0, (1e-3, 1e3)) * RBF([5, 5, 5, 5], (1e-2, 1e2))
    elif config._type == 'two-dim':
      print('Choosing Kernel for 2D BEC')
      kernel = C(1.0, (1e-3, 1e3)) * RBF([5, 5, 5], (1e-2, 1e2))

    # kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
    self.config = config

  def fit(self, X, y):
    self.gp.fit(X, y)

  def evaluate(self, X, y):
    self.error = ((self.gp.predict(X) - y) ** 2).sum() / len(y)
    return self.error

  def predict(self, X):
    out = self.gp.predict(X, return_std=True)
    return out

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


def fit(config, ssc, testset=None, sampling_method=None):
  if sampling_method == 'weighted':
    sample = weighted_sample
  else:  #  sampling_method == 'random':
    sample = random_sample

  # read data from file
  df = OneDimensionalData(
        os.path.join(config.path_to_results, config.name)
        ).load()
  assert len(df) > 0
  dataset = sample(df, ssc * 2)
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
  D = OneDimensionalData
  if config._type == 'twod':
    D = TwoDimensionalData
  df = D(os.path.join(config.path_to_results, config.name)).load()
  assert len(df) > 0
  return df.sample(ssc)


def fit2(config, ssc, testset=None):
  # read data from file
  df = OneDimensionalTwoComponentData(
        os.path.join(config.path_to_results, config.name, '2')
        ).load()
  assert len(df) > 0
  dataset = weighted_sample(df, ssc * 2)
  trainset = dataset[:ssc]
  testset = dataset[ssc:] if testset is None else testset
  # instantiate model
  model = config.model(config=config)
  # fit model
  model.fit(trainset[['x', 'g11', 'g12', 'g22']], trainset[['psi1', 'psi2']])
  # evaluate on test set
  error = model.evaluate(testset[['x', 'g11', 'g12', 'g22']], testset[['psi1', 'psi2']])
  print(f'Error : [ {model.error} ]')
  # write model to disk
  model.save(
      os.path.join(config.path_to_results, config.name, '2', f'ssc={ssc}.model')
      )

  return model


def fit2d(config, ssc, testset=None):
  # read data from file
  df = TwoDimensionalData(
        os.path.join(config.path_to_results, config.name)
        ).load()
  assert len(df) > 0
  dataset = weighted_sample(df, ssc * 2)
  trainset = dataset[:ssc]
  testset = dataset[ssc:] if testset is None else testset
  # instantiate model
  model = config.model(config=config)
  # fit model
  model.fit(trainset[['x', 'y', 'g']], trainset.psi)
  # evaluate on test set
  error = model.evaluate(testset[['x', 'y', 'g']], testset.psi)
  print(f'Error : [ {model.error} ]')
  # write model to disk
  model.save(
      os.path.join(config.path_to_results, config.name, f'ssc={ssc}.model')
      )

  return model
