import pickle
import logging


def save(d, filename):
  pickle.dump(d, open(filename, 'wb'))


def get_logger(name):
  # setup logger
  logging.basicConfig(level=logging.INFO)
  return logging.getLogger(name)


def split_dataset(dataset, ratio=0.8):
  """Split dataset into train set and test set

  Parameters
  ----------
  dataset : list
    List of data points
  ratio : float
    train/test split ratio

  Returns
  -------
  tuple
    train and test set lists
  """
  inputs, outputs = dataset
  n = len(inputs)
  m = int(n * 0.8)
  return ( (inputs[:m], outputs[:m]), (inputs[m:], outputs[m:]) )
