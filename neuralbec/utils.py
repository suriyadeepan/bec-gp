import pickle
import logging


def save(d, filename):
  pickle.dump(d, open(filename, 'wb'))


def get_logger(name):
  # setup logger
  logging.basicConfig(level=logging.INFO)
  return logging.getLogger(name)
