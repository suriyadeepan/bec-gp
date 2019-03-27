import pickle


def save(d, filename):
  pickle.dump(d, open(filename, 'wb'))
