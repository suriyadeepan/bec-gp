import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('ggplot')


def plot_wave_fn(g, x, psi):
  plt.plot(x, psi)
  plt.title(f'g={g}')
  plt.show()


def plot_from_file(filename):
  df = pd.read_csv(filename, sep='\t')
  assert len(df.g.unique()) == 1
  plot_wave_fn(df.g.iloc[0], df.x, df.psi)
