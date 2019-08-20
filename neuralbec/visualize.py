import pandas as pd

from matplotlib import pyplot as plt
import plotly.graph_objs as go
plt.style.use('ggplot')


def mat_plot_wave_fn(g, x, psi):
  plt.plot(x, psi)
  plt.title(f'g={g}')
  plt.show()


def plotly_plot_wave_fn(g, x, psi):
  fig = go.Figure(data=go.Scatter(x=x, y=psi))
  fig.update_layout(
    title_text=f'g = {g}',
  )
  fig.show()


def plot_from_file(filename):
  df = pd.read_csv(filename, sep='\t')
  assert len(df.g.unique()) == 1
  plotly_plot_wave_fn(df.g.iloc[0], df.x, df.psi)
