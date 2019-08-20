import pandas as pd
import numpy as np

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt
plt.style.use('ggplot')


def mat_plot_wave_fn(g, x, psi):
  plt.plot(x, psi)
  plt.title(f'g={g}')
  plt.show()


def plotly_plot_wave_fn(g, x, psi):
  fig = go.Figure(data=go.Scatter(x=x, y=psi))
  fig = plotly_layout_setup(fig, title=f'g={g}')
  fig.show()


def sample_data_from_file(filename):
    df = pd.read_csv(filename, sep='\t')
    x = df.x.unique()
    g = df.g.unique()
    g_sub_sampled = g[np.random.randint(0, len(g), (9, ))]
    psi_sub_sampled = [ df.loc[df.g == g, 'psi'].to_numpy() for g in g_sub_sampled ]
    return [ (g, pd.DataFrame({ 'x' : x, 'y' : psi })) for g, psi in zip(g_sub_sampled, psi_sub_sampled) ]


def plot_multi(data):
  n = 3  # 3-column plot
  m = len(data) // n  # num rows
  fig = make_subplots(rows=m, cols=n, start_cell="top-left",
      subplot_titles=['g = {0:.2f}'.format(g) for g, df in data]
      )
  k = 0
  for i in range(n):
    for j in range(m):
      g, df = data[k]
      k = k + 1
      fig.add_trace(go.Scatter(x=df.x, y=df.y),
          row=1 + i, col=1 + j
          )
  fig.layout.showlegend = False
  fig.show()


def plotly_layout_setup(fig, title):
  fig.layout.title.text = title
  fig.layout.title.x = 0.5
  fig.layout.title.xanchor = 'center'
  fig.layout.xaxis.title.text = 'x'
  fig.layout.xaxis.dtick = 2
  fig.layout.xaxis.title.font.size = 20
  fig.layout.yaxis.title.text = 'ψ'
  fig.layout.yaxis.title.font.size = 20
  return fig


def plot_from_file(filename):
  if 'g=' in filename:  # single simulation plot
    df = pd.read_csv(filename, sep='\t')
    assert len(df.g.unique()) == 1
    plotly_plot_wave_fn(df.g.iloc[0], df.x, df.psi)
  else:  # multi simulation plot
    data = sample_data_from_file(filename)
    plot_multi(data)
