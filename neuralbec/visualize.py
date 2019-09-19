import pandas as pd
import numpy as np
import os
import pickle

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt
plt.style.use('ggplot')


def build_visuals_from_file(filename, save_to_file=False, overlay=False):
  plot_fn = plot_multi_overlay if overlay else plot_multi_grid
  # figure out the type of file
  if 'sim_g=' in filename:
    df = pd.read_csv(filename)
    assert len(df.g.unique()) == 1
    plotly_plot_wave_fn(df.g.iloc[0], df.x, df.psi)
  elif 'sim.csv' in filename:
    data = sample_data_from_file(filename, n=9)
    plot_fn(data)
  elif 'sim_prediction.csv' in filename:
    data = read_data_from_file(filename)
    plot_fn(data)


def plotly_plot_wave_fn(g, x, psi):
  fig = go.Figure(data=go.Scatter(x=x, y=psi))
  fig = plotly_layout_setup(fig, title=f'g={g}')
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


def make_prediction_plot(model, sample):
  # run sample through model
  y_pred, sigma = model.predict(sample[['x', 'g']])
  # get `g` the scalar
  # plots
  data = []
  if 'psi' in sample:
    data.append(go.Scatter(x=sample.x, y=sample.psi,
        line=dict(color='red', dash='dot'), name='observations'))
  data.extend( [ go.Scatter(
    x=sample.x, y=y_pred, line=dict(color='blue'), name='prediction'),
    go.Scatter(x=np.concatenate(
      [sample.x.to_numpy(), sample.x.to_numpy()[::-1]]), y=np.concatenate([y_pred - 1.9600 * sigma, ]), line=dict(color='blue'), fill='tonexty', opacity=0.1, name='95% confidence interval')] )

  return data


def render_prediction_plot(model, sample):
  fig = go.Figure(make_prediction_plot(model, sample))
  fig.show()


def read_data_from_file(filename):
  df = pd.read_csv(filename)
  x = df.x.unique()
  couplings = df.g.unique()
  psi_values = [ df.loc[df.g == g, 'psi'].to_numpy() for g in couplings ]
  return [ (g, pd.DataFrame({ 'x' : x, 'y' : psi }))
      for g, psi in zip(couplings, psi_values) ]


def sample_data_from_file(filename, n):
  n = 9 if not n else n
  df = pd.read_csv(filename)
  x = df.x.unique()
  g = df.g.unique()
  g_sub_sampled = g[np.random.randint(0, len(g), (n, ))]
  psi_sub_sampled = [ df.loc[df.g == g, 'psi'].to_numpy() for g in g_sub_sampled ]
  return [ (g, pd.DataFrame({ 'x' : x, 'y' : psi }))
      for g, psi in zip(g_sub_sampled, psi_sub_sampled) ]


def sample_structured_data_from_file(filename, n):
  df = pd.read_csv(filename, sep='\t')
  # x = df.x.unique()
  g = df.g.unique()
  g_sub_sampled = g[np.random.randint(0, len(g), (n, ))]
  return [ df.loc[df.g == g, ['g', 'x', 'psi']] for g in g_sub_sampled ]


def plot_multi_grid(data):
  n = 3  # 3-column plot
  m = len(data) // n  # num rows
  fig = make_subplots(rows=m, cols=n, start_cell="top-left",
      subplot_titles=['g = {0:.2f}'.format(g) for g, df in data]
      )
  k = 0
  for i in range(m):
    for j in range(n):
      g, df = data[k]
      k = k + 1
      fig.add_trace(go.Scatter(x=df.x, y=df.y),
          row=1 + i, col=1 + j
          )
      if i == m - 1:
        # set axes names
        fig.update_xaxes(title_text='x', row=1 + i, col=1 + j)
      if j == 0:
        fig.update_yaxes(title_text='ψ', row=1 + i, col=1 + j)
  # show off
  fig.layout.showlegend = False
  fig.show()


def plot_multi_overlay(data):
  plots = []
  for g, df in data:
    plots.append(go.Scatter(x=df.x, y=df.y))
  fig = go.Figure(plots)
  #fig.layout.showlegend = False
  fig.show()


def plot_predictions_overlay(data):
  fig = go.Figure(data)
  fig.show()
