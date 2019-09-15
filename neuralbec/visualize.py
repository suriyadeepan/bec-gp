import pandas as pd
import numpy as np
import os
import pickle

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


def sample_data_from_file(filename, n):
  n = 9 if not n else n
  df = pd.read_csv(filename, sep='\t')
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


def plot_multi(data):
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


def plot_prediction_multi(data):
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


def plot_from_file(filename, num_plots=None):
  if 'g=' in filename:  # single simulation plot
    df = pd.read_csv(filename, sep='\t')
    assert len(df.g.unique()) == 1
    plotly_plot_wave_fn(df.g.iloc[0], df.x, df.psi)
  else:  # multi simulation plot
    data = sample_data_from_file(filename, n=num_plots)
    plot_multi(data)


def plot(config):
  for f in [ f'bec_g={config.coupling}.{config.name}.csv', f'bec_{config.name}.csv']:
    path = os.path.join(config.path, f)
    if os.path.exists(path):
      plot_from_file(path, num_plots=config.num_plots)


def plot_prediction(config):
  # read saved model from file
  with open(os.path.join(
    config.path, f'model_{config.name}.sav'), 'rb') as bf:
    model = pickle.load(bf)

  # get data samples from file
  samples = sample_structured_data_from_file(
      os.path.join(config.path, f'bec_{config.name}.csv'),
      config.num_prediction_plots)

  plot_sample_prediction(model, samples)


def make_prediction_plot(model, sample):
  # run sample through model
  y_pred, sigma = model.predict(sample[['x', 'g']])
  # get `g` the scalar
  # plots
  p1 = go.Scatter(x=sample.x, y=sample.psi,
      line=dict(color='red', dash='dot'), name='observations')
  p2 = go.Scatter(x=sample.x, y=y_pred, line=dict(color='blue'), name='prediction')
  p3 = go.Scatter(x=np.concatenate(
    [sample.x.to_numpy(), sample.x.to_numpy()[::-1]]),
                  y=np.concatenate([y_pred - 1.9600 * sigma, ]),
                  line=dict(color='blue'),
                  fill='tonexty', opacity=0.1,
                  name='95% confidence interval')
  # combine plots
  data = [p2, p3, p1]

  return data


def plot_sample_prediction(model, samples):
  n = 3
  m = len(samples) // n
  # setup figure
  fig = make_subplots(rows=m, cols=n, start_cell="top-left",
      subplot_titles=[ 'g = {0:.2f}'.format(sample.g.iloc[0])
        for sample in samples ]
      )
  k = 0
  for i in range(m):
    for j in range(n):
      # make prediction plot
      data = make_prediction_plot(model, samples[k])
      k = k + 1
      [ fig.add_trace(p, row=1 + i, col=1 + j) for p in data ]
      # set axes names
      if j == 0:
        fig.update_yaxes(title_text='ψ', row=1 + i, col=1 + j)
      if i == m - 1:
        fig.update_xaxes(title_text='x', row=1 + i, col=1 + j)
  # show off
  fig.layout.showlegend = False
  fig.show()


def plot_sample_prediction_1(model, sample):
  # run sample through model
  y_pred, sigma = model.predict(sample[['x', 'g']])
  # get `g` the scalar
  g = sample.g.iloc[0]
  # plots
  p1 = go.Scatter(x=sample.x, y=sample.psi,
      line=dict(color='red', dash='dot'), name='observations')
  p2 = go.Scatter(x=sample.x, y=y_pred, line=dict(color='blue'), name='prediction')
  p3 = go.Scatter(x=np.concatenate(
    [sample.x.to_numpy(), sample.x.to_numpy()[::-1]]),
                  y=np.concatenate([y_pred - 1.9600 * sigma, ]),
                  line=dict(color='blue'),
                  fill='tonexty', opacity=0.1,
                  name='95% confidence interval')
  # combine plots
  data = [p2, p3, p1]
  # setup layout
  layout = go.Layout(xaxis=dict(title='x'),
                     yaxis=dict(title='ψ'),
                     title=dict(text='g = {:0.2f}'.format(g),
                       x=0.40, xanchor='center')
                    )
  # setup figure
  fig = go.Figure(data=data, layout=layout)
  # show off
  fig.show()
