import pandas as pd
import numpy as np
import math

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


def build_visuals_from_file2(filename, save_to_file=False, overlay=False):

  if 'sim.csv' in filename:
    df = pd.read_csv(filename)
    g11 = df.g11.unique()[0]
    g12 = df.g12.unique()[0]
    g22 = df.g22.unique()[0]
    plotly_plot_wave_fn2((g11, g12, g22), df.x, df.psi1, df.psi2)
  # elif 'sim_prediction.csv' in filename:


def build_2d_visuals_from_file(filename, save_to_file=False):

  if 'sim_g=' in filename:
    df = pd.read_csv(filename)
    assert len(df.g.unique()) == 1
    d = int(math.sqrt(df.psi.shape[0]))
    plotly_plot_2d_wave_fn(df.g.iloc[0], x=df.x.unique(), y=df.y.unique(),
        psi=df.psi.to_numpy().reshape(d, d))


def plotly_plot_2d_wave_fn(coupling, x, y, psi):
  fig = go.Figure(data=go.Heatmap(z=psi, x=x, y=y))
  fig.layout.title.text = f'g = {coupling}'
  fig.update_layout(width=850, height=720)
  fig.show()


def render_prediction_plot2d(model, sample):
  fig = make_subplots(rows=1, cols=2, start_cell="top-left",
      subplot_titles=['Ground Truth', 'Prediction'])
  d = int(math.sqrt(sample.psi.shape[0]))
  ground_truth = go.Heatmap(x=sample.x.unique(), y=sample.y.unique(),
      z=sample.psi.to_numpy().reshape(d, d))
  # model prediction
  y_hat, sigma = model.predict(sample[['x', 'y', 'g']])
  y_hat = y_hat / y_hat.max()
  prediction = go.Heatmap(x=sample.x.unique(), y=sample.y.unique(),
      z=y_hat.reshape(d, d))  # , showlegend=False)
  fig.add_trace(ground_truth, row=1, col=1)
  fig.add_trace(prediction, row=1, col=2)
  fig.update_layout(width=1250, height=600)
  fig.show()


def plotly_plot_wave_fn2(couplings, x, psi1, psi2):
  g11, g12, g22 = couplings
  plots = [ go.Scatter(x=x, y=psi1, name='ψ₁'),
      go.Scatter(x=x, y=psi2, name='ψ₂') ]
  # setup title
  title = f'g11 : {g11}, g12 : {g12}, g22 : {g22}'
  fig = go.Figure(data=plots)
  fig = plotly_layout_setup2(fig, title=title)
  fig.show()


def plotly_plot_wave_fn(g, x, psi):
  fig = go.Figure(data=go.Scatter(x=x, y=psi))
  fig = plotly_layout_setup(fig, title=f'g={g}')
  fig.show()


def plotly_layout_setup2(fig, title):
  fig = plotly_layout_setup(fig, title)
  fig.layout.yaxis.title.text = 'ψ₁, ψ₂'
  return fig


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


def make_prediction_plot(model, sample, colors=('#1f77b4', '#d62728')):
  # run sample through model
  y_pred, sigma = model.predict(sample[['x', 'g']])
  g = sample.g.unique()[0]
  # get colors
  c1, c2 = colors
  # plots
  data = []
  data.extend( [ go.Scatter(
    x=sample.x, y=y_pred, line=dict(color=c1), name=f'(g={g}) prediction'),
    go.Scatter(x=np.concatenate(
      [sample.x.to_numpy(), sample.x.to_numpy()[::-1]]), y=np.concatenate([y_pred - 1.9600 * sigma, ]), line=dict(color=c1), fill='tonexty', opacity=0.1, name=f'(g={g}) 95% confidence interval')] )
  if 'psi' in sample:
    data.append(go.Scatter(x=sample.x, y=sample.psi,
        line=dict(color=c2, dash='dot'), name=f'(g={g}) observations'))

  return data, np.square(y_pred - sample.psi).sum() / len(sample.psi)


def make_prediction_plot2(model, sample, colors=('#1f77b4', '#d62728')):
  # run forward
  y_pred, sigma = model.predict(sample[['x', 'g11', 'g12', 'g22']])
  y_pred = y_pred / y_pred.max()
 # get colors
  c1, c2 = colors

  def plot_fn(psi, psi_hat, showlegend=False):
    data = []
    data.append(go.Scatter(x=sample.x, y=psi,
      line=dict(color=c2, dash='dot'), name='observations',
      showlegend=showlegend))
    data.append(go.Scatter(x=sample.x, y=psi_hat,
      line=dict(color=c1), name='prediction',
      showlegend=showlegend))
    data.append(go.Scatter(x=np.concatenate(
      [sample.x.to_numpy(), sample.x.to_numpy()[::-1]]), y=np.concatenate([psi_hat - 1.9600 * sigma, ]), line=dict(color=c1), fill='tonexty', opacity=0.1, name='95% confidence interval', showlegend=showlegend))

    return data

  # plot psi 1
  data1 = plot_fn(sample.psi1, y_pred[:, 0], showlegend=True)
  # plot psi 2
  data2 = plot_fn(sample.psi2, y_pred[:, 1])
  # create figure
  # fig = make_subplots(rows=m, cols=n, start_cell="top-left",
  # subplot_titles=['g = {0:.2f}'.format(g) for g, df in data]
  # )

  return (data1, data2)


def render_prediction_plot(model, sample, save_to_file=False):
  data, error = make_prediction_plot(model, sample)
  fig = go.Figure(data)
  print(f'MSE : {error}')
  if save_to_file:
    fig.write_image(f'{save_to_file}.png', scale=4)
    return
  fig.show()


def make_title2(sample):
  g11 = sample.g11.unique()[0]
  g12 = sample.g12.unique()[0]
  g22 = sample.g22.unique()[0]
  return f'g11 : {g11}, g12 : {g12}, g22 : {g22}'

def render_prediction_plot2_overlayed(model, sample, save_to_file=False):
  title = make_title2(sample)
  data1, data2 = make_prediction_plot2(model, sample)
  fig = go.Figure(data2)
  if not save_to_file:
    fig.show()
    return
  fig.write_image(f'{save_to_file}.png', height=750, width=1100, scale=4)

def render_prediction_plot2(model, sample, save_to_file=False):
  title = make_title2(sample)
  data1, data2 = make_prediction_plot2(model, sample)
  assert len(data1) == 3 and len(data2) == 3
  fig = make_subplots(rows=2, cols=2, start_cell="top-left",
      subplot_titles=[None, None])
  fig.add_trace(data1[0], row=1, col=1)
  fig.update_xaxes(title_text='x', row=1, col=1)
  fig.update_yaxes(title_text='ψ₁', row=1, col=1)
  fig.add_trace(data2[0], row=2, col=1)
  fig.update_xaxes(title_text='x', row=2, col=1)
  fig.update_yaxes(title_text='ψ₂', row=2, col=1)
  for d in data1[1:]:
    fig.add_trace(d, row=1, col=2)
    fig.update_xaxes(title_text='x', row=1, col=2)
    fig.update_yaxes(title_text='ψ₁', row=1, col=2)
  for d in data2[1:]:
    fig.add_trace(d, row=2, col=2)
    fig.update_xaxes(title_text='x', row=2, col=2)
    fig.update_yaxes(title_text='ψ₂', row=2, col=2)
  if not save_to_file:
    fig.show()
    return
  fig.write_image(f'{save_to_file}.png', scale=4, height=1000, width=1200)


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


def plot_multi_overlay(data, save_to_file=False):
  plots = []
  for g, df in data:
    plots.append(go.Scatter(x=df.x, y=df.y, name=f'g={g}'))
  fig = go.Figure(plots)
  fig.update_layout(legend=go.layout.Legend(x=0.9, y=0.95))
  fig.layout.yaxis.title.text = 'ψ / max(ψ)'
  fig.layout.xaxis.title.text = 'x'
  if not save_to_file:
    fig.show()
    return
  fig.write_image(f'{save_to_file}.png', scale=4)


def plot_predictions_overlay(data, save_to_file=False):
  fig = go.Figure(data)
  # fig.update_layout(legend=go.layout.Legend(x=0.8, y=0.95))
  fig.layout.yaxis.title.text = 'ψ'
  fig.layout.xaxis.title.text = 'x'
  if not save_to_file:
    fig.show()
    return
  fig.write_image(f'{save_to_file}.png', scale=4, height=700, width=1050)
