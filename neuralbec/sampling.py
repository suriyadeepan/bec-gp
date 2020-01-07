import numpy as np
import pandas as pd


def calc_psi_std(df):
  if 'psi' in df:
    return { xi : np.std(df.loc[df.x == xi].psi)
        for xi in df.x.unique() }
  elif 'psi1' in df and 'psi2' in df:
    return { xi : np.std(df.loc[df.x == xi].psi1) + np.std(df.loc[df.x == xi].psi2)
        for xi in df.x.unique() }


def repr_sample(df, n, invert_p=False, lower_threshold=0.075):
  psi_std = calc_psi_std(df)
  df_p = np.zeros((len(df),))
  for i in range(len(df)):
    df_p[i] = psi_std[df.x[i]]
    df_p[i] = max(lower_threshold, df_p[i])
  if invert_p:  # invert probabilities
    df_p = df_p.max() - df_p
  return df.iloc[np.random.choice(
    np.arange(0, len(df)), size=(n,), p=df_p / df_p.sum())]


def weighted_sample(df, n, repr_sample_ratio=0.75):
  print('weighted sampling')
  m = int(n * repr_sample_ratio)
  samples = repr_sample(df, m)
  # random_samples = repr_sample(df, n-m, invert_p=True)  # df.sample(n - m)
  random_samples = df.sample(n - m)
  df_sampled = pd.concat([samples, random_samples])
  return df_sampled.sample(frac=1)  # shuffle


def random_sample(df, n, **kwargs):
  print('random sampling')
  return df.sample(n)
