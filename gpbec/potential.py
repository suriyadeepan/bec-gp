"""List of Potential Functions and Wave Functions"""

import numpy as np


def harmonic(x, y):
  """Harmonic Potential"""
  return 0.5 * (x ** 2 + y ** 2)


def gaussian(x, y):
  """Gaussian Form of Wave Function"""
  return np.exp(-0.5 * (x**2 + y**2)) / np.sqrt(np.pi)


def cosine(x, y):
  """Cosine Potential from Liang et. al."""
  return 0.5 * (x ** 2) + 24 * (math.cos(x)) ** 2


def optical_lat(x, y):
  """Optical Lattice Potential"""
  return (x ** 2) / ( 2 + 12 * (math.sin(4 * x) ** 2) )


def double_well(x, y):
  """Double-Well Potential"""
  return (4. - x**2) ** 2 / 2.
