from gpbec import potential as pot


c1 = {
    'dim' : 512, 'radius' : 24,   # space
    'angular_momentum' : 1.,      # angular momentum
    'time_step' : 1e-4, 'g' : 1.,  # coupling strength
    'iterations' : 100000,
    'potential' : pot.harmonic,
    'wave_function' : pot.gaussian,
    }

c2 = {
    'dim' : 300, 'radius' : 20,  # space
    'time_step' : 1e-2,
    'iterations' : 100000,
    'g11' : 1., 'g12' : 1., 'g22' : 1.,  # coupling strengths
    'potential' : pot.harmonic,
    'omega' : -1
    }


def get_params(**kwargs):
  return { name : value if name not in kwargs else kwargs.get(name)
      for name, value in c1.items() }


def get_params2(**kwargs):
  return { name : value if name not in kwargs else kwargs.get(name)
      for name, value in c2.items() }
