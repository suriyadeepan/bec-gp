import numpy as np


def gaussian_disorder_potential(sigma_d=0.39, low=-20, high=20, step=0.002):

  def gaussian_disorder_fn(x, y):
    # sample Ai from uniform [-20, 20]
    Xi = np.arange(-20., 0., 0.002)  # -20 : 20 : 2e-3
    assert Xi.shape[0] == 1e4
    # Ai = np.random.uniform(-20, 20, Xi.shape[0])
    Ai = np.random.uniform(-20, 20, Xi.shape[0])
    exp_term = np.exp( - np.square(x - Xi) / ( 2 * np.square(sigma_d) ) )
    return (Ai * exp_term).sum()

  return gaussian_disorder_fn


def generatepot(style, param, bins):
  """0=step,1=linear,2=fourier; 0-1 "jaggedness" scale"""
  cosval = np.cos([[np.pi*i*j/bins for i in range(1,bins)] for j in range(1,bins//2)])
  sinval = np.sin([[np.pi*i*j/bins for i in range(1,bins)] for j in range(1,bins//2)])
  mu = 1. + bins * param #mean number of jump points for styles 0 + 1
  forxp = 2.5 - 2 * param #fourier exponent for style 2
  scale = 5.0 * (np.pi*np.pi*0.5) # energy scale
  if style < 2:
    dx = bins/mu
    xlist = [-dx/2]
    while xlist[-1] < bins:
      xlist.append(xlist[-1]+dx*subexp(1.))
    vlist = [scale*subexp(2.) for k in range(len(xlist))]
    k = 0
    poten = []
    for l in range(1, bins):
      while xlist[k+1] < l:
        k = k + 1
      if style == 0:
        poten.append(vlist[k])
      else:
        poten.append(vlist[k]+(vlist[k+1]-vlist[k])*(l-xlist[k])/(xlist[k+1]-xlist[k]))
  else:
    sincoef = [(2*np.random.randint(2)-1.)*scale*subexp(2.)/np.power(k,forxp) for k in range(1,bins//2)]
    coscoef = [(2*np.random.randint(2)-1.)*scale*subexp(2.)/np.power(k,forxp) for k in range(1,bins//2)]
    zercoef = scale*subexp(2.)
    poten = np.maximum(np.add(np.add(np.matmul(sincoef,sinval),np.matmul(coscoef,cosval)),zercoef),0).tolist()
  return poten


def get_potential_fn(jaggedness, X, style=0):
  poten = generatepot(style, jaggedness, 1 + X.shape[0])
  assert X.shape[0] == len(poten), (X.shape[0], len(poten))
  poten_dict = { xi : pi for xi, pi in zip(X, poten) }
  def fn(x, y):
    return poten_dict[x]

  return fn
