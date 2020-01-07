import numpy as np
import gpytorch
import torch


def t(a):
  if isinstance(a, torch.Tensor):
    return a
  return torch.tensor(a).float()


def torch_em(*ts):
  return [ torch.tensor(t).float() for t in ts ]


def choose_kernel(outputs):
  no = t(outputs).dim()

  if no == 1:  # 2D input -> 1D output
    return (
        gpytorch.means.ConstantMean(),
        gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        )


def num_tasks(outputs):
  outputs = t(outputs)
  no = outputs.dim()
  if no == 2:
    return outputs.shape[1]
  else:
    return 1


def generate_test_io(params):
  if 'g' in params:  # one-component BEC
    return torch.rand(10, 2), torch.rand(10)

  if 'g11' in params and 'g12' in params and 'g21' in params:
    return torch.rand(10, 4), torch.rand(10, 2)

  raise Exception('Check your parameters')


def generate_uniform_inputs(params, inputs):
  uinputs = []
  x_steps = abs((params['x.high'] - params['x.low']) / params['x.step'])
  # iterate through variable names
  for name in inputs:
    # get bounds
    lb, ub = params[f'{name}.low'], params[f'{name}.high']
    # find step size for variable generation
    step = abs((lb - ub) / x_steps)
    # generate uniform inputs
    #  add to list
    uinputs.append(torch.arange(lb, ub, step))
    # TODO : verify lengths are equals
        
  return torch.stack(uinputs).transpose(1, 0)


def sample_from_sim_results(simdata, inputs, outputs, n=100):
  n_simset = n // len(simdata)  # include weights
  examples = []
  for _, simset in simdata.items():
    idx = np.random.randint(0, len(simset), n_simset)
    samples = simset[inputs + outputs].iloc[idx].values
    examples.append(samples)
  return np.concatenate(examples, axis=0)
