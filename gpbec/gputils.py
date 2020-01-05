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


def sample_from_evidence(evidence, n):
  pass
